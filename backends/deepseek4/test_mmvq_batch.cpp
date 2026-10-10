// Compare the optional token-batch path against independent single-token CUDA
// results, including routed/broadcast inputs and padded token/route strides.
// Timing reuses fixed device-resident tensors; it is not a model benchmark.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using json=nlohmann::ordered_json;
using Floats=std::vector<float>;
static void require(bool b,const char * msg) {if(!b)throw std::runtime_error(msg);}
static void env(const char * name,const char * value) {
#ifdef _WIN32
    _putenv_s(name,value);
#else
    setenv(name,value,1);
#endif
}
static Floats values(size_t size,uint32_t seed) {
    Floats result(size);
    for(auto & x:result) {seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;x=(float(seed&65535)/32767.5f-1)*0.1f;}
    return result;
}
struct Matrix {
    ggml_type type;
    int k,m,experts=8,used=6;
    std::vector<uint8_t> packed;
    Matrix(ggml_type t,int k_,int m_):type(t),k(k_),m(m_) {
        auto w=values(size_t(k)*m*experts,0x53e001);Floats importance(k,1);
        packed.resize(ggml_row_size(type,k)*m*experts);
        require(ggml_quantize_chunk(type,w.data(),packed.data(),0,m*experts,k,importance.data())==packed.size(),"quantize failed");
    }
};
struct Graph {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * out;
    ggml_cgraph * graph;
    Graph(ggml_backend_t gpu,const Matrix & f,int n,int mode,bool strided,const Floats & input,const std::vector<int32_t> & ids)
        :ctx(ggml_init({ggml_tensor_overhead()*12+ggml_graph_overhead(),nullptr,true})) {
        require(bool(ctx),"metadata allocation failed");
        auto * a=ggml_new_tensor_3d(ctx.get(),f.type,f.k,f.m,mode?f.experts:1);
        const int lanes=mode==2?f.used:1, pad=strided?2:1;
        auto * raw=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,f.k*pad,lanes*pad,n);
        auto * b=mode?ggml_view_3d(ctx.get(),raw,f.k,lanes,n,raw->nb[1],raw->nb[2],0):
                     ggml_view_2d(ctx.get(),raw,f.k,n,raw->nb[2],0);
        auto * route_raw=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,f.used*pad,n);
        auto * route=ggml_view_2d(ctx.get(),route_raw,f.used,n,route_raw->nb[1],0);
        out=mode?ggml_mul_mat_id(ctx.get(),a,b,route):ggml_mul_mat(ctx.get(),a,b);
        require(ggml_backend_supports_op(gpu,out),"unsupported matrix fixture");
        graph=ggml_new_graph(ctx.get());ggml_build_forward_expand(graph,out);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(),gpu));require(bool(buffer),"device allocation failed");
        ggml_backend_buffer_clear(buffer.get(),0);
        ggml_backend_tensor_set(a,f.packed.data(),0,ggml_nbytes(a));
        for(int token=0;token<n;++token) {
            for(int lane=0;lane<lanes;++lane)
                ggml_backend_tensor_set(raw,input.data()+size_t(token*lanes+lane)*f.k,
                    token*raw->nb[2]+lane*raw->nb[1],f.k*sizeof(float));
            ggml_backend_tensor_set(route_raw,ids.data()+token*f.used,token*route_raw->nb[1],f.used*sizeof(int32_t));
        }
    }
    Floats run(ggml_backend_t gpu) {
        require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"matrix compute failed");
        Floats result(ggml_nelements(out));ggml_backend_tensor_get(out,result.data(),0,ggml_nbytes(out));return result;
    }
    double time(ggml_backend_t gpu,int repeats) {
        for(int i=0;i<8;++i)require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"warmup failed");
        ggml_backend_synchronize(gpu);
        auto start=std::chrono::steady_clock::now();
        for(int i=0;i<repeats;++i) require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"benchmark failed");
        ggml_backend_synchronize(gpu);
        return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;
    }
};
int main() {
    json report={{"status","error"},{"scope","synthetic device-resident matvec; CUDA graphs disabled; CPU wall microseconds/op"},{"checks",json::array()}};
    try {
        env("NVIDIA_TF32_OVERRIDE","0");
        env("GGML_CUDA_DISABLE_GRAPHS","1");ggml_backend_load_all();
        auto * device=ggml_backend_dev_by_name("CUDA0");if(!device) {std::cerr<<"SKIP: CUDA0 required\n";return 77;}
        ggml_backend_ptr gpu(ggml_backend_dev_init(device,nullptr));require(bool(gpu),"backend init failed");
        report["device"]=ggml_backend_dev_description(device);
        bool pass=true;
        for(auto shape:std::vector<std::pair<int,int>>{{512,64},{512,65},{4096,2048},{2048,4096}}) {
            for(auto type:{GGML_TYPE_Q8_0,GGML_TYPE_IQ3_XXS,GGML_TYPE_MXFP4,GGML_TYPE_Q6_K}) {
                Matrix f(type,shape.first,shape.second);
                std::cerr<<"MMVQ fixture "<<f.k<<'x'<<f.m<<' '<<ggml_type_name(type)<<'\n';
                for(int n:{1,2,3,4,5,8}) for(int mode:{0,1,2}) for(bool strided:{false,true}) {
                    // Upstream single-token routed small-K can overwrite the
                    // adjacent route for odd M. It is not a valid oracle here.
                    // Odd-M multi-token graphs must stay on the upstream path.
                    if(f.m%32 && mode && n==1)continue;
                    const int lanes=mode==2?f.used:1;
                    const auto input=values(size_t(f.k)*lanes*n,0x53f001);
                    std::vector<int32_t> ids(f.used*n);
                    for(int t=0;t<n;++t) for(int lane=0;lane<f.used;++lane) ids[t*f.used+lane]=(t*3+lane*5)%f.experts;
                    Graph batch(gpu.get(),f,n,mode,strided,input,ids);
                    env("STRATA_DS4_MMVQ_TOKEN_BATCH","0");const auto old=batch.run(gpu.get());
                    env("STRATA_DS4_MMVQ_TOKEN_BATCH","2");const auto actual=batch.run(gpu.get());
                    Floats serial;
                    const bool serial_valid=!(f.m%32 && mode);
                    if(serial_valid)for(int t=0;t<n;++t) {
                        Graph one(gpu.get(),f,1,mode,strided,
                            {input.begin()+size_t(t)*f.k*lanes,input.begin()+size_t(t+1)*f.k*lanes},
                            {ids.begin()+t*f.used,ids.begin()+(t+1)*f.used});
                        auto row=one.run(gpu.get());serial.insert(serial.end(),row.begin(),row.end());
                    }
                    const bool eligible=n>=2 && n<=4 && type!=GGML_TYPE_Q6_K && f.m%32==0;
                    auto identical=[](const Floats &a,const Floats &b) {
                        return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0;
                    };
                    bool finite=std::all_of(actual.begin(),actual.end(),[](float x){return std::isfinite(x);});
                    const bool exact=identical(actual,eligible?serial:old) && finite;
                    double max_diff=0,stock_diff=0;
                    if(serial_valid)for(size_t i=0;i<actual.size();++i) {
                        max_diff=std::max(max_diff,double(std::abs(actual[i]-serial[i])));
                        stock_diff=std::max(stock_diff,double(std::abs(old[i]-serial[i])));
                    }
                    pass &= exact;
                    json row={{"k",f.k},{"m",f.m},{"type",ggml_type_name(type)},{"tokens",n},{"mode",mode},
                        {"strided",strided},{"elements",actual.size()},{"eligible",eligible},{"passed",exact},
                        {"serial_reference_valid",serial_valid},
                        {"candidate_serial_exact",identical(actual,serial)},{"stock_serial_exact",identical(old,serial)},
                        {"candidate_stock_exact",identical(actual,old)},{"candidate_serial_max_diff",max_diff},
                        {"stock_serial_max_diff",stock_diff}};
                    if(eligible && !strided && shape.first!=512) {
                        // Alternate order within one allocation; changing the knob cannot
                        // reuse a captured old CUDA graph because graphs are disabled.
                        std::vector<double> control,optimized;
                        for(int repeat=0;repeat<4;++repeat) for(int j=0;j<2;++j) {
                            const bool enabled=(repeat+j)%2;
                            env("STRATA_DS4_MMVQ_TOKEN_BATCH",enabled?"2":"0");
                            (enabled?optimized:control).push_back(batch.time(gpu.get(),50));
                        }
                        row["control_us"]=control;row["token_batch_us"]=optimized;
                    }
                    report["checks"].push_back(row);
                }
            }
        }
        report["status"]=pass?"pass":"fail";
    } catch(const std::exception & e) {report["error"]=e.what();}
    ggml_quantize_free();std::cout<<report.dump(2)<<'\n';return report["status"]=="pass"?0:1;
}
