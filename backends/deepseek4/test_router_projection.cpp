// BF16 router on CUDA, with an independent FP64 dot-product reference.
// CPU dot products only verify the fixture; model computation stays on CUDA.
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
#include <map>
#include <set>
#include <stdexcept>
#include <vector>
using Json=nlohmann::ordered_json;
using Floats=std::vector<float>;
static void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
static void env(const char *name,const char *value){
#ifdef _WIN32
    _putenv_s(name,value);
#else
    setenv(name,value,1);
#endif
}
static const char *test_mode(){const char *s=std::getenv("STRATA_DS4_ROUTER_TEST_MODE");return s?s:"1";}
static void mode(bool enabled){env("STRATA_DS4_ROUTER_SAMPLE_MMVF",enabled?test_mode():"0");}
static Floats values(size_t n,uint32_t seed,float scale){
    Floats a(n);for(auto &x:a){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;x=(float(seed&0xffffff)/8388607.5f-1)*scale;}return a;
}
static bool exact(const Floats &a,const Floats &b){return a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*sizeof(float));}
template<class T> static Json difference(const Floats &a,const std::vector<T> &b){
    require(a.size()==b.size() && !a.empty(),"bad reference shape");double max=0,sum=0;size_t unequal=0;
    for(size_t i=0;i<a.size();++i){require(std::isfinite(a[i]) && std::isfinite(b[i]),"nonfinite output");
        const double d=double(a[i])-double(b[i]);max=std::max(max,std::abs(d));sum+=d*d;unequal+=d!=0;}
    return {{"max_abs",max},{"rms",std::sqrt(sum/a.size())},{"unequal",unequal}};
}
struct Matrix {
    int k,m;ggml_type type;Floats w;std::vector<uint8_t> bytes;
    Matrix(int k_=4096,int m_=256,ggml_type type_=GGML_TYPE_BF16):k(k_),m(m_),type(type_),
        w(values(size_t(k)*m,0xa453,0.125f)),bytes(ggml_row_size(type,k)*m){
        if(type==GGML_TYPE_F32)std::memcpy(bytes.data(),w.data(),bytes.size());
        else {require(ggml_quantize_chunk(type,w.data(),bytes.data(),0,m,k,nullptr)==bytes.size(),"quantize failed");
            ggml_get_type_traits(type)->to_float(bytes.data(),w.data(),int64_t(k)*m);}
    }
    std::vector<double> reference(const Floats &x,int t,bool bf16_input=false)const{
        std::vector<double> input(k),y(m);
        for(int i=0;i<k;++i){const float v=x[size_t(t)*k+i];input[i]=bf16_input?ggml_bf16_to_fp32(ggml_fp32_to_bf16(v)):v;}
        for(int row=0;row<m;++row)for(int i=0;i<k;++i)y[row]+=double(w[size_t(row)*k+i])*input[i];return y;
    }
};
struct Graph {
    ggml_context_ptr ctx;ggml_backend_buffer_ptr buffer;ggml_tensor *out;ggml_cgraph *graph;
    Graph(ggml_backend_t gpu,const Matrix &a,const Floats &input,int n,const char *name,int layout=0,bool precise=true):
        ctx(ggml_init({ggml_tensor_overhead()*8+ggml_graph_overhead(),nullptr,true})){
        require(bool(ctx),"metadata allocation failed");
        auto *raw_w=ggml_new_tensor_2d(ctx.get(),a.type,a.k,a.m*(layout==2?2:1));
        auto *w=layout==2?ggml_view_2d(ctx.get(),raw_w,a.k,a.m,2*raw_w->nb[1],0):raw_w;
        ggml_set_name(w,name);
        auto *raw_x=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,a.k*(layout==1?2:1),n);
        auto *x=ggml_view_2d(ctx.get(),raw_x,a.k,n,raw_x->nb[1],0);
        out=ggml_mul_mat(ctx.get(),w,x);if(precise)ggml_prec_set_acc(out,GGML_PREC_F32);require(ggml_backend_supports_op(gpu,out),"unsupported CUDA graph");
        graph=ggml_new_graph(ctx.get());ggml_build_forward_expand(graph,out);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(),gpu));require(bool(buffer),"CUDA allocation failed");
        ggml_backend_buffer_clear(buffer.get(),0);
        const size_t row=ggml_row_size(a.type,a.k);
        for(int m=0;m<a.m;++m)ggml_backend_tensor_set(raw_w,a.bytes.data()+size_t(m)*row,size_t(m)*w->nb[1],row);
        for(int t=0;t<n;++t)ggml_backend_tensor_set(raw_x,input.data()+size_t(t)*a.k,size_t(t)*raw_x->nb[1],a.k*sizeof(float));
    }
    Floats run(ggml_backend_t gpu){
        require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"compute failed");
        Floats y(ggml_nelements(out));ggml_backend_tensor_get(out,y.data(),0,ggml_nbytes(out));return y;
    }
    double time(ggml_backend_t gpu){
        for(int i=0;i<8;++i)require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"warmup failed");
        ggml_backend_synchronize(gpu);const auto begin=std::chrono::steady_clock::now();
        for(int i=0;i<80;++i)require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"timing failed");
        ggml_backend_synchronize(gpu);return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/80;
    }
};
int main(){
    Json report={{"status","error"},{"scope","Router sample-axis MMVF; sampled FP64 and serial, complete common prefix, guards and token isolation"},
#ifdef STRATA_PROJECTION_EXPERIMENT
        {"experiment_compiled",true},
#else
        {"experiment_compiled",false},
#endif
        {"tf32_override",std::getenv("NVIDIA_TF32_OVERRIDE")?std::getenv("NVIDIA_TF32_OVERRIDE"):"unset"},
        {"router_mode",test_mode()},
        {"checks",Json::array()},{"guards",Json::array()}};
    try{
        env("GGML_CUDA_DISABLE_GRAPHS","1");env("STRATA_DS4_PROJECTION_MMQ","0");env("STRATA_DS4_ROUTER_MMVF","1");
        ggml_backend_load_all();auto *dev=ggml_backend_dev_by_name("CUDA0");if(!dev){std::cerr<<"SKIP: CUDA0 required\n";return 77;}
        ggml_backend_ptr gpu(ggml_backend_dev_init(dev,nullptr));require(bool(gpu),"CUDA init failed");
        report["device"]=ggml_backend_dev_description(dev);bool pass=true;
        const bool enabled=report["experiment_compiled"].get<bool>();
        Matrix a;const auto input=values(size_t(a.k)*4097,0xb823,1);
        for(const char *name:{"blk.0.ffn_gate_inp.weight"}){
            std::map<int,Floats> serial;std::map<int,std::vector<double>> reference,rounded;Floats stock371,fixed371;
            for(int n:{1,2,3,4,5,6,7,8,9,15,16,17,32,128,371,409,512,1024,4096}){
                Graph batch(gpu.get(),a,input,n,name);mode(false);auto stock=batch.run(gpu.get());mode(true);auto fixed=batch.run(gpu.get());
                if(!enabled)pass &= exact(stock,fixed);
                Json row={{"name",name},{"tokens",n},{"columns",Json::array()}};
                std::set<int> columns={0,n-1};if(n>282)columns.insert(282);if(n==409)columns.insert(371);
                for(int t:columns){
                    if(!serial.count(t)){
                        Graph one(gpu.get(),a,Floats(input.begin()+size_t(t)*a.k,input.begin()+size_t(t+1)*a.k),1,name);
                        mode(false);serial[t]=one.run(gpu.get());reference[t]=a.reference(input,t);
                        if(!enabled)rounded[t]=a.reference(input,t,true);
                    }
                    Floats col(fixed.begin()+size_t(t)*a.m,fixed.begin()+size_t(t+1)*a.m);
                    auto error=difference(col,reference[t]);
                    const double tolerance=enabled || report["tf32_override"]=="0"?4e-5:0.01;
                    bool ok=error["max_abs"].get<double>()<tolerance;
                    Json column={{"token",t},{"fixed_serial",difference(col,serial[t])},{"fixed_fp64",error}};
                    if(enabled)ok &= exact(col,serial[t]);
                    else {
                        // Stock MMF rounds activations to BF16; compare that
                        // path to its own oracle, without loosening MMVF's bound.
                        const auto bf16=difference(col,rounded[t]);
                        column["fallback_bf16_input_fp64"]=bf16;
                        ok |= bf16["max_abs"].get<double>()<4e-5;
                    }
                    pass &= ok;column["passed"]=ok;row["columns"].push_back(column);
                }
                if(n==371){stock371=stock;fixed371=fixed;}
                if(n==409){Floats sp(stock.begin(),stock.begin()+stock371.size()),fp(fixed.begin(),fixed.begin()+fixed371.size());
                    row["stock_371_409"]=difference(sp,stock371);row["fixed_371_409"]=difference(fp,fixed371);if(enabled)pass &= exact(fp,fixed371);}
                if(n==1 || n==4 || n==371 || n==409 || n==4096){
                    row["stock_us"]=Json::array();row["fixed_us"]=Json::array();
                    for(int repeat=0;repeat<3;++repeat)for(int j=0;j<2;++j){bool f=(repeat+j)%2;mode(f);row[f?"fixed_us":"stock_us"].push_back(batch.time(gpu.get()));}
                }
                report["checks"].push_back(row);
            }
            auto changed=input;std::fill_n(changed.data()+a.k,a.k,0);
            for(bool fixed:{false,true}){
                Graph before(gpu.get(),a,input,9,name),after(gpu.get(),a,changed,9,name);mode(fixed);
                auto b=before.run(gpu.get()),c=after.run(gpu.get());bool ok=true;
                for(size_t i=0;i<c.size();++i)ok &= i>=size_t(a.m) && i<size_t(a.m)*2?c[i]==0:c[i]==b[i];
                pass &= ok;report["guards"].push_back({{"name",name},{"isolation",true},{"fixed",fixed},{"passed",ok}});
            }
        }
        for(int guard=0;guard<10;++guard){
            Matrix a(guard==3?2048:4096,guard==4?128:256,guard==5?GGML_TYPE_F32:guard==9?GGML_TYPE_F16:GGML_TYPE_BF16);
            const char *name=guard==0?"blk.0.unrelated.weight":guard==7?"blk.0.ffn_gate_inp.weight.extra":"blk.0.ffn_gate_inp.weight";
            const int n=guard==6?4097:4;
            Graph g(gpu.get(),a,input,n,name,guard==1?1:guard==2?2:0,guard!=8);mode(false);auto stock=g.run(gpu.get());mode(true);auto fixed=g.run(gpu.get());
            const bool ok=exact(stock,fixed);pass &= ok;report["guards"].push_back({{"case",guard},{"passed",ok}});
        }
        report["status"]=pass?"pass":"fail";
    }catch(const std::exception &e){report["error"]=e.what();}
    ggml_quantize_free();std::cout<<report.dump(2)<<'\n';return report["status"]=="pass"?0:1;
}
