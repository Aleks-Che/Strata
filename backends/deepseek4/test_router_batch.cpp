// The router's BF16 weights must not imply BF16 activations. Compare the
// opt-in short-batch MMVF with serial CUDA and independent scalar FP64.
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
static void require(bool ok,const char *msg) { if(!ok)throw std::runtime_error(msg); }
static void env(const char *name,const char *value) {
#ifdef _WIN32
    _putenv_s(name,value);
#else
    setenv(name,value,1);
#endif
}
static Floats values(size_t n,uint32_t seed,float scale) {
    Floats a(n); for(float &x:a) { seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;x=(float(seed&65535)/32767.5f-1)*scale; }return a;
}
struct Matrix {
    int k,m;ggml_type type;Floats floats;std::vector<uint8_t> bytes;
    Matrix(int k_,int m_,ggml_type t):k(k_),m(m_),type(t),floats(values(size_t(k)*m,0xd54b,0.125f)),bytes(ggml_row_size(t,k)*m) {
        for(size_t i=0;i<floats.size();++i) {
            if(t==GGML_TYPE_BF16) { auto h=ggml_fp32_to_bf16(floats[i]);std::memcpy(bytes.data()+2*i,&h,2);floats[i]=ggml_bf16_to_fp32(h); }
            else if(t==GGML_TYPE_F16) { auto h=ggml_fp32_to_fp16(floats[i]);std::memcpy(bytes.data()+2*i,&h,2);floats[i]=ggml_fp16_to_fp32(h); }
            else std::memcpy(bytes.data()+4*i,&floats[i],4);
        }
    }
    std::vector<double> reference(const Floats &input,bool bf16_input) const {
        std::vector<double> out(input.size()/k*m);
        for(size_t t=0;t<input.size()/k;++t)for(int row=0;row<m;++row) {
            double sum=0;
            for(int col=0;col<k;++col) {
                float x=input[t*k+col];if(bf16_input)x=ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
                sum+=double(x)*double(floats[row*k+col]);
            }
            out[t*m+row]=sum;
        }
        return out;
    }
};
struct Graph {
    ggml_context_ptr ctx;ggml_backend_buffer_ptr buffer;ggml_tensor *out;ggml_cgraph *graph;
    Graph(ggml_backend_t gpu,const Matrix &a,const Floats &input,int n,bool strided,bool named,bool precise)
        :ctx(ggml_init({ggml_tensor_overhead()*8+ggml_graph_overhead(),nullptr,true})) {
        require(bool(ctx),"metadata allocation failed");
        auto *w=ggml_new_tensor_2d(ctx.get(),a.type,a.k,a.m);
        ggml_set_name(w,named?"blk.0.ffn_gate_inp.weight":"blk.0.other.weight");
        auto *raw=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,a.k*(strided?2:1),n);
        auto *x=ggml_view_2d(ctx.get(),raw,a.k,n,raw->nb[1],0);
        out=ggml_mul_mat(ctx.get(),w,x);if(precise)ggml_prec_set_acc(out,GGML_PREC_F32);
        require(ggml_backend_supports_op(gpu,out),"unsupported router graph");
        graph=ggml_new_graph(ctx.get());ggml_build_forward_expand(graph,out);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(),gpu));require(bool(buffer),"device allocation failed");
        ggml_backend_buffer_clear(buffer.get(),0);ggml_backend_tensor_set(w,a.bytes.data(),0,a.bytes.size());
        for(int t=0;t<n;++t)ggml_backend_tensor_set(raw,input.data()+t*a.k,t*raw->nb[1],a.k*sizeof(float));
    }
    Floats run(ggml_backend_t gpu) {
        require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"router compute failed");
        Floats out_data(ggml_nelements(out));ggml_backend_tensor_get(out,out_data.data(),0,ggml_nbytes(out));return out_data;
    }
    double time(ggml_backend_t gpu) {
        for(int i=0;i<8;++i)require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"warmup failed");
        ggml_backend_synchronize(gpu);auto start=std::chrono::steady_clock::now();
        for(int i=0;i<100;++i)require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"timing failed");
        ggml_backend_synchronize(gpu);return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/100;
    }
};
template<class T> static double diff(const Floats &a,const std::vector<T> &b) {
    require(a.size()<=b.size(),"reference too small");double max=0;
    for(size_t i=0;i<a.size();++i) {require(std::isfinite(a[i]) && std::isfinite(b[i]),"nonfinite router result");max=std::max(max,std::abs(double(a[i])-double(b[i])));}return max;
}
static bool exact(const Floats &a,const Floats &b) {return a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*sizeof(float));}
int main() {
    json report={{"status","error"},{"scope","device-resident router; FP64 oracle; wall microseconds; CUDA graphs disabled"},{"checks",json::array()}};
    try {
        env("NVIDIA_TF32_OVERRIDE","0");env("GGML_CUDA_DISABLE_GRAPHS","1");ggml_backend_load_all();
        auto *dev=ggml_backend_dev_by_name("CUDA0");if(!dev){std::cerr<<"SKIP: CUDA0 required\n";return 77;}
        ggml_backend_ptr gpu(ggml_backend_dev_init(dev,nullptr));require(bool(gpu),"CUDA init failed");report["device"]=ggml_backend_dev_description(dev);
        bool pass=true;
        for(auto shape:std::vector<std::pair<int,int>>{{4096,256},{512,64}})for(auto type:{GGML_TYPE_BF16,GGML_TYPE_F16,GGML_TYPE_F32}) {
            Matrix a(shape.first,shape.second,type);auto input=values(size_t(a.k)*16,0xda45,1);
            const auto fp64=a.reference(input,false),rounded=a.reference(input,true);
            for(bool strided:{false,true})for(int n:{1,2,3,4,5,16})for(int mode:{0,1,2}) {
                const bool named=mode!=1,precise=mode!=2;
                Graph batch(gpu.get(),a,input,n,strided,named,precise);
                env("STRATA_DS4_ROUTER_MMVF","0");auto stock=batch.run(gpu.get());
                env("STRATA_DS4_ROUTER_MMVF","1");auto candidate=batch.run(gpu.get());
                const bool eligible=type==GGML_TYPE_BF16 && a.k==4096 && a.m==256 && named && precise && n>=2 && n<=4;
                Floats serial;
                if(eligible)for(int t=0;t<n;++t) {Graph one(gpu.get(),a,{input.begin()+t*a.k,input.begin()+(t+1)*a.k},1,strided,named,precise);auto row=one.run(gpu.get());serial.insert(serial.end(),row.begin(),row.end());}
                bool ok=exact(candidate,eligible?serial:stock);double error=diff(candidate,fp64);
                if(eligible)ok &= error<2e-5;
                pass &= ok;
                json r={{"k",a.k},{"m",a.m},{"type",ggml_type_name(type)},{"tokens",n},{"strided",strided},{"named",named},{"precise",precise},
                    {"eligible",eligible},{"passed",ok},{"candidate_fp64_max",error},{"stock_fp64_max",diff(stock,fp64)},
                    {"stock_bf16_input_max",diff(stock,rounded)},{"candidate_stock_exact",exact(candidate,stock)}};
                if(eligible) {
                    r["candidate_serial_exact"]=exact(candidate,serial);r["stock_serial_max"]=diff(stock,serial);
                    r["stock_us"]=json::array();r["candidate_us"]=json::array();
                    for(int repeat=0;repeat<4;++repeat)for(int j=0;j<2;++j) {bool on=(repeat+j)%2;env("STRATA_DS4_ROUTER_MMVF",on?"1":"0");r[on?"candidate_us":"stock_us"].push_back(batch.time(gpu.get()));}
                }
                report["checks"].push_back(r);
            }
        }
        report["status"]=pass?"pass":"fail";
    }catch(const std::exception &e){report["error"]=e.what();}
    std::cout<<report.dump(2)<<'\n';return report["status"]=="pass"?0:1;
}
