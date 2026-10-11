// Grouped DeepSeek output-A: CUDA execution, independent decoded-weight FP64
// reference, and exact checks across tokens/groups/layouts. No model needed.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "nlohmann/json.hpp"
#include <cuda_runtime_api.h>
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
static void require(bool value,const char *message){if(!value)throw std::runtime_error(message);}
static void mode(const char *value){
#ifdef _WIN32
    _putenv_s("STRATA_DS4_PROJECTION_MMQ",value);
#else
    setenv("STRATA_DS4_PROJECTION_MMQ",value,1);
#endif
}
static Floats values(size_t n,uint32_t seed,float scale){
    Floats a(n);for(auto &x:a){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;x=(float(seed&0xffffff)/8388607.5f-1)*scale;}return a;
}
static bool exact(const Floats &a,const Floats &b){return a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*sizeof(float));}
template<class T> static Json difference(const Floats &a,const std::vector<T> &b){
    require(a.size()==b.size() && !a.empty(),"bad reference shape");double maximum=0,sum=0;size_t unequal=0;
    for(size_t i=0;i<a.size();++i){require(std::isfinite(a[i]) && std::isfinite(b[i]),"nonfinite projection");
        const double d=double(a[i])-double(b[i]);maximum=std::max(maximum,std::abs(d));sum+=d*d;unequal+=d!=0;}
    return {{"max_abs",maximum},{"rms",std::sqrt(sum/a.size())},{"unequal",unequal}};
}
struct Matrix {
    int k,m,g;ggml_type type;Floats w;std::vector<uint8_t> bytes;
    Matrix(int k_=4096,int m_=1024,int g_=8,ggml_type type_=GGML_TYPE_Q8_0):k(k_),m(m_),g(g_),type(type_),
        w(values(size_t(k)*m*g,0x53682,0.125f)),bytes(ggml_row_size(type,k)*m*g){
        if(type==GGML_TYPE_F32)std::memcpy(bytes.data(),w.data(),bytes.size());
        else {require(ggml_quantize_chunk(type,w.data(),bytes.data(),0,m*g,k,nullptr)==bytes.size(),"quantize failed");
            ggml_get_type_traits(type)->to_float(bytes.data(),w.data(),int64_t(k)*m*g);}
    }
    // Input is canonical token,group,K order, independent of the CUDA view.
    std::vector<double> reference(const Floats &x,int t)const{
        std::vector<double> y(size_t(m)*g);
        for(int group=0;group<g;++group)for(int row=0;row<m;++row)for(int i=0;i<k;++i)
            y[size_t(group)*m+row]+=double(w[(size_t(group)*m+row)*k+i])*x[(size_t(t)*g+group)*k+i];
        return y;
    }
};
struct Graph {
    ggml_context_ptr ctx;ggml_backend_buffer_ptr buffer;ggml_tensor *out;ggml_cgraph *graph;
    Graph(ggml_backend_t gpu,const Matrix &a,const Floats &input,int n,int layout,bool named=true):
        ctx(ggml_init({ggml_tensor_overhead()*8+ggml_graph_overhead(),nullptr,true})){
        require(bool(ctx),"metadata allocation failed");
        auto *w=ggml_new_tensor_3d(ctx.get(),a.type,a.k,a.m,a.g);
        ggml_set_name(w,named?"blk.0.attn_output_a.weight":"blk.0.unrelated.weight");
        auto *raw=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,a.k*(layout==2?2:1),layout==1?a.g:n,layout==1?n:a.g);
        auto *x=layout==1?ggml_permute(ctx.get(),raw,0,2,1,3):
            ggml_view_3d(ctx.get(),raw,a.k,n,a.g,raw->nb[1],raw->nb[2],0);
        out=ggml_mul_mat(ctx.get(),w,x);require(ggml_backend_supports_op(gpu,out),"unsupported CUDA graph");
        graph=ggml_new_graph(ctx.get());ggml_build_forward_expand(graph,out);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(),gpu));require(bool(buffer),"CUDA allocation failed");
        ggml_backend_buffer_clear(buffer.get(),0);ggml_backend_tensor_set(w,a.bytes.data(),0,a.bytes.size());
        Floats packed(ggml_nelements(raw),0);
        for(int t=0;t<n;++t)for(int group=0;group<a.g;++group){
            const size_t at=(layout==1?size_t(t)*raw->nb[2]+group*raw->nb[1]:size_t(t)*raw->nb[1]+group*raw->nb[2])/sizeof(float);
            std::copy_n(input.data()+(size_t(t)*a.g+group)*a.k,a.k,packed.data()+at);
        }
        ggml_backend_tensor_set(raw,packed.data(),0,ggml_nbytes(raw));
    }
    Floats run(ggml_backend_t gpu){
        require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"compute failed");
        Floats y(ggml_nelements(out));ggml_backend_tensor_get(out,y.data(),0,ggml_nbytes(out));return y;
    }
    double time(ggml_backend_t gpu){
        for(int i=0;i<4;++i)require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"warmup failed");
        ggml_backend_synchronize(gpu);const auto begin=std::chrono::steady_clock::now();
        for(int i=0;i<40;++i)require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"timing failed");
        ggml_backend_synchronize(gpu);return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/40;
    }
};
static Floats column(const Floats &data,const Matrix &a,int n,int t){
    Floats y(size_t(a.m)*a.g);for(int g=0;g<a.g;++g)std::copy_n(data.data()+(size_t(g)*n+t)*a.m,a.m,y.data()+size_t(g)*a.m);return y;
}
static Floats prefix(const Floats &data,const Matrix &a,int n,int count){
    Floats y(size_t(a.m)*a.g*count);for(int g=0;g<a.g;++g)std::copy_n(data.data()+size_t(g)*n*a.m,size_t(count)*a.m,y.data()+size_t(g)*count*a.m);return y;
}
int main(){
    Json report={{"status","error"},{"scope","grouped output-A; sampled FP64 reference; complete common-prefix and group-isolation checks"},
#ifdef STRATA_PROJECTION_EXPERIMENT
        {"experiment_compiled",true},
#else
        {"experiment_compiled",false},
#endif
        {"checks",Json::array()},{"guards",Json::array()}};
    try{
#ifdef _WIN32
        _putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
#else
        setenv("GGML_CUDA_DISABLE_GRAPHS","1",1);
#endif
        ggml_backend_load_all();auto *device=ggml_backend_dev_by_name("CUDA0");if(!device){std::cerr<<"SKIP: CUDA0 required\n";return 77;}
        ggml_backend_ptr gpu(ggml_backend_dev_init(device,nullptr));require(bool(gpu),"CUDA init failed");
        cudaDeviceProp properties{};require(cudaGetDeviceProperties(&properties,0)==cudaSuccess,"CUDA properties failed");
        report["device"]=ggml_backend_dev_description(device);
        const bool eligible=report["experiment_compiled"].get<bool>() && properties.major>=8;bool pass=true;
        Matrix a;auto input=values(size_t(a.k)*a.g*4096,0x5491,1);
        std::map<int,Floats> serial;std::map<int,std::vector<double>> references;
        std::map<int,Floats> contiguous;
        for(int layout:{0,1}){
            Floats stock371,fixed371;
            for(int n:{1,2,3,4,5,8,9,16,32,128,371,409,1024,4096}){
                Graph batch(gpu.get(),a,input,n,layout);mode("3");auto stock=batch.run(gpu.get());mode("4");auto fixed=batch.run(gpu.get());
                if(!eligible)pass &= exact(stock,fixed);
                Json row={{"tokens",n},{"layout",layout==0?"contiguous":"permuted"},{"columns",Json::array()}};
                std::set<int> columns={0,n-1};if(n>282)columns.insert(282);if(n==409)columns.insert(371);
                for(int t:columns){
                    if(!serial.count(t)){
                        const size_t width=size_t(a.k)*a.g;
                        Graph one(gpu.get(),a,Floats(input.begin()+t*width,input.begin()+(t+1)*width),1,0);
                        mode("4");serial[t]=one.run(gpu.get());references[t]=a.reference(input,t);
                    }
                    auto f=column(fixed,a,n,t),s=column(stock,a,n,t);auto error=difference(f,references[t]);
                    bool ok=error["max_abs"].get<double>()<0.05 && difference(s,references[t])["max_abs"].get<double>()<0.05;
                    if(eligible)ok &= exact(f,serial[t]);pass &= ok;
                    row["columns"].push_back({{"token",t},{"fixed_serial",difference(f,serial[t])},{"stock_fixed",difference(s,f)},{"fixed_fp64",error},{"passed",ok}});
                }
                if(n==371){stock371=stock;fixed371=fixed;}
                if(n==409){row["stock_371_409"]=difference(prefix(stock,a,n,371),stock371);
                    row["fixed_371_409"]=difference(prefix(fixed,a,n,371),fixed371);if(eligible)pass &= exact(prefix(fixed,a,n,371),fixed371);}
                if(layout==0 && n<=409)contiguous[n]=fixed;
                if(layout==1 && n<=409){row["layout_exact"]=exact(fixed,contiguous[n]);if(eligible)pass &= exact(fixed,contiguous[n]);}
                if(n==1 || n==4 || n==371 || n==409){
                    row["stock_us"]=Json::array();row["fixed_us"]=Json::array();
                    for(int repeat=0;repeat<3;++repeat)for(int j=0;j<2;++j){bool f=(repeat+j)%2;mode(f?"4":"3");row[f?"fixed_us":"stock_us"].push_back(batch.time(gpu.get()));}
                }
                report["checks"].push_back(row);
            }
        }
        // These inputs are outside the exact shape/name/layout contract.
        for(int guard=0;guard<7;++guard){
            Matrix a(guard==3?1024:4096,guard==4?512:1024,guard==5?4:8,guard==6?GGML_TYPE_F32:GGML_TYPE_Q8_0);
            Graph graph(gpu.get(),a,values(size_t(a.k)*a.g*4,0x7524,1),4,guard==2?2:guard==1?1:0,guard>1);
            mode("3");auto stock=graph.run(gpu.get());mode("4");auto fixed=graph.run(gpu.get());
            bool ok=exact(stock,fixed);pass &= ok;report["guards"].push_back({{"case",guard},{"passed",ok}});
        }
        {
            const int n=4097;Graph over(gpu.get(),a,values(size_t(a.k)*a.g*n,0x7524,1),n,1);
            mode("3");auto stock=over.run(gpu.get());mode("4");auto fixed=over.run(gpu.get());
            bool ok=exact(stock,fixed);pass &= ok;report["guards"].push_back({{"over_batch_limit",n},{"passed",ok}});
        }
        // Zero one group of one token; every other output must stay bitwise equal.
        Floats small(input.begin(),input.begin()+size_t(a.k)*a.g*9);auto changed=small;
        std::fill_n(changed.data()+(1*a.g+3)*a.k,a.k,0);
        for(int layout:{0,1})for(const char *m:{"3","4"}){
            Graph original(gpu.get(),a,small,9,layout),zero(gpu.get(),a,changed,9,layout);mode(m);
            auto before=original.run(gpu.get()),after=zero.run(gpu.get());bool ok=true;
            for(size_t i=0;i<after.size();++i){const bool modified=i/a.m==3*9+1;ok &= modified?after[i]==0:after[i]==before[i];}
            pass &= ok;report["guards"].push_back({{"isolation",true},{"layout",layout},{"mode",m},{"passed",ok}});
        }
        report["status"]=pass?"pass":"fail";
    }catch(const std::exception &e){report["error"]=e.what();}
    ggml_quantize_free();std::cout<<report.dump(2)<<'\n';return report["status"]=="pass"?0:1;
}
