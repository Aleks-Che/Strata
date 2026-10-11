// Compressor Q8_0 projections: CUDA results versus independent FP64 dots,
// batch invariance, dispatch boundaries and isolation of neighbouring tokens.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "nlohmann/json.hpp"
#include <algorithm>
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
static Floats values(size_t n,uint32_t seed){
    Floats out(n);for(auto &v:out){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;v=(float(seed&0xffffff)/8388607.5f-1);}return out;
}
static bool exact(const Floats &a,const Floats &b){return a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*sizeof(float));}
template<class T> static Json diff(const Floats &a,const std::vector<T> &b){
    require(!a.empty()&&a.size()==b.size(),"reference shape");double max=0,sum=0;size_t unequal=0;
    for(size_t i=0;i<a.size();++i){require(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite");
        double d=double(a[i])-double(b[i]);max=std::max(max,std::abs(d));sum+=d*d;unequal+=d!=0;}
    return {{"max_abs",max},{"rms",std::sqrt(sum/a.size())},{"unequal",unequal}};
}
struct Matrix {
    int k,m;ggml_type type;Floats weights;std::vector<uint8_t> bytes;
    Matrix(int k_,int m_,ggml_type type_=GGML_TYPE_Q8_0):k(k_),m(m_),type(type_),
        weights(values(size_t(k)*m,0xa453)),bytes(ggml_row_size(type,k)*m){
        for(auto &v:weights)v*=0.125f;
        if(type==GGML_TYPE_F32)std::memcpy(bytes.data(),weights.data(),bytes.size());
        else{require(ggml_quantize_chunk(type,weights.data(),bytes.data(),0,m,k,nullptr)==bytes.size(),"quantize");
            ggml_get_type_traits(type)->to_float(bytes.data(),weights.data(),int64_t(k)*m);}
    }
    std::vector<double> reference(const Floats &input,int row)const{
        std::vector<double> out(m);
        for(int j=0;j<m;++j)for(int i=0;i<k;++i)out[j]+=double(weights[size_t(j)*k+i])*double(input[size_t(row)*k+i]);
        return out;
    }
};
struct Graph {
    ggml_context_ptr ctx;ggml_backend_buffer_ptr buffer;ggml_cgraph *graph;ggml_tensor *out;
    Graph(ggml_backend_t gpu,const Matrix &w,const Floats &x,int n,const char *name,int layout=0):
        ctx(ggml_init({ggml_tensor_overhead()*8+ggml_graph_overhead(),nullptr,true})){
        require(bool(ctx),"metadata");
        auto *rw=ggml_new_tensor_2d(ctx.get(),w.type,w.k,w.m*(layout==2?2:1));
        auto *a=layout==2?ggml_view_2d(ctx.get(),rw,w.k,w.m,2*rw->nb[1],0):rw;ggml_set_name(a,name);
        auto *rx=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,w.k*(layout==1?2:1),n);
        auto *b=ggml_view_2d(ctx.get(),rx,w.k,n,rx->nb[1],0);out=ggml_mul_mat(ctx.get(),a,b);
        require(ggml_backend_supports_op(gpu,out),"CUDA graph");graph=ggml_new_graph(ctx.get());ggml_build_forward_expand(graph,out);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(),gpu));require(bool(buffer),"CUDA buffer");ggml_backend_buffer_clear(buffer.get(),0);
        const size_t row=ggml_row_size(w.type,w.k);
        for(int j=0;j<w.m;++j)ggml_backend_tensor_set(rw,w.bytes.data()+size_t(j)*row,size_t(j)*a->nb[1],row);
        for(int t=0;t<n;++t)ggml_backend_tensor_set(rx,x.data()+size_t(t)*w.k,size_t(t)*rx->nb[1],w.k*sizeof(float));
    }
    Floats run(ggml_backend_t gpu,const char *mode){
        env("STRATA_DS4_PROJECTION_MMQ",mode);require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"CUDA compute");
        Floats outv(ggml_nelements(out));ggml_backend_tensor_get(out,outv.data(),0,ggml_nbytes(out));return outv;
    }
};
int main(){
    Json report={{"status","error"},{"scope","CSA/HCA/indexer compressor projections; FP64, serial, common prefix and fallback guards"},
#ifdef STRATA_PROJECTION_EXPERIMENT
        {"experiment_compiled",true},
#else
        {"experiment_compiled",false},
#endif
        {"checks",Json::array()},{"guards",Json::array()}};
    try{
        env("GGML_CUDA_DISABLE_GRAPHS","1");env("STRATA_DS4_PROJECTION_TILE","64");env("STRATA_DS4_PROJECTION_SPLITS","4");
        ggml_backend_load_all();auto *dev=ggml_backend_dev_by_name("CUDA0");if(!dev){std::cerr<<"SKIP: CUDA0 required\n";return 77;}
        ggml_backend_ptr gpu(ggml_backend_dev_init(dev,nullptr));require(bool(gpu),"CUDA init");report["device"]=ggml_backend_dev_description(dev);
        const bool enabled=report["experiment_compiled"].get<bool>();bool pass=true;
        for(int m:{256,512,1024})for(const char *suffix:{"kv","gate"}){
            Matrix a(4096,m);std::string name=std::string(m==256?"blk.2.indexer_compressor_":"blk.2.attn_compressor_")+suffix+".weight";
            auto input=values(size_t(a.k)*4097,0xb823);std::map<int,Floats> serial;std::map<int,std::vector<double>> reference;
            Floats stock371,fixed371;
            for(int n:{1,2,3,4,5,8,9,16,32,128,371,409,512,1024,4096}){
                Graph g(gpu.get(),a,input,n,name.c_str());auto stock=g.run(gpu.get(),"5"),fixed=g.run(gpu.get(),"6");
                if(!enabled)pass &= exact(stock,fixed);
                Json row={{"name",name},{"k",a.k},{"m",a.m},{"type",ggml_type_name(a.type)},{"tokens",n},{"columns",Json::array()}};
                std::set<int> columns={0,n-1};if(n>55)columns.insert(55);if(n==409)columns.insert(371);
                for(int t:columns){
                    if(!serial.count(t)){Graph one(gpu.get(),a,Floats(input.begin()+size_t(t)*a.k,input.begin()+size_t(t+1)*a.k),1,name.c_str());
                        serial[t]=one.run(gpu.get(),"6");reference[t]=a.reference(input,t);}
                    Floats f(fixed.begin()+size_t(t)*m,fixed.begin()+size_t(t+1)*m),s(stock.begin()+size_t(t)*m,stock.begin()+size_t(t+1)*m);
                    auto error=diff(f,reference[t]),stock_error=diff(s,reference[t]);
                    bool ok=error["max_abs"].get<double>()<0.05 && stock_error["max_abs"].get<double>()<0.05 && (!enabled||exact(f,serial[t]));
                    row["columns"].push_back({{"token",t},{"fixed_serial",diff(f,serial[t])},{"fixed_fp64",error},{"stock_fp64",stock_error},{"passed",ok}});pass &= ok;
                }
                if(n==371){stock371=stock;fixed371=fixed;}
                if(n==409){Floats s(stock.begin(),stock.begin()+stock371.size()),f(fixed.begin(),fixed.begin()+fixed371.size());
                    row["stock_common_prefix"]=diff(s,stock371);row["fixed_common_prefix"]=diff(f,fixed371);if(enabled)pass &= exact(f,fixed371);}
                report["checks"].push_back(row);
            }
            auto changed=input;std::fill_n(changed.data()+a.k,a.k,0);
            Graph before(gpu.get(),a,input,9,name.c_str()),after(gpu.get(),a,changed,9,name.c_str());
            for(const char *mode:{"5","6"}){auto b=before.run(gpu.get(),mode),c=after.run(gpu.get(),mode);bool ok=true;
                for(size_t i=0;i<c.size();++i)ok &= i>=size_t(m)&&i<size_t(2*m)?c[i]==0:c[i]==b[i];
                pass &= ok;report["guards"].push_back({{"name",name},{"m",a.m},{"isolation",true},{"mode",mode},{"passed",ok}});}
            for(int guard=0;guard<9;++guard){
                Matrix w(guard==3?2048:4096,guard==4?128:m,guard==5?GGML_TYPE_F32:GGML_TYPE_Q8_0);
                std::string label=guard==0?"blk.2.unrelated.weight":guard==7?name+".extra":name;
                int n=guard==6?4097:4;Graph g(gpu.get(),w,input,n,label.c_str(),guard==1?1:guard==2?2:0);
                auto stock=g.run(gpu.get(),"0"),candidate=g.run(gpu.get(),guard==8?"5":"6");
                bool ok=exact(stock,candidate);pass &= ok;report["guards"].push_back({{"name",name},{"k",w.k},{"m",w.m},{"type",ggml_type_name(w.type)},{"case",guard},{"passed",ok}});
            }
        }
        report["status"]=pass?"pass":"fail";
    }catch(const std::exception &e){report["error"]=e.what();}
    ggml_quantize_free();std::cout<<report.dump(2)<<'\n';return report["status"]=="pass"?0:1;
}
