// Indexer CUDA projections: sampled independent FP64 dots, exact token-window
// invariance, isolation, and dispatch guards. CPU is only the test oracle.
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
    Floats out(n);for(auto &v:out){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;v=float(seed&0xffffff)/8388607.5f-1;}return out;
}
static bool exact(const Floats &a,const Floats &b){return a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*sizeof(float));}
// The stock F32 MMA path uses TF32 operands. Test that arithmetic against
// its own FP64 oracle; do not loosen the new full-F32 MMVF error bound.
static float tf32(float value,bool nearest){
    uint32_t bits;std::memcpy(&bits,&value,4);
    if(nearest)bits+=0xfff+((bits>>13)&1);
    bits&=~uint32_t(0x1fff);std::memcpy(&value,&bits,4);return value;
}
template<class T> static Json diff(const Floats &a,const std::vector<T> &b){
    require(!a.empty()&&a.size()==b.size(),"reference shape");double max=0,sum=0;size_t unequal=0;
    for(size_t i=0;i<a.size();++i){require(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite");
        double d=double(a[i])-double(b[i]);max=std::max(max,std::abs(d));sum+=d*d;unequal+=d!=0;}
    return {{"max_abs",max},{"rms",std::sqrt(sum/a.size())},{"unequal",unequal}};
}
struct Matrix {
    int k,m;ggml_type type;Floats weights;std::vector<uint8_t> bytes;
    Matrix(int k_,int m_,ggml_type type_):k(k_),m(m_),type(type_),weights(values(size_t(k)*m,0xa453)),bytes(ggml_row_size(type,k)*m){
        for(auto &v:weights)v*=0.125f;
        if(type==GGML_TYPE_F32)std::memcpy(bytes.data(),weights.data(),bytes.size());
        else{require(ggml_quantize_chunk(type,weights.data(),bytes.data(),0,m,k,nullptr)==bytes.size(),"quantize");
            ggml_get_type_traits(type)->to_float(bytes.data(),weights.data(),int64_t(k)*m);}
    }
    std::vector<double> reference(const Floats &input,int row,int rounding=0)const{
        std::vector<double> out(m);
        for(int j=0;j<m;++j)for(int i=0;i<k;++i){
            const float a=weights[size_t(j)*k+i],b=input[size_t(row)*k+i];
            out[j]+=double(rounding?tf32(a,rounding==2):a)*double(rounding?tf32(b,rounding==2):b);
        }
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
        env("STRATA_DS4_INDEXER_PROJECTION",mode);require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"CUDA compute");
        Floats outv(ggml_nelements(out));ggml_backend_tensor_get(out,outv.data(),0,ggml_nbytes(out));return outv;
    }
};
int main(){
    Json report={{"status","error"},{"scope","Indexer Q8 query and F32 head weights; FP64, serial, common window and guards"},
#ifdef STRATA_PROJECTION_EXPERIMENT
        {"experiment_compiled",true},
#else
        {"experiment_compiled",false},
#endif
        {"checks",Json::array()},{"guards",Json::array()}};
    try{
        env("GGML_CUDA_DISABLE_GRAPHS","1");env("STRATA_DS4_PROJECTION_TILE","64");env("STRATA_DS4_PROJECTION_SPLITS","4");
        env("STRATA_DS4_PROJECTION_MMQ","0");env("NVIDIA_TF32_OVERRIDE","0");
        ggml_backend_load_all();auto *dev=ggml_backend_dev_by_name("CUDA0");if(!dev){std::cerr<<"SKIP: CUDA0 required\n";return 77;}
        ggml_backend_ptr gpu(ggml_backend_dev_init(dev,nullptr));require(bool(gpu),"CUDA init");report["device"]=ggml_backend_dev_description(dev);
        const bool enabled=report["experiment_compiled"].get<bool>();bool pass=true;
        for(bool query:{true,false}){
            Matrix a(query?1024:4096,query?8192:64,query?GGML_TYPE_Q8_0:GGML_TYPE_F32);
            const std::string name=query?"blk.6.indexer.attn_q_b.weight":"blk.6.indexer.proj.weight";
            const auto input=values(size_t(a.k)*4097,0xb823);std::map<int,Floats> serial;std::map<int,std::vector<double>> reference;
            Floats stock_full,fixed_full;
            for(int n:{1,2,3,7,8,9,128,258,3071,4096}){
                Graph g(gpu.get(),a,input,n,name.c_str());auto stock=g.run(gpu.get(),"0"),fixed=g.run(gpu.get(),"1");
                if(!enabled)pass &= exact(stock,fixed);
                Json row={{"name",name},{"tokens",n},{"columns",Json::array()}};
                std::set<int> columns={0,n-1};if(n>2347)columns.insert(2347);
                for(int t:columns){
                    if(!serial.count(t)){Graph one(gpu.get(),a,Floats(input.begin()+size_t(t)*a.k,input.begin()+size_t(t+1)*a.k),1,name.c_str());
                        serial[t]=one.run(gpu.get(),"1");reference[t]=a.reference(input,t);}
                    Floats f(fixed.begin()+size_t(t)*a.m,fixed.begin()+size_t(t+1)*a.m);
                    const auto error=diff(f,reference[t]);const double tolerance=query?0.05:4e-5;
                    bool accurate=error["max_abs"].get<double>()<tolerance;
                    Json column={{"token",t},{"serial",diff(f,serial[t])},{"fp64",error},{"tolerance",tolerance}};
                    if(!enabled && !query){
                        const auto truncated=diff(f,a.reference(input,t,1)),nearest=diff(f,a.reference(input,t,2));
                        column["stock_tf32_truncate_fp64"]=truncated;column["stock_tf32_nearest_fp64"]=nearest;
                        accurate |= truncated["max_abs"].get<double>()<tolerance || nearest["max_abs"].get<double>()<tolerance;
                    }
                    const bool ok=accurate && (!enabled||exact(f,serial[t]));
                    column["passed"]=ok;row["columns"].push_back(column);pass &= ok;
                }
                if(n==3071){stock_full=std::move(stock);fixed_full=std::move(fixed);}
                report["checks"].push_back(row);
            }
            Graph window(gpu.get(),a,Floats(input.begin()+size_t(2303)*a.k,input.begin()+size_t(2561)*a.k),258,name.c_str());
            const auto sw=window.run(gpu.get(),"0"),fw=window.run(gpu.get(),"1");
            Floats sf(stock_full.begin()+size_t(2303)*a.m,stock_full.begin()+size_t(2561)*a.m),ff(fixed_full.begin()+size_t(2303)*a.m,fixed_full.begin()+size_t(2561)*a.m);
            bool same=!enabled||exact(ff,fw);pass &= same;
            report["guards"].push_back({{"name",name},{"common_window",true},{"tokens",258},{"stock",diff(sf,sw)},{"fixed",diff(ff,fw)},{"passed",same}});
            auto changed=input;std::fill_n(changed.data()+a.k,a.k,0);
            Graph before(gpu.get(),a,input,9,name.c_str()),after(gpu.get(),a,changed,9,name.c_str());
            auto b=before.run(gpu.get(),"1"),c=after.run(gpu.get(),"1");bool isolated=true;
            for(size_t i=0;i<c.size();++i)isolated &= i>=size_t(a.m)&&i<size_t(2*a.m)?c[i]==0:c[i]==b[i];
            pass &= isolated;report["guards"].push_back({{"name",name},{"isolation",true},{"passed",isolated}});
            for(int guard=0;guard<9;++guard){
                Matrix w(guard==3?a.k/2:a.k,guard==4?a.m/2:a.m,guard==5?(query?GGML_TYPE_F32:GGML_TYPE_Q8_0):a.type);
                std::string label=guard==0?"blk.6.unrelated.weight":guard==7?name+".extra":name;
                int n=guard==6?4097:4;Graph g(gpu.get(),w,input,n,label.c_str(),guard==1?1:guard==2?2:0);
                auto stock=g.run(gpu.get(),"0"),candidate=g.run(gpu.get(),guard==8?"2":"1");
                bool ok=exact(stock,candidate);pass &= ok;report["guards"].push_back({{"name",name},{"case",guard},{"passed",ok}});
            }
        }
        report["status"]=pass?"pass":"fail";
    }catch(const std::exception &e){report["error"]=e.what();}
    ggml_quantize_free();std::cout<<report.dump(2)<<'\n';return report["status"]=="pass"?0:1;
}
