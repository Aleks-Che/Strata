// Research-only short routed MMQ against upstream long-batch MMQ.
// Real K/M, eight resident experts, six selected routes, synthetic inputs.
// Timing reuses fixed device-resident tensors; it is not a model benchmark.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include <cuda_runtime_api.h>
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <tuple>
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
    std::string name;
    std::vector<uint8_t> packed;
    Matrix(ggml_type t,int k_,int m_,std::string name_):type(t),k(k_),m(m_),name(std::move(name_)) {
        // All cases use the same seed. Reuse the expensive IQ3 test-weight
        // encoding across name/stride guards and whole-graph checks.
        using Key=std::tuple<int,int,int>;
        static std::map<Key,std::vector<uint8_t>> cache;
        const Key key(int(type),k,m);auto found=cache.find(key);
        if(found!=cache.end()){packed=found->second;return;}
        auto w=values(size_t(k)*m*experts,0x53e001);Floats importance(k,1);
        packed.resize(ggml_row_size(type,k)*m*experts);
        require(ggml_quantize_chunk(type,w.data(),packed.data(),0,m*experts,k,importance.data())==packed.size(),"quantize failed");
        cache.emplace(key,packed);
    }
};
struct Graph {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * out;
    ggml_cgraph * graph;
    Graph(ggml_backend_t gpu,const Matrix & f,int n,int mode,bool strided,const Floats & input,const std::vector<int32_t> & ids,int id_stride=0)
        :ctx(ggml_init({ggml_tensor_overhead()*12+ggml_graph_overhead(),nullptr,true})) {
        require(bool(ctx),"metadata allocation failed");
        auto * a=ggml_new_tensor_3d(ctx.get(),f.type,f.k,f.m,mode?f.experts:1);
        ggml_set_name(a,f.name.c_str());
        const int lanes=mode==2?f.used:1, pad=strided?2:1;
        auto * raw=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,f.k*pad,lanes*pad,n);
        auto * b=mode?ggml_view_3d(ctx.get(),raw,f.k,lanes,n,raw->nb[1],raw->nb[2],0):
                     ggml_view_2d(ctx.get(),raw,f.k,n,raw->nb[2],0);
        auto * route_raw=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,id_stride?id_stride:f.used*pad,n);
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
static bool exact(const Floats&a,const Floats&b){return a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*4);}
static json delta(const Floats&a,const Floats&b){
    require(a.size()==b.size() && !a.empty(),"reference size mismatch");double mx=0,sum=0;size_t neq=0;
    for(size_t i=0;i<a.size();++i){require(std::isfinite(a[i]) && std::isfinite(b[i]),"nonfinite matrix");double d=double(a[i])-b[i];mx=std::max(mx,std::abs(d));sum+=d*d;neq+=d!=0;}
    return {{"max_abs",mx},{"rms",std::sqrt(sum/a.size())},{"unequal",neq}};
}
static void mode(int m){env("STRATA_DS4_ROUTED_MMQ",m==2?"2":m==1?"1":"0");}
int main(){
    json report={{"status","error"},{"scope","short routed MMQ; upstream long MMQ reference; real K/M, 8 experts, 6 routes"},
#ifdef STRATA_PROJECTION_EXPERIMENT
        {"experiment_compiled",true},
#else
        {"experiment_compiled",false},
#endif
        {"checks",json::array()},{"guards",json::array()},{"chains",json::array()},{"strided_ids",json::array()}};
    try{
        env("GGML_CUDA_DISABLE_GRAPHS","1");env("GGML_CUDA_DISABLE_FUSION","0");
        env("STRATA_DS4_MMVQ_TOKEN_BATCH","2");ggml_backend_load_all();
        auto *dev=ggml_backend_dev_by_name("CUDA0");if(!dev)return 77;
        ggml_backend_ptr gpu(ggml_backend_dev_init(dev,nullptr));require(bool(gpu),"CUDA init failed");
        cudaDeviceProp properties{};require(cudaGetDeviceProperties(&properties,0)==cudaSuccess,"CUDA properties failed");
        if(properties.major<8)return 77;
        report["device"]=ggml_backend_dev_description(dev);bool pass=true;
        const bool enabled=report["experiment_compiled"].get<bool>();
        for(int shape=0;shape<3;++shape)for(auto type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_MXFP4}){
            std::cerr<<"routed fixture shape="<<shape<<" type="<<ggml_type_name(type)<<'\n';
            const bool down=shape==2;const int lanes=down?6:1,k=down?2048:4096,m=down?4096:2048;
            Matrix f(type,k,m,shape==0?"blk.0.ffn_gate_exps.weight":shape==1?"CUDA0#blk.0.ffn_up_exps.weight#0":"blk.0.ffn_down_exps.weight");
            const auto input=values(size_t(k)*lanes*409,0x123456);
            std::vector<int32_t> ids(6*409);for(int t=0;t<409;++t)for(int j=0;j<6;++j)ids[t*6+j]=(t*3+j*5)%8;
            // This graph uses the upstream long-batch path even when enabled.
            Graph long_graph(gpu.get(),f,409,down?2:1,false,input,ids);mode(0);auto reference=long_graph.run(gpu.get());
            for(int n:{1,2,3,4,5,8,9,371,409}){
                Graph graph(gpu.get(),f,n,down?2:1,false,input,ids);Floats out[3];
                for(int setting=0;setting<3;++setting){mode(setting);out[setting]=graph.run(gpu.get());}
                const Floats expected(reference.begin(),reference.begin()+size_t(n)*6*m);
                const auto difference=delta(out[2],expected);const double max=difference["max_abs"].get<double>();
                bool ok=enabled?max<1e-5:exact(out[0],out[2]);
                ok &= exact(out[1],out[type==GGML_TYPE_MXFP4?0:2]);
                if(n>8)ok &= exact(out[0],out[2]);
                json row={{"shape",shape},{"type",ggml_type_name(type)},{"tokens",n},{"passed",ok},
                    {"candidate_long",difference},{"stock_long",delta(out[0],expected)}};
                if(n<=8){row["stock_us"]=json::array();row["mmq_us"]=json::array();
                    for(int rep=0;rep<3;++rep)for(int j=0;j<2;++j){bool on=(rep+j)%2;mode(on?2:0);row[on?"mmq_us":"stock_us"].push_back(graph.time(gpu.get(),40));}}
                pass &= ok;report["checks"].push_back(row);
            }
            // Layer 3+ top-k is a view of the full 256-expert argsort: its
            // columns have a 1024-byte stride despite selecting only 6 IDs.
            // With one token ggml_is_contiguous ignores this unused stride;
            // requiring contiguous IDs used to choose different kernels for
            // N=1 and N>1. MMQ's ID helper already accepts a column stride.
            for(int n:{1,2,3,4,8,9}) {
                Graph dense(gpu.get(),f,n,down?2:1,false,input,ids);
                Graph padded(gpu.get(),f,n,down?2:1,false,input,ids,256);
                mode(0);auto stock_dense=dense.run(gpu.get()),stock_padded=padded.run(gpu.get());
                mode(2);auto candidate_dense=dense.run(gpu.get()),candidate_padded=padded.run(gpu.get());
                bool ok=exact(stock_dense,stock_padded) && exact(candidate_dense,candidate_padded);
                if(!enabled || n>8)ok &= exact(stock_padded,candidate_padded);
                pass &= ok;report["strided_ids"].push_back({{"shape",shape},{"type",ggml_type_name(type)},
                    {"tokens",n},{"id_stride_bytes",1024},{"passed",ok},
                    {"stock_dense_padded",delta(stock_dense,stock_padded)},
                    {"candidate_dense_padded",delta(candidate_dense,candidate_padded)}});
            }
            // Wrong names, types, shapes and strides must preserve dispatch.
            for(int guard=0;guard<5;++guard){
                Matrix b(guard==4?GGML_TYPE_Q8_0:type,guard==2?512:k,m,guard==0?"blk.0.unrelated.weight":guard==3?f.name+".extra":f.name);
                Graph graph(gpu.get(),b,4,down?2:1,guard==1,input,ids);mode(0);auto a=graph.run(gpu.get());mode(2);auto c=graph.run(gpu.get());
                bool ok=exact(a,c);pass &= ok;report["guards"].push_back({{"shape",shape},{"type",ggml_type_name(type)},{"guard",guard},{"passed",ok}});
            }
        }
        // Whole graph exercises gate/up fusion and down * route weights.
        for(int n:{1,4,9}){
            Matrix g(GGML_TYPE_IQ3_XXS,4096,2048,"CUDA0#blk.0.ffn_gate_exps.weight#0"),
                   u(GGML_TYPE_IQ3_XXS,4096,2048,"CUDA0#blk.0.ffn_up_exps.weight#0"),
                   d(GGML_TYPE_MXFP4,2048,4096,"CUDA0#blk.0.ffn_down_exps.weight#0");
            // Make up distinct without changing the quantized representation.
            const size_t expert_bytes=u.packed.size()/8;
            std::rotate(u.packed.begin(),u.packed.begin()+expert_bytes,u.packed.end());
            ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead()*24+6*ggml_graph_overhead(),nullptr,true}));require(bool(ctx),"metadata allocation failed");
            auto weight=[&](const Matrix&a){auto*t=ggml_new_tensor_3d(ctx.get(),a.type,a.k,a.m,8);ggml_set_name(t,a.name.c_str());return t;};
            auto *wg=weight(g),*wu=weight(u),*wd=weight(d),*x=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,4096,1,n),
                 *ids=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,6,n),*weights=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,1,6,n);
            auto *gate=ggml_mul_mat_id(ctx.get(),wg,x,ids),*up=ggml_mul_mat_id(ctx.get(),wu,x,ids),
                 *act=ggml_swiglu_clamp(ctx.get(),gate,up,0.3f),*down=ggml_mul_mat_id(ctx.get(),wd,act,ids),*out=ggml_mul(ctx.get(),down,weights);
            auto *whole=ggml_new_graph(ctx.get());ggml_build_forward_expand(whole,out);
            ggml_cgraph *steps[5];int j=0;for(auto*t:{gate,up,act,down,out}){steps[j]=ggml_new_graph(ctx.get());ggml_graph_add_node(steps[j++],t);}
            ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),gpu.get()));require(bool(buffer),"allocation failed");
            ggml_backend_buffer_clear(buffer.get(),0);auto input=values(4096*n,0xdef567);std::vector<int32_t> route(6*n);Floats probs(6*n);
            for(int i=0;i<6*n;++i){route[i]=(i*5)%8;probs[i]=0.03f*(i%6+1);}
            for(auto pair:{std::make_pair(wg,&g),std::make_pair(wu,&u),std::make_pair(wd,&d)})ggml_backend_tensor_set(pair.first,pair.second->packed.data(),0,pair.second->packed.size());
            ggml_backend_tensor_set(x,input.data(),0,input.size()*4);ggml_backend_tensor_set(ids,route.data(),0,route.size()*4);ggml_backend_tensor_set(weights,probs.data(),0,probs.size()*4);
            auto read=[&](ggml_tensor*t){Floats v(ggml_nelements(t));ggml_backend_tensor_get(t,v.data(),0,v.size()*4);return v;};
            auto run=[&](bool staged){if(staged){for(auto*graph:steps)require(ggml_backend_graph_compute(gpu.get(),graph)==GGML_STATUS_SUCCESS,"stage failed");}
                else require(ggml_backend_graph_compute(gpu.get(),whole)==GGML_STATUS_SUCCESS,"whole failed");return read(out);};
            mode(0);auto stock=run(false);
            Floats poison(size_t(2048)*6*n,std::numeric_limits<float>::quiet_NaN());ggml_backend_tensor_set(gate,poison.data(),0,poison.size()*4);ggml_backend_tensor_set(up,poison.data(),0,poison.size()*4);
            mode(2);auto candidate=run(false);bool written=true;for(auto*t:{gate,up})for(float v:read(t))written &= std::isfinite(v);
            auto staged=run(true);bool ok=enabled?exact(candidate,staged):exact(candidate,stock);if(enabled)ok &= written;
            pass &= ok;report["chains"].push_back({{"tokens",n},{"passed",ok},{"intermediates_written",written},{"whole_staged",delta(candidate,staged)}});
        }
        report["status"]=pass?"pass":"fail";
    }catch(const std::exception&e){report["error"]=e.what();}
    ggml_quantize_free();std::cout<<report.dump(2)<<'\n';return report["status"]=="pass"?0:1;
}
