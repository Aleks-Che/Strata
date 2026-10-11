// Real DeepSeek shared-expert shapes with synthetic data. FP64 uses decoded
// stored weights and original F32 activations; extra oracles isolate Q8 scales.
// All model-like matrix operations run on CUDA. CPU work is reference checking.
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
#include <set>
#include <stdexcept>
#include <vector>
using Json = nlohmann::ordered_json;
using Floats = std::vector<float>;
static void require(bool ok, const char *msg) { if (!ok) throw std::runtime_error(msg); }
static void env(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
static Floats values(size_t n, uint32_t seed, float scale) {
    Floats a(n);
    for (auto &x : a) { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        x = (float(seed & 0xffffff) / 8388607.5f - 1) * scale; }
    return a;
}
static bool exact(const Floats &a, const Floats &b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size()*sizeof(float));
}
template<class T> static Json difference(const Floats &a, const std::vector<T> &b) {
    require(a.size() == b.size() && !a.empty(), "bad reference shape");
    double max = 0, sum = 0; size_t unequal = 0;
    for (size_t i=0; i<a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite projection");
        const double d = double(a[i])-double(b[i]); max=std::max(max,std::abs(d)); sum+=d*d; unequal+=d!=0;
    }
    return {{"max_abs", max}, {"rms", std::sqrt(sum/a.size())}, {"unequal", unequal}};
}
struct Matrix {
    int k,m; ggml_type type; std::string name; Floats w; std::vector<uint8_t> bytes;
    Matrix(int k_, int m_, ggml_type type_, std::string name_, uint32_t seed=0xa453)
        : k(k_),m(m_),type(type_),name(std::move(name_)),w(values(size_t(k)*m,seed,0.125f)),
          bytes(ggml_row_size(type,k)*m) {
        if (type==GGML_TYPE_F32) std::memcpy(bytes.data(),w.data(),bytes.size());
        else {
            require(ggml_quantize_chunk(type,w.data(),bytes.data(),0,m,k,nullptr)==bytes.size(),"quantize failed");
            ggml_get_type_traits(type)->to_float(bytes.data(),w.data(),int64_t(k)*m);
        }
    }
    std::vector<double> reference(const Floats &x, int token, int rounding) const {
        // 0: original F32. 1: MMVQ-style int8 with FP16 scale.
        // 2: MMQ-style int8 with F32 scale. CPU arithmetic is independent of
        // CUDA kernels; float reciprocal rounding may differ at quantization ties.
        std::vector<double> input(k), out(m);
        for(int b=0;b<k;b+=32) {
            float amax=0; for(int j=0;j<32;++j) amax=std::max(amax,std::abs(x[token*k+b+j]));
            const float d=amax/127.0f, inv=amax ? 127.0f/amax : 0;
            const float stored=rounding==1 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(d)) : (amax ? 1.0f/inv : 0);
            for(int j=0;j<32;++j) {
                const float v=x[token*k+b+j];
                input[b+j]=!rounding ? double(v) : double(stored)*(amax ? std::round(rounding==1 ? v/d : v*inv) : 0);
            }
        }
        for(int row=0;row<m;++row) for(int j=0;j<k;++j) out[row]+=double(w[size_t(row)*k+j])*input[j];
        return out;
    }
    // Bound input rounding instead of imposing a constant error tolerance on
    // the much larger, nonuniform SwiGLU activations. Also cover stock's FP16
    // scale rounding and a conservative F32 dot-product accumulation error.
    std::vector<double> quantization_bounds(const Floats &x,int token)const{
        std::vector<double> error(k),bound(m,1e-4);
        for(int b=0;b<k;b+=32){double amax=0;for(int j=0;j<32;++j)amax=std::max(amax,std::abs(double(x[size_t(token)*k+b+j])));
            for(int j=0;j<32;++j)error[b+j]=amax*(0.5/127+1.0/1024)+1e-5*std::abs(double(x[size_t(token)*k+b+j]));}
        for(int row=0;row<m;++row)for(int j=0;j<k;++j)bound[row]+=std::abs(double(w[size_t(row)*k+j]))*error[j];
        return bound;
    }
};
struct Graph {
    ggml_context_ptr ctx; ggml_backend_buffer_ptr buffer; ggml_tensor *out; ggml_cgraph *graph;
    Graph(ggml_backend_t gpu,const Matrix &a,const Floats &input,int n,bool strided=false,bool named=true,bool precise=false,bool weight_view=false)
        :ctx(ggml_init({ggml_tensor_overhead()*8+ggml_graph_overhead(),nullptr,true})) {
        require(bool(ctx),"metadata allocation failed");
        auto *raw_w=ggml_new_tensor_2d(ctx.get(),a.type,a.k,a.m);
        auto *w=weight_view?ggml_view_tensor(ctx.get(),raw_w):raw_w;
        ggml_set_name(w,named?a.name.c_str():"blk.0.unrelated.weight");
        auto *raw=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,a.k*(strided?2:1),n);
        auto *x=ggml_view_2d(ctx.get(),raw,a.k,n,raw->nb[1],0);
        out=ggml_mul_mat(ctx.get(),w,x);if(precise)ggml_prec_set_acc(out,GGML_PREC_F32);
        require(ggml_backend_supports_op(gpu,out),"unsupported graph");
        graph=ggml_new_graph(ctx.get());ggml_build_forward_expand(graph,out);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(),gpu));require(bool(buffer),"CUDA allocation failed");
        ggml_backend_buffer_clear(buffer.get(),0);ggml_backend_tensor_set(raw_w,a.bytes.data(),0,a.bytes.size());
        for(int t=0;t<n;++t) ggml_backend_tensor_set(raw,input.data()+size_t(t)*a.k,t*raw->nb[1],a.k*sizeof(float));
    }
    Floats run(ggml_backend_t gpu) {
        require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"compute failed");
        Floats data(ggml_nelements(out));ggml_backend_tensor_get(out,data.data(),0,ggml_nbytes(out));return data;
    }
    double time(ggml_backend_t gpu) {
        for(int i=0;i<8;++i) require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"warmup failed");
        ggml_backend_synchronize(gpu); const auto start=std::chrono::steady_clock::now();
        for(int i=0;i<80;++i) require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"timing failed");
        ggml_backend_synchronize(gpu);
        return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/80;
    }
};
static void mode(int value){env("STRATA_DS4_SHARED_MMQ",value==2?"2":value==1?"1":"0");}
static Floats column(const Floats &x,int m,int t){return {x.begin()+size_t(t)*m,x.begin()+size_t(t+1)*m};}
struct Chain {
    ggml_context_ptr ctx;ggml_backend_buffer_ptr buffer;
    ggml_tensor *gate,*up,*activation,*out;ggml_cgraph *whole,*steps[4];
    Chain(ggml_backend_t gpu,const Matrix &g,const Matrix &u,const Matrix &d,const Floats &input,int n,float limit,bool named=true):
        ctx(ggml_init({ggml_tensor_overhead()*20+5*ggml_graph_overhead(),nullptr,true})) {
        require(bool(ctx),"chain metadata allocation failed");
        auto weight=[&](const Matrix &a){auto *t=ggml_new_tensor_2d(ctx.get(),a.type,a.k,a.m);ggml_set_name(t,named?a.name.c_str():"blk.0.unrelated.weight");return t;};
        auto *wg=weight(g),*wu=weight(u),*wd=weight(d),*x=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,g.k,n);
        gate=ggml_mul_mat(ctx.get(),wg,x);up=ggml_mul_mat(ctx.get(),wu,x);
        activation=ggml_swiglu_clamp(ctx.get(),gate,up,limit);out=ggml_mul_mat(ctx.get(),wd,activation);
        whole=ggml_new_graph(ctx.get());ggml_build_forward_expand(whole,out);
        int i=0;for(auto *t:{gate,up,activation,out}){
            require(ggml_backend_supports_op(gpu,t),"unsupported chain operation");
            steps[i]=ggml_new_graph(ctx.get());ggml_graph_add_node(steps[i++],t);
        }
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(),gpu));require(bool(buffer),"chain CUDA allocation failed");
        ggml_backend_buffer_clear(buffer.get(),0);
        ggml_backend_tensor_set(wg,g.bytes.data(),0,g.bytes.size());ggml_backend_tensor_set(wu,u.bytes.data(),0,u.bytes.size());
        ggml_backend_tensor_set(wd,d.bytes.data(),0,d.bytes.size());ggml_backend_tensor_set(x,input.data(),0,ggml_nbytes(x));
    }
    Floats read(ggml_tensor *t){Floats v(ggml_nelements(t));ggml_backend_tensor_get(t,v.data(),0,ggml_nbytes(t));return v;}
    Floats run(ggml_backend_t gpu,bool staged){
        if(staged)for(auto *graph:steps)require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"chain stage failed");
        else require(ggml_backend_graph_compute(gpu,whole)==GGML_STATUS_SUCCESS,"whole chain failed");
        return read(out);
    }
};
int main(){
    Json report={{"status","error"},{"scope","shared expert: Q8_0 matrices, FP64 references, whole/staged SwiGLU clamp chain, prefix and guards"},
#ifdef STRATA_PROJECTION_EXPERIMENT
        {"experiment_compiled",true},
#else
        {"experiment_compiled",false},
#endif
        {"tf32_override",std::getenv("NVIDIA_TF32_OVERRIDE")?std::getenv("NVIDIA_TF32_OVERRIDE"):"unset"},
        {"checks",Json::array()},{"chains",Json::array()},{"guards",Json::array()}};
    try{
        env("GGML_CUDA_DISABLE_GRAPHS","1");env("GGML_CUDA_DISABLE_FUSION","0");
        env("STRATA_DS4_PROJECTION_MMQ","0");env("STRATA_DS4_MMVQ_TOKEN_BATCH","2");
        env("STRATA_DS4_HC_MMVF","0");env("STRATA_DS4_ROUTER_SAMPLE_MMVF","0");
        ggml_backend_load_all();auto *dev=ggml_backend_dev_by_name("CUDA0");if(!dev){std::cerr<<"SKIP: CUDA0 required\n";return 77;}
        ggml_backend_ptr gpu(ggml_backend_dev_init(dev,nullptr));require(bool(gpu),"CUDA init failed");
        cudaDeviceProp props{};require(cudaGetDeviceProperties(&props,0)==cudaSuccess,"CUDA properties failed");
        report["device"]=ggml_backend_dev_description(dev);bool pass=true;
        const bool enabled=report["experiment_compiled"].get<bool>() && props.major>=8;
        for(int shape=0;shape<3;++shape){
            Matrix a(shape==2?2048:4096,shape==2?4096:2048,GGML_TYPE_Q8_0,
                shape==0?"blk.0.ffn_gate_shexp.weight":shape==1?"blk.0.ffn_up_shexp.weight":"blk.0.ffn_down_shexp.weight",0xa453+shape);
            const auto input=values(size_t(a.k)*4097,0xb823,1);
            std::map<int,Floats> serial;std::map<int,std::vector<std::vector<double>>> refs;Floats prefix[3];
            for(int n:{1,2,3,4,5,8,9,16,17,32,64,128,371,409,512,1024,4096}){
                Graph graph(gpu.get(),a,input,n);Floats outputs[3];
                for(int m=0;m<3;++m){mode(m);outputs[m]=graph.run(gpu.get());}
                if(!enabled)pass &= exact(outputs[0],outputs[1]) && exact(outputs[0],outputs[2]);
                else pass &= exact(outputs[1],outputs[shape==2?0:2]);
                Json row={{"name",a.name},{"tokens",n},{"columns",Json::array()}};
                std::set<int> cols={0,n-1};if(n>282)cols.insert(282);if(n==409)cols.insert(371);
                for(int t:cols){
                    if(!serial.count(t)){
                        Graph one(gpu.get(),a,column(input,a.k,t),1);mode(2);serial[t]=one.run(gpu.get());
                        for(int rounding=0;rounding<3;++rounding)refs[t].push_back(a.reference(input,t,rounding));
                    }
                    auto fixed=column(outputs[2],a.m,t),stock=column(outputs[0],a.m,t);
                    auto f=difference(fixed,refs[t][0]);bool ok=f["max_abs"].get<double>()<0.05;
                    ok &= difference(stock,refs[t][0])["max_abs"].get<double>()<0.05;
                    if(enabled)ok &= exact(fixed,serial[t]);pass &= ok;
                    row["columns"].push_back({{"token",t},{"passed",ok},{"fixed_serial",difference(fixed,serial[t])},
                        {"fixed_fp64",f},{"stock_fp64",difference(stock,refs[t][0])},
                        {"stock_half_scale_fp64",difference(stock,refs[t][1])},
                        {"stock_float_scale_fp64",difference(stock,refs[t][2])},
                        {"fixed_float_scale_fp64",difference(fixed,refs[t][2])}});
                }
                if(n==371)for(int m=0;m<3;++m)prefix[m]=outputs[m];
                if(n==409){
                    for(int m:{0,2}){
                        Floats p(outputs[m].begin(),outputs[m].begin()+prefix[m].size());
                        row[m?"fixed_371_409":"stock_371_409"]=difference(p,prefix[m]);if(enabled && m)pass &= exact(p,prefix[m]);
                    }
                }
                if(n==1 || n==4 || n==371 || n==409 || n==4096){
                    row["stock_us"]=Json::array();row["fixed_us"]=Json::array();
                    for(int repeat=0;repeat<3;++repeat)for(int j=0;j<2;++j){bool fixed=(repeat+j)%2;mode(fixed?2:0);row[fixed?"fixed_us":"stock_us"].push_back(graph.time(gpu.get()));}
                }
                report["checks"].push_back(row);
            }
            for(int guard=0;guard<8;++guard){
                Matrix b(guard==2?1024:a.k,guard==3?128:a.m,guard==4?GGML_TYPE_F32:GGML_TYPE_Q8_0,
                    guard==5?a.name+".extra":a.name);
                Graph graph(gpu.get(),b,input,guard==6?4097:4,guard==1,guard!=0,guard==7);
                mode(0);auto stock=graph.run(gpu.get());mode(2);auto fixed=graph.run(gpu.get());
                bool ok=exact(stock,fixed);pass &= ok;report["guards"].push_back({{"name",a.name},{"case",guard},{"passed",ok}});
            }
            // A contiguous weight view is deliberately outside the research path.
            Graph view(gpu.get(),a,input,4,false,true,false,true);mode(0);auto stock=view.run(gpu.get());mode(2);auto fixed=view.run(gpu.get());
            bool ok=exact(stock,fixed);pass &= ok;report["guards"].push_back({{"name",a.name},{"weight_view",true},{"passed",ok}});
            auto zero_input=input;std::fill_n(zero_input.data()+a.k,a.k,0);
            for(int m:{0,2}){
                Graph before(gpu.get(),a,input,9),after(gpu.get(),a,zero_input,9);mode(m);auto x=before.run(gpu.get()),y=after.run(gpu.get());bool ok=true;
                for(size_t i=0;i<x.size();++i)ok &= i>=size_t(a.m) && i<size_t(a.m)*2?y[i]==0:x[i]==y[i];
                pass &= ok;report["guards"].push_back({{"name",a.name},{"zero_isolation",true},{"mode",m},{"passed",ok}});
            }
        }
        Matrix g(4096,2048,GGML_TYPE_Q8_0,"blk.0.ffn_gate_shexp.weight",0x1234),
               u(4096,2048,GGML_TYPE_Q8_0,"blk.0.ffn_up_shexp.weight",0x5678),
               d(2048,4096,GGML_TYPE_Q8_0,"blk.0.ffn_down_shexp.weight",0x9abc);
        const auto input=values(size_t(4096)*4096,0x76543,1);
        for(float limit:{10.0f,0.3f}){
            Floats prefix;std::map<int,Floats> serial;
            for(int n:{1,2,5,9,371,409,4096}){
                Chain graph(gpu.get(),g,u,d,input,n,limit);
                if(n==1){Floats poison(2048,std::numeric_limits<float>::quiet_NaN());
                    ggml_backend_tensor_set(graph.gate,poison.data(),0,poison.size()*sizeof(float));
                    ggml_backend_tensor_set(graph.up,poison.data(),0,poison.size()*sizeof(float));}
                mode(0);auto stock=graph.run(gpu.get(),false);bool stock_fused=true,fixed_writes=true;
                if(n==1)for(auto *t:{graph.gate,graph.up}){auto v=graph.read(t);stock_fused &= std::all_of(v.begin(),v.end(),[](float x){return std::isnan(x);});}
                mode(2);auto whole=graph.run(gpu.get(),false);
                if(n==1)for(auto *t:{graph.gate,graph.up}){auto v=graph.read(t);fixed_writes &= std::all_of(v.begin(),v.end(),[](float x){return std::isfinite(x);});}
                auto staged=graph.run(gpu.get(),true);
                bool ok=enabled?exact(whole,staged):exact(whole,stock);
                Json row={{"tokens",n},{"limit",limit},{"whole_staged",difference(whole,staged)},{"columns",Json::array()}};
                if(n==1){row["stock_fused_skips_intermediates"]=stock_fused;row["fixed_writes_intermediates"]=fixed_writes;
                    if(props.major>=8)ok &= stock_fused;if(enabled)ok &= fixed_writes;}
                auto gate=graph.read(graph.gate),up=graph.read(graph.up),act=graph.read(graph.activation);
                for(int t:std::set<int>{0,n-1}){
                    if(!serial.count(t)){Chain one(gpu.get(),g,u,d,column(input,4096,t),1,limit);mode(2);serial[t]=one.run(gpu.get(),false);}
                    auto col=column(whole,4096,t);if(enabled)ok &= exact(col,serial[t]);
                    std::vector<double> glu(2048);
                    for(int i=0;i<2048;++i){const size_t j=size_t(t)*2048+i;const double gv=std::min(double(gate[j]),double(limit)),uv=std::clamp(double(up[j]),-double(limit),double(limit));glu[i]=gv/(1+std::exp(-gv))*uv;}
                    const auto down_ref=d.reference(act,t,0),bound=d.quantization_bounds(act,t);
                    const auto down=column(staged,4096,t);double max_fraction=0;
                    for(int i=0;i<4096;++i)max_fraction=std::max(max_fraction,std::abs(double(down[i])-down_ref[i])/bound[i]);
                    auto glu_error=difference(column(act,2048,t),glu),down_error=difference(down,down_ref);
                    ok &= glu_error["max_abs"].get<double>()<5e-5 && max_fraction<=1;
                    row["columns"].push_back({{"token",t},{"whole_serial",difference(col,serial[t])},{"glu_fp64",glu_error},
                        {"down_fp64_actual_input",down_error},{"down_float_scale_fp64",difference(down,d.reference(act,t,2))},
                        {"down_error_fraction_of_quantization_bound",max_fraction}});
                }
                if(n==371)prefix=whole;
                if(n==409){Floats p(whole.begin(),whole.begin()+prefix.size());row["prefix_371_409"]=difference(p,prefix);if(enabled)ok &= exact(p,prefix);}
                row["passed"]=ok;pass &= ok;report["chains"].push_back(row);
            }
        }
        // This whole graph can fuse; an unrelated name must still follow stock.
        Chain guard(gpu.get(),g,u,d,input,1,10,false);mode(0);auto stock=guard.run(gpu.get(),false);mode(2);auto fixed=guard.run(gpu.get(),false);
        const bool ok=exact(stock,fixed);pass &= ok;report["guards"].push_back({{"unrelated_chain",true},{"passed",ok}});
        report["status"]=pass?"pass":"fail";
    }catch(const std::exception &e){report["error"]=e.what();}
    ggml_quantize_free();std::cout<<report.dump(2)<<'\n';return report["status"]=="pass"?0:1;
}
