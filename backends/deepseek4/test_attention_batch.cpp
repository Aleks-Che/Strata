// Isolated CUDA Flash Attention, with a double-precision scalar oracle over
// exactly the stored F32 Q, F16 K/V/mask and F32 sinks. No model or CPU graph.
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
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>
using json = nlohmann::ordered_json;
using Floats = std::vector<float>;
constexpr int D = 512, H = 64, N = 5;
static void require(bool ok, const char * msg) { if (!ok) throw std::runtime_error(msg); }
static void env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
static float value(uint32_t i, uint32_t seed) {
    i ^= seed; i ^= i >> 16; i *= 0x7feb352d; i ^= i >> 15; i *= 0x846ca68b; i ^= i >> 16;
    return (float(i & 65535) / 32767.5f - 1.0f);
}
static float half(float x) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)); }
struct Fixture {
    std::string name;
    int first, max_k, cap;
    bool alias, analytic, sparse, sinks, strided;
    float scale = 1.0f / std::sqrt(float(D));
    Floats q, k, v, sink;
    std::vector<ggml_fp16_t> kh, vh;
    Fixture(std::string name_, int first_, int max_k_, int cap_, bool alias_, bool analytic_, bool sparse_, bool sinks_, float amplitude, bool strided_=false)
        : name(name_), first(first_), max_k(max_k_), cap(cap_), alias(alias_), analytic(analytic_), sparse(sparse_), sinks(sinks_), strided(strided_),
          q(D*H*N), k(D*max_k), v(D*max_k), sink(H), kh(D*max_k), vh(D*max_k) {
        // Index-based values keep every visible byte fixed as padding changes.
        for (size_t i=0; i<q.size(); ++i) q[i]=analytic ? 0 : amplitude*value(uint32_t(i),0xd54a);
        for (size_t i=0; i<k.size(); ++i) {
            kh[i]=ggml_fp32_to_fp16(value(uint32_t(i),0xa741)); k[i]=ggml_fp16_to_fp32(kh[i]);
            vh[i]=ggml_fp32_to_fp16(analytic ? 0.25f : (alias ? k[i] : value(uint32_t(i),0xb659)));
            v[i]=ggml_fp16_to_fp32(vh[i]);
        }
        for (int h=0; h<H; ++h) sink[h]=analytic ? 0 : 2*value(h,0x1784);
    }
    float mask(int t, int key) const {
        const int pos=first+t;
        const bool visible=key<=pos && (key>=std::max(0,pos-127) || (sparse && key%31==0));
        return visible ? (sparse ? half(-float(key%7)*0.125f) : 0.0f) : -std::numeric_limits<float>::infinity();
    }
    int visible(int t) const {
        int n=0; for(int i=0; i<=first+t; ++i) n+=std::isfinite(mask(t,i)); return n;
    }
    std::vector<double> reference(bool rounded_q) const {
        std::vector<double> out(size_t(N)*H*D);
        for(int t=0; t<N; ++t) for(int h=0; h<H; ++h) {
            std::vector<int> ids; std::vector<double> scores;
            double maximum=sinks ? double(sink[h]) : -std::numeric_limits<double>::infinity();
            for(int key=0; key<=first+t; ++key) if(std::isfinite(mask(t,key))) {
                double dot=0;
                for(int d=0; d<D; ++d) {
                    const float x=q[(t*H+h)*D+d];
                    dot += (rounded_q ? double(half(half(x)*half(scale))) : double(x)*double(scale))*double(k[key*D+d]);
                }
                const double score=dot+mask(t,key); ids.push_back(key); scores.push_back(score); maximum=std::max(maximum,score);
            }
            double sum=sinks ? std::exp(double(sink[h])-maximum) : 0;
            for(auto & score:scores) { score=std::exp(score-maximum); sum+=score; }
            for(size_t j=0; j<ids.size(); ++j) {
                const double p=scores[j]/sum;
                for(int d=0; d<D; ++d) out[(t*H+h)*D+d]+=p*double(v[ids[j]*D+d]);
            }
            if(analytic) for(int d=0; d<D; ++d)
                require(std::abs(out[(t*H+h)*D+d]-0.25*ids.size()/(ids.size()+(sinks?1:0)))<1e-14,"analytic oracle mismatch");
        }
        return out;
    }
};
struct Graph {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * out;
    ggml_cgraph * graph;
    Graph(ggml_backend_t gpu, const Fixture & f, int n, int length, int row=0)
        : ctx(ggml_init({ggml_tensor_overhead()*12+ggml_graph_overhead(),nullptr,true})) {
        require(bool(ctx),"metadata allocation failed");
        require(length>=f.first+row+n,"KV length too short");
        auto * q=f.strided ? ggml_permute(ctx.get(),ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,D,H,n),0,2,1,3) :
                            ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,D,n,H);
        auto * k=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,D,length);
        auto * v=f.alias ? k : ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,D,length);
        auto * mask=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,length,n);
        auto * sinks=f.sinks ? ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,H) : nullptr;
        out=ggml_flash_attn_ext(ctx.get(),q,k,v,mask,f.scale,0,0);
        ggml_flash_attn_ext_add_sinks(out,sinks);
        ggml_flash_attn_ext_set_n_kv_max(out,f.cap);
        ggml_prec_set_acc(out,GGML_PREC_F32);
        require(ggml_backend_supports_op(gpu,out),"CUDA attention unsupported");
        graph=ggml_new_graph(ctx.get()); ggml_build_forward_expand(graph,out);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(),gpu)); require(bool(buffer),"device allocation failed");
        for(int h=0; h<H; ++h) for(int t=0; t<n; ++t)
            ggml_backend_tensor_set(q,f.q.data()+((row+t)*H+h)*D,h*q->nb[2]+t*q->nb[1],D*sizeof(float));
        ggml_backend_tensor_set(k,f.kh.data(),0,ggml_nbytes(k));
        if(!f.alias) ggml_backend_tensor_set(v,f.vh.data(),0,ggml_nbytes(v));
        std::vector<ggml_fp16_t> masks(size_t(length)*n);
        for(int t=0; t<n; ++t) for(int key=0; key<length; ++key) masks[t*length+key]=ggml_fp32_to_fp16(f.mask(row+t,key));
        ggml_backend_tensor_set(mask,masks.data(),0,ggml_nbytes(mask));
        if(sinks) ggml_backend_tensor_set(sinks,f.sink.data(),0,ggml_nbytes(sinks));
    }
    Floats run(ggml_backend_t gpu) {
        require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"attention failed");
        Floats values(ggml_nelements(out)); ggml_backend_tensor_get(out,values.data(),0,ggml_nbytes(out)); return values;
    }
    double time(ggml_backend_t gpu) {
        for(int i=0;i<5;++i) require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"warmup failed");
        ggml_backend_synchronize(gpu); const auto start=std::chrono::steady_clock::now();
        for(int i=0;i<50;++i) require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"timing failed");
        ggml_backend_synchronize(gpu);
        return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/50;
    }
};
template<class T> static json difference(const Floats & actual, const std::vector<T> & reference) {
    require(actual.size()<=reference.size(),"reference too small");
    double max=0,squared=0,ref_squared=0; size_t unequal=0; bool finite=true;
    for(size_t i=0;i<actual.size();++i) {
        finite &= std::isfinite(actual[i]) && std::isfinite(reference[i]);
        const double e=std::abs(double(actual[i])-double(reference[i]));
        max=std::max(max,e); squared+=e*e; ref_squared+=double(reference[i])*double(reference[i]); unequal+=e!=0;
    }
    json result={{"finite",finite},{"max_abs",max},{"rms",std::sqrt(squared/actual.size())},
        {"reference_rms",std::sqrt(ref_squared/actual.size())},{"unequal",unequal}};
    if constexpr(std::is_same_v<T,float>)
        result["exact_bits"]=std::memcmp(actual.data(),reference.data(),actual.size()*sizeof(float))==0;
    return result;
}
static std::string checksum(const Floats & values) {
    uint64_t hash=14695981039346656037ull;
    const auto * bytes=reinterpret_cast<const uint8_t *>(values.data());
    for(size_t i=0;i<values.size()*sizeof(float);++i) { hash^=bytes[i]; hash*=1099511628211ull; }
    std::ostringstream out; out<<std::hex<<std::setw(16)<<std::setfill('0')<<hash; return out.str();
}
int main() {
    json report={{"status","error"},{"scope","fixed GPU attention; scalar FP64 oracle; no model; CPU wall microseconds/op"},
        {"dimensions",{D,H,N}},{"checks",json::array()}};
    try {
        env("NVIDIA_TF32_OVERRIDE","0"); env("GGML_CUDA_DISABLE_GRAPHS","1"); ggml_backend_load_all();
        auto * device=ggml_backend_dev_by_name("CUDA0"); if(!device) { std::cerr<<"SKIP: CUDA0 required\n"; return 77; }
        ggml_backend_ptr gpu(ggml_backend_dev_init(device,nullptr)); require(bool(gpu),"CUDA init failed");
        report["device"]=ggml_backend_dev_description(device);
        const char * compact=std::getenv("STRATA_DS4_FA_COMPACT");
        report["compact_mode"]=compact ? compact : "0";
#ifdef STRATA_ATTENTION_EXPERIMENT
        report["experiment_compiled"]=true;
#else
        report["experiment_compiled"]=false;
#endif
        bool pass=true;
        for(const auto & f:std::vector<Fixture>{
            {"analytic",255,8192,0,false,true,false,true,0},
            {"analytic_bounded",255,8192,128,false,true,false,true,0},
            {"prefix",35,8192,0,true,false,false,true,1},
            {"prefix_bounded",35,8192,128,true,false,false,true,1},
            {"boundary",255,8192,0,true,false,false,true,1},
            {"boundary_bounded",255,8192,128,true,false,false,true,1},
            {"boundary_strided",255,8192,128,true,false,false,true,1,true},
            {"sharp",255,8192,0,false,false,false,false,6},
            {"sharp_bounded",255,8192,128,false,false,false,false,6},
            {"sparse_dense_path",4091,8192,0,true,false,true,true,1},
            {"sparse_gather_path",4091,8192,260,true,false,true,true,1},
            {"sparse_wide_bound",4091,8192,2048,true,false,true,true,1}}) {
            std::cerr<<"Attention fixture "<<f.name<<'\n';
            const auto ref=f.reference(false), rounded=f.reference(true);
            for(int t=0;t<N;++t) require(f.cap==0 || f.visible(t)<=f.cap,"sparse cap truncates visible keys");
            // Serial minimum padded lengths cross 256 -> 512 at position 256.
            Floats natural;
            for(int t=0;t<N;++t) {
                Graph one(gpu.get(),f,1,((f.first+t+1+255)/256)*256,t);
                auto output=one.run(gpu.get()); natural.insert(natural.end(),output.begin(),output.end());
            }
            for(int n:{1,2,3,4,5}) {
                Floats minimal;
                const int base=((f.first+n+255)/256)*256;
                std::vector<int> lengths={base,512,4096,8192};
                std::sort(lengths.begin(),lengths.end()); lengths.erase(std::unique(lengths.begin(),lengths.end()),lengths.end());
                for(int length:lengths) if(length>=base) {
                    Graph batch(gpu.get(),f,n,length); const auto output=batch.run(gpu.get());
                    const auto replay=batch.run(gpu.get()); if(minimal.empty()) minimal=output;
                    Floats serial;
                    for(int t=0;t<n;++t) {
                        Graph one(gpu.get(),f,1,length,t); auto row=one.run(gpu.get()); serial.insert(serial.end(),row.begin(),row.end());
                    }
                    auto error=difference(output,ref);
                    // Deliberately broad guard on these bounded inputs; exact batch
                    // and padding invariance are measurements, not assumed properties.
                    bool ok=error["finite"].get<bool>() && error["max_abs"].get<double>()<0.02 &&
                        std::memcmp(output.data(),replay.data(),output.size()*sizeof(float))==0;
                    // Separate-query compact mode must be invariant on the
                    // explicitly supported cases, even across the 256 boundary.
                    const bool invariant_required=report["experiment_compiled"].get<bool>() && compact &&
                        std::strcmp(compact,"2")==0 && f.cap>0 && n<=4;
                    if(invariant_required) ok &= difference(output,serial)["exact_bits"].get<bool>() &&
                        difference(output,natural)["exact_bits"].get<bool>() && difference(output,minimal)["exact_bits"].get<bool>();
                    pass &= ok;
                    report["checks"].push_back({{"fixture",f.name},{"tokens",n},{"kv_length",length},{"n_kv_max",f.cap},{"strided_q",f.strided},
                        {"passed",ok},{"exact_invariance_required",invariant_required},{"output_fnv1a64",checksum(output)},
                        {"fp64",error},{"fp64_with_half_scaled_q",difference(output,rounded)},
                        {"serial_same_padding",difference(output,serial)},{"serial_natural_padding",difference(output,natural)},
                        {"minimal_padding",difference(output,minimal)},{"wall_us",batch.time(gpu.get())}});
                }
            }
        }
        report["status"]=pass?"pass":"fail";
    } catch(const std::exception & e) { report["error"]=e.what(); }
    std::cout<<report.dump(2)<<'\n'; return report["status"]=="pass"?0:1;
}
