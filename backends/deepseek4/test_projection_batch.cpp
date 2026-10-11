// Real DeepSeek HC/Q/K/output-B shapes with synthetic data. FP64 uses decoded
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
    Matrix(int k_, int m_, ggml_type type_, std::string name_)
        : k(k_),m(m_),type(type_),name(std::move(name_)),w(values(size_t(k)*m,0xa453,0.125f)),
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
};
struct Graph {
    ggml_context_ptr ctx; ggml_backend_buffer_ptr buffer; ggml_tensor *out; ggml_cgraph *graph;
    Graph(ggml_backend_t gpu,const Matrix &a,const Floats &input,int n,bool strided=false,bool named=true)
        :ctx(ggml_init({ggml_tensor_overhead()*8+ggml_graph_overhead(),nullptr,true})) {
        require(bool(ctx),"metadata allocation failed");
        auto *w=ggml_new_tensor_2d(ctx.get(),a.type,a.k,a.m);
        ggml_set_name(w,named?a.name.c_str():"blk.0.unrelated.weight");
        auto *raw=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,a.k*(strided?2:1),n);
        auto *x=ggml_view_2d(ctx.get(),raw,a.k,n,raw->nb[1],0);
        out=ggml_mul_mat(ctx.get(),w,x);
        require(ggml_backend_supports_op(gpu,out),"unsupported graph");
        graph=ggml_new_graph(ctx.get());ggml_build_forward_expand(graph,out);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(),gpu));require(bool(buffer),"CUDA allocation failed");
        ggml_backend_buffer_clear(buffer.get(),0);ggml_backend_tensor_set(w,a.bytes.data(),0,a.bytes.size());
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
int main() {
    Json report={{"status","error"},{"scope","synthetic device-resident HC/Q/K/output-B; sampled columns; FP64 reference; CUDA graphs disabled; wall us/op"},
#ifdef STRATA_PROJECTION_EXPERIMENT
        {"experiment_compiled",true},
#else
        {"experiment_compiled",false},
#endif
        {"tf32_override",std::getenv("NVIDIA_TF32_OVERRIDE")?std::getenv("NVIDIA_TF32_OVERRIDE"):"unset"},
        {"fixed_tile_env",std::getenv("STRATA_DS4_PROJECTION_TILE")?std::getenv("STRATA_DS4_PROJECTION_TILE"):"unset"},
        {"fixed_splits_env",std::getenv("STRATA_DS4_PROJECTION_SPLITS")?std::getenv("STRATA_DS4_PROJECTION_SPLITS"):"unset"},
        {"checks",Json::array()},{"guards",Json::array()}};
    try {
        env("GGML_CUDA_DISABLE_GRAPHS","1");env("STRATA_DS4_MMVQ_TOKEN_BATCH","2");
        ggml_backend_load_all();auto *device=ggml_backend_dev_by_name("CUDA0");
        if(!device){std::cerr<<"SKIP: CUDA0 required\n";return 77;}
        ggml_backend_ptr gpu(ggml_backend_dev_init(device,nullptr));require(bool(gpu),"CUDA init failed");
        cudaDeviceProp properties{};require(cudaGetDeviceProperties(&properties,0)==cudaSuccess,"CUDA properties failed");
        report["cuda_major"]=properties.major;report["cuda_minor"]=properties.minor;
        report["device"]=ggml_backend_dev_description(device);bool pass=true;
        for(int shape=0;shape<5;++shape) {
            const char *fixed_mode=shape==4?"5":shape==3?"3":"2";
            Matrix a(shape==4?8192:shape==3?1024:shape?4096:16384,shape==1?1024:shape==2?512:shape==3?32768:shape==4?4096:24,shape?GGML_TYPE_Q8_0:GGML_TYPE_F32,
                shape==1?"blk.0.attn_q_a.weight":shape==2?"blk.0.attn_kv.weight":shape==3?"blk.0.attn_q_b.weight":shape==4?"blk.0.attn_output_b.weight":"blk.0.hc_attn_fn.weight");
            const auto input=values(size_t(a.k)*(shape?4096:409),0xb823,1);
            std::map<int,Floats> serial,serial_candidate,serial_fixed;
            Floats stock371,fixed371;
            std::map<int,std::vector<std::vector<double>>> references;
            for(int n:{1,2,3,4,5,8,9,16,32,371,409,512,1024,4096}) {
                if(!shape && n>409)continue;
                Graph batch(gpu.get(),a,input,n);
                env("STRATA_DS4_PROJECTION_MMQ","0");auto stock=batch.run(gpu.get());
                env("STRATA_DS4_PROJECTION_MMQ","1");auto candidate=batch.run(gpu.get());
                env("STRATA_DS4_PROJECTION_MMQ",fixed_mode);auto fixed=batch.run(gpu.get());
                const bool eligible=report["experiment_compiled"].get<bool>() && shape>0 && shape<3 && n<=8;
                const bool fixed_eligible=report["experiment_compiled"].get<bool>() && shape && properties.major>=8;
                if(!eligible)pass &= exact(candidate,stock);
                if(!fixed_eligible)pass &= exact(fixed,stock);
                Json row={{"name",a.name},{"type",ggml_type_name(a.type)},{"k",a.k},{"m",a.m},{"tokens",n},
                    {"candidate_eligible",eligible},{"candidate_stock_exact",exact(candidate,stock)},{"columns",Json::array()}};
                std::set<int> columns={0,n-1};if(n==409)columns.insert(371);if(n>282)columns.insert(282);
                for(int t:columns) {
                    if(!serial.count(t)) {
                        Graph one(gpu.get(),a,Floats(input.begin()+size_t(t)*a.k,input.begin()+size_t(t+1)*a.k),1);
                        env("STRATA_DS4_PROJECTION_MMQ","0");serial[t]=one.run(gpu.get());
                        env("STRATA_DS4_PROJECTION_MMQ","1");serial_candidate[t]=one.run(gpu.get());
                        env("STRATA_DS4_PROJECTION_MMQ",fixed_mode);serial_fixed[t]=one.run(gpu.get());
                        for(int rounding=0;rounding<(shape?3:1);++rounding)references[t].push_back(a.reference(input,t,rounding));
                    }
                    Floats s(stock.begin()+size_t(t)*a.m,stock.begin()+size_t(t+1)*a.m);
                    Floats c(candidate.begin()+size_t(t)*a.m,candidate.begin()+size_t(t+1)*a.m);
                    Floats f(fixed.begin()+size_t(t)*a.m,fixed.begin()+size_t(t+1)*a.m);
                    Json col={{"token",t},{"stock_serial",difference(s,serial[t])},{"candidate_serial",difference(c,serial_candidate[t])},
                        {"stock_fp64",difference(s,references[t][0])},{"candidate_fp64",difference(c,references[t][0])}};
                    col["fixed_serial"]=difference(f,serial_fixed[t]);col["fixed_fp64"]=difference(f,references[t][0]);
                    if(fixed_eligible)pass &= exact(f,serial_fixed[t]);
                    // Check accuracy as well as finiteness and dispatch guards.
                    // Q8 input quantization has a much larger error budget than
                    // F32 summation. These limits apply to the seeded fixture.
                    const double tolerance=shape==4?0.08:shape?0.05:(report["tf32_override"]=="0"?4e-5:0.01);
                    const bool accurate=col["stock_fp64"]["max_abs"].get<double>()<tolerance &&
                        col["candidate_fp64"]["max_abs"].get<double>()<tolerance && col["fixed_fp64"]["max_abs"].get<double>()<tolerance;
                    col["passed"]=accurate;pass &= accurate;
                    if(shape) {
                        col["stock_q8_half_scale_fp64"]=difference(s,references[t][1]);
                        col["stock_q8_float_scale_fp64"]=difference(s,references[t][2]);
                        col["candidate_q8_float_scale_fp64"]=difference(c,references[t][2]);
                    }
                    row["columns"].push_back(col);
                }
                if(n==371){stock371=stock;fixed371=fixed;}
                if(n==409 && shape) {
                    Floats sp(stock.begin(),stock.begin()+stock371.size()),fp(fixed.begin(),fixed.begin()+fixed371.size());
                    row["stock_371_409"]=difference(sp,stock371);row["fixed_371_409"]=difference(fp,fixed371);
                    if(fixed_eligible)pass &= exact(fp,fixed371);
                }
                if(n==1 || n==4 || n==371 || n==409) {
                    for(auto key:{"stock_us","candidate_us","fixed_us"})row[key]=Json::array();
                    for(int repeat=0;repeat<3;++repeat) for(int j=0;j<3;++j) {
                        const int mode=(repeat+j)%3;env("STRATA_DS4_PROJECTION_MMQ",mode==2?fixed_mode:mode==1?"1":"0");
                        row[mode==2?"fixed_us":mode==1?"candidate_us":"stock_us"].push_back(batch.time(gpu.get()));
                    }
                }
                report["checks"].push_back(row);
            }
            for(bool named:{false,true})for(bool strided:{false,true}) {
                if(named && !strided)continue;
                Graph guard(gpu.get(),a,input,4,strided,named);
                env("STRATA_DS4_PROJECTION_MMQ","0");auto stock=guard.run(gpu.get());
                env("STRATA_DS4_PROJECTION_MMQ","1");auto candidate=guard.run(gpu.get());
                env("STRATA_DS4_PROJECTION_MMQ",fixed_mode);auto fixed=guard.run(gpu.get());
                const bool ok=exact(stock,candidate) && exact(stock,fixed);pass &= ok;
                report["guards"].push_back({{"name",a.name},{"named",named},{"strided",strided},{"passed",ok}});
            }
            // A zero column beside nonzero columns must not acquire data from
            // neighbouring tokens in either kernel family.
            auto zero_input=input;std::fill(zero_input.begin()+a.k,zero_input.begin()+2*a.k,0.0f);
            Graph zero(gpu.get(),a,zero_input,9);
            for(const char *mode:{"0","1",fixed_mode}) {
                env("STRATA_DS4_PROJECTION_MMQ",mode);auto result=zero.run(gpu.get());
                const bool ok=std::all_of(result.begin()+a.m,result.begin()+2*a.m,[](float x){return x==0;});
                pass &= ok;report["guards"].push_back({{"name",a.name},{"zero_column",true},{"mode",mode},{"passed",ok}});
            }
        }
        // Matching suffix alone is not sufficient: unsupported shapes and
        // non-Q8 weights must retain the upstream result bit for bit.
        for(bool output:{false,true})for(int guard=0;guard<3;++guard) {
            Matrix a(guard==0?512:output?8192:4096,guard==1?64:output?4096:512,guard==2?GGML_TYPE_F32:GGML_TYPE_Q8_0,
                output?"blk.0.attn_output_b.weight":"blk.0.attn_kv.weight");
            Graph g(gpu.get(),a,values(size_t(a.k)*4,0xc435,1),4);
            env("STRATA_DS4_PROJECTION_MMQ","0");auto stock=g.run(gpu.get());
            env("STRATA_DS4_PROJECTION_MMQ","1");auto candidate=g.run(gpu.get());
            env("STRATA_DS4_PROJECTION_MMQ",output?"5":"2");auto fixed=g.run(gpu.get());
            const bool ok=exact(stock,candidate) && exact(stock,fixed);pass &= ok;
            report["guards"].push_back({{"k",a.k},{"m",a.m},{"type",ggml_type_name(a.type)},{"passed",ok}});
        }
        report["status"]=pass?"pass":"fail";
    }catch(const std::exception &e){report["error"]=e.what();}
    ggml_quantize_free();std::cout<<report.dump(2)<<'\n';return report["status"]=="pass"?0:1;
}
