// Matrix kernels for every tensor type found in the two local GLM profiles.
// Synthetic packed weights, direct CUDA execution, explicit CPU/scalar oracles.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using json=nlohmann::ordered_json;
using Floats=std::vector<float>;
static void require(bool ok,const char * what) { if(!ok) throw std::runtime_error(what); }

struct Fixture {
    static constexpr int k=512, m=64, experts=16, used=8;
    ggml_type type;
    std::vector<uint8_t> packed;
    Floats dequant;
    explicit Fixture(ggml_type t):type(t) {
        Floats w(k*m*experts),importance(k,1.0f);
        uint32_t seed=0x53a001u;
        for(auto & x:w) { seed^=seed<<13; seed^=seed>>17; seed^=seed<<5; x=(float(seed&0xffffu)/32767.5f-1)*0.06f; }
        packed.resize(ggml_row_size(type,k)*m*experts);
        require(ggml_quantize_chunk(type,w.data(),packed.data(),0,m*experts,k,importance.data())==packed.size(),"quantization size mismatch");
        dequant.resize(w.size());
        if(type==GGML_TYPE_F32) std::memcpy(dequant.data(),packed.data(),packed.size());
        else {
            const auto * traits=ggml_get_type_traits(type);
            require(traits->to_float!=nullptr,"missing CPU dequantization reference");
            for(int row=0;row<m*experts;++row)
                traits->to_float(packed.data()+row*ggml_row_size(type,k),dequant.data()+row*k,k);
        }
    }
};

static Floats inputs(int n,int lanes) {
    Floats x(Fixture::k*n*lanes);
    uint32_t seed=0x53b001u;
    for(auto & v:x) { seed^=seed<<13; seed^=seed>>17; seed^=seed<<5; v=(float(seed&0xffffu)/32767.5f-1)*0.5f; }
    return x;
}
static std::vector<int32_t> routes(int n) {
    std::vector<int32_t> ids(n*Fixture::used);
    for(int i=0;i<n;++i) for(int lane=0;lane<Fixture::used;++lane)
        ids[i*Fixture::used+lane]=(i*3+lane*5)%Fixture::experts;
    return ids;
}

static Floats compute(ggml_backend_t backend,const Fixture & f,int n,int mode,const Floats & x,const std::vector<int32_t> & ids) {
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead()*8+ggml_graph_overhead(),nullptr,true}));
    require(bool(ctx),"metadata allocation failed");
    auto * a=ggml_new_tensor_3d(ctx.get(),f.type,Fixture::k,Fixture::m,mode==0?1:Fixture::experts);
    const int lanes=mode==2?Fixture::used:1;
    auto * b=mode==0?ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,Fixture::k,n):
                    ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,Fixture::k,lanes,n);
    auto * selected=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,Fixture::used,n);
    auto * out=mode==0?ggml_mul_mat(ctx.get(),a,b):ggml_mul_mat_id(ctx.get(),a,b,selected);
    require(ggml_backend_supports_op(backend,out),"matrix operation unsupported on explicit backend");
    auto * graph=ggml_new_graph(ctx.get()); ggml_build_forward_expand(graph,out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend));
    require(bool(buffer),"matrix buffer allocation failed");
    ggml_backend_tensor_set(a,f.packed.data(),0,ggml_nbytes(a));
    ggml_backend_tensor_set(b,x.data(),0,ggml_nbytes(b));
    ggml_backend_tensor_set(selected,ids.data(),0,ggml_nbytes(selected));
    require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"matrix compute failed");
    Floats result(ggml_nelements(out)); ggml_backend_tensor_get(out,result.data(),0,ggml_nbytes(out)); return result;
}

static Floats scalar(const Fixture & f,int n,int mode,const Floats & x,const std::vector<int32_t> & ids) {
    const int used=mode==0?1:Fixture::used, lanes=mode==2?Fixture::used:1;
    Floats result(Fixture::m*n*used);
    for(int token=0;token<n;++token) for(int lane=0;lane<used;++lane) for(int row=0;row<Fixture::m;++row) {
        const int expert=mode==0?0:ids[token*used+lane];
        const auto * w=f.dequant.data()+(expert*Fixture::m+row)*Fixture::k;
        const auto * input=x.data()+(token*lanes+(lanes==1?0:lane))*Fixture::k;
        double total=0; for(int i=0;i<Fixture::k;++i) total+=double(w[i])*input[i];
        result[(token*used+lane)*Fixture::m+row]=float(total);
    }
    return result;
}

static json error(const Floats & a,const Floats & b,double limit) {
    require(a.size()==b.size() && !a.empty(),"result shape mismatch");
    double sq=0,energy=0,max_abs=0; bool finite=true;
    for(size_t i=0;i<a.size();++i) {
        finite &= std::isfinite(a[i]) && std::isfinite(b[i]);
        const double d=double(a[i])-b[i]; sq+=d*d; energy+=double(b[i])*b[i]; max_abs=std::max(max_abs,std::abs(d));
    }
    // CUDA/CPU quantized matmuls can quantize F32 activations internally.
    // Bound relative RMS error to 1%; compare packed-weight outputs, not pre-quant weights.
    const double nmse=sq/std::max(energy,1e-30);
    return {{"pass",finite && nmse<=limit},{"finite",finite},{"nmse",nmse},{"max_abs",max_abs},{"nmse_limit",limit}};
}

int main() {
    json report={{"schema_version",1},{"status","error"},
        {"scope","synthetic packed matrix kernels; no GGUF weights or streamed dispatch"},
        {"requested_revision",STRATA_GLM_SOURCE_SHA},{"archive_sha256",STRATA_GLM_ARCHIVE_SHA256},
        {"patch_set",STRATA_GLM_PATCH_SET},{"NVIDIA_TF32_OVERRIDE","0"},
        {"shape",{{"input",Fixture::k},{"output",Fixture::m},{"experts",Fixture::experts},{"selected",Fixture::used}}},
        {"nmse_limit",1e-4},{"weight_seed",0x53a001u},{"input_seed",0x53b001u},{"results",json::array()}};
    try {
        const char * tf32=std::getenv("NVIDIA_TF32_OVERRIDE");
        require(tf32 && std::string(tf32)=="0","set NVIDIA_TF32_OVERRIDE=0");
        ggml_backend_load_all();
        auto * device=ggml_backend_dev_by_name("CUDA0");
        require(device && ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_GPU,"CUDA0 required");
        ggml_backend_ptr gpu(ggml_backend_dev_init(device,nullptr)),cpu(ggml_backend_cpu_init());
        require(bool(gpu)&&bool(cpu),"backend initialization failed");
        ggml_backend_cpu_set_n_threads(cpu.get(),4); report["device"]=ggml_backend_dev_description(device);
        bool pass=true;
        for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q8_0,GGML_TYPE_Q6_K,
                      GGML_TYPE_IQ2_S,GGML_TYPE_IQ3_S,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ4_XS,
                      GGML_TYPE_Q2_K,GGML_TYPE_Q3_K,GGML_TYPE_Q4_K}) {
            Fixture fixture(type);
            for(int n:{1,4,17}) for(int mode:{0,1,2}) {
                const auto x=inputs(n,mode==2?Fixture::used:1);
                const auto ids=routes(n);
                const auto actual=compute(gpu.get(),fixture,n,mode,x,ids);
                const auto reference=compute(cpu.get(),fixture,n,mode,x,ids);
                const auto direct=scalar(fixture,n,mode,x,ids);
                // F32 must not use the custom TF32 MMA path in correctness mode.
                const double limit=type==GGML_TYPE_F32?1e-10:1e-4;
                const auto cpu_error=error(actual,reference,limit),scalar_error=error(actual,direct,limit);
                const bool ok=cpu_error["pass"] && scalar_error["pass"];
                pass &= ok;
                report["results"].push_back({{"type",ggml_type_name(type)},{"tokens",n},
                    {"op",mode==0?"MUL_MAT":mode==1?"MUL_MAT_ID broadcast":"MUL_MAT_ID per-route"},
                    {"status",ok?"pass":"fail"},{"elements",actual.size()},
                    {"cuda_vs_cpu",cpu_error},{"cuda_vs_scalar",scalar_error}});
            }
        }
        report["status"]=pass?"pass":"fail";
    } catch(const std::exception & e) { report["error"]=e.what(); }
    ggml_quantize_free();
    std::cout<<report.dump(2)<<'\n'; return report["status"]=="pass"?0:1;
}
