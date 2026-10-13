// Stagewise numerical audit. With no arguments this is a small regression test;
// fixture paths add original GGUF weights/activations captured by the engine.
#include "cpu_expert_fixture.hpp"
#include "cpu_mmvq_reference.hpp"
#include "swiglu_compat.hpp"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

using json=nlohmann::ordered_json;
using Floats=std::vector<float>;
using namespace strata_ds4;
void strata_test_cuda_quantize(const float *,int,CpuQ8_1 *);
void strata_test_cuda_swiglu(const float *,const float *,float *,int,float);
bool strata_test_generic_mmvq_device();
static void require(bool b,const char *why) {if(!b)throw std::runtime_error(why);}
static void env(const char *name,const char *value) {
#ifdef _WIN32
    _putenv_s(name,value);
#else
    setenv(name,value,1);
#endif
}
static bool exact(const Floats &a,const Floats &b) {return a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*4);}
static json delta(const Floats &a,const Floats &b) {
    require(a.size()==b.size() && !a.empty(),"comparison shape");
    double mx=0,err=0,norm=0;size_t different=0;
    for(size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]),"nonfinite result");
        const double d=double(a[i])-b[i];mx=std::max(mx,std::abs(d));err+=d*d;norm+=double(b[i])*b[i];different+=d!=0;
    }
    return {{"max_abs",mx},{"rms",std::sqrt(err/a.size())},{"relative_l2",std::sqrt(err/std::max(norm,1e-30))},{"unequal",different}};
}
static Floats activation(const Floats &g,const Floats &u,float clamp) {
    Floats h(g.size());for(size_t i=0;i<g.size();++i) {
        float a=g[i],b=u[i];if(clamp>0){a=std::min(a,clamp);b=std::clamp(b,-clamp,clamp);}
        h[i]=(a/(1.0f+std::exp(-a)))*b;
    }return h;
}
static Floats compatible_activation(const Floats &g,const Floats &u,float clamp) {
    Floats h(g.size());for(size_t i=0;i<g.size();++i)h[i]=swiglu_compat(g[i],u[i],clamp);return h;
}
static Floats cpu_dot(ggml_type type,int k,int m,const void *weights,const Floats &x) {
    std::vector<uint8_t> q(cpu_lane_activation_bytes(type,k));Floats y(m);
    cpu_lane_quantize(type,k,x.data(),q.data());cpu_lane_rows(type,k,weights,q.data(),y.data(),0,m);return y;
}
static Floats oracle(ggml_type type,int k,int m,const void *weights,const Floats &x) {
    Floats row(k),out(m);const size_t stride=ggml_row_size(type,k);
    for(int r=0;r<m;++r) {
        ggml_get_type_traits(type)->to_float(static_cast<const uint8_t *>(weights)+r*stride,row.data(),k);
        double sum=0;for(int j=0;j<k;++j)sum+=double(row[j])*x[j];out[r]=float(sum);
    }return out;
}
static Floats decode(const std::vector<CpuQ8_1> &q) {
    Floats f(q.size()*32);for(size_t i=0;i<q.size();++i)for(int j=0;j<32;++j)f[i*32+j]=ggml_fp16_to_fp32(q[i].d)*q[i].qs[j];return f;
}
static json quantization(ggml_type type,const Floats &x) {
    auto host=cpu_mmvq_quantize(x.data(),int(x.size())),gpu=host;
    strata_test_cuda_quantize(x.data(),int(x.size()),gpu.data());
    require(!std::memcmp(host.data(),gpu.data(),host.size()*sizeof(CpuQ8_1)),"host Q8_1 differs from CUDA quantizer");
    auto qt=ggml_get_type_traits_cpu(type)->vec_dot_type;
    std::vector<uint8_t> packed(cpu_lane_activation_bytes(type,int(x.size())));Floats old(x.size());
    cpu_lane_quantize(type,int(x.size()),x.data(),packed.data());
    if(qt==GGML_TYPE_Q8_K)dequantize_row_q8_K(reinterpret_cast<const block_q8_K *>(packed.data()),old.data(),x.size());
    else {auto fn=ggml_get_type_traits(qt)->to_float;require(fn!=nullptr,"activation dequantizer");fn(packed.data(),old.data(),x.size());}
    auto converted=decode(gpu);
    return {{"cpu_type",ggml_type_name(qt)},{"cuda_type","q8_1"},{"host_q8_1_bytes_exact",true},
            {"cpu_vs_input",delta(old,x)},{"cuda_vs_input",delta(converted,x)},{"cpu_vs_cuda",delta(old,converted)}};
}
struct GpuExpert {
    ggml_backend_t gpu;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor *w[3],*x,*ids,*g,*u,*h,*d;
    ggml_cgraph *steps[4],*whole;
    int k,m,n;
    GpuExpert(ggml_backend_t gpu_,const CpuExpertWork &f,const CpuExpert &e,int tokens=1,int selected=1)
        :gpu(gpu_),ctx(ggml_init({ggml_tensor_overhead()*32+5*ggml_graph_overhead(),nullptr,true})),k(f.columns),m(f.hidden),n(tokens) {
        require(bool(ctx),"graph context");
        w[0]=ggml_new_tensor_3d(ctx.get(),f.gate_type,k,m,1);w[1]=ggml_new_tensor_3d(ctx.get(),f.up_type,k,m,1);
        w[2]=ggml_new_tensor_3d(ctx.get(),f.down_type,m,k,1);
        // Real names retain the private dispatch guards used by the engine.
        ggml_set_name(w[0],"blk.0.ffn_gate_exps.weight");ggml_set_name(w[1],"blk.0.ffn_up_exps.weight");ggml_set_name(w[2],"blk.0.ffn_down_exps.weight");
        x=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,k,1,n);ids=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,selected,n);
        g=ggml_mul_mat_id(ctx.get(),w[0],x,ids);u=ggml_mul_mat_id(ctx.get(),w[1],x,ids);
        h=f.clamp>0?ggml_swiglu_clamp(ctx.get(),g,u,f.clamp):ggml_swiglu_split(ctx.get(),g,u);
        d=ggml_mul_mat_id(ctx.get(),w[2],h,ids);
        int j=0;for(auto *t:{g,u,h,d}){steps[j]=ggml_new_graph(ctx.get());ggml_graph_add_node(steps[j++],t);}
        whole=ggml_new_graph(ctx.get());ggml_build_forward_expand(whole,d);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(),gpu));require(bool(buffer),"graph allocation");ggml_backend_buffer_clear(buffer.get(),0);
        ggml_backend_tensor_set(w[0],e.gate,0,ggml_nbytes(w[0]));ggml_backend_tensor_set(w[1],e.up,0,ggml_nbytes(w[1]));
        ggml_backend_tensor_set(w[2],e.down,0,ggml_nbytes(w[2]));
    }
    Floats get(ggml_tensor *t) {Floats out(ggml_nelements(t));ggml_backend_tensor_get(t,out.data(),0,out.size()*4);return out;}
    void set(ggml_tensor *t,const Floats &v) {require(ggml_nelements(t)==v.size(),"input shape");ggml_backend_tensor_set(t,v.data(),0,v.size()*4);}
    void run(ggml_cgraph *graph) {require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"CUDA graph compute");}
};
static json matrix(ggml_type type,int k,int m,const void *w,const Floats &x,const Floats &gpu) {
    auto cpu=cpu_dot(type,k,m,w,x),q=decode(cpu_mmvq_quantize(x.data(),k));
    auto ref=cpu_mmvq_reference(type,k,m,w,cpu_mmvq_quantize(x.data(),k).data());
    auto fp=oracle(type,k,m,w,x),fpq=oracle(type,k,m,w,q);
    // This test is restricted to the generic NVIDIA dispatch table. A future
    // SIMD implementation must preserve these bits, not a percentage tolerance.
    auto diff=delta(ref,gpu);
    if(!exact(ref,gpu))
        throw std::runtime_error(std::string("MMVQ reference arithmetic mismatch ")+ggml_type_name(type)+": "+diff.dump());
    return {{"type",ggml_type_name(type)},{"activation",quantization(type,x)},
            {"legacy_cpu_vs_gpu",delta(cpu,gpu)},{"q8_1_reference_vs_gpu",diff},
            {"legacy_cpu_vs_float64",delta(cpu,fp)},{"gpu_vs_float64",delta(gpu,fp)},
            {"gpu_vs_q8_1_float64",delta(gpu,fpq)}};
}
static json audit(ggml_backend_t gpu,CpuExpertWork &f,size_t index) {
    const auto &e=f.experts.at(index);const int k=f.columns,m=f.hidden;
    Floats x(f.input+e.token*k,f.input+(e.token+1)*k);GpuExpert graph(gpu,f,e);graph.set(graph.x,x);
    graph.run(graph.steps[0]);auto gg=graph.get(graph.g);graph.run(graph.steps[1]);auto gu=graph.get(graph.u);
    graph.run(graph.steps[2]);auto gh=graph.get(graph.h);graph.run(graph.steps[3]);auto gd=graph.get(graph.d);
    auto cg=cpu_dot(f.gate_type,k,m,e.gate,x),cu=cpu_dot(f.up_type,k,m,e.up,x);
    auto ch=activation(cg,cu,f.clamp),cd=cpu_dot(f.down_type,m,k,e.down,ch);
    auto q=cpu_mmvq_quantize(x.data(),k);
    auto rg=cpu_mmvq_reference(f.gate_type,k,m,e.gate,q.data()),ru=cpu_mmvq_reference(f.up_type,k,m,e.up,q.data());
    auto rh=compatible_activation(rg,ru,f.clamp),rd=cpu_mmvq_reference(f.down_type,m,k,e.down,cpu_mmvq_quantize(rh.data(),m).data());
    require(exact(rh,gh),"compatible SwiGLU differs from production CUDA");
    require(exact(rd,gd),"compatible expert chain differs from CUDA");
    CpuExpertWork candidate=f;candidate.experts={e};Floats parallel(k);candidate.output=parallel.data();
    for(int threads:{12,16}) {
        CpuLanePool pool(threads);int phases=0;
        cpu_mmvq_reference_compute(candidate,pool,nullptr,[&](CpuWeightPhase phase) {
            require((phase==CpuWeightPhase::gate_up && phases==0) || (phase==CpuWeightPhase::down && phases==1),"reference release order");++phases;
        });
        require(phases==2 && exact(rd,parallel),"parallel reference changed arithmetic");
    }
    json row={{"expert_index",index},{"token",e.token},{"slot",e.slot},{"columns",k},{"hidden",m},{"clamp",f.clamp},
        {"gate",matrix(f.gate_type,k,m,e.gate,x,gg)},{"up",matrix(f.up_type,k,m,e.up,x,gu)},
        {"legacy_swiglu_vs_compat_gpu",delta(activation(gg,gu,f.clamp),gh)},
        {"down_same_input",matrix(f.down_type,m,k,e.down,gh,gd)},
        {"legacy_chain_vs_gpu",delta(cd,gd)},{"q8_1_reference_chain_vs_gpu",delta(rd,gd)}};
    row["reference_threads_12_16_exact"]=true;
    if(f.output)require(!std::memcmp(cd.data(),f.output+index*k,k*4),"stagewise CPU disagrees with CpuExpertWork");
    graph.run(graph.whole);row["gpu_fused_vs_steps"]=delta(graph.get(graph.d),gd);
    require(exact(graph.get(graph.d),gd),"fused expert chain differs from separate operations");
    // Run the actual down kernel on the mirrored hidden activation: distinguish
    // amplification at a new quantization boundary from the down dot product.
    graph.set(graph.h,rh);graph.run(graph.steps[3]);row["reference_down_on_reference_hidden_vs_gpu"]=delta(rd,graph.get(graph.d));
    row["hidden_reference_vs_gpu"]=delta(rh,gh);
    return row;
}
static Floats values(int n,uint32_t seed) {
    Floats f(n);for(auto &v:f){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;v=(float(seed&65535)/32767.5f-1)*0.25f;}return f;
}
static json swiglu_boundary(ggml_backend_t gpu,int tokens=1,int selected=1,float limit=0) {
    // Deliberately put hidden values on Q8_1 half-integer boundaries. Ordinary
    // model captures can hide libm/libdevice differences after requantization.
    const int n=4096*tokens*selected;Floats gate(n),up(n);
    for(int i=0;i<n;++i) {
        gate[i]=i%32==0?128.0f:(float(i%997)+0.37f)*0.01f-5.0f;
        up[i]=i%32==0?127.0f/128.0f:(float(i%5)+0.5f)/(gate[i]/(1.0f+std::exp(-gate[i])));
        if(limit>0) {
            const float peak=(limit/(1.0f+std::exp(-limit)))*limit;
            gate[i]=i%32==0?limit:limit*(0.25f+0.75f*float(i%997)/997.0f);
            up[i]=i%32==0?limit:((float(i%30)+0.5f)*(peak/127.0f))/(gate[i]/(1.0f+std::exp(-gate[i])));
        }
    }
    CpuExpertWork f;f.columns=256;f.hidden=4096;f.clamp=limit;
    f.gate_type=f.up_type=f.down_type=GGML_TYPE_MXFP4;
    std::vector<uint8_t> gu(ggml_row_size(f.gate_type,f.columns)*f.hidden),dw(ggml_row_size(f.down_type,f.hidden)*f.columns);
    CpuExpert e{gu.data(),gu.data(),dw.data(),0,0};GpuExpert graph(gpu,f,e,tokens,selected);
    // Overwrite actual gate/up result tensors, then execute only their real
    // production GLU node. This targets rounding boundaries independently of
    // how often the model's dot products happen to land there.
    graph.set(graph.g,gate);graph.set(graph.u,up);
    env("STRATA_DS4_SWIGLU_COMPAT","0");graph.run(graph.steps[2]);auto old_gpu=graph.get(graph.h);
    env("STRATA_DS4_SWIGLU_COMPAT","1");graph.run(graph.steps[2]);auto gpu_hidden=graph.get(graph.h);
    auto host_hidden=compatible_activation(gate,up,limit),old_host=activation(gate,up,limit);
    require(exact(host_hidden,gpu_hidden),"SwiGLU boundary bits differ");
    auto host_q=cpu_mmvq_quantize(host_hidden.data(),n),gpu_q=cpu_mmvq_quantize(gpu_hidden.data(),n);
    auto old_hq=cpu_mmvq_quantize(old_host.data(),n),old_gq=cpu_mmvq_quantize(old_gpu.data(),n);
    require(!std::memcmp(host_q.data(),gpu_q.data(),host_q.size()*sizeof(CpuQ8_1)),"boundary Q8_1 blocks differ");
    auto fast_q=host_q;
    env("STRATA_DS4_SWIGLU_COMPAT","0");strata_test_cuda_quantize(gpu_hidden.data(),n,fast_q.data());
    env("STRATA_DS4_SWIGLU_COMPAT","1");
    size_t fast_changed=0,fast_scales=0;
    for(size_t b=0;b<host_q.size();++b) {
        fast_scales+=host_q[b].d!=fast_q[b].d;
        for(int j=0;j<32;++j)fast_changed+=host_q[b].qs[j]!=fast_q[b].qs[j];
    }
    size_t changed=0;for(size_t b=0;b<host_q.size();++b)for(int j=0;j<32;++j)changed+=host_q[b].qs[j]!=gpu_q[b].qs[j];
    size_t old_changed=0;for(size_t b=0;b<old_hq.size();++b)for(int j=0;j<32;++j)old_changed+=old_hq[b].qs[j]!=old_gq[b].qs[j];
    return {{"scope","adversarial SwiGLU rounding boundaries; bit-exact assertions"},{"tokens",tokens},{"selected",selected},{"limit",limit},{"values",n},
        {"legacy_hidden",delta(old_host,old_gpu)},{"legacy_changed_q8_values",old_changed},
        {"old_cuda_quantizer_on_identical_hidden_changed_q8_values",fast_changed},
        {"old_cuda_quantizer_on_identical_hidden_changed_scales",fast_scales},
        {"compatible_vs_previous_gpu",delta(gpu_hidden,old_gpu)},
        {"hidden",delta(host_hidden,gpu_hidden)},{"changed_q8_values",changed},
        {"quantizer_on_gpu_hidden",quantization(GGML_TYPE_MXFP4,gpu_hidden)}};
}
static json swiglu_range() {
    constexpr int n=262144;Floats g(n),u(n),device(n);uint32_t seed=773123;
    // The wide exponent sweep includes signed zeros, subnormals, exp overflow
    // and the subnormal flush boundary. Products remain finite.
    for(int i=0;i<n;++i) {
        seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;
        g[i]=float(int(seed%200001)-100000)*0.001f;
        u[i]=float(int((seed>>12)%10001)-5000)*0.001f;
        if(i%8==0) {uint32_t b=seed&0x807fffffu;std::memcpy(&g[i],&b,4);}
        if(i%8==1)g[i]=float(int(seed%30001)-15000)*0.000001f;
        if(i%8==2) {uint32_t b=(seed&0x807fffffu)|0x00800000u;std::memcpy(&u[i],&b,4);}
    }
    json result=json::array();
    for(float limit:{0.0f,0.1f,7.0f}) {
        auto host=compatible_activation(g,u,limit);
        strata_test_cuda_swiglu(g.data(),u.data(),device.data(),n,limit);
        require(exact(host,device),"fast-math CUDA/CPU SwiGLU range bits differ");
        Floats oracle(n);
        for(int i=0;i<n;++i) {
            float a=sw_flush(g[i]),b=sw_flush(u[i]);if(limit>0){a=std::min(a,limit);b=std::clamp(b,-limit,limit);}
            const float ex=sw_to_float(std::exp(-double(a)));
            oracle[i]=sw_mul(sw_div(a,sw_add(1.0f,ex)),b);
        }
        auto error=delta(host,oracle);require(error["relative_l2"].get<double>()<1e-7,"common exp approximation accuracy");
        result.push_back({{"values",n},{"limit",limit},{"bit_exact",true},{"vs_host_double_exp",error}});
    }
    return result;
}
static json synthetic(ggml_backend_t gpu,ggml_type type,bool mixed=false,float limit=0.1f) {
    CpuExpertWork f;f.columns=type==GGML_TYPE_Q8_0?4096:1024;f.hidden=type==GGML_TYPE_Q8_0?2048:256;f.tokens=1;f.topk=1;f.clamp=limit;
    f.gate_type=mixed?GGML_TYPE_Q8_0:type;f.up_type=type;f.down_type=mixed?GGML_TYPE_MXFP4:type;
    std::array<std::vector<uint8_t>,3> w;
    for(int j=0;j<3;++j) {
        int k=j==2?f.hidden:f.columns,m=j==2?f.columns:f.hidden;auto v=values(k*m,17+j*123);
        auto wt=j==0?f.gate_type:j==1?f.up_type:f.down_type;
        Floats importance(k,1);w[j].resize(ggml_row_size(wt,k)*m);
        require(ggml_quantize_chunk(wt,v.data(),w[j].data(),0,m,k,importance.data())==w[j].size(),"weight quantize");
    }
    auto x=values(f.columns,895),out=Floats(f.columns);f.input=x.data();f.output=out.data();f.experts.push_back({w[0].data(),w[1].data(),w[2].data(),0,0});
    CpuLanePool pool(12);f.compute(pool);auto result=audit(gpu,f,0);
    GpuExpert one(gpu,f,f.experts[0]);one.set(one.x,x);one.run(one.whole);auto base=one.get(one.d);
    result["prefix"]=json::array();
    for(int n:{2,3,4,8}) {
        GpuExpert batch(gpu,f,f.experts[0],n);auto input=values(f.columns*n,958);std::copy(x.begin(),x.end(),input.begin());
        batch.set(batch.x,input);batch.run(batch.whole);auto output=batch.get(batch.d);output.resize(f.columns);
        const bool same=exact(base,output);if(n<=4)require(same,"short-window prefix changed");
        if(n<=4) {
            const auto all=batch.get(batch.d);
            for(int t=0;t<n;++t) {
                one.set(one.x,Floats(input.begin()+t*f.columns,input.begin()+(t+1)*f.columns));one.run(one.whole);
                require(exact(one.get(one.d),Floats(all.begin()+t*f.columns,all.begin()+(t+1)*f.columns)),"short-window token differs from single-token reduction");
            }
            batch.run(batch.steps[0]);batch.run(batch.steps[1]);batch.run(batch.steps[2]);batch.run(batch.steps[3]);
            require(exact(all,batch.get(batch.d)),"multi-token fused/unfused expert differs");
        }
        result["prefix"].push_back({{"tokens",n},{"bit_exact",same},{"delta",delta(output,base)}});
    }
    return result;
}
int main(int argc,char **argv) {
    json report={{"status","error"},{"scope","resident expert arithmetic, not full-model logits or a speed benchmark"},
        {"synthetic",json::array()},{"fixtures",json::array()}};
    try {
        env("GGML_CUDA_DISABLE_GRAPHS","1");env("GGML_CUDA_DISABLE_FUSION","0");env("STRATA_DS4_MMVQ_TOKEN_BATCH","0");env("STRATA_DS4_ROUTED_MMQ","0");
        env("STRATA_DS4_CPU_MMVQ_REFERENCE","0");env("STRATA_DS4_SWIGLU_COMPAT","1");
        ggml_backend_load_all();auto *dev=ggml_backend_dev_by_name("CUDA0");if(!dev)return 77;
        ggml_backend_ptr gpu(ggml_backend_dev_init(dev,nullptr));require(bool(gpu),"CUDA init");report["device"]=ggml_backend_dev_description(dev);
        if(!strata_test_generic_mmvq_device())return 77;
        // Round-to-even vs round-away, zero, per-block outliers, scale boundaries.
        Floats edges(1024);for(size_t i=0;i<edges.size();++i)edges[i]=i%32==0?127.0f:float(int(i%31)-15)+0.5f;
        for(int i=0;i<32;++i)edges[i]=0;edges[64]=10000;edges[128]=-10000;
        report["quantizer_edges"]=quantization(GGML_TYPE_Q8_0,edges);
        Floats tiny(256);for(int i=0;i<256;++i) {
            const uint32_t bits=(i%2?0x80000000u:0u)|uint32_t(i/32)*0x00400000u;
            std::memcpy(&tiny[i],&bits,4);
        }
        auto tq=cpu_mmvq_quantize(tiny.data(),256),tq_gpu=tq;
        strata_test_cuda_quantize(tiny.data(),256,tq_gpu.data());
        require(!std::memcmp(tq.data(),tq_gpu.data(),tq.size()*sizeof(CpuQ8_1)),"Q8_1 subnormal/scale flush bits differ");
        report["quantizer_tiny_blocks_exact"]=true;
        report["swiglu_boundary"]=json::array({swiglu_boundary(gpu.get()),swiglu_boundary(gpu.get(),4,3),swiglu_boundary(gpu.get(),4,3,7.0f),swiglu_boundary(gpu.get(),2,2,0.1f)});
        report["swiglu_range"]=swiglu_range();
        report["universal_chain_compatibility_proven"]=false;
        for(auto type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_MXFP4,GGML_TYPE_Q8_0})report["synthetic"].push_back(synthetic(gpu.get(),type));
        report["synthetic"].push_back(synthetic(gpu.get(),GGML_TYPE_IQ3_XXS,true));
        report["synthetic"].push_back(synthetic(gpu.get(),GGML_TYPE_MXFP4,false,0));
        for(int arg=1;arg<argc;++arg) {
            CpuExpertFixture fixture(argv[arg]);CpuLanePool p12(12),p16(16);
            const bool captured_reference=std::getenv("STRATA_DS4_TEST_REFERENCE_FIXTURES")!=nullptr;
            if(captured_reference)cpu_mmvq_reference_compute(fixture.work,p12,nullptr,{});else fixture.work.compute(p12);
            require(exact(fixture.output,fixture.expected),"12-thread capture replay");
            if(captured_reference)cpu_mmvq_reference_compute(fixture.work,p16,nullptr,{});else fixture.work.compute(p16);
            require(exact(fixture.output,fixture.expected),"16-thread capture replay");
            if(captured_reference)fixture.work.output=nullptr;
            json item={{"path",argv[arg]},{"threads_12_16_capture_exact",true},{"tokens",fixture.work.tokens},{"experts",json::array()}};
            for(size_t i=0;i<fixture.work.experts.size();++i)item["experts"].push_back(audit(gpu.get(),fixture.work,i));
            report["fixtures"].push_back(item);
        }
        report["status"]="passed";std::cout<<report.dump(2)<<'\n';return 0;
    }catch(const std::exception &e){report["error"]=e.what();std::cout<<report.dump(2)<<'\n';std::cerr<<e.what()<<'\n';return 1;}
}
