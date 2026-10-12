#include "expert_transfer.h"
#include "cpu_lane.hpp"
#include "ggml-cpp.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
static void check(bool b,const char *s){if(!b)throw std::runtime_error(s);}
static void env(const char *k,const char *v){
#ifdef _WIN32
    _putenv_s(k,v);
#else
    setenv(k,v,1);
#endif
}
struct Graph {
    ggml_context_ptr wc,ctx;ggml_backend_buffer_ptr wb,alternate_raw;ggml_backend_sched_t sched=nullptr;
    void *original_raw=nullptr;
    ggml_backend_t gpu_backend;
    ggml_tensor *w,*x,*scores,*raw,*out;ggml_cgraph *g;
    std::vector<float> input,route;
    Graph(ggml_backend_t cpu,ggml_backend_t gpu,ggml_type type,int n,bool parallel,int lanes,const std::vector<uint8_t>& packed):
        wc(ggml_init({ggml_tensor_overhead(),nullptr,true})),ctx(ggml_init({32*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true})),gpu_backend(gpu) {
        w=ggml_new_tensor_3d(wc.get(),type,256,256,16);
        wb.reset(ggml_backend_alloc_ctx_tensors(wc.get(),cpu));check(bool(wb),"weights");
        ggml_backend_buffer_set_usage(wb.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(w,packed.data(),0,packed.size());
        x=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,256,lanes,n);ggml_set_input(x);
        // An input leaf with the same padded row stride as a top-k view. A
        // leading argsort in this tiny single-op graph would copy all weights
        // before the routes exist and never exercise the selected-copy path.
        scores=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,6,n);
        scores->nb[1]=16*4;scores->nb[2]=scores->nb[3]=scores->nb[1]*n;ggml_set_input(scores);
        auto *ids=scores;
        raw=ggml_mul_mat_id(ctx.get(),w,x,ids);ggml_set_output(raw);
        out=ggml_silu(ctx.get(),raw);ggml_set_output(out);
        g=ggml_new_graph(ctx.get());ggml_build_forward_expand(g,out);
        ggml_backend_t backends[]={gpu,cpu};sched=ggml_backend_sched_new(backends,nullptr,2,GGML_DEFAULT_GRAPH_SIZE,parallel,true);
        ggml_backend_sched_set_tensor_backend(sched,x,gpu);ggml_backend_sched_set_tensor_backend(sched,scores,gpu);
        for(int i=0;i<ggml_graph_n_nodes(g);++i)ggml_backend_sched_set_tensor_backend(sched,ggml_graph_node(g,i),gpu);
        check(ggml_backend_sched_alloc_graph(sched,g),"allocate");
        original_raw=raw->data;
        alternate_raw.reset(ggml_backend_alloc_buffer(gpu,ggml_nbytes(raw)));
        check(bool(alternate_raw),"alternate output allocation");
    }
    ~Graph(){if(sched)ggml_backend_sched_free(sched);}
    std::vector<float> run(int seed,int threads) {
        env("STRATA_DS4_CPU_THREADS",std::to_string(threads).c_str());
        // Capture the initial layout on GPU, then change an output address
        // during two CPU-lane executions before returning to GPU replay.
        // Bypassing capture must not mark unrecorded properties as captured.
        if(seed==4) {
            // Reallocation supplies fresh split UIDs, just like a rebuilt
            // model graph. Mutating addresses under a reused UID is invalid.
            ggml_backend_sched_reset(sched);
            raw->src[0]=w;raw->src[1]=x;raw->src[2]=scores;
            ggml_backend_sched_set_tensor_backend(sched,x,gpu_backend);
            ggml_backend_sched_set_tensor_backend(sched,scores,gpu_backend);
            for(int i=0;i<ggml_graph_n_nodes(g);++i)
                ggml_backend_sched_set_tensor_backend(sched,ggml_graph_node(g,i),gpu_backend);
            check(ggml_backend_sched_alloc_graph(sched,g),"reallocate graph");
        }
        raw->data=seed>=4?ggml_backend_buffer_get_base(alternate_raw.get()):original_raw;
        input.resize(ggml_nelements(x));route.resize(size_t(scores->ne[1])*16);
        for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i+seed)*0.071f)*0.2f;
        for(size_t i=0;i<route.size();++i)route[i]=float((i*7+seed*3+(i/16)*5)%16);
        std::vector<int32_t> routed(route.size());
        for(int t=0;t<scores->ne[1];++t) {
            for(int j=0;j<16;++j)routed[t*16+j]=j;
            std::sort(routed.begin()+t*16,routed.begin()+(t+1)*16,[&](int l,int r){return route[t*16+l]>route[t*16+r];});
        }
        ggml_backend_tensor_set(x,input.data(),0,input.size()*4);ggml_backend_tensor_set(scores,routed.data(),0,ggml_nbytes(scores));
        check(ggml_backend_sched_graph_compute(sched,g)==GGML_STATUS_SUCCESS,"compute");
        std::vector<float> r(ggml_nelements(raw));ggml_backend_tensor_get(raw,r.data(),0,r.size()*4);
        std::vector<float> post(ggml_nelements(out));ggml_backend_tensor_get(out,post.data(),0,post.size()*4);
        for(size_t i=0;i<r.size();++i)check(std::isfinite(post[i]) && std::abs(post[i]-r[i]/(1+std::exp(-r[i])))<1e-6f,"downstream saw stale CPU output");
        return r;
    }
};
int main(){try {
    env("STRATA_DS4_CPU_MATRIX_LEGACY","1");
    env("STRATA_EXPERT_PIPELINE","1");env("STRATA_EXPERT_STAGE_MIB","1");env("STRATA_EXPERT_CACHE_MIB","0");
    env("GGML_OP_OFFLOAD_MIN_BATCH","1");
    env("STRATA_DS4_DIRECT_CACHE","1");env("STRATA_DS4_ORDERED_EXPERTS","1");env("STRATA_DS4_EARLY_HITS","0");
    ggml_backend_load_all();ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr));
    auto *dev=ggml_backend_dev_by_name("CUDA0");if(!dev || !cpu)return 77;
    auto stats=(StrataExpertStats)ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),"strata_expert_stats");
    int cases=0;uint64_t ops=0,bytes=0;double max_diff=0;
    for(auto type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_MXFP4,GGML_TYPE_Q8_0}) {
        check(strata_ds4::cpu_lane_format(type,256),"CPU type unavailable");
        std::vector<float> f(256*256*16),importance(256,1);
        for(size_t i=0;i<f.size();++i)f[i]=std::cos(float(i)*0.017f)*0.05f;
        std::vector<uint8_t> packed(ggml_row_size(type,256)*256*16);
        check(ggml_quantize_chunk(type,f.data(),packed.data(),0,256*16,256,importance.data())==packed.size(),"quantize");
        for(int nt:{1,3,8,9})for(int lanes:{1,6})for(int cap:{1,6})for(bool parallel:{false,true}) {
            env("STRATA_DS4_CPU_MAX_EXPERTS",std::to_string(cap).c_str());
            ggml_backend_ptr ga(ggml_backend_dev_init(dev,nullptr)),gb(ggml_backend_dev_init(dev,nullptr));
            Graph a(cpu.get(),ga.get(),type,nt,parallel,lanes,packed),b(cpu.get(),gb.get(),type,nt,parallel,lanes,packed);
            for(int seed=0;seed<8;++seed) {
                const int threads=(seed==1 || seed==2 || seed==3 || seed==6)?0:4;
                auto expected=a.run(seed,0),actual=b.run(seed,threads);
                std::vector<uint8_t> q(strata_ds4::cpu_lane_activation_bytes(type,256));
                std::vector<float> ref(256);std::vector<int> ids(16);
                std::vector<int> chosen;
                for(int t=0;t<nt;++t) {
                    for(int j=0;j<16;++j)ids[j]=j;
                    std::sort(ids.begin(),ids.end(),[&](int l,int r){return b.route[t*16+l]>b.route[t*16+r];});
                    if(t==0)chosen.assign(ids.begin(),ids.begin()+cap);
                    for(int j=0;j<6;++j) {
                        const size_t off=(size_t(t)*6+j)*256;
                        if(threads && !parallel && nt<=8 && std::find(chosen.begin(),chosen.end(),ids[j])!=chosen.end()) {
                            strata_ds4::cpu_lane_quantize(type,256,b.input.data()+(size_t(t)*lanes+j%lanes)*256,q.data());
                            strata_ds4::cpu_lane_rows(type,256,packed.data()+ids[j]*b.w->nb[2],q.data(),ref.data(),0,256);
                            if(std::memcmp(ref.data(),actual.data()+off,256*4)!=0) {
                                StrataExpertCounters debug;stats(gb.get(),&debug);
                                std::fprintf(stderr,"type=%s n=%d lanes=%d cap=%d seed=%d token=%d expert=%d ops=%llu gpu_diff=%g cpu_diff=%g\n",
                                    ggml_type_name(type),nt,lanes,cap,seed,t,ids[j],(unsigned long long)debug.cpu_lane_ops,
                                    actual[off]-expected[off],actual[off]-ref[0]);
                                check(false,"CPU rows/strided IDs differ from native reference");
                            }
                        } else check(std::memcmp(expected.data()+off,actual.data()+off,256*4)==0,"GPU rows changed");
                        for(int k=0;k<256;++k)max_diff=std::max(max_diff,double(std::abs(actual[off+k]-expected[off+k])));
                    }
                }
                ++cases;
            }
            StrataExpertCounters s;stats(gb.get(),&s);ops+=s.cpu_lane_ops;bytes+=s.cpu_lane_saved_bytes;
            check(!s.pipeline_unused && !s.pipeline_fallbacks,"pipeline work lost");
            if(parallel || nt==9)check(!s.cpu_lane_ops,"guard failed");else check(s.cpu_lane_ops>0,"CPU lane not used");
        }
    }
    check(ops && bytes,"missing work");
    std::printf("CPU lane: %d cases, %llu operations, %llu skipped bytes, max CPU/GPU difference %.9g\n",cases,(unsigned long long)ops,(unsigned long long)bytes,max_diff);
    return 0;
}catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
