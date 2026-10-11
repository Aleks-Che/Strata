// Prove that cached MMVQ reads work with poisoned scratch, not merely that the
// output agrees after an unnoticed D2D materialization. All matrix math is CUDA.
#include "expert_transfer.h"
#include "ggml-cpp.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
static void check(bool b,const char *s){if(!b)throw std::runtime_error(s);}
static void env(const char *k,const char *v){
#ifdef _WIN32
    _putenv_s(k,v);
#else
    setenv(k,v,1);
#endif
}
struct Api {
    StrataExpertCopy copy; StrataExpertPlan plan; StrataExpertFinish finish;
    StrataExpertPrepare prepare; StrataExpertFlush flush; StrataExpertStats stats;
    StrataExpertControl control; StrataExpertBudget budget;
    Api(ggml_backend_t gpu) {
        auto r=ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu));
#define LOAD(field,Type) field=(Type)ggml_backend_reg_get_proc_address(r,"strata_expert_" #field);check(field,"missing " #field)
        LOAD(copy,StrataExpertCopy);LOAD(plan,StrataExpertPlan);LOAD(finish,StrataExpertFinish);
        LOAD(prepare,StrataExpertPrepare);LOAD(flush,StrataExpertFlush);LOAD(stats,StrataExpertStats);
        LOAD(control,StrataExpertControl);LOAD(budget,StrataExpertBudget);
#undef LOAD
    }
};
struct Graph {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor *w,*ids,*consumer,*out;
    ggml_cgraph *graph;
    Graph(ggml_backend_t gpu,ggml_type type,int n,int lanes,int fused,const std::vector<uint8_t>& packed)
      :ctx(ggml_init({ggml_tensor_overhead()*16+ggml_graph_overhead(),nullptr,true})) {
        w=ggml_new_tensor_3d(ctx.get(),type,512,128,8);
        auto *x=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,512,lanes,n);
        auto *raw_ids=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,256,n);
        ids=ggml_view_2d(ctx.get(),raw_ids,6,n,raw_ids->nb[1],0);
        consumer=ggml_mul_mat_id(ctx.get(),w,x,ids);out=consumer;
        ggml_tensor *other=nullptr;
        if(fused) {
            other=ggml_new_tensor_3d(ctx.get(),type,512,128,8);
            auto *gate=ggml_mul_mat_id(ctx.get(),other,x,ids);
            out=fused==1?ggml_swiglu_split(ctx.get(),consumer,gate):ggml_swiglu_split(ctx.get(),gate,consumer);
        }
        graph=ggml_new_graph(ctx.get());ggml_build_forward_expand(graph,out);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(),gpu));check(bool(buffer),"allocation");
        ggml_backend_buffer_clear(buffer.get(),0);
        ggml_backend_tensor_set(w,packed.data(),0,packed.size());
        if(other)ggml_backend_tensor_set(other,packed.data(),0,packed.size());
        std::vector<float> input(ggml_nelements(x));
        for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i)*0.03f)*0.1f;
        ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
        route(0);
    }
    void route(int turn) {
        for(int t=0;t<ids->ne[1];++t) {
            int32_t row[6];for(int i=0;i<6;++i)row[i]=(i+t+turn)%8;
            ggml_backend_tensor_set(ids,row,size_t(t)*ids->nb[1],sizeof(row));
        }
    }
    std::vector<float> run(ggml_backend_t gpu) {
        check(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"graph compute");
        std::vector<float> v(ggml_nelements(out));ggml_backend_tensor_get(out,v.data(),0,v.size()*4);return v;
    }
};
int main() {
    try {
        env("STRATA_DS4_DIRECT_CACHE","1");env("STRATA_EXPERT_CACHE_MIB","4");
        env("STRATA_EXPERT_STAGE_MIB","1");env("STRATA_DS4_CACHE_COPY_BATCH","1");
        ggml_backend_load_all();
        ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr));
        auto *dev=ggml_backend_dev_by_name("CUDA0");if(!cpu || !dev)return 77;
        size_t cases=0;uint64_t hits=0,skipped=0;
        for(auto type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_MXFP4,GGML_TYPE_Q8_0}) {
            std::vector<float> w(512*128*8),importance(512,1);
            for(size_t i=0;i<w.size();++i)w[i]=std::cos(float(i)*0.017f)*0.1f;
            std::vector<uint8_t> packed(ggml_row_size(type,512)*128*8);
            check(ggml_quantize_chunk(type,w.data(),packed.data(),0,128*8,512,importance.data())==packed.size(),"quantize");
            for(int n:{1,2,3,4,8,9})for(int lanes:{1,6})for(int fused:{0,1,2}) {
                ggml_backend_ptr gpu(ggml_backend_dev_init(dev,nullptr));Api a(gpu.get());a.budget(4);
                ggml_context_ptr hc(ggml_init({ggml_tensor_overhead(),nullptr,true}));
                auto *src=ggml_new_tensor_3d(hc.get(),type,512,128,8);
                ggml_backend_buffer_ptr hb(ggml_backend_alloc_ctx_tensors(hc.get(),cpu.get()));
                ggml_backend_tensor_set(src,packed.data(),0,packed.size());
                Graph reference(gpu.get(),type,n,lanes,fused,packed),candidate(gpu.get(),type,n,lanes,fused,packed);
                std::vector<StrataExpertSlice> slices;
                for(int id=0;id<8;++id)slices.push_back({(uint8_t *)src->data+id*src->nb[2],src->nb[2]+(id<7?512:0),true});
                // Warm only half: first prepared graph includes hits and misses.
                a.plan(gpu.get(),slices.data(),slices.size());a.copy(gpu.get(),src,candidate.w,0,3,1);a.finish(gpu.get());
                auto partial=*candidate.w;auto unsupported=*candidate.consumer;
                partial.ne[1]=127;partial.nb[2]=partial.nb[1]*127;partial.nb[3]=partial.nb[2]*8;
                unsupported.src[0]=&partial;check(!a.prepare(gpu.get(),&unsupported),"partial row tile accepted");
                partial=*candidate.w;partial.view_src=candidate.w;
                check(!a.prepare(gpu.get(),&unsupported),"weight view accepted");
                for(int turn=0;turn<7;++turn) {
                    reference.route(turn);candidate.route(turn);auto expected=reference.run(gpu.get());
                    std::vector<uint8_t> poison(packed.size(),0xA5);ggml_backend_tensor_set(candidate.w,poison.data(),0,poison.size());
                    a.plan(gpu.get(),slices.data(),slices.size());
                    bool direct=a.prepare(gpu.get(),candidate.consumer);
                    StrataExpertCounters before,after;a.stats(gpu.get(),&before);
                    a.copy(gpu.get(),src,candidate.w,0,7,n);a.flush(gpu.get());a.stats(gpu.get(),&after);
                    if(direct) {
                        check(after.direct_hits>before.direct_hits,"no direct hits");
                        ggml_backend_synchronize(gpu.get());
                        std::vector<uint8_t> interior(64);
                        ggml_backend_tensor_get(candidate.w,interior.data(),1024,interior.size());
                        check(std::all_of(interior.begin(),interior.end(),[](uint8_t b){return b==0xA5;}),"cached matrix was materialized");
                        if(turn>0)check(after.d2d_bytes==before.d2d_bytes,"all-hit direct still copied D2D");
                    }
                    auto actual=candidate.run(gpu.get());
                    check(actual.size()==expected.size() && std::memcmp(actual.data(),expected.data(),actual.size()*4)==0,"direct output differs");
                    for(float v:actual)check(std::isfinite(v),"nonfinite output");
                    a.finish(gpu.get());++cases;
                }
                // Abandon a prepared operation before launch. Retirement must
                // materialize its held matrices before a policy change/free.
                a.plan(gpu.get(),slices.data(),slices.size());
                if(a.prepare(gpu.get(),candidate.consumer)) {
                    a.copy(gpu.get(),src,candidate.w,0,7,n);a.flush(gpu.get());a.finish(gpu.get());
                    ggml_backend_synchronize(gpu.get());
                    std::vector<uint8_t> restored(packed.size());ggml_backend_tensor_get(candidate.w,restored.data(),0,restored.size());
                    check(restored==packed,"abandoned binding was not materialized");
                }
                // Reuse the already captured graph while trimming to mixed
                // hits, zero hits, regrowth and different cached addresses.
                for(uint64_t limit:{4,0,8,4}) {
                    StrataVramPolicy policy;policy.mode=1;policy.matrices=limit;
                    StrataVramStatus status;check(a.control(0,&policy,&status),"cache control");
                    check(status.matrices<=limit,"cache did not shrink");
                    for(int turn=0;turn<3;++turn) {
                        reference.route(turn+3);candidate.route(turn+3);auto expected=reference.run(gpu.get());
                        std::vector<uint8_t> poison(packed.size(),0xA5);ggml_backend_tensor_set(candidate.w,poison.data(),0,poison.size());
                        a.plan(gpu.get(),slices.data(),slices.size());a.prepare(gpu.get(),candidate.consumer);
                        a.copy(gpu.get(),src,candidate.w,0,7,n);a.flush(gpu.get());
                        auto actual=candidate.run(gpu.get());
                        check(actual.size()==expected.size() && std::memcmp(actual.data(),expected.data(),actual.size()*4)==0,"pressure/replay output differs");
                        a.finish(gpu.get());++cases;
                    }
                }
                StrataVramPolicy configured;StrataVramStatus status;a.control(0,&configured,&status);
                StrataExpertCounters total;a.stats(gpu.get(),&total);hits+=total.direct_hits;skipped+=total.direct_bytes;
                // Free the CUDA cache before the host mapping backing its keys.
                gpu.reset();
            }
        }
        std::printf("direct cache: %zu exact cases, %llu hits, %llu skipped D2D bytes\n",cases,(unsigned long long)hits,(unsigned long long)skipped);
        check(hits>0,"feature never exercised");return 0;
    } catch(const std::exception &e) {std::fprintf(stderr,"direct cache failure: %s\n",e.what());return 1;}
}
