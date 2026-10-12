// Exercise the real scheduler with three changing exact hash routes, nine
// CPU-backed matrices, cache pressure, input copies and CUDA graph replay.
#include "expert_transfer.h"
#include "ggml-cpp.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
static void check(bool ok,const char *what) {if(!ok)throw std::runtime_error(what);}
static void env(const char *key,const char *value) {
#ifdef _WIN32
    _putenv_s(key,value);
#else
    setenv(key,value,1);
#endif
}
struct Graph {
    ggml_context_ptr weights,tables,ctx;
    ggml_backend_buffer_ptr weight_buffer,table_buffer;
    ggml_backend_sched_t sched=nullptr;
    ggml_tensor *x,*tokens,*output,*table[3];
    ggml_cgraph *graph;
    Graph(ggml_backend_t cpu,ggml_backend_t gpu,int n,bool parallel,bool cpu_input,const std::vector<uint8_t> &packed):
        weights(ggml_init({9*ggml_tensor_overhead(),nullptr,true})),
        tables(ggml_init({3*ggml_tensor_overhead(),nullptr,true})),
        ctx(ggml_init({64*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true})) {
        ggml_tensor *w[9];
        for(auto &t:w)t=ggml_new_tensor_3d(weights.get(),GGML_TYPE_Q8_0,256,256,16);
        weight_buffer.reset(ggml_backend_alloc_ctx_tensors(weights.get(),cpu));
        check(bool(weight_buffer),"weight allocation");
        ggml_backend_buffer_set_usage(weight_buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for(auto *t:w)ggml_backend_tensor_set(t,packed.data(),0,packed.size());
        for(auto &t:table)t=ggml_new_tensor_2d(tables.get(),GGML_TYPE_I32,6,32);
        table_buffer.reset(ggml_backend_alloc_ctx_tensors(tables.get(),gpu));
        check(bool(table_buffer),"table allocation");
        ggml_backend_buffer_set_usage(table_buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for(int h=0;h<3;++h) {
            int32_t rows[6*32];
            for(int i=0;i<6*32;++i)rows[i]=(i*7+h*3)%16;
            ggml_backend_tensor_set(table[h],rows,0,sizeof(rows));
        }
        x=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,256,1,n);ggml_set_input(x);
        tokens=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,n);ggml_set_input(tokens);
        output=x;
        for(int h=0;h<3;++h) {
            auto *ids=ggml_get_rows(ctx.get(),table[h],tokens);
            for(int j=0;j<3;++j)output=ggml_silu(ctx.get(),ggml_mul_mat_id(ctx.get(),w[h*3+j],output,ids));
        }
        ggml_set_output(output);
        graph=ggml_new_graph(ctx.get());ggml_build_forward_expand(graph,output);
        ggml_backend_t backends[]={gpu,cpu};
        sched=ggml_backend_sched_new(backends,nullptr,2,GGML_DEFAULT_GRAPH_SIZE,parallel,true);
        ggml_backend_sched_set_tensor_backend(sched,x,gpu);
        ggml_backend_sched_set_tensor_backend(sched,tokens,cpu_input?cpu:gpu);
        for(int i=0;i<ggml_graph_n_nodes(graph);++i)
            ggml_backend_sched_set_tensor_backend(sched,ggml_graph_node(graph,i),gpu);
        check(ggml_backend_sched_alloc_graph(sched,graph),"scheduler allocation");
    }
    ~Graph(){if(sched)ggml_backend_sched_free(sched);}
    std::vector<float> run(int seed,int early) {
        env("STRATA_DS4_HASH_PREFETCH",early==2?"2":early==1?"1":"0");
        std::vector<float> input(ggml_nelements(x));
        std::vector<int32_t> ids(ggml_nelements(tokens));
        for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i+seed)*0.07f)*0.05f;
        for(size_t i=0;i<ids.size();++i)ids[i]=int32_t((i*3+seed*7)%32);
        ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
        ggml_backend_tensor_set(tokens,ids.data(),0,ids.size()*4);
        check(ggml_backend_sched_graph_compute(sched,graph)==GGML_STATUS_SUCCESS,"scheduler compute");
        std::vector<float> out(ggml_nelements(output));
        ggml_backend_tensor_get(output,out.data(),0,out.size()*4);return out;
    }
};
int main() {
    try {
        env("STRATA_EXPERT_PIPELINE","1");env("STRATA_EXPERT_STAGE_MIB","1");
        env("STRATA_DS4_DIRECT_CACHE","1");env("STRATA_DS4_CACHE_COPY_BATCH","1");
        env("STRATA_DS4_ORDERED_EXPERTS","1");
        ggml_backend_load_all();
        ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr));
        auto *dev=ggml_backend_dev_by_name("CUDA0");if(!cpu || !dev)return 77;
        auto reg=ggml_backend_dev_backend_reg(dev);
        auto stats=(StrataExpertStats)ggml_backend_reg_get_proc_address(reg,"strata_expert_stats");
        auto read=(StrataExpertHashRead)ggml_backend_reg_get_proc_address(reg,"strata_expert_hash_read");
        auto early=(StrataExpertEarlyPlan)ggml_backend_reg_get_proc_address(reg,"strata_expert_early_plan");
        auto plan=(StrataExpertPlan)ggml_backend_reg_get_proc_address(reg,"strata_expert_plan");
        auto finish=(StrataExpertFinish)ggml_backend_reg_get_proc_address(reg,"strata_expert_finish");
        check(stats && read && early && plan && finish,"missing hooks");
        std::vector<float> weight(256*256*16);
        for(size_t i=0;i<weight.size();++i)weight[i]=std::cos(float(i)*0.017f)*0.03f;
        std::vector<uint8_t> packed(ggml_row_size(GGML_TYPE_Q8_0,256)*256*16);
        check(ggml_quantize_chunk(GGML_TYPE_Q8_0,weight.data(),packed.data(),0,256*16,256,nullptr)==packed.size(),"quantize");
        int cases=0;
        for(int mode:{1,2})for(bool cpu_input:{false,true})for(bool parallel:{false,true})
        for(const char *budget:{"0","1","4"})for(int n:{1,2,3,8,9}) {
            env("STRATA_EXPERT_CACHE_MIB",budget);
            ggml_backend_ptr a(ggml_backend_dev_init(dev,nullptr)),b(ggml_backend_dev_init(dev,nullptr));
            Graph control(cpu.get(),a.get(),n,parallel,cpu_input,packed),candidate(cpu.get(),b.get(),n,parallel,cpu_input,packed);
            for(int seed=0;seed<4;++seed) {
                const auto expected=control.run(seed,0),actual=candidate.run(seed,mode);
                check(expected.size()==actual.size() && std::memcmp(expected.data(),actual.data(),expected.size()*4)==0,"hash scheduler changed output");
                ++cases;
            }
            StrataExpertCounters total;stats(b.get(),&total);
            const int expected=(!parallel && n<=8)?12:0;
            check(total.early_plans==expected && total.early_reuses==expected && !total.early_mismatches,"prefetch activation/reuse");
            check(!total.pipeline_fallbacks && !total.pipeline_unused,"prefetch changed transfer sequence");
            StrataHashRoutes routes;routes.count=1;routes.n_tokens=1;routes.tables[0]=candidate.table[0];
            routes.tokens[0]=-1;check(!read(b.get(),&routes),"negative token accepted");
            routes.tokens[0]=32;check(!read(b.get(),&routes),"out of range token accepted");
            routes.tokens[0]=0;routes.count=4;check(!read(b.get(),&routes),"too many tables accepted");
            // Drain an abandoned early plan and reject a stale transfer plan.
            StrataExpertSlice slice{packed.data(),packed.size(),true};
            early(b.get(),&slice,1);finish(b.get());
            early(b.get(),&slice,1);auto changed=slice;changed.bytes-=512;
            plan(b.get(),&changed,1);finish(b.get());stats(b.get(),&total);
            check(total.early_mismatches==1,"mismatched plan was reused");
        }
        std::printf("Hash prefetch: %d exact comparisons, guards and abandoned plans passed\n",cases);return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
