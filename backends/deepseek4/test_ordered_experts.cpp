// Exercise the actual scheduler: CPU-backed weights, GPU-computed strided
// routes, three dependent expert projections, cache pressure and graph replay.
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
    ggml_context_ptr weights,ctx;
    ggml_backend_buffer_ptr weight_buffer;
    ggml_backend_sched_t sched=nullptr;
    ggml_tensor *x,*scores,*output;
    ggml_cgraph *graph;
    Graph(ggml_backend_t cpu,ggml_backend_t gpu,int n,bool parallel,const std::vector<uint8_t> &packed):
        weights(ggml_init({3*ggml_tensor_overhead(),nullptr,true})),
        ctx(ggml_init({32*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true})) {
        ggml_tensor *w[3];
        for(auto &t:w)t=ggml_new_tensor_3d(weights.get(),GGML_TYPE_Q8_0,256,256,16);
        weight_buffer.reset(ggml_backend_alloc_ctx_tensors(weights.get(),cpu));
        check(bool(weight_buffer),"weight allocation");
        ggml_backend_buffer_set_usage(weight_buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for(auto *t:w)ggml_backend_tensor_set(t,packed.data(),0,packed.size());
        x=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,256,1,n);ggml_set_input(x);
        scores=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,16,n);ggml_set_input(scores);
        auto *sorted=ggml_argsort(ctx.get(),scores,GGML_SORT_ORDER_DESC);
        auto *ids=ggml_view_2d(ctx.get(),sorted,6,n,sorted->nb[1],0);
        output=x;
        for(auto *t:w)output=ggml_silu(ctx.get(),ggml_mul_mat_id(ctx.get(),t,output,ids));
        ggml_set_output(output);
        graph=ggml_new_graph(ctx.get());ggml_build_forward_expand(graph,output);
        ggml_backend_t backends[]={gpu,cpu};
        sched=ggml_backend_sched_new(backends,nullptr,2,GGML_DEFAULT_GRAPH_SIZE,parallel,true);
        ggml_backend_sched_set_tensor_backend(sched,x,gpu);
        ggml_backend_sched_set_tensor_backend(sched,scores,gpu);
        for(int i=0;i<ggml_graph_n_nodes(graph);++i)
            ggml_backend_sched_set_tensor_backend(sched,ggml_graph_node(graph,i),gpu);
        check(ggml_backend_sched_alloc_graph(sched,graph),"scheduler allocation");
    }
    ~Graph(){if(sched)ggml_backend_sched_free(sched);}
    std::vector<float> run(int seed,bool ordered) {
        env("STRATA_DS4_ORDERED_EXPERTS",ordered?"1":"0");
        std::vector<float> input(ggml_nelements(x)),route(ggml_nelements(scores));
        for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i+seed)*0.07f)*0.05f;
        for(size_t i=0;i<route.size();++i)route[i]=float((i*7+seed*3)%16);
        ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
        ggml_backend_tensor_set(scores,route.data(),0,route.size()*4);
        check(ggml_backend_sched_graph_compute(sched,graph)==GGML_STATUS_SUCCESS,"scheduler compute");
        std::vector<float> out(ggml_nelements(output));
        ggml_backend_tensor_get(output,out.data(),0,out.size()*4);return out;
    }
};
int main() {
    try {
        env("STRATA_EXPERT_PIPELINE","1");env("STRATA_EXPERT_STAGE_MIB","1");
        env("STRATA_DS4_DIRECT_CACHE","1");env("STRATA_DS4_CACHE_COPY_BATCH","1");
        ggml_backend_load_all();
        ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr));
        auto *dev=ggml_backend_dev_by_name("CUDA0");if(!cpu || !dev)return 77;
        std::vector<float> weight(256*256*16);
        for(size_t i=0;i<weight.size();++i)weight[i]=std::cos(float(i)*0.017f)*0.03f;
        std::vector<uint8_t> packed(ggml_row_size(GGML_TYPE_Q8_0,256)*256*16);
        check(ggml_quantize_chunk(GGML_TYPE_Q8_0,weight.data(),packed.data(),0,256*16,256,nullptr)==packed.size(),"quantize");
        int cases=0;
        for(bool parallel:{false,true})for(const char *budget:{"0","1","4"})for(int n:{1,2,3,8,9}) {
            env("STRATA_EXPERT_CACHE_MIB",budget);
            ggml_backend_ptr a(ggml_backend_dev_init(dev,nullptr)),b(ggml_backend_dev_init(dev,nullptr));
            Graph control(cpu.get(),a.get(),n,parallel,packed),candidate(cpu.get(),b.get(),n,parallel,packed);
            for(int seed=0;seed<12;++seed) {
                const auto expected=control.run(seed,false),actual=candidate.run(seed,true);
                check(expected.size()==actual.size() && std::memcmp(expected.data(),actual.data(),expected.size()*4)==0,"ordered scheduler changed output");
                ++cases;
            }
        }
        std::printf("Ordered scheduler: %d exact comparisons passed\n",cases);return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
