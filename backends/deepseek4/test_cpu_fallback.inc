#include "cpu_cost_profile.hpp"
#include "cpu_cost_refresh.hpp"
#include "cpu_cost_device.cuh"
#include "nlohmann/json.hpp"

// Artificial costs are a correctness fixture, never performance evidence.
// Alternate feasible/impossible shapes INSIDE one graph to exercise retiring
// queue consumers before ordinary transport reuses their scratch and cache.
static void gpu_fallback_test(ggml_backend_t cpu,ggml_backend_dev_t dev,
        StrataExpertStats stats,StrataCpuExpertPhase phase) {
#ifdef _WIN32
    using json=nlohmann::json;
    env("STRATA_DS4_CPU_POLICY","auto");env("STRATA_DS4_CPU_THREADS","4");
    env("STRATA_DS4_CPU_SHARED_RAM","1");env("STRATA_DS4_RAM_CACHE_MIB","32");
    env("STRATA_DS4_CPU_RAM_MIB","0");env("STRATA_DS4_CPU_MAX_EXPERTS","1");
    env("STRATA_DS4_GPU_FETCH_QUEUE","1");env("STRATA_DS4_CPU_GPU_QUEUE","1");
    env("STRATA_DS4_CPU_MMVQ_FAST","1");env("STRATA_DS4_SWIGLU_COMPAT","1");
    env("STRATA_DS4_CPU_EMPTY_GPU_FALLBACK","1");env("STRATA_DS4_CPU_COST_AUTO","1");
    env("STRATA_DS4_CPU_BUDGET_US","11.5");env("STRATA_DS4_CPU_COST_BURST","0");
    json profile={{"version",1},{"build",strata_ds4::cpu_cost_build_id()},
        {"machine",strata_ds4::cpu_cost_machine()},{"gpu",strata_cpu_cost_gpu(0)},
        {"mode",strata_ds4::cpu_cost_mode()},{"threads",4},{"profiles",json::array()},
        {"test_fixture","synthetic costs and declared working set for transport transitions only"}};
    for(auto type:{GGML_TYPE_Q8_0,GGML_TYPE_IQ3_XXS}) {
        json p={{"gate",int(GGML_TYPE_Q8_0)},{"up",int(type)},{"down",int(type)},
            {"columns",D},{"hidden",H},{"clamp",0.1f},{"resident_weight_bytes",128ULL<<20},
            {"copy_us_per_byte",0.1},{"cases",json::array()},{"boundary",json::array()}};
        for(int t=1;t<=4;++t) {
            for(int count=1;count<=t;++count)p["cases"].push_back({{"tokens",t},{"counts",{count}},
                {"compute_us",type==GGML_TYPE_Q8_0?10:10000}});
            for(int r=1;r<=8;++r)p["boundary"].push_back({{"tokens",t},{"rows",r},{"us",1}});
        }
        profile["profiles"].push_back(p);
    }
    const auto path=std::filesystem::absolute("fallback-profile-"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".json");
    {std::ofstream file(path);file<<profile.dump();}
    env("STRATA_DS4_CPU_COST_PROFILE",path.string().c_str());
    int replays=0;
    for(int n:{1,3,4})for(bool ends_cpu:{false,true}) {
        Weights a(cpu,ends_cpu?GGML_TYPE_Q8_0:GGML_TYPE_IQ3_XXS,1),
            b(cpu,ends_cpu?GGML_TYPE_IQ3_XXS:GGML_TYPE_Q8_0,2),
            c(cpu,ends_cpu?GGML_TYPE_Q8_0:GGML_TYPE_IQ3_XXS,3);
        a.register_files();b.register_files();c.register_files();
        ggml_backend_ptr hybrid(ggml_backend_dev_init(dev,nullptr)),gpu(ggml_backend_dev_init(dev,nullptr));
        Graph actual(a,cpu,hybrid.get(),n,false,0.1f,&b,&c,true),reference(a,cpu,gpu.get(),n,false,0.1f,&b,&c,true);
        std::vector<float> x(n*D);std::vector<int32_t> ids(n*E);
        for(int t=0;t<n;++t){ids[t*E]=1;ids[t*E+1]=2;}
        for(size_t i=0;i<x.size();++i)x[i]=0.8f*std::cos(float(i)*0.031f);
        phase(false);for(int i=0;i<24;++i)actual.run(x,ids);phase(true);
        for(int transition=0;transition<4;++transition) {
            for(size_t i=0;i<x.size();++i)x[i]=0.8f*std::cos(float(i+23*transition)*0.031f);
            env("STRATA_DS4_CPU_THREADS","0");const auto expected=reference.run(x,ids);
            for(int i=0;i<3;++i)check(actual.run(x,ids)==expected,"GPU phase retained CPU graph state");
            env("STRATA_DS4_CPU_THREADS","4");StrataExpertCounters before,after;stats(hybrid.get(),&before);
            for(int i=0;i<6;++i){check(actual.run(x,ids)==expected,"fallback transition changed logits");++replays;}
            stats(hybrid.get(),&after);
            check(after.cpu_gpu_fallback_branches-before.cpu_gpu_fallback_branches==uint64_t(6*(ends_cpu?1:2)),
                "impossible branch did not use ordinary GPU transport exactly once");
            check(after.cpu_queue_jobs>before.cpu_queue_jobs,"feasible branch lost actual CPU jobs");
            check(after.gpu_fetch_branches-before.gpu_fetch_branches==uint64_t(6*(ends_cpu?2:1)),
                "fallback branch unexpectedly published a GPU fetch queue");
        }
        StrataExpertCounters end;stats(hybrid.get(),&end);
        if(!std::getenv("GGML_CUDA_DISABLE_GRAPHS"))
            check(end.cpu_graph_changes && end.cuda_graph_captures && end.cuda_graph_replays,"no real graph capture/replay");
        else check(!end.cuda_graph_captures && !end.cuda_graph_replays,"disabled graphs control captured");
        std::printf("FALLBACK tokens=%d ends_cpu=%d bypasses=%llu cpu_jobs=%llu captures=%llu replays=%llu\n",n,int(ends_cpu),
            (unsigned long long)end.cpu_gpu_fallback_branches,(unsigned long long)end.cpu_queue_jobs,
            (unsigned long long)end.cuda_graph_captures,(unsigned long long)end.cuda_graph_replays);
    }
    std::printf("GPU fallback: %d bit-exact mixed-transport replays\n",replays);
    {
        Weights w(cpu,GGML_TYPE_Q8_0);w.register_files();
        ggml_backend_ptr hybrid(ggml_backend_dev_init(dev,nullptr)),gpu(ggml_backend_dev_init(dev,nullptr));
        Graph actual(w,cpu,hybrid.get(),1,false,0.1f),reference(w,cpu,gpu.get(),1,false,0.1f);
        std::vector<float> x(D,0.125f);std::vector<int32_t> ids(E);ids[0]=1;ids[1]=2;
        phase(false);for(int i=0;i<24;++i)actual.run(x,ids);phase(true);
        env("STRATA_DS4_CPU_THREADS","0");const auto expected=reference.run(x,ids);env("STRATA_DS4_CPU_THREADS","4");
        strata_ds4::CpuExpertWork work;work.gate_type=work.up_type=work.down_type=GGML_TYPE_Q8_0;
        work.columns=D;work.hidden=H;work.clamp=0.1f;work.topk=K;
        std::string reason;auto base=strata_ds4::cpu_cost_load(path.string(),work,4,strata_cpu_cost_gpu(0),reason);
        check(base.enabled,"refresh fixture profile rejected");base.budget_us_per_token=11.5f;
        auto refresh=strata_ds4::cpu_cost_refresh_for(path.string(),work,4,strata_cpu_cost_gpu(0),base);
        check(bool(refresh),"refresh registry unavailable");
        auto replay=[&](bool fallback) {
            StrataExpertCounters before,after;stats(hybrid.get(),&before);
            for(int i=0;i<6;++i)check(actual.run(x,ids)==expected,"profile generation changed logits");
            stats(hybrid.get(),&after);
            check(after.cpu_gpu_fallback_branches-before.cpu_gpu_fallback_branches==uint64_t(fallback?6:0),"fallback cached a stale profile generation");
            check((after.cpu_queue_jobs>before.cpu_queue_jobs)==!fallback,"profile generation lost CPU/GPU transition");
        };
        replay(false);
        for(double observed:{30.,1.}) {
            const auto before=refresh->snapshot().generation;
            for(int request=0;request<2;++request) {
                strata_ds4::cpu_cost_request_boundary(0);
                for(int i=0;i<16;++i)refresh->observe(before,1,1,false,observed);
                check(refresh->snapshot().generation==before,"profile changed inside request");
                strata_ds4::cpu_cost_request_boundary(2);
            }
            check(refresh->snapshot().generation==before+1,"synthetic observations did not publish successor");
            replay(observed>10);
        }
        std::printf("GPU fallback refresh: generations 0/1/2, CPU/GPU/CPU, 18 exact replays\n");
    }
#else
    throw std::runtime_error("GPU fallback fixture requires native Windows file registration");
#endif
}
