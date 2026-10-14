static void ordinary_cache_test(ggml_backend_t cpu,ggml_backend_dev_t dev,StrataExpertStats stats) {
    auto reg=ggml_backend_dev_backend_reg(dev);
    auto control=(StrataExpertControl)ggml_backend_reg_get_proc_address(reg,"strata_expert_control");
    auto plan=(StrataExpertPlanRoutes)ggml_backend_reg_get_proc_address(reg,"strata_expert_plan_routes");
    auto finish=(StrataExpertFinish)ggml_backend_reg_get_proc_address(reg,"strata_expert_finish");
    auto copy=(StrataExpertCopy)ggml_backend_reg_get_proc_address(reg,"strata_expert_copy");
    check(control && plan && finish && copy,"missing ordinary-cache extension");
    env("STRATA_EXPERT_CACHE_POLICY","frequency");env("STRATA_DS4_CPU_SHARED_RAM","1");
    env("STRATA_DS4_RAM_CACHE_MIB","16");env("STRATA_DS4_RAM_RESIDENCY","0");
    env("STRATA_DS4_RAM_DEMOTE_MIB","0");env("STRATA_DS4_RAM_ON_BYPASS","0");
    env("STRATA_DS4_GPU_FETCH_RAM_MIB","0");env("STRATA_DS4_GPU_FETCH_KEEP_RAM","0");
    env("STRATA_DS4_CPU_GPU_QUEUE","1");env("STRATA_DS4_GPU_FETCH_QUEUE","1");
    env("STRATA_DS4_GPU_FETCH_CACHE_SLOTS","8");env("STRATA_DS4_GPU_FETCH_DEMAND_CACHE","1");
    env("STRATA_DS4_GPU_FETCH_DEMAND_RANK","1");env("STRATA_DS4_GPU_FETCH_DEMAND_RANK_MIN_CACHE_MIB","0");
    env("STRATA_DS4_GPU_FETCH_DEMAND_RANK_SCAN","256");env("STRATA_DS4_CPU_EMPTY_GPU_FALLBACK","0");
    int cases=0;
    for(bool managed:{false,true})for(auto type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_MXFP4,GGML_TYPE_Q8_0})for(int tokens:{1,3,8}) {
        Weights w(cpu,type);w.register_files();
        std::vector<std::vector<float>> expected;
        StrataExpertCounters queue_stats;
        for(int mode=0;mode<3;++mode) {
            env("STRATA_EXPERT_CACHE_MIB",mode?"1":"0");
            env("STRATA_DS4_CPU_THREADS",mode==1?"4":"0");
            env("STRATA_DS4_ORDINARY_DEMAND_CACHE",mode==2?"1":"0");
            ggml_backend_ptr gpu(ggml_backend_dev_init(dev,nullptr));
            Graph g(w,cpu,gpu.get(),tokens,false,0.1f);
            if(managed){StrataVramPolicy policy;StrataVramStatus status;check(control(0,&policy,&status),"managed ordinary cache");}
            std::vector<float> x(tokens*D);std::vector<int32_t> ids(tokens*E);
            for(int iteration=0;iteration<24;++iteration) {
                for(int t=0;t<tokens;++t){ids[t*E]=(iteration+t)%8;ids[t*E+1]=1;}
                for(size_t i=0;i<x.size();++i)x[i]=0.8f*std::cos(float(i+23*iteration)*0.031f);
                auto result=g.run(x,ids);
                if(!mode)expected.push_back(result);else check(result==expected[iteration],"ordinary/O3 cache changed GPU output");
            }
            StrataExpertCounters end;stats(gpu.get(),&end);
            if(mode==1) {
                check(end.gpu_fetch_branches==24 && !end.cpu_queue_jobs,"queue reference must be GPU-only");queue_stats=end;
            }
            if(mode==2) {
                check(end.ordinary_cache_branches==24 && !end.gpu_fetch_branches && !end.cpu_queue_jobs,"ordinary path created queue or missed planning");
                check(end.h2d_bytes==queue_stats.h2d_bytes && end.hits==queue_stats.hits && end.misses==queue_stats.misses &&
                    end.gpu_admissions==queue_stats.gpu_admissions,"ordinary/O3 admission history diverged");
                check(end.gpu_fetch_cache_offered==end.gpu_fetch_cache_admitted+end.gpu_fetch_cache_restored,"ordinary loans leaked");
                // Explicit cancel, then a partial branch with no staged items.
                // This exercises the synchronous upload fallback while other
                // loan reservations remain live. Only copied payloads publish.
                Weights cold(cpu,type,31);cold.register_files();StrataCpuExpertBranch branch;
                for(int m=0;m<3;++m)branch.weights[m]=cold.w[m];
                // Empty the cache through its public controller so these
                // cancellations actually own loans, even after hot-frequency
                // entries would reject every new candidate.
                StrataVramPolicy clear;clear.mode=1;clear.matrices=0;StrataVramStatus status;
                check(control(0,&clear,&status),"clear before cancel fixture");
                StrataVramPolicy restore;check(control(0,&restore,&status),"restore cache budget");
                StrataExpertCounters before,after;stats(gpu.get(),&before);
                plan(gpu.get(),nullptr,0,&branch,g.ids,ids.data());finish(gpu.get());
                ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead(),nullptr,true}));
                auto *dst=ggml_dup_tensor(ctx.get(),cold.w[0]);
                ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),gpu.get()));
                plan(gpu.get(),nullptr,0,&branch,g.ids,ids.data());copy(gpu.get(),cold.w[0],dst,ids[0],ids[0],tokens);finish(gpu.get());
                stats(gpu.get(),&after);
                check(after.pipeline_fallbacks>before.pipeline_fallbacks,"partial fixture missed unstaged upload");
                check(after.gpu_fetch_cache_restored>before.gpu_fetch_cache_restored && after.gpu_admissions>before.gpu_admissions,
                    "cancel/partial fixture did not restore and publish real loans");
                check(after.gpu_fetch_cache_offered==after.gpu_fetch_cache_admitted+after.gpu_fetch_cache_restored,"cancel/partial leaked loans");
                Graph recovery(cold,cpu,gpu.get(),tokens,false,0.1f);
                const auto recovered=recovery.run(x,ids);
                env("STRATA_EXPERT_CACHE_MIB","0");
                ggml_backend_ptr clean(ggml_backend_dev_init(dev,nullptr));Graph clean_graph(cold,cpu,clean.get(),tokens,false,0.1f);
                const auto cold_expected=clean_graph.run(x,ids);env("STRATA_EXPERT_CACHE_MIB","1");
                check(recovered==cold_expected && recovery.run(x,ids)==recovered && g.run(x,ids)==expected.back(),
                    "cancel/partial corrupted cached weights");
                ++cases;
            }
        }
    }
    std::printf("Ordinary demand cache: %d cases, 432 exact replays, identical O3 H2D/admissions, cancel/partial recovery\n",cases);
}
