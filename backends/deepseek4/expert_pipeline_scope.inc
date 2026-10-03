    const char *strata_pipeline_env=std::getenv("STRATA_EXPERT_PIPELINE");
    const bool strata_pipeline_enabled=strata_pipeline_env && std::strcmp(strata_pipeline_env,"1")==0;
    struct StrataPipelineScope {
        std::vector<std::pair<ggml_backend_t,StrataExpertFinish>> contexts;
        ~StrataPipelineScope() {
            for(auto &entry:contexts)entry.second(entry.first);
        }
    } strata_pipeline_scope;
    if(strata_pipeline_enabled)for(int i=0;i<sched->n_backends;++i) {
        auto *backend=sched->backends[i];
        auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        auto finish=(StrataExpertFinish)ggml_backend_reg_get_proc_address(reg,"strata_expert_finish");
        if(finish)strata_pipeline_scope.contexts.push_back({backend,finish});
    }
