// Deliberately change activations after the early pass. Cached expert rows must
// retain the old result, while misses must use the new input. This detects an
// unnoticed full recomputation even when ordinary parity tests would pass.
static void prove_early_results(ggml_backend_t cpu,ggml_backend_dev_t dev,int mode) {
    env("STRATA_DS4_EARLY_HITS",mode==2?"2":"1");env("STRATA_EXPERT_CACHE_MIB","4");
    ggml_backend_ptr gpu(ggml_backend_dev_init(dev,nullptr));
    auto reg=ggml_backend_dev_backend_reg(dev);
    auto plan=(StrataExpertPlan)ggml_backend_reg_get_proc_address(reg,"strata_expert_plan");
    auto copy=(StrataExpertCopy)ggml_backend_reg_get_proc_address(reg,"strata_expert_copy");
    auto prepare=(StrataExpertPrepare)ggml_backend_reg_get_proc_address(reg,"strata_expert_prepare");
    auto early=(StrataExpertEarlyHits)ggml_backend_reg_get_proc_address(reg,"strata_expert_early_hits");
    auto flush=(StrataExpertFlush)ggml_backend_reg_get_proc_address(reg,"strata_expert_flush");
    auto finish=(StrataExpertFinish)ggml_backend_reg_get_proc_address(reg,"strata_expert_finish");
    check(plan && copy && prepare && early && flush && finish,"proof hooks");
    std::vector<float> weights(256*128*8);
    for(size_t i=0;i<weights.size();++i)weights[i]=std::cos(float(i)*0.037f)*0.1f;
    std::vector<uint8_t> packed(ggml_row_size(GGML_TYPE_Q8_0,256)*128*8);
    check(ggml_quantize_chunk(GGML_TYPE_Q8_0,weights.data(),packed.data(),0,128*8,256,nullptr)==packed.size(),"proof quantize");
    ggml_context_ptr host(ggml_init({2*ggml_tensor_overhead(),nullptr,true}));
    auto *source=ggml_new_tensor_3d(host.get(),GGML_TYPE_Q8_0,256,128,8);
    auto *alternate=ggml_new_tensor_3d(host.get(),GGML_TYPE_Q8_0,256,128,8);
    ggml_backend_buffer_ptr hb(ggml_backend_alloc_ctx_tensors(host.get(),cpu));
    ggml_backend_tensor_set(source,packed.data(),0,packed.size());
    ggml_backend_tensor_set(alternate,packed.data(),0,packed.size());
    for(int n:{1,3}) {
        ggml_context_ptr ctx(ggml_init({16*ggml_tensor_overhead()+2*ggml_graph_overhead(),nullptr,true}));
        ggml_tensor *w[2],*x[2],*ids[2],*out[2];ggml_cgraph *graphs[2];
        for(int i=0;i<2;++i) {
            w[i]=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_Q8_0,256,128,8);
            x[i]=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,256,1,n);
            auto *raw=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,256,n);
            ids[i]=ggml_view_2d(ctx.get(),raw,6,n,raw->nb[1],0);
            out[i]=ggml_mul_mat_id(ctx.get(),w[i],x[i],ids[i]);
            graphs[i]=ggml_new_graph(ctx.get());ggml_build_forward_expand(graphs[i],out[i]);
        }
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),gpu.get()));
        check(bool(buffer),"proof GPU allocation");
        for(int i=0;i<2;++i) {
            ggml_backend_tensor_set(w[i],packed.data(),0,packed.size());
            for(int t=0;t<n;++t) {
                int32_t row[6];for(int j=0;j<6;++j)row[j]=(j+t)%8;
                ggml_backend_tensor_set(ids[i],row,size_t(t)*ids[i]->nb[1],sizeof(row));
            }
        }
        std::vector<float> old_input(256*n),new_input(256*n);
        for(size_t i=0;i<old_input.size();++i) {
            old_input[i]=std::sin(float(i)*0.09f)*0.3f;
            new_input[i]=std::cos(float(i)*0.07f)*0.2f;
        }
        auto reference=[&](const std::vector<float> &input) {
            ggml_backend_tensor_set(x[0],input.data(),0,input.size()*4);
            check(ggml_backend_graph_compute(gpu.get(),graphs[0])==GGML_STATUS_SUCCESS,"proof reference");
            std::vector<float> result(ggml_nelements(out[0]));
            ggml_backend_tensor_get(out[0],result.data(),0,result.size()*4);return result;
        };
        auto expected_old=reference(old_input),expected_new=reference(new_input);
        check(std::memcmp(expected_old.data(),expected_new.data(),expected_new.size()*4)!=0,"proof inputs produce same result");
        std::vector<StrataExpertSlice> slices;
        for(int e=0;e<8;++e)slices.push_back({(uint8_t *)source->data+size_t(e)*source->nb[2],source->nb[2]+(e<7?512:0),true});
        // Use a new cache instance for each n below, so only 0..3 are resident.
        plan(gpu.get(),slices.data(),slices.size());copy(gpu.get(),source,w[1],0,3,1);finish(gpu.get());
        ggml_backend_tensor_set(x[1],old_input.data(),0,old_input.size()*4);
        int32_t selected[8]={0,1,2,3,4,5,6,7};
        // Abandon work before flush/graph submission. finish must join the
        // private reader before releasing the layer's allocation protection.
        plan(gpu.get(),slices.data(),slices.size());check(prepare(gpu.get(),out[1]),"abandon prepare");
        check(early(gpu.get(),source,out[1],ids[1],selected,8),"abandon early pass");finish(gpu.get());
        plan(gpu.get(),slices.data(),slices.size());check(prepare(gpu.get(),out[1]),"proof prepare");
        check(early(gpu.get(),source,out[1],ids[1],selected,8),"proof early pass");
        flush(gpu.get());
        ggml_backend_synchronize(gpu.get());
        ggml_backend_tensor_set(x[1],new_input.data(),0,new_input.size()*4);
        copy(gpu.get(),source,w[1],0,7,n);flush(gpu.get());
        check(ggml_backend_graph_compute(gpu.get(),graphs[1])==GGML_STATUS_SUCCESS,"proof candidate");
        std::vector<float> actual(ggml_nelements(out[1]));ggml_backend_tensor_get(out[1],actual.data(),0,actual.size()*4);
        for(int t=0;t<n;++t)for(int j=0;j<6;++j) {
            size_t offset=(size_t(t)*6+j)*128;
            auto &expected=(j+t)%8<4?expected_old:expected_new;
            if(std::memcmp(actual.data()+offset,expected.data()+offset,128*4)!=0) {
                std::fprintf(stderr,"proof n=%d token=%d lane=%d id=%d first actual=%g old=%g new=%g\n",
                    n,t,j,(j+t)%8,actual[offset],expected_old[offset],expected_new[offset]);
                check(false,"early outputs were not reused");
            }
        }
        finish(gpu.get());ggml_backend_synchronize(gpu.get());
        // Replay the same captured graph with all hits, then no hits. Neither
        // execution may reuse the previous pass's old-input output rows.
        for(auto *src:{source,alternate}) {
            slices.clear();
            for(int e=0;e<8;++e)slices.push_back({(uint8_t *)src->data+size_t(e)*src->nb[2],src->nb[2]+(e<7?512:0),true});
            plan(gpu.get(),slices.data(),slices.size());check(prepare(gpu.get(),out[1]),"replay prepare");
            check(!early(gpu.get(),src,out[1],ids[1],selected,8),"all/zero hits should not launch");
            copy(gpu.get(),src,w[1],0,7,n);flush(gpu.get());
            check(ggml_backend_graph_compute(gpu.get(),graphs[1])==GGML_STATUS_SUCCESS,"proof replay");
            ggml_backend_tensor_get(out[1],actual.data(),0,actual.size()*4);
            check(std::memcmp(actual.data(),expected_new.data(),actual.size()*4)==0,"stale early mask on replay");
            finish(gpu.get());ggml_backend_synchronize(gpu.get());
        }
        // Clearing the backend also tests destruction before a graph's buffers.
        gpu.reset();gpu.reset(ggml_backend_dev_init(dev,nullptr));
    }
    gpu.reset();
}
