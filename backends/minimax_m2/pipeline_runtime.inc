// Producers only write their own ring. Scheduler scratch is delivered on demand,
// after its previous compute consumer; future source mappings and hits stay live.
static void mm27_pipeline_abort() noexcept {
    try {mm27_pipeline_dispose();}catch(...) {}
    try {strata_mm27_cache_clear();}catch(...) {}
    if(mm27_pipeline_trace)try {mm27_pipeline_trace->end(true);}catch(...) {}
}
static bool mm27_pipeline_error(const std::exception &e) {
    const std::string message=e.what();mm27_pipeline_abort();mm27_error=message;
    GGML_LOG_ERROR("STRATA_MM27 pipeline: %s\n",message.c_str());return false;
}
static bool mm27_pipeline_complete() {
    if(!mm27_pipeline_active)return true;
    try {
        if(mm27_plan_next!=mm27_plan.size())throw std::runtime_error("incomplete MiniMax matrix plan");
        mm27_pipeline_queue->finish();step35::cuda_check(cudaStreamSynchronize(mm27_pipeline_stream));
        mm27_pipeline_active=false;mm27_plan_pins.reset();mm27_plan.clear();mm27_plan_next=0;
        mm27_pipeline_account();if(mm27_pipeline_trace)mm27_pipeline_trace->end(false);
        return true;
    }catch(const std::exception &e) {return mm27_pipeline_error(e);}
}
// Every scheduler exit, including callbacks, cancellation and failed compute,
// drains any abandoned plan before runtime/model ownership can change.
struct mm27_pipeline_graph_scope {
    ~mm27_pipeline_graph_scope() {if(mm27_pipeline_active)mm27_pipeline_abort();}
};
static bool mm27_pipeline_begin(ggml_backend_t backend,std::vector<mm27_plan_tensor> plan) {
    try {
        if(!mm27_pipeline_complete())return false;
        if(plan.empty() || plan.size()>3)throw std::runtime_error("invalid MiniMax matrix plan size");
        mm27_pipeline_create();
        if(mm27_cancelled() || !mm27_live_memory())return false;
        ggml_backend_synchronize(backend);
        const bool entries=mm27_cache && (mm27_cache_admit || mm27_cache->size());
        std::vector<step35::MatrixKey> keys;std::vector<StrataExpertSlice> misses;
        for(size_t i=0;i<plan.size();++i) {
            const auto *src=plan[i].source;const auto &ranges=plan[i].ranges;
            if(!src || ranges.empty() || ranges.size()>256 || !mm27_expert(src) || !src->data ||
                !src->nb[2] || src->ne[2]>256 || !ggml_is_contiguous(src) || src->ne[3]!=1)
                throw std::runtime_error("invalid MiniMax pipeline tensor plan");
            for(size_t j=0;j<i;++j)if(plan[j].source==src)throw std::runtime_error("duplicate MiniMax plan tensor");
            const auto file=strata_expert_file::find(src->data,ggml_nbytes(src));
            if(!file)throw std::runtime_error("unregistered MiniMax pipeline mapping");
            const auto id=entries?mm27_tensor_id(src):0;
            const size_t stride=src->nb[2];size_t previous_end=0;
            for(const auto &range:ranges) {
                const size_t at=range.first,bytes=range.second,count=bytes/stride,tail=bytes%stride;
                if(at%stride || !count || at<previous_end || at>ggml_nbytes(src) || bytes>ggml_nbytes(src)-at ||
                    (tail && tail!=std::min<size_t>(512,stride)))throw std::runtime_error("invalid MiniMax pipeline range");
                previous_end=at+count*stride;
                if(!entries)misses.push_back({static_cast<const char *>(src->data)+at,bytes,mm27_cache_admit});
                else for(size_t e=0;e<count;++e) {
                    const size_t offset=at+e*stride,stored=stride+std::min<size_t>(512,ggml_nbytes(src)-offset-stride);
                    const size_t needed=stride+(e+1==count?tail:0),delivered=mm27_cache_admit?stored:needed;
                    const step35::MatrixKey key{mm27_generation,id,uint32_t(offset/stride)};keys.push_back(key);
                    if(!mm27_cache->contains(key,stored))misses.push_back({static_cast<const char *>(src->data)+offset,delivered,mm27_cache_admit});
                }
            }
        }
        if(entries)mm27_plan_pins=mm27_cache->protect(keys);
        *mm27_pipeline_job={mm27_cancel,mm27_test_limits.pipeline_read_failure,mm27_test_limits.pipeline_copy_failure};
        if(mm27_pipeline_trace)mm27_pipeline_trace->begin();
        mm27_plan=std::move(plan);mm27_plan_next=0;
        mm27_pipeline_active=true;mm27_pipeline_queue->start(misses);
        ++mm27_stats.pipeline_plans;mm27_stats.pipeline_matrices+=mm27_plan.size();
        mm27_stats.pipeline_plan_peak=std::max<uint64_t>(mm27_stats.pipeline_plan_peak,mm27_plan.size());
        if(mm27_plan.size()>1)++mm27_stats.pipeline_lookahead_plans;
        return true;
    }catch(const std::exception &e) {return mm27_pipeline_error(e);}
}
// All copies of one scheduler input use the same ordered delivery stream.
// Existing hits are plan-pinned. New fills acquire their own residency pins
// before submission: admit/refresh may neither reuse nor free a pending slot.
// No newly admitted key is looked up until this tensor has completed.
static void mm27_pipeline_copy_batch(ggml_backend_t backend,ggml_tensor *dst,const ggml_tensor *src,const mm27_ranges &ranges) {
    struct Pending {
        std::vector<std::unique_ptr<step35::ExpertCache::PlanPins>> fills;
        bool complete=false;
        ~Pending() {
            // The destructor body precedes pin destruction, including during
            // unwinding. The outer handler may clear the cache only afterwards.
            if(!complete) {++mm27_stats.pipeline_abort_fences;cudaStreamSynchronize(mm27_pipeline_stream);}
        }
    };
    if(!dst->data || !ggml_are_same_shape(src,dst) || src->type!=dst->type ||
        !ggml_is_contiguous(dst) || !dst->buffer || ggml_backend_buffer_is_host(dst->buffer))
        throw std::runtime_error("invalid MiniMax batched destination");
    const auto file=strata_expert_file::find(src->data,ggml_nbytes(src));
    if(!file)throw std::runtime_error("unregistered MiniMax batched mapping");
    const bool entries=mm27_cache && (mm27_cache_admit || mm27_cache->size());
    const auto id=entries?mm27_tensor_id(src):0;
    const uint64_t base=file->offset_of(src->data,ggml_nbytes(src));
    const size_t stride=src->nb[2];
    // Scratch can alias the previous split's activations, even on all hits.
    ++mm27_stats.pipeline_scratch_fences;ggml_backend_synchronize(backend);
    Pending pending;pending.fills.reserve(256);int queued=0;
    auto check_live=[] {
        if(mm27_cancelled() || !mm27_live_memory())throw std::runtime_error(mm27_error);
    };
    auto submitted=[&] {
        ++queued;
        if(mm27_test_limits.batch_fail_after==queued)throw std::runtime_error("injected MiniMax batched enqueue failure");
        if(mm27_test_limits.batch_cancel && mm27_test_limits.batch_cancel_after==queued)
            mm27_test_limits.batch_cancel->store(true);
        if(mm27_test_limits.batch_pressure_after==queued) {
            mm27_test_limits.ram_available=0;mm27_next_memory_check={};
        }
    };
    for(const auto &range:ranges) {
        check_live();const size_t offset=range.first,bytes=range.second,count=bytes/stride,tail=bytes%stride;
        if(offset>ggml_nbytes(dst) || bytes>ggml_nbytes(dst)-offset)
            throw std::runtime_error("MiniMax batched range outside destination");
        ++mm27_stats.ranges;mm27_stats.selected_bytes+=bytes;
        if(!entries) {
            if(!mm27_upload(backend,dst,*file,base+offset,offset,bytes,true))throw std::runtime_error(mm27_error);
            submitted();continue;
        }
        for(size_t e=0;e<count;++e) {
            check_live();const size_t at=offset+e*stride;
            const size_t stored=stride+std::min<size_t>(512,ggml_nbytes(src)-at-stride);
            const size_t needed=stride+(e+1==count?tail:0);
            const step35::MatrixKey key{mm27_generation,id,uint32_t(at/stride)};
            auto *dest=static_cast<char *>(dst->data)+at;
            if(void *hit=mm27_cache->get(key,stored,mm27_cache_admit)) {
                mm27_cache_copy(dest,hit,needed,true);mm27_stats.cache_hit_bytes+=needed;
            } else {
                const size_t delivered=mm27_cache_admit?stored:needed;
                if(!mm27_upload(backend,dst,*file,base+at,at,delivered,true))throw std::runtime_error(mm27_error);
                mm27_stats.cache_guard_bytes+=delivered-needed;
                if(mm27_cache_admit)if(void *slot=mm27_cache->admit(key,stored)) {
                    pending.fills.push_back(mm27_cache->protect({key}));
                    mm27_stats.pipeline_pending_fills_peak=std::max<uint64_t>(mm27_stats.pipeline_pending_fills_peak,pending.fills.size());
                    if(mm27_test_limits.cache_fill_failure)throw std::runtime_error("injected MiniMax cache fill failure");
                    mm27_cache_copy(slot,dest,stored,true);mm27_stats.cache_fill_bytes+=stored;
                }
            }
            submitted();
        }
    }
    check_live();++mm27_stats.pipeline_copy_fences;
    step35::cuda_check(cudaStreamSynchronize(mm27_pipeline_stream));pending.complete=true;
    pending.fills.clear();++mm27_stats.pipeline_copy_batches;
    // Diagnostics receive completed bytes. They can cancel or inject pressure;
    // future-hit plan pins remain live until the outer drain on every exit.
    if(mm27_observer)for(const auto &range:ranges) {
        check_live();mm27_observer(backend,dst,src,range.first,range.second,mm27_owner);
    }
    check_live();
}
static bool mm27_pipeline_deliver(ggml_backend_t backend,ggml_tensor *dst,const ggml_tensor *src,const mm27_ranges &ranges) {
    try {
        if(!mm27_pipeline_active || mm27_plan_next>=mm27_plan.size() ||
            mm27_plan[mm27_plan_next].source!=src || mm27_plan[mm27_plan_next].ranges!=ranges)
            throw std::runtime_error("MiniMax matrix order/ranges differ from router plan");
        if(mm27_pipeline_d2d_batch)mm27_pipeline_copy_batch(backend,dst,src,ranges);
        else for(const auto &range:ranges)if(!mm27_copy_range(backend,dst,src,range.first,range.second))
            throw std::runtime_error(mm27_error.empty()?"MiniMax pipeline transfer failed":mm27_error);
        ++mm27_plan_next;return true;
    }catch(const std::exception &e) {return mm27_pipeline_error(e);}
}
static bool mm27_pipeline_tensor(ggml_backend_t backend,ggml_tensor *dst,const ggml_tensor *src,const mm27_ranges &ranges) {
    return mm27_pipeline_begin(backend,{{src,ranges}}) && mm27_pipeline_deliver(backend,dst,src,ranges) && mm27_pipeline_complete();
}
