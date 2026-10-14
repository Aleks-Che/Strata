// Members of StrataExpertTransfer. Only the owning scheduler thread enters
// here, under the transfer mutex. The I/O service sees addresses, never Entries.
void prepare_fetch_cache(const strata_ds4::GpuFetchRequest *selected=nullptr) {
    auto &f=gpu_fetch();
    GGML_ASSERT(!fetch_loan_count());
    if(!f.cache_slots)return;
    f.demotion_epoch=++demotion_clock;
    // Keep the ordinary victim policy. A handoff orders old reads before new
    // writes on this backend's stream. Loans retain their allocations, and
    // cross-backend eviction separately fences reuse/free on the owner stream.
    // Without a handoff, wait for the old consumer before offering its keys.
    if(!cpu_whole().handoff_pending)cpu_whole().finish_consumer();
    deferred_keys().clear();
    const auto start=std::chrono::steady_clock::now();
    for(int m=0;m<3;++m) {
        auto &p=f.plan[m];
        const int count=selected?selected[m].used_count:p.experts;
        for(int i=0;i<count;++i) {
            const int id=selected?selected[m].used_ids[i]:i;
            p.scores[id]=uint8_t(frequency?std::min(255u,frequency->score(f.weights[m]+size_t(id)*p.width)+1):255u);
        }
        if(f.cache_demand)continue; // route publisher starts with scratch destinations
        size_t scanned=0;
        for(auto it=lru.rbegin();it!=lru.rend() && p.cache_count<f.cache_slots && scanned<64;++it,++scanned) {
            const void *key=*it;
            if(direct.protects(key))continue;
            auto found=entries.find(key);
            if(found==entries.end())continue; // already loaned by this/another matrix
            auto &e=found->second;
            if(e.bytes!=p.width && e.bytes!=p.width+std::min<size_t>(p.width,512))continue;
            // Only the final expert uses an unpadded slot. Several such slots
            // can otherwise occupy the entire bounded offer window forever.
            // One tail candidate is sufficient for every possible route.
            if(f.cache_tail_limit && p.experts>1 && e.bytes==p.width) {
                bool tail=false;for(int k=0;k<p.cache_count;++k)tail|=p.cache[k].bytes==p.width;
                if(tail){++f.cache_tail_skipped;continue;}
            }
            // A current-branch candidate keeps its published pointer in the
            // immutable plan, but the GPU may overwrite it ONLY if its expert
            // is absent from the complete route. Other reader epochs still win.
            int victim=-1;
            for(int j=0;j<3;++j) {
                const auto address=uintptr_t(key),base=uintptr_t(f.weights[j]);
                if(address>=base && address-base<size_t(f.plan[j].experts)*f.plan[j].width) {
                    GGML_ASSERT((address-base)%f.plan[j].width==0);
                    victim=int((address-base)/f.plan[j].width);break;
                }
            }
            if(victim>=0) {
                if(!f.cache_current_victims)continue;
                bool other=false;
                for(size_t i=0;i<branch_contexts.size();++i) {
                    const auto &c=branch_contexts[i];
                    other|=c.deferred_keys.count(key) || (i!=current_context && c.protected_keys.count(key));
                }
                if(other)continue;
            } else if(protected_in_flight(key))continue;
            const int k=p.cache_count++;
            p.cache[k]={managed?e.allocation:arena+e.offset,e.bytes,frequency?frequency->score(key):0u,victim};
            if(e.bytes==p.width)++f.cache_tail_offered;
            if(victim>=0)++f.cache_current_offered;
            protected_keys().insert(key);
            // D2H precedes every possible overwrite on the SAME owner stream.
            // Unused loans retain their old payload and publication. The RAM
            // policy limits speculative demotions to its existing byte/job cap.
            residency.evict(e.residency,p.cache[k].address,stream,!strata_cpu_decode,f.demotion_epoch);
            // Revoke publication BEFORE a possible overwrite. The bounded
            // demotion above is submitted by the owner, never the I/O service.
            e.residency.reset();
            fetch_loans()[m][k].node=entries.extract(found);
            ++fetch_loan_count();++f.cache_offered;
        }
        // Reserve empty slots on the owner. No allocation is performed by
        // either service, and reservations remain charged to the GPU budget.
        // Before the route is known, assigned-only protection must not remove
        // possible GPU misses from the bounded empty-slot reservation budget.
        int missing=0;for(int id=0;id<p.experts;++id)missing+=!p.cached[id] && (!p.keep_ram[id] || p.keep_ram_assigned);
        while(p.cache_count<std::min(f.cache_slots,missing)) {
            bool tail_slot=false;for(int i=0;i<p.cache_count;++i)tail_slot|=p.cache[i].bytes==p.width;
            const bool tail=!tail_slot && !p.cached[p.experts-1] && (!p.keep_ram[p.experts-1] || p.keep_ram_assigned) &&
                p.cache_count+1==std::min(f.cache_slots,missing);
            const size_t bytes=p.width+(tail?0:std::min<size_t>(p.width,512));
            auto slot=reserve_fetch_slot(bytes);if(slot==entries.end())break;
            const void *key=slot->first;auto &e=slot->second;
            const int k=p.cache_count++;
            p.cache[k]={managed?e.allocation:arena+e.offset,e.bytes,0u};
            if(e.bytes==p.width)++f.cache_tail_offered;
            protected_keys().insert(key);fetch_loans()[m][k].node=entries.extract(slot);
            ++fetch_loan_count();++f.cache_offered;
        }
    }
    f.cache_prepare_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
}
void prepare_demand_cache_matrix(int m,strata_ds4::GpuFetchRequest &job) {
    auto &f=gpu_fetch();auto &p=f.plan[m];
        bool selected[256]={};
        for(int i=0;i<job.used_count;++i) {
            const int id=job.used_ids[i];GGML_ASSERT(id>=0 && id<p.experts && !selected[id]);selected[id]=true;
        }
        ++f.cache_demand_plans;f.cache_demand_misses+=job.count;
        const bool ranked=f.cache_rank && capacity>=f.cache_rank_min_bytes && bool(frequency);
        int order[64];for(int i=0;i<job.count;++i)order[i]=i;
        if(ranked) {
            ++f.cache_rank_plans;
            // Change admission priority only; request order, scratch indices
            // and GPU accumulation order must remain unchanged.
            std::stable_sort(order,order+job.count,[&](int a,int b){return p.scores[job.ids[a]]>p.scores[job.ids[b]];});
        }
        for(int priority=0;priority<job.count && p.cache_count<f.cache_slots;++priority) {
            const int i=order[priority];
            const int id=job.ids[i];GGML_ASSERT(id>=0 && id<p.experts && selected[id] && !p.cached[id]);
            if(strata_ds4::gpu_fetch_keep_ram(p,job,id))continue;
            const size_t bytes=p.width+(id<p.experts-1?std::min<size_t>(p.width,512):0);
            auto chosen=entries.end();int victim=-1;unsigned score=0;
            uint64_t best_cost=UINT64_MAX;bool costly=false;
            size_t scanned=0;
            for(auto it=lru.rbegin();it!=lru.rend() && scanned<size_t(ranked?f.cache_rank_scan:64);++it,++scanned) {
                ++f.cache_demand_scanned;
                const void *key=*it;if(direct.protects(key))continue;
                auto found=entries.find(key);if(found==entries.end())continue;
                const auto &e=found->second;if(e.bytes!=bytes)continue;
                const unsigned candidate_score=e.empty?0u:frequency?frequency->score(key):0u;
                // A relative byte-cost proxy, not a measured latency or a
                // probability: decayed frequency times reload bytes. Margin
                // charges each nonempty replacement extra reload equivalents.
                const uint64_t cost=e.empty?0:uint64_t(candidate_score+(ranked?f.cache_rank_margin:0))*e.bytes;
                if(cost>uint64_t(p.scores[id])*bytes){costly=true;continue;}
                int current=-1;
                for(int j=0;j<3;++j) {
                    const auto address=uintptr_t(key),base=uintptr_t(f.weights[j]);
                    if(address>=base && address-base<size_t(f.plan[j].experts)*f.plan[j].width) {
                        GGML_ASSERT((address-base)%f.plan[j].width==0);
                        current=int((address-base)/f.plan[j].width);break;
                    }
                }
                if(current>=0) {
                    if(!f.cache_current_victims || selected[current])continue;
                    bool other=false;
                    for(size_t c=0;c<branch_contexts.size();++c)
                        other|=branch_contexts[c].deferred_keys.count(key) ||
                            (c!=current_context && branch_contexts[c].protected_keys.count(key));
                    if(other)continue;
                } else if(protected_in_flight(key))continue;
                if(cost>=best_cost)continue; // preserve LRU order on ties
                chosen=found;victim=current;score=candidate_score;best_cost=cost;
                if(!ranked || !cost)break;
            }
            if(chosen==entries.end())chosen=reserve_fetch_slot(bytes);
            if(chosen==entries.end()) {
                if(ranked && costly)++f.cache_rank_rejected;
                continue; // same bounded scratch fallback
            }
            const void *key=chosen->first;auto &e=chosen->second;const int k=p.cache_count++;
            p.cache[k]={managed?e.allocation:arena+e.offset,e.bytes,score,victim};
            if(e.bytes==p.width)++f.cache_tail_offered;
            if(victim>=0)++f.cache_current_offered;
            protected_keys().insert(key);
            residency.evict(e.residency,p.cache[k].address,stream,!strata_cpu_decode,f.demotion_epoch);
            e.residency.reset();fetch_loans()[m][k].node=entries.extract(chosen);
            ++fetch_loan_count();++f.cache_offered;
            if(ranked){++f.cache_rank_admitted;f.cache_rank_cost+=best_cost==UINT64_MAX?0:best_cost;}
            job.cache_slots[i]=k;job.destinations[i]=p.cache[k].address;
        }
}
// The existing owner-DMA fence has retired all published route producers.
// Gate the service until every destination/loan is stable. With early sources,
// approve all three matrices together so later reads still overlap computation.
void prepare_demand_fetch_cache() {
    auto &f=gpu_fetch();GGML_ASSERT(f.cache_demand && f.queue->owner_dma);
    const auto start=std::chrono::steady_clock::now();
    for(int m=0;m<3;++m)if(f.published[m]) {
        auto &slot=f.queue->host[m];const auto sequence=slot.request.load(cuda::memory_order_acquire);
        if(slot.admitted.load(cuda::memory_order_acquire)==sequence)continue;
        auto &p=f.plan[m];auto &job=slot.job;
        GGML_ASSERT(!slot.failure.load(cuda::memory_order_acquire) && sequence && job.epoch==p.epoch &&
            job.count>=0 && job.count<=64 && job.used_count>=0 && job.used_count<=64 && !p.cache_count);
        prepare_demand_cache_matrix(m,job);
    }
    f.cache_prepare_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
    for(int m=0;m<3;++m)if(f.published[m]) {
        auto &slot=f.queue->host[m];
        slot.admitted.store(slot.request.load(cuda::memory_order_acquire),cuda::memory_order_release);
    }
}
// Selected current-branch hits may themselves be unused loans. Their payload
// was protected by the GPU route, and their LRU node lives until retirement.
Entry *fetch_hit_entry(const void *key) {
    const auto it=entries.find(key);if(it!=entries.end())return &it->second;
    for(auto &matrix:fetch_loans())for(auto &loan:matrix)
        if(!loan.node.empty() && loan.node.key()==key && !loan.replacement)return &loan.node.mapped();
    return nullptr;
}
void finish_fetch_cache() {
    if(!fetch_loan_count())return;
    const auto start=std::chrono::steady_clock::now();
    for(auto &matrix:fetch_loans())for(auto &loan:matrix)if(!loan.node.empty()) {
        auto &e=loan.node.mapped();
        if(loan.replacement) {
            const bool was_empty=e.empty;
            if(!was_empty)record_fetch_eviction(loan.node.key(),e.bytes);
            // An older context may have restored an unused loan for this key
            // after our snapshot. Both payloads are valid. Keep its canonical
            // entry and recycle our allocation as an empty, budgeted slot.
            const bool duplicate=entries.count(loan.replacement)!=0;
            if(duplicate){loan.replacement=managed?e.allocation:arena+e.offset;++gpu_fetch().cache_duplicates;}
            loan.node.key()=loan.replacement;*e.lru=loan.replacement;
            lru.splice(lru.begin(),lru,e.lru);
            if(managed)e.touched=strata_expert_touch_clock(device);
            if(!was_empty)++stats.evictions;e.empty=duplicate;
            if(!duplicate)++stats.gpu_admissions;
            ++stats.direct_admissions;stats.direct_bytes+=e.bytes;
            ++gpu_fetch().cache_admitted;gpu_fetch().cache_bytes+=e.bytes;
        } else {
            ++gpu_fetch().cache_restored;
            // The converse race is possible too: this unused loan may be
            // retired AFTER an older context admitted a second copy of its key.
            if(entries.count(loan.node.key())) {
                loan.node.key()=managed?e.allocation:arena+e.offset;*e.lru=loan.node.key();
                e.empty=true;++gpu_fetch().cache_duplicates;
            }
        }
        auto inserted=entries.insert(std::move(loan.node));GGML_ASSERT(inserted.inserted);
        if(!inserted.position->second.empty)promoted(inserted.position->first,inserted.position->second,true);
        loan.replacement=nullptr;--fetch_loan_count();
    }
    gpu_fetch().cache_retire_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
}
