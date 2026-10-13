// Members of StrataExpertTransfer. Only the owning scheduler thread enters
// here, under the transfer mutex. The I/O service sees addresses, never Entries.
void prepare_fetch_cache() {
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
        for(int id=0;id<p.experts;++id)
            p.scores[id]=uint8_t(frequency?std::min(255u,frequency->score(f.weights[m]+size_t(id)*p.width)+1):255u);
        size_t scanned=0;
        for(auto it=lru.rbegin();it!=lru.rend() && p.cache_count<f.cache_slots && scanned<64;++it,++scanned) {
            const void *key=*it;
            if(direct.protects(key) || protected_in_flight(key))continue;
            auto found=entries.find(key);GGML_ASSERT(found!=entries.end());
            auto &e=found->second;
            if(e.bytes!=p.width && e.bytes!=p.width+std::min<size_t>(p.width,512))continue;
            // Never loan a current-branch weight, even if its cached width
            // differs from the requested padding. All three matrices matter.
            bool current=false;
            for(int j=0;j<3;++j) {
                const auto address=uintptr_t(key),base=uintptr_t(f.weights[j]);
                current|=address>=base && address-base<size_t(f.plan[j].experts)*f.plan[j].width;
            }
            if(current)continue;
            const int k=p.cache_count++;
            p.cache[k]={managed?e.allocation:arena+e.offset,e.bytes,frequency?frequency->score(key):0u};
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
        int missing=0;for(int id=0;id<p.experts;++id)missing+=!p.cached[id] && !p.keep_ram[id];
        while(p.cache_count<std::min(f.cache_slots,missing)) {
            bool tail_slot=false;for(int i=0;i<p.cache_count;++i)tail_slot|=p.cache[i].bytes==p.width;
            const bool tail=!tail_slot && !p.cached[p.experts-1] && !p.keep_ram[p.experts-1] &&
                p.cache_count+1==std::min(f.cache_slots,missing);
            const size_t bytes=p.width+(tail?0:std::min<size_t>(p.width,512));
            auto slot=reserve_fetch_slot(bytes);if(slot==entries.end())break;
            const void *key=slot->first;auto &e=slot->second;
            const int k=p.cache_count++;
            p.cache[k]={managed?e.allocation:arena+e.offset,e.bytes,0u};
            protected_keys().insert(key);fetch_loans()[m][k].node=entries.extract(slot);
            ++fetch_loan_count();++f.cache_offered;
        }
    }
    f.cache_prepare_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
}
void finish_fetch_cache() {
    if(!fetch_loan_count())return;
    const auto start=std::chrono::steady_clock::now();
    for(auto &matrix:fetch_loans())for(auto &loan:matrix)if(!loan.node.empty()) {
        auto &e=loan.node.mapped();
        if(loan.replacement) {
            const bool was_empty=e.empty;
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
