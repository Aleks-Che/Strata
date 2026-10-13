// Members of StrataExpertTransfer. Only the owning scheduler thread enters
// here, under the transfer mutex. The I/O service sees addresses, never Entries.
void prepare_fetch_cache() {
    auto &f=gpu_fetch;
    GGML_ASSERT(!fetch_loan_count);
    if(!f.cache_slots)return;
    // Demotion must copy OLD bytes before reuse. Until a demand-triggered
    // demotion protocol exists, preserve that policy by not offering loans.
    if(host_cache && host_cache->demotion_enabled())return;
    const auto start=std::chrono::steady_clock::now();
    for(int m=0;m<3;++m) {
        auto &p=f.plan[m];
        for(int id=0;id<p.experts;++id)
            p.scores[id]=uint8_t(frequency?std::min(255u,frequency->score(f.weights[m]+size_t(id)*p.width)+1):255u);
        size_t scanned=0;
        for(auto it=lru.rbegin();it!=lru.rend() && p.cache_count<f.cache_slots && scanned<64;++it,++scanned) {
            const void *key=*it;
            if(direct.protects(key) || protected_keys.count(key))continue;
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
            protected_keys.insert(key);
            // Revoke publication BEFORE a possible overwrite. No speculative
            // D2H, no extra weight copy, no CUDA calls from the I/O service.
            e.residency.reset();
            fetch_loans[m][k].node=entries.extract(found);
            ++fetch_loan_count;++f.cache_offered;
        }
    }
    f.cache_prepare_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
}
void finish_fetch_cache() {
    if(!fetch_loan_count)return;
    const auto start=std::chrono::steady_clock::now();
    for(auto &matrix:fetch_loans)for(auto &loan:matrix)if(!loan.node.empty()) {
        auto &e=loan.node.mapped();
        if(loan.replacement) {
            loan.node.key()=loan.replacement;*e.lru=loan.replacement;
            lru.splice(lru.begin(),lru,e.lru);
            if(managed)e.touched=strata_expert_touch_clock(device);
            ++stats.evictions;++stats.direct_admissions;stats.direct_bytes+=e.bytes;
            ++gpu_fetch.cache_admitted;gpu_fetch.cache_bytes+=e.bytes;
        } else ++gpu_fetch.cache_restored; // same key, payload and LRU position
        auto inserted=entries.insert(std::move(loan.node));GGML_ASSERT(inserted.inserted);
        promoted(inserted.position->first,inserted.position->second);
        loan.replacement=nullptr;--fetch_loan_count;
    }
    gpu_fetch.cache_retire_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
}
