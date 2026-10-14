// Members of StrataExpertTransfer. Reuse O3 admission on the owning host
// thread, with the route the ordinary scheduler already synchronized.
// No GPU request queue, CPU snapshot, or service thread is needed.
void prepare_ordinary_cache(const StrataCpuExpertBranch &branch,const ggml_tensor *route,const int32_t *ids) {
    auto &f=gpu_fetch();auto &o=ordinary_cache();
    const char *flag=std::getenv("STRATA_DS4_ORDINARY_DEMAND_CACHE");
    if(!flag || !std::atoi(flag) || !ids || !route || !capacity || !f.cache_slots || !f.cache_demand ||
       (f.cache_demand_max_bytes && capacity>f.cache_demand_max_bytes) || cpu_whole().configured || f.active ||
       route->ne[0]<1 || route->ne[0]>8 || route->ne[1]<1 || route->ne[1]>8 || route->nb[0]!=4)return;
    for(int m=0;m<3;++m) {
        auto *w=branch.weights[m];
        if(!w || !w->data || w->ne[2]<1 || w->ne[2]>256 || w->nb[2]==0 ||
           w->ne[2]!=branch.weights[0]->ne[2])return;
    }
    bool selected[256]={};int used[64],n=0;
    for(int t=0;t<route->ne[1];++t)for(int k=0;k<route->ne[0];++k) {
        const int id=ids[size_t(t)*route->nb[1]/4+k];
        if(id<0 || id>=branch.weights[0]->ne[2])return;
        if(!selected[id]){selected[id]=true;used[n++]=id;}
    }
    GGML_ASSERT(!o.active && !fetch_loan_count());o={};o.active=true;++stats.ordinary_cache_branches;
    const char *sparse_flag=std::getenv("STRATA_DS4_ORDINARY_PREP_SPARSE");
    const bool sparse=sparse_flag && std::atoi(sparse_flag);
    for(int m=0;m<3;++m) {
        auto *w=branch.weights[m];auto &p=f.plan[m];auto &job=o.jobs[m];p={};
        f.weights[m]=(const uint8_t *)w->data;
        f.files[m]=strata_expert_file::find(w->data,ggml_nbytes(w));
        f.offsets[m]=f.files[m]?f.files[m]->offset_of(w->data,ggml_nbytes(w)):0;
        p.width=w->nb[2];p.experts=int(w->ne[2]);p.tokens=int(route->ne[1]);p.topk=int(route->ne[0]);
        // Match the queue's complete immutable hit snapshot, including current
        // branch victims. The selector may replace only unselected experts.
        for(int id=0;id<p.experts;++id) {
            // Ordinary routing already owns the complete selection. No other
            // expert can read this snapshot; O3 still checks current-branch
            // victims and every other context's protection separately.
            if(sparse && !selected[id])continue;
            const void *key=f.weights[m]+size_t(id)*p.width;
            auto it=entries.find(key);
            const size_t bytes=p.width+(id<p.experts-1?std::min<size_t>(p.width,512):0);
            if(it!=entries.end() && !it->second.empty && it->second.bytes==bytes) {
                p.cached[id]=managed?it->second.allocation:arena+it->second.offset;
                protected_keys().insert(key);
            }
        }
        for(int i=0;i<n;++i) {
            const int id=used[i];job.used_ids[job.used_count++]=id;
            if(!p.cached[id]){const int j=job.count++;job.ids[j]=id;job.cache_slots[j]=-1;}
        }
    }
    prepare_fetch_cache(sparse?o.jobs:nullptr);
    const auto start=std::chrono::steady_clock::now();
    for(int m=0;m<3;++m)prepare_demand_cache_matrix(m,o.jobs[m]);
    f.cache_prepare_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
}
bool copy_ordinary_cache(const ggml_tensor *source,ggml_tensor *dest,int id) {
    auto &o=ordinary_cache();if(!o.active)return false;
    auto &f=gpu_fetch();int m=0;for(;m<3 && f.weights[m]!=source->data;++m){}
    if(m==3)return false;
    auto &p=f.plan[m];auto &job=o.jobs[m];
    bool selected=false;for(int i=0;i<job.used_count;++i)selected|=job.used_ids[i]==id;
    if(!selected)return false;
    GGML_ASSERT(!o.copied[m][id]);o.copied[m][id]=true;
    const void *key=f.weights[m]+size_t(id)*p.width;
    const size_t bytes=p.width+(id<p.experts-1?std::min<size_t>(p.width,512):0);
    void *destination=direct.destination(dest,id);
    if(p.cached[id]) {
        if(pipeline)pipeline->transfer(nullptr,key,bytes,stream);
        if(!read_direct(dest,id,key,p.cached[id],bytes,true)) {
            copy_cached(destination,p.cached[id],bytes);stats.d2d_bytes+=bytes;
        }
    } else {
        int index=0;for(;index<job.count && job.ids[index]!=id;++index){}
        GGML_ASSERT(index<job.count);const int slot=job.cache_slots[index];
        if(slot<0)upload(destination,key,bytes,true);
        else {
            auto &loan=fetch_loans()[m][slot];GGML_ASSERT(!loan.node.empty() && !loan.replacement);
            loan.replacement=key;
            void *cached=p.cache[slot].address;upload(cached,key,bytes);
            // O3 admissions are read directly too. Retirement counts their
            // bytes once, alongside the queue's admissions.
            if(!direct.bind(dest,id,key,cached,bytes)) {
                copy_cached(destination,cached,bytes);stats.d2d_bytes+=bytes;
            }
        }
    }
    return true;
}
void finish_ordinary_cache() {
    auto &o=ordinary_cache();if(!o.active)return;
    auto &f=gpu_fetch();
    // Train in the same up/gate/down and route-appearance order as O3.
    // Cancellation trains only copies actually submitted. Unused loans retain
    // their original payload and are restored by finish_fetch_cache().
    for(int m=0;m<3;++m) {
        const auto &p=f.plan[m];const auto &job=o.jobs[m];bool admitted[256]={};
        for(int i=0;i<job.count;++i)if(job.cache_slots[i]>=0 && o.copied[m][job.ids[i]]) {
            admitted[job.ids[i]]=true;
            if(p.cache[job.cache_slots[i]].victim>=0)++f.cache_current_admitted;
        }
        for(int i=0;i<job.used_count;++i) {
            const int id=job.used_ids[i];if(!o.copied[m][id])continue;
            const void *key=f.weights[m]+size_t(id)*p.width;
            const size_t bytes=p.width+(id<p.experts-1?std::min<size_t>(p.width,512):0);
            if(frequency)frequency->record(key);
            if(p.cached[id]) {
                auto *e=fetch_hit_entry(key);GGML_ASSERT(e);lru.splice(lru.begin(),lru,e->lru);
                if(managed)e->touched=strata_expert_touch_clock(device);
                if(host_cache && host_cache->demotion_enabled() && f.files[m])
                    host_cache->observe_gpu(f.files[m],f.offsets[m]+size_t(id)*p.width,bytes);
                ++stats.hits;
            } else {++stats.misses;if(!admitted[id])++stats.bypass;record_fetch_refault(key,bytes);}
        }
    }
    o.active=false;
}
