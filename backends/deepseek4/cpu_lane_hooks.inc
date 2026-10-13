// Called only for sole, unfused, short MUL_MAT_ID consumers. The ordinary
// scheduler CPU audit remains active; this is an explicit partial-matrix lane.
static bool strata_cpu_eligible(ggml_backend_t backend,const ggml_tensor *src,
        const ggml_tensor *node,const ggml_tensor *route,size_t ids_bytes) {
#ifndef STRATA_DS4_CPU_LANE_COMPILED
    return false;
#else
    if(!strata_cuda_expert_ordered(backend,node) || !src || !src->data || !route ||
       !ggml_backend_buffer_is_host(src->buffer) || !ggml_is_contiguous(src))return false;
    const auto *x=node->src[1],*w=node->src[0];
    const auto *ctx=(ggml_backend_cuda_context *)backend->context;
    const int cc=ggml_cuda_info().devices[ctx->device].cc;
    // The scalar numerical oracle mirrors generic NVIDIA four-warp MMVQ.
    // Other dispatch tables/window sizes retain the GPU arithmetic path.
    if(strata_ds4::cpu_mmvq_reference_enabled() &&
       (route->ne[1]>4 || !GGML_CUDA_CC_IS_NVIDIA(cc) || cc<GGML_CUDA_CC_AMPERE || cc==GGML_CUDA_CC_DGX_SPARK))return false;
    if(!ggml_is_contiguous(node) || node->ne[3]!=1 || x->ne[3]!=1 || x->nb[0]!=4 ||
       !x->buffer || x->buffer->buft!=ggml_backend_cuda_buffer_type(ctx->device) ||
       src->type!=w->type || !ggml_are_same_shape(src,w) ||
       !ggml_are_same_shape(route,node->src[2]) || route->type!=GGML_TYPE_I32 || route->nb[0]!=4 ||
       ggml_nbytes(route)>ids_bytes || route->ne[0]<1 || route->ne[0]>8 || route->ne[1]>8 ||
       x->ne[0]!=w->ne[0] || (x->ne[1]!=1 && x->ne[1]!=route->ne[0]) || x->ne[2]!=route->ne[1] ||
       node->ne[2]>get_mmvq_mmid_max_batch(w->type,cc) ||
       ggml_nbytes(x)>(2ULL<<20) || ggml_nbytes(node)>(2ULL<<20) ||
       !strata_ds4::cpu_lane_format(w->type,int(w->ne[0])))return false;
    if(const char *v=std::getenv("STRATA_DS4_ROUTED_MMQ"))if(std::strcmp(v,"0"))return false;
    return true;
#endif
}
static void strata_cuda_expert_cpu_select(ggml_backend_t backend,const ggml_tensor *src,
        const ggml_tensor *node,const ggml_tensor *route,const int32_t *ids,size_t ids_bytes,uint8_t *selected) {
    if(!strata_cpu_eligible(backend,src,node,route,ids_bytes))return;
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    auto &s=strata_expert_state(backend);auto &c=s.cpu_lane;
    if(!s.cpu_whole().legacy || !c.threads || !s.pipeline || !s.direct.enabled || s.early_hits.enabled)return;
    int count=0;
    for(int t=0;t<route->ne[1];++t)for(int j=0;j<route->ne[0];++j) {
        int id=ids[size_t(t)*route->nb[1]/4+j];
        if(id<0 || id>=src->ne[2])return;
        const void *key=(const uint8_t *)src->data+size_t(id)*src->nb[2];
        if(!selected[id] && count<c.max_experts && s.entries.find(key)==s.entries.end() &&
           strata_expert_file::resident(key,src->nb[2])) {selected[id]=1;++count;}
    }
}
static bool strata_cuda_expert_cpu_begin(ggml_backend_t backend,const ggml_tensor *src,
        ggml_tensor *node,const ggml_tensor *route,const int32_t *ids,size_t ids_bytes) {
    if(!strata_cpu_eligible(backend,src,node,route,ids_bytes))return false;
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    auto &s=strata_expert_state(backend);auto &c=s.cpu_lane;
    if(s.cpu_whole().configured) {
        if(node==s.cpu_whole().branch.nodes[0] || node==s.cpu_whole().branch.nodes[1])s.cpu_whole().begin(s.stream);
        return true;
    }
    if(!s.cpu_whole().legacy)return false;
    if(!c.threads || s.direct.tensor!=node->src[0])return false;
    c.clear();
    for(int t=0;t<route->ne[1];++t)for(int j=0;j<route->ne[0];++j) {
        int id=ids[size_t(t)*route->nb[1]/4+j];
        if(id<0 || id>=src->ne[2])return false;
        const void *key=(const uint8_t *)src->data+size_t(id)*src->nb[2];
        if(c.planned.find(key)==c.planned.end())continue;
        c.mask[size_t(id)]=1;
        const auto *x=node->src[1];
        c.rows.push_back({key,size_t(t)*x->nb[2]+size_t(j%x->ne[1])*x->nb[1],
                              size_t(t)*node->nb[2]+size_t(j)*node->nb[1]});
    }
    if(c.rows.empty())return false;
    ggml_cuda_set_device(s.device);c.allocate(ggml_nbytes(node->src[1]),ggml_nbytes(node));
    const auto start=ggml_time_us();
    CUDA_CHECK(cudaMemcpyAsync(c.input,node->src[1]->data,ggml_nbytes(node->src[1]),cudaMemcpyDeviceToHost,s.stream));
    CUDA_CHECK(cudaMemcpyAsync(c.device_mask,c.mask.data(),c.mask.size(),cudaMemcpyHostToDevice,s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
    c.input_us+=ggml_time_us()-start;c.weight=node->src[0];
    c.columns=int(src->ne[0]);c.nrows=int(src->ne[1]);c.type=src->type;
    c.experts+=std::count(c.mask.begin(),c.mask.end(),uint8_t(1));
    c.launch();return true;
}
const uint8_t *strata_expert_cpu_mask(ggml_backend_cuda_context &ctx,const ggml_tensor *weight,const int **slots) {
    *slots=nullptr;
    const char *v=std::getenv("STRATA_DS4_CPU_THREADS");if(!v || !std::atoi(v))return nullptr;
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    for(auto &entry:strata_expert_transfers) {
        auto &s=*entry.second;if(s.context!=&ctx)continue;
        if(s.cpu_whole().matches(weight)){*slots=s.cpu_whole().device_packet->slots;return s.cpu_whole().device_packet->mask;}
        if(s.cpu_lane.weight==weight)return s.cpu_lane.device_mask;
    }
    return nullptr;
}
static bool strata_expert_cpu_active(ggml_backend_cuda_context *ctx) {
    const char *v=std::getenv("STRATA_DS4_CPU_THREADS");if(!v || !std::atoi(v))return false;
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    for(auto &entry:strata_expert_transfers)if(entry.second->context==ctx && entry.second->cpu_lane.weight)return true;
    return false;
}
void strata_expert_cpu_join(ggml_backend_cuda_context &ctx,const ggml_tensor *weight,ggml_tensor *dst) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    for(auto &entry:strata_expert_transfers) {
        auto &s=*entry.second;if(s.context!=&ctx)continue;
        if(s.cpu_whole().matches(weight)){s.cpu_whole().merge(dst,ctx.stream());return;}
        if(s.cpu_lane.weight==weight) {s.cpu_lane.join(dst,ctx.stream());return;}
    }
}

static bool strata_cuda_cpu_expert_plan(ggml_backend_t backend,const StrataCpuExpertBranch *branch,
        const ggml_tensor *route,const int32_t *ids,size_t ids_bytes,uint8_t *selected) {
    for(int i=0;i<3;++i)if(!branch->nodes[i] || !branch->weights[i] ||
        !strata_cpu_eligible(backend,branch->weights[i],branch->nodes[i],route,ids_bytes))return false;
    auto *up=branch->nodes[0],*gate=branch->nodes[1],*down=branch->nodes[2],*x=up->src[1];
    auto *act=down->src[1];
    if(gate->src[1]!=x || x->ne[1]!=1 || !ggml_is_contiguous(x) || act->op!=GGML_OP_GLU ||
       act->src[0]!=gate || act->src[1]!=up || !ggml_are_same_shape(up,gate) ||
       down->ne[0]!=x->ne[0] || down->src[0]->ne[0]!=up->ne[0] || route->ne[0]*route->ne[1]>64 ||
       (ggml_get_glu_op(act)!=GGML_GLU_OP_SWIGLU && ggml_get_glu_op(act)!=GGML_GLU_OP_SWIGLU_CLAMP) ||
       act->op_params[1]!=0)return false;
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    auto &s=strata_expert_state(backend);
    if(!ids)s.next_context(*branch);
    auto &c=s.cpu_whole();
    if(!c.threads || c.legacy || !s.pipeline || !s.direct.enabled || s.early_hits.enabled)return false;
    if(ids && c.gpu_queue_active && c.branch.nodes[0]==branch->nodes[0]) {
        c.queue_adopt(selected);
        if(c.trace.enabled) {
            c.trace.route_timing=strata_ds4::CpuRouteProfile::current();c.trace.route_measured=true;
            c.trace.route.resize(size_t(route->ne[0]*route->ne[1]));
            for(int t=0;t<route->ne[1];++t)for(int j=0;j<route->ne[0];++j)
                c.trace.route[size_t(t)*route->ne[0]+j]=ids[size_t(t)*route->nb[1]/4+j];
        }
        return true;
    }
    if(!ids && (!c.gpu_queue_enabled || !c.shared_ram || !c.ram || !strata_cpu_decode || c.cap<1 ||
        !route->buffer || route->buffer->buft!=ggml_backend_cuda_buffer_type(s.device)))return false;
    std::shared_ptr<strata_expert_file::Source> files[3];uint64_t offsets[3]={};
    int first_ready=-1;uint64_t snapshot_time=0;bool prepared=false;
    auto prepare_snapshot=[&](strata_ds4::CpuGpuTable &table,StrataCpuExpertLane::QueueLeases &leases) {
        const auto start=StrataCpuExpertLane::now();table={};
        table.experts=int(branch->weights[0]->ne[2]);
        for(int m=0;m<3;++m) {
            auto *w=branch->weights[m];files[m]=strata_expert_file::find(w->data,ggml_nbytes(w));
            if(files[m])offsets[m]=files[m]->offset_of(w->data,ggml_nbytes(w));
        }
        for(int id=0;id<table.experts;++id) {
            auto &e=table.expert[id];e.ready=true;bool all_miss=true;
            for(int m=0;m<3;++m) {
                auto *w=branch->weights[m];const void *key=(const uint8_t *)w->data+size_t(id)*w->nb[2];
                const bool miss=s.entries.find(key)==s.entries.end();all_miss&=miss;if(miss)e.missing+=w->nb[2];
                if(s.frequency && !c.fixed)e.score+=s.frequency->score(key);
            }
            e.eligible=c.fixed || all_miss;
            // Ineligible GPU residents cannot be sent to the CPU. Nor can an
            // incomplete RAM triple. Avoid speculative locks/refcounts for
            // addresses which the router is guaranteed never to consume.
            int queried=0;
            if(!c.snapshot_prune || e.eligible)for(int m=0;m<3;++m) {
                auto *w=branch->weights[m];++queried;
                auto &lease=leases[id][m];
                lease=c.ram->acquire(files[m],offsets[m]+size_t(id)*w->nb[2],w->nb[2],false,false);
                e.ready&=bool(lease);e.matrix[m]={uint64_t(uintptr_t(lease.data())),lease.generation()};
                if(c.snapshot_prune && !e.ready)break;
            }
            if(!queried)e.ready=false;
            c.snapshot_queries+=queried;c.snapshot_skipped+=3-queried;
            if(!e.ready || !e.eligible)for(auto &lease:leases[id])lease={};
            else if(first_ready<0)first_ready=id;
        }
        snapshot_time=StrataCpuExpertLane::now()-start;c.snapshot_us+=snapshot_time;
    };
    if(!ids && c.snapshot_overlap && !c.context_overlap) {
        s.finish_context(false,[&] {
            if(!c.next_snapshot)c.next_snapshot=std::make_unique<StrataCpuExpertLane::NextSnapshot>();
            const auto before=cudaEventQuery(c.consumed);
            if(before!=cudaErrorNotReady)CUDA_CHECK(before);
            prepare_snapshot(c.next_snapshot->table,c.next_snapshot->leases);
            prepared=true;++c.snapshot_prepared;c.snapshot_prepare_us+=snapshot_time;
            if(before==cudaErrorNotReady) {
                ++c.snapshot_tail_pending;c.snapshot_candidate_us+=snapshot_time;
                const auto after=cudaEventQuery(c.consumed);
                if(after!=cudaErrorNotReady){CUDA_CHECK(after);++c.snapshot_tail_finished;}
            }
        },false);
        if(!prepared)++c.snapshot_fallbacks;
    } else s.finish_context();
    ggml_cuda_set_device(s.device);c.allocate(s.device);
    c.branch=*branch;c.configured=true;c.pending=true;c.prepare_work();++c.branches;
    c.trace.begin(branch->weights[0]->name,route);
    if(!ids) {
        const auto start=StrataCpuExpertLane::now();auto &table=c.queue_table;
        if(prepared) {
            table=c.next_snapshot->table;c.queue_leases.swap(c.next_snapshot->leases);
            // clear() drained the old branch and left the receiving leases empty.
        } else prepare_snapshot(table,c.queue_leases);
        table.backend=uint64_t(uintptr_t(&c));table.epoch=++c.queue_epoch;
        table.layer=uint64_t(uintptr_t(branch->weights[0]));
        table.columns=c.work.columns;table.tokens=c.work.tokens;table.topk=c.work.topk;
        table.cap=std::min(8,c.cap);table.force=c.force;
        auto &fetch=s.gpu_fetch();
        const bool fetch_ready=fetch.enabled && fetch.epoch<UINT32_MAX && files[0] && files[1] && files[2];
        // An empty CPU snapshot is valid: publish an all-GPU decision and let
        // actual GPU demand warm RAM/VRAM without reading route IDs on host.
        if(first_ready<0 && !fetch_ready) {c.clear();return false;}
        if(first_ready>=0 && !c.force) {
            // Keep the old arithmetic/cost formula; broader calibration is stage 4.
            const auto key=std::make_tuple(int(c.work.gate_type),int(c.work.up_type),int(c.work.down_type),c.work.columns,c.work.hidden,1);
            if(!c.costs.count(key))for(auto &other:s.branch_contexts)
                if(&other.cpu!=&c && other.cpu.started && other.cpu.completion)CUDA_CHECK(cudaEventSynchronize(other.cpu.consumed));
            if(auto *blob=c.lease(*branch,first_ready)) {
                const auto cost=c.calibrate(*blob);table.compute_us=cost.compute_us;
                table.copy_us_per_byte=cost.bytes?cost.copy_us/cost.bytes:0;table.overhead_us=cost.overhead_us;
            } else table.cap=0;
        }
        if(fetch_ready) {
            fetch.allocate();++fetch.epoch;++fetch.branches;
            fetch.ram=c.ram;
            if(fetch.release_ram)fetch.release_cpu_candidates=[&c](const bool *selected){c.queue_release_unused(selected);};
            for(int m=0;m<3;++m) {
                auto *w=branch->weights[m];auto *dst=branch->nodes[m]->src[0];auto &p=fetch.plan[m];p={};
                fetch.files[m]=files[m];fetch.offsets[m]=offsets[m];fetch.submitted[m]=false;
                fetch.weights[m]=(const uint8_t *)w->data;
                p.epoch=fetch.epoch;p.width=w->nb[2];p.experts=int(w->ne[2]);p.topk=int(route->ne[0]);p.tokens=int(route->ne[1]);
                p.scratch=dst->data;p.compact=strata_compact_slots(dst);
                for(int id=0;id<p.experts;++id) {
                    p.keep_ram[id]=fetch.cache_keep_ram && fetch.cache_slots && table.expert[id].ready && table.expert[id].eligible;
                    const size_t bytes=p.width+(id<p.experts-1?std::min<size_t>(p.width,512):0);
                    const void *key=(const uint8_t *)w->data+size_t(id)*p.width;
                    const auto it=s.entries.find(key);
                    if(it!=s.entries.end() && it->second.bytes==bytes) {
                        p.cached[id]=s.managed?it->second.allocation:s.arena+it->second.offset;
                        s.protected_keys().insert(key);
                    } else fetch.leases[m][id]=c.ram->acquire(files[m],offsets[m]+size_t(id)*p.width,bytes,false,false);
                    fetch.ram_hit[m][id]=bool(fetch.leases[m][id]);
                }
            }

            if(c.context_overlap)fetch.epoch_leases=fetch.leases;
            s.prepare_fetch_cache();fetch.prepare_fills();
            fetch.active=true;fetch.queue->arm();
        }
        const auto selection=StrataCpuExpertLane::now()-start+(prepared?snapshot_time:0);
        c.select_us+=selection;if(c.trace.enabled)c.trace.select_us=selection;
        c.queue_publish(route,s.stream);s.deferred_keys().clear();return true;
    }
    if(c.trace.enabled) {
        c.trace.route.resize(size_t(route->ne[0]*route->ne[1]));
        for(int t=0;t<route->ne[1];++t)for(int j=0;j<route->ne[0];++j)
            c.trace.route[size_t(t)*route->ne[0]+j]=ids[size_t(t)*route->nb[1]/4+j];
    }
    std::memset(c.packet.mask,0,sizeof(c.packet.mask));std::fill(std::begin(c.packet.slots),std::end(c.packet.slots),-1);
    auto start=StrataCpuExpertLane::now();
    // Decide each token before inspecting subsequent route IDs. MTP drafts
    // cannot change the device chosen for a preceding target token.
    struct Candidate {int id,order;unsigned score;size_t missing;};
    std::array<StrataCpuExpertLane::Blob *,256> blobs{};
    std::array<bool,256> gpu_required{},used{};
    double prefix_cpu=0;size_t prefix_missing=0;
    for(int t=0;strata_cpu_decode && t<route->ne[1];++t) {
        std::vector<Candidate> candidates;std::array<bool,256> seen{};
        std::array<size_t,256> missing{};size_t total_missing=prefix_missing;
        for(int j=0;j<route->ne[0];++j) {
            const int id=ids[size_t(t)*route->nb[1]/4+j];
            if(id<0 || id>=branch->weights[0]->ne[2])GGML_ABORT("Invalid CPU expert route");
            used[id]=true;if(seen[id])continue;seen[id]=true;
            bool all_miss=true;unsigned score=0;
            for(auto *w:branch->weights) {
                const void *key=(const uint8_t *)w->data+size_t(id)*w->nb[2];
                const bool miss=s.entries.find(key)==s.entries.end();all_miss&=miss;
                if(miss)missing[id]+=w->nb[2];
                if(s.frequency && !c.fixed)score+=s.frequency->score(key);
            }
            if(!gpu_required[id])total_missing+=missing[id];
            if(c.fixed || all_miss)candidates.push_back({id,j,score,missing[id]});
        }
        std::sort(candidates.begin(),candidates.end(),[](const auto &a,const auto &b){return a.score!=b.score?a.score<b.score:a.order>b.order;});
        size_t verified=0;
        for(size_t i=0;i<candidates.size() && verified<size_t(c.cap);++i) {
            const int id=candidates[i].id;bool resident=true;
            if(c.shared_ram)resident=c.lease(*branch,id)!=nullptr;
            else if(!c.fixed && !c.find(*branch,id))for(auto *w:branch->weights)
                if(!strata_expert_file::resident((const uint8_t *)w->data+size_t(id)*w->nb[2],w->nb[2])){resident=false;break;}
            if(resident)candidates[verified++]=candidates[i];
        }
        candidates.resize(verified);
        int take=std::min(c.cap,int(candidates.size()));
        StrataCpuExpertLane::Cost cost;
        if(take && !c.force) {
            // Single-token calibration retains fixed pool/quantization costs.
            // Never divide a batched compute time by the draft window length.
            auto key=std::make_tuple(int(c.work.gate_type),int(c.work.up_type),int(c.work.down_type),c.work.columns,c.work.hidden,1);
            auto measured=c.costs.find(key);
            if(measured!=c.costs.end())cost=measured->second;
            else if(auto *blob=c.lease(*branch,candidates[0].id))cost=c.calibrate(*blob);
            else take=0;
            if(take) {
                int best_n=0;double cpu=prefix_cpu,gpu=double(total_missing)*cost.copy_us/cost.bytes;
                double best=std::max(cpu,gpu);
                if(c.work.experts.empty())cpu+=cost.overhead_us;
                for(int i=0;i<take;++i) {
                    const auto &e=candidates[i];cpu+=cost.compute_us;
                    auto *b=c.find(*branch,e.id);if(!b || (!b->filled && !blobs[e.id]))cpu+=cost.pack_us;
                    if(!gpu_required[e.id])gpu-=double(e.missing)*cost.copy_us/cost.bytes;
                    const double next=std::max(cpu,std::max(0.0,gpu));
                    if(next<best*0.98){best=next;best_n=i+1;}
                }
                take=best_n;
            }
        }
        std::array<bool,256> token_cpu{};
        for(int i=0;i<take;++i)if(auto *p=c.lease(*branch,candidates[i].id)) {
            const int id=candidates[i].id;token_cpu[id]=true;
            if(!c.force) {
                if(c.work.experts.empty() && prefix_cpu==0)prefix_cpu+=cost.overhead_us;
                prefix_cpu+=cost.compute_us;
                if(!p->filled && !blobs[id])prefix_cpu+=cost.pack_us;
            }
            if(!blobs[id])c.leases.push_back(p);blobs[id]=p;
        }
        for(int j=0;j<route->ne[0];++j) {
            const int id=ids[size_t(t)*route->nb[1]/4+j];
            if(!token_cpu[id]) {
                if(!gpu_required[id])prefix_missing+=missing[id];gpu_required[id]=true;continue;
            }
            auto *b=blobs[id];int slot=t*int(route->ne[0])+j;c.packet.slots[slot]=int(c.work.experts.size());
            c.work.experts.push_back({b->weight(1),b->weight(0),b->weight(2),t,slot});
        }
    }
    // Mixed placements must still upload the matrix for its GPU occurrences.
    for(int id=0;id<256;++id)c.packet.mask[id]=selected[id]=used[id] && !gpu_required[id];
    if(c.work.experts.empty())++c.declined;
    const auto selection=StrataCpuExpertLane::now()-start;
    c.select_us+=selection;if(c.trace.enabled)c.trace.select_us=selection;return true;
}

static bool strata_cuda_expert_gpu_fetch(ggml_backend_t backend,const ggml_tensor *source,
        ggml_tensor *node,const ggml_tensor *route) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    auto &s=strata_expert_state(backend);auto &f=s.gpu_fetch();auto &c=s.cpu_whole();
    if(!f.active || !c.gpu_queue_active)return false;
    int m=0;while(m<3 && c.branch.nodes[m]!=node)++m;
    if(m==3 || c.branch.weights[m]!=source || route!=node->src[2])return false;
    GGML_ASSERT(!f.submitted[m] && s.stream==((ggml_backend_cuda_context *)backend->context)->stream());
    s.retire_direct();s.direct.prepare(node->src[0]);
    // The GPU publisher writes the same stable table used by captured MMVQ.
    // Protected cache keys cover the full branch until its final consumer.
    s.direct.dirty=false;f.submitted[m]=true;
    strata_ds4::gpu_fetch_publish<<<1,256,0,s.stream>>>(f.queue->mapped+m,f.plan[m],
        (const int32_t *)route->data,int(route->nb[1]/4),c.device_packet,s.direct.table);
    CUDA_CHECK(cudaGetLastError());
    f.queue->wait(m,s.stream);
    if(f.early_retire && f.submitted[0] && f.submitted[1] && f.submitted[2])f.queue->mark_copies(s.stream);
    return true;
}
