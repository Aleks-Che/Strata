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
    if(!s.cpu_whole.legacy || !c.threads || !s.pipeline || !s.direct.enabled || s.early_hits.enabled)return;
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
    if(s.cpu_whole.configured) {
        if(node==s.cpu_whole.branch.nodes[0] || node==s.cpu_whole.branch.nodes[1])s.cpu_whole.begin(s.stream);
        return true;
    }
    if(!s.cpu_whole.legacy)return false;
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
        if(s.cpu_whole.matches(weight)){*slots=s.cpu_whole.device_packet->slots;return s.cpu_whole.device_packet->mask;}
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
        if(s.cpu_whole.matches(weight)){s.cpu_whole.merge(dst,ctx.stream());return;}
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
    auto &s=strata_expert_state(backend);auto &c=s.cpu_whole;
    if(!c.threads || c.legacy || !s.pipeline || !s.direct.enabled || s.early_hits.enabled)return false;
    s.finish_plan();ggml_cuda_set_device(s.device);c.allocate(s.device);
    c.branch=*branch;c.configured=true;c.pending=true;c.prepare_work();++c.branches;
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
    c.select_us+=StrataCpuExpertLane::now()-start;return true;
}
