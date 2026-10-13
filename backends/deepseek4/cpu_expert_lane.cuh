#pragma once
#include "cpu_lane.hpp"
#include "host_cache.hpp"
#include "cpu_lane_trace.cuh"
#include "cpu_gpu_router.cuh"
#include <array>
#include <chrono>
#include <exception>
#include <tuple>
#include <cmath>

// The CPU callback only touches CPU memory and the CPU pool. All CUDA calls
// are enqueued by the engine before it starts. This permits an external event
// wait inside a captured down graph without a host join or device spin loop.
struct StrataCpuExpertLane {
    static constexpr size_t scratch_bytes=2ULL<<20;
    // mask skips transfers only when every occurrence is on CPU. slots skips
    // compute per token, including experts shared with GPU rows in this window.
    using Packet=strata_ds4::CpuQueuePacket;
    Packet packet{};
    bool gpu_queue_enabled=false,gpu_queue_active=false,queue_adopted=false;
    bool snapshot_prune=false;
    bool snapshot_overlap=false;
    bool context_overlap=false,consumer_pending=false;
    uint64_t consumer_epoch=0,context_deferred=0,context_tail_pending=0,context_wait_us=0;
    bool consumer_handoff=false,handoff_pending=false;
    uint64_t handoffs=0,handoff_fenced=0,handoff_waits=0,handoff_ram_fallbacks=0;
    using QueueLeases=std::array<std::array<strata_ds4::HostCache::Lease,3>,256>;
    // Owner-only staging. This preparation can precede the service replies,
    // so it must not mutate the active queue_table/leases.
    struct NextSnapshot { strata_ds4::CpuGpuTable table{};QueueLeases leases; };
    std::unique_ptr<NextSnapshot> next_snapshot;
    uint64_t snapshot_prepared=0,snapshot_tail_pending=0,snapshot_tail_finished=0;
    uint64_t snapshot_prepare_us=0,snapshot_candidate_us=0,snapshot_fallbacks=0;
    bool async_park=false;
    bool release_weights=false;
    // The CPU service writes these; live statistics may read them concurrently.
    std::atomic<uint64_t> weight_release_gate_up{0},weight_release_down{0},weight_release_bytes{0};
    std::atomic<uint64_t> weight_release_deferred{0},weight_release_us{0},weight_release_jobs{0},weight_capture_jobs{0};
    uint64_t snapshot_queries=0,snapshot_skipped=0,snapshot_us=0;
    uint64_t snapshot_released=0,snapshot_release_bytes=0,snapshot_release_deferred=0;
    uint64_t queue_epoch=0,queue_jobs=0,queue_rows=0,queue_branches=0;
    uint64_t verified_matrices=0,verified_bytes=0;
    cudaStream_t queue_owner=nullptr;
    std::unique_ptr<strata_ds4::CpuGpuQueue> gpu_queue;
    strata_ds4::CpuGpuTable queue_table{};
    strata_ds4::CpuGpuTable *device_queue_table=nullptr;
    strata_ds4::CpuGpuTable *upload_queue_table=nullptr;
    QueueLeases queue_leases;
    QueueLeases epoch_leases;
    struct Blob {
        const void *key[3]={}; size_t widths[3]={};
        strata_ds4::HostCache::Lease shared[3];
        const void *weight(int i)const {
            if(shared[i])return shared[i].data();
            size_t off=0;for(int j=0;j<i;++j)off+=widths[j];return data+off;
        }
        uint8_t *data=nullptr;size_t capacity=0;uint64_t stamp=0;
        bool filled=false,held=false;
        ~Blob(){if(data)CUDA_CHECK(cudaFreeHost(data));}
        size_t bytes()const{return widths[0]+widths[1]+widths[2];}
        void fill(){if(!filled){size_t off=0;for(int i=0;i<3;++i){std::memcpy(data+off,key[i],widths[i]);off+=widths[i];}filled=true;}}
    };
    struct Cost {double compute_us=0,copy_us=0,overhead_us=0,pack_us=0;size_t bytes=0;};
    std::map<std::tuple<int,int,int,int,int,int>,Cost> costs;
    std::vector<std::unique_ptr<Blob>> cache;
    std::vector<Blob *> leases;
    std::shared_ptr<strata_ds4::HostCache> ram;
    bool shared_ram=false;
    size_t cache_bytes=0,cache_limit=256ULL<<20;
    uint64_t clock=0;
    int threads=0,cap=1;
    bool force=false,fixed=false,legacy=false,pending=false,configured=false,started=false,completion=false;
    int device=0;
    StrataCpuExpertBranch branch{};
    strata_ds4::CpuExpertWork work;
    StrataCpuLaneTrace trace;
    std::shared_ptr<strata_ds4::CpuLanePool> pool;
    // Captured graphs keep these stable across the bounded host-context ring.
    // The owner stream binds the current queue before each branch. Individual
    // contexts retain their own work, leases, trace, upload source and replies.
    struct GraphStorage {
        float *output=nullptr;Packet *packet=nullptr;
        strata_ds4::CpuQueueBinding *binding=nullptr;cudaEvent_t ready=nullptr;
        ~GraphStorage(){if(output)cudaFree(output);if(packet)cudaFree(packet);if(binding)cudaFree(binding);if(ready)cudaEventDestroy(ready);}
    };
    std::shared_ptr<GraphStorage> graph_storage;
    float *input=nullptr,*output=nullptr,*device_output=nullptr;
    Packet *device_packet=nullptr;
    cudaStream_t stream=nullptr;
    cudaEvent_t input_ready=nullptr,input_copied=nullptr,ready=nullptr,consumed=nullptr;
    std::exception_ptr error;
    uint64_t operations=0,experts=0,saved_bytes=0,compute_us=0,pack_us=0,retire_us=0,select_us=0;
    uint64_t input_bytes=0,output_bytes=0,ram_hits=0,ram_misses=0,declined=0,branches=0;
    static int64_t now(){return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
    StrataCpuExpertLane() {
        if(const char *v=std::getenv("STRATA_DS4_CPU_THREADS"))threads=std::atoi(v);
        if(const char *v=std::getenv("STRATA_DS4_CPU_SHARED_RAM"))shared_ram=std::atoi(v)!=0;
        if(const char *v=std::getenv("STRATA_DS4_CPU_GPU_QUEUE"))gpu_queue_enabled=std::atoi(v)!=0;
        if(const char *v=std::getenv("STRATA_DS4_CPU_SNAPSHOT_PRUNE"))snapshot_prune=std::atoi(v)!=0;
        if(const char *v=std::getenv("STRATA_DS4_CPU_SNAPSHOT_OVERLAP"))snapshot_overlap=std::atoi(v)!=0;
        if(const char *v=std::getenv("STRATA_DS4_CPU_CONTEXT_OVERLAP"))context_overlap=std::atoi(v)!=0;
        if(const char *v=std::getenv("STRATA_DS4_CPU_CONSUMER_HANDOFF"))consumer_handoff=std::atoi(v)!=0;
        if(const char *v=std::getenv("STRATA_DS4_QUEUE_ASYNC_PARK"))async_park=std::atoi(v)!=0;
        if(const char *v=std::getenv("STRATA_DS4_CPU_RELEASE_WEIGHTS"))release_weights=std::atoi(v)!=0;
        if(const char *v=std::getenv("STRATA_DS4_CPU_MAX_EXPERTS"))cap=std::atoi(v);
        if(const char *v=std::getenv("STRATA_DS4_CPU_POLICY"))force=std::strcmp(v,"force")==0;
        // Diagnostic only: stable device placement separates numerical changes
        // caused by adaptive routing from KV/session lifecycle defects.
        if(const char *v=std::getenv("STRATA_DS4_CPU_TEST_FIXED"))fixed=std::atoi(v)!=0;
        force|=fixed;
        if(const char *v=std::getenv("STRATA_DS4_CPU_MATRIX_LEGACY"))legacy=std::atoi(v)!=0;
        if(const char *v=std::getenv("STRATA_DS4_CPU_RAM_MIB"))cache_limit=size_t(std::max(0,std::min(4096,std::atoi(v))))<<20;
    }
    ~StrataCpuExpertLane() {
        const size_t queue_bytes=gpu_queue?sizeof(strata_ds4::CpuQueueSlot)*strata_ds4::cpu_queue_depth:0;
        clear();cache.clear();
        if(gpu_queue_enabled)GGML_LOG_INFO("STRATA_CPU_CONSUMER_HANDOFF enabled=%d submitted=%llu fenced=%llu waits=%llu ram_fallbacks=%llu upload_bytes=%zu\n",
            int(consumer_handoff),(unsigned long long)handoffs,(unsigned long long)handoff_fenced,(unsigned long long)handoff_waits,
            (unsigned long long)handoff_ram_fallbacks,upload_queue_table?sizeof(queue_table):0);
        if(gpu_queue_enabled)GGML_LOG_INFO("STRATA_CPU_CONTEXT_OVERLAP enabled=%d deferred=%llu tail_pending=%llu wait_us=%llu\n",
            int(context_overlap),(unsigned long long)context_deferred,(unsigned long long)context_tail_pending,
            (unsigned long long)context_wait_us);
        if(gpu_queue)GGML_LOG_INFO("STRATA_CPU_QUEUE_PARK enabled=%d async_disarms=%llu waits=%llu wait_us=%llu\n",
            int(async_park),(unsigned long long)gpu_queue->async_disarms,(unsigned long long)gpu_queue->park_waits,(unsigned long long)gpu_queue->park_us);
        gpu_queue.reset();
        if(device_queue_table)CUDA_CHECK(cudaFree(device_queue_table));
        if(upload_queue_table)CUDA_CHECK(cudaFreeHost(upload_queue_table));
        if(gpu_queue_enabled)GGML_LOG_INFO("STRATA_CPU_GPU_QUEUE branches=%llu jobs=%llu rows=%llu slots=2 mapped_bytes=%zu\n",
            (unsigned long long)queue_branches,(unsigned long long)queue_jobs,(unsigned long long)queue_rows,
            queue_bytes);
        if(verified_matrices)GGML_LOG_INFO("STRATA_CPU_RAM_VERIFY matrices=%llu bytes=%llu mismatches=0\n",
            (unsigned long long)verified_matrices,(unsigned long long)verified_bytes);
        if(gpu_queue_enabled)GGML_LOG_INFO("STRATA_CPU_SNAPSHOT prune=%d queries=%llu skipped=%llu snapshot_us=%llu select_us=%llu\n",
            int(snapshot_prune),(unsigned long long)snapshot_queries,(unsigned long long)snapshot_skipped,(unsigned long long)snapshot_us,(unsigned long long)select_us);
        if(gpu_queue_enabled)GGML_LOG_INFO("STRATA_CPU_SNAPSHOT_RELEASE matrices=%llu lease_bytes=%llu deferred=%llu\n",
            (unsigned long long)snapshot_released,(unsigned long long)snapshot_release_bytes,(unsigned long long)snapshot_release_deferred);
        if(gpu_queue_enabled)GGML_LOG_INFO("STRATA_CPU_SNAPSHOT_OVERLAP enabled=%d prepared=%llu tail_pending=%llu tail_finished=%llu prepare_us=%llu candidate_us=%llu fallback=%llu staging_bytes=%zu\n",
            int(snapshot_overlap),(unsigned long long)snapshot_prepared,(unsigned long long)snapshot_tail_pending,
            (unsigned long long)snapshot_tail_finished,(unsigned long long)snapshot_prepare_us,
            (unsigned long long)snapshot_candidate_us,(unsigned long long)snapshot_fallbacks,next_snapshot?sizeof(NextSnapshot):0);
        if(gpu_queue_enabled)GGML_LOG_INFO("STRATA_CPU_WEIGHT_RELEASE enabled=%d jobs=%llu gate_up=%llu down=%llu lease_bytes=%llu deferred=%llu capture_jobs=%llu release_us=%llu\n",
            int(release_weights),(unsigned long long)weight_release_jobs.load(),(unsigned long long)weight_release_gate_up.load(),
            (unsigned long long)weight_release_down.load(),(unsigned long long)weight_release_bytes.load(),
            (unsigned long long)weight_release_deferred.load(),(unsigned long long)weight_capture_jobs.load(),
            (unsigned long long)weight_release_us.load());
        if(stream)CUDA_CHECK(cudaStreamDestroy(stream));
        for(auto e:{input_ready,input_copied,consumed})if(e)CUDA_CHECK(cudaEventDestroy(e));
        if(input)CUDA_CHECK(cudaFreeHost(input));if(output)CUDA_CHECK(cudaFreeHost(output));
    }
    void allocate(int dev) {
        if(stream)return;device=dev;
        if(!pool)pool=std::make_shared<strata_ds4::CpuLanePool>(threads);
        trace.allocate();
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        for(auto p:{&input_ready,&input_copied,&consumed})CUDA_CHECK(cudaEventCreateWithFlags(p,cudaEventDisableTiming));
        CUDA_CHECK(cudaMallocHost((void **)&input,scratch_bytes));CUDA_CHECK(cudaMallocHost((void **)&output,scratch_bytes));
        if(!graph_storage) {
            graph_storage=std::make_shared<GraphStorage>();
            CUDA_CHECK(cudaMalloc((void **)&graph_storage->output,scratch_bytes));
            CUDA_CHECK(cudaMalloc((void **)&graph_storage->packet,sizeof(Packet)));
            CUDA_CHECK(cudaMalloc((void **)&graph_storage->binding,sizeof(strata_ds4::CpuQueueBinding)));
            CUDA_CHECK(cudaEventCreateWithFlags(&graph_storage->ready,cudaEventDisableTiming));
        }
        device_output=graph_storage->output;device_packet=graph_storage->packet;ready=graph_storage->ready;
        if(gpu_queue_enabled && shared_ram) {
            gpu_queue=std::make_unique<strata_ds4::CpuGpuQueue>([this](auto &slot){queue_execute(slot);});
            CUDA_CHECK(cudaMalloc((void **)&device_queue_table,sizeof(queue_table)));
            CUDA_CHECK(cudaMallocHost((void **)&upload_queue_table,sizeof(queue_table)));
            strata_ds4::cpu_queue_reset<<<1,1,0,stream>>>(gpu_queue->mapped,gpu_queue->device);
            CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
    Blob *find(const StrataCpuExpertBranch &b,int id) {
        for(auto &p:cache){bool same=true;for(int j=0;j<3;++j)
            same&=p->key[j]==(const uint8_t *)b.weights[j]->data+size_t(id)*b.weights[j]->nb[2];
            if(same)return p.get();}
        return nullptr;
    }
    Blob *lease(const StrataCpuExpertBranch &b,int id) {
        const auto t=trace.enabled?now():0;auto *result=lease_impl(b,id);
        if(trace.enabled){trace.lease_us+=now()-t;++trace.lease_calls;if(!result)++trace.lease_failures;}
        return result;
    }
    Blob *lease_impl(const StrataCpuExpertBranch &b,int id) {
        if(auto *p=find(b,id)){p->held=true;p->stamp=++clock;++ram_hits;return p;}
        if(shared_ram) {
            if(!ram)return nullptr;
            auto p=std::make_unique<Blob>();
            for(int j=0;j<3;++j) {
                auto *w=b.weights[j];p->key[j]=(const uint8_t *)w->data+size_t(id)*w->nb[2];p->widths[j]=w->nb[2];
                auto file=strata_expert_file::find(p->key[j],p->widths[j]);
                if(!file){++ram_misses;return nullptr;}
                p->shared[j]=ram->acquire(file,file->offset_of(p->key[j],p->widths[j]),p->widths[j]);
                if(!p->shared[j]){++ram_misses;return nullptr;}
            }
            p->filled=p->held=true;++ram_hits;auto *result=p.get();cache.push_back(std::move(p));return result;
        }
        size_t bytes=0;for(auto *w:b.weights)bytes+=w->nb[2];
        if(bytes>cache_limit)return nullptr;
        Blob *p=nullptr;
        for(auto &v:cache)if(!v->held && v->capacity>=bytes && (!p || v->stamp<p->stamp))p=v.get();
        // Grow the bounded cache before replacing an old expert. No CUDA
        // allocation/free occurs on the callback thread.
        if(cache_bytes+bytes<=cache_limit) {
            auto v=std::make_unique<Blob>();CUDA_CHECK(cudaMallocHost((void **)&v->data,bytes));
            v->capacity=bytes;cache_bytes+=bytes;p=v.get();cache.push_back(std::move(v));
        }
        if(!p)return nullptr;
        for(int j=0;j<3;++j){p->key[j]=(const uint8_t *)b.weights[j]->data+size_t(id)*b.weights[j]->nb[2];p->widths[j]=b.weights[j]->nb[2];}
        p->filled=false;p->held=true;p->stamp=++clock;++ram_misses;return p;
    }
    void finish_consumer(bool owner_fenced=false) {
        if(handoff_pending) {
            // Normally the NEXT fetch's copy fence already covers this receipt.
            // Fallback/teardown without that fetch must drain the owner stream.
            if(!owner_fenced){CUDA_CHECK(cudaStreamSynchronize(queue_owner));++handoff_waits;}
            else ++handoff_fenced;
            gpu_queue->check_receipt(consumer_epoch);handoff_pending=false;
        }
        if(!consumer_pending)return;
        const auto start=now();CUDA_CHECK(cudaEventSynchronize(consumed));
        gpu_queue->check_consumed(consumer_epoch);
        context_wait_us+=now()-start;consumer_pending=false;
    }
    // Only called while preparing another GPU-fetch branch, under the owner
    // mutex. Old CPU callbacks have replied; device result/table/slot reuse is
    // ordered on the same stream. No allocation is freed by this handoff.
    void handoff_consumer(cudaStream_t owner,bool ram_fills) {
        if(!consumer_handoff || !consumer_pending)return;
        // Early CPU weight release makes RAM victim availability depend on
        // CPU/I/O timing. This combination failed model placement parity after
        // removing the final fence. Keep the established boundary until RAM
        // admissions can preserve their victim policy under that overlap.
        if(ram_fills && release_weights){++handoff_ram_fallbacks;return;}
        GGML_ASSERT(!handoff_pending && owner==queue_owner);
        strata_ds4::cpu_queue_retire<<<1,1,0,owner>>>(gpu_queue->mapped,consumer_epoch);
        CUDA_CHECK(cudaGetLastError());
        consumer_pending=false;handoff_pending=true;++handoffs;
    }
    // defer_consumer requires a completed route producer AND a copy fence.
    // The CPU reply releases work/trace/leases; the mapped result stays owned
    // by the GPU. Its event or stream-ordered receipt validates the consumer.
    void clear(bool defer_consumer=false) {
        finish_consumer();
        const auto retirement=trace.enabled?now():0;
        if(gpu_queue_active) {
            const auto t=now();
            GGML_ASSERT(!defer_consumer || completion);
            if(!defer_consumer) {
                if(completion)CUDA_CHECK(cudaEventSynchronize(consumed));
                else CUDA_CHECK(cudaStreamSynchronize(queue_owner));
            }
            gpu_queue->drain(!async_park || !completion);
            if(defer_consumer) {
                consumer_epoch=queue_epoch;consumer_pending=true;++context_deferred;
                const auto result=cudaEventQuery(consumed);
                if(result==cudaErrorNotReady)++context_tail_pending;else CUDA_CHECK(result);
            } else if(!completion)gpu_queue->abandon_after_drain();
            else for(int i=0;i<strata_ds4::cpu_queue_depth;++i) {
                auto &slot=gpu_queue->host[i];const auto seq=slot.request.load(cuda::memory_order_acquire);
                if(seq && slot.job.epoch==queue_epoch && slot.consumed.load(cuda::memory_order_acquire)!=seq)
                    GGML_ABORT("CPU GPU queue consumer missing from graph replay");
            }
            const auto failure=gpu_queue->host[0].gpu_failure.load(cuda::memory_order_acquire);
            if(failure)GGML_ABORT("CPU GPU queue failed: status=%u",failure);
            if(!queue_adopted) {uint8_t selected[256];queue_adopt(selected);}
            retire_us+=now()-t;
        } else if(started){auto t=now();CUDA_CHECK(cudaStreamSynchronize(stream));
            if(completion)CUDA_CHECK(cudaEventSynchronize(consumed));retire_us+=now()-t;}
        trace.finish(this,work,trace.enabled?now()-retirement:0,started && completion && !error);
        if(error){try{std::rethrow_exception(error);}catch(const std::exception &e){GGML_ABORT("CPU expert failed: %s",e.what());}catch(...){GGML_ABORT("CPU expert failed");}}
        for(auto &p:cache)p->held=false;
        leases.clear();if(shared_ram)cache.clear();work.experts.clear();configured=pending=started=completion=false;branch={};
        for(auto &row:queue_leases)for(auto &lease:row)lease={};
        for(auto &row:epoch_leases)for(auto &lease:row)lease={};gpu_queue_active=false;queue_adopted=false;
    }
    bool matches(const ggml_tensor *w)const {
        if(!configured)return false;
        for(auto *n:branch.nodes)if(n->src[0]==w)return true;
        return false;
    }
    bool selected(const ggml_tensor *w,int id)const{return matches(w) && packet.mask[id];}
    void prepare_work() {
        auto *up=branch.weights[0],*gate=branch.weights[1],*down=branch.weights[2];
        work.up_type=up->type;work.gate_type=gate->type;work.down_type=down->type;
        work.columns=int(up->ne[0]);work.hidden=int(up->ne[1]);
        work.tokens=int(branch.nodes[0]->ne[2]);work.topk=int(branch.nodes[0]->ne[1]);work.clamp=branch.clamp;
        work.input=input;work.output=output;
    }
    static void CUDART_CB execute(void *arg) noexcept {
        auto &c=*(StrataCpuExpertLane *)arg;
        if(c.trace.enabled)c.trace.callback_us=now();
        try {
            auto t=now();if(!c.shared_ram){for(auto *p:c.leases)p->fill();c.pack_us+=now()-t;}
            if(c.trace.enabled)c.trace.pack_us=c.shared_ram?0:now()-t;
            t=now();c.work.compute(*c.pool,c.trace.enabled?&c.trace.cpu:nullptr);c.compute_us+=now()-t;
        } catch(...) {c.error=std::current_exception();std::fill(c.output,c.output+scratch_bytes/sizeof(float),NAN);}
        if(c.trace.enabled)c.trace.callback_end_us=now();
    }
    static void CUDART_CB noop(void *) {}
    Cost calibrate(Blob &blob) {
        auto key=std::make_tuple(int(work.gate_type),int(work.up_type),int(work.down_type),work.columns,work.hidden,1);
        auto found=costs.find(key);if(found!=costs.end())return found->second;
        const int saved_tokens=work.tokens;work.tokens=1;
        Cost c;c.bytes=blob.bytes();auto t=now();blob.fill();c.pack_us=shared_ram?0:now()-t;
        std::fill(input,input+size_t(work.tokens)*work.columns,0.125f);
        const auto saved=work.experts;work.experts.clear();
        for(int i=0;i<work.tokens;++i)work.experts.push_back({blob.weight(1),blob.weight(0),blob.weight(2),i,i});
        work.compute(*pool);
        for(int i=0;i<3;++i){t=now();work.compute(*pool);c.compute_us+=now()-t;}c.compute_us/=3;
        work.experts=saved;
        void *probe=nullptr;cudaEvent_t a,b;CUDA_CHECK(cudaMalloc(&probe,c.bytes));
        CUDA_CHECK(cudaEventCreate(&a));CUDA_CHECK(cudaEventCreate(&b));float ms;
        CUDA_CHECK(cudaEventRecord(a,stream));
        for(int i=0;i<4;++i){size_t off=0;for(int j=0;j<3;++j){
            CUDA_CHECK(cudaMemcpyAsync((uint8_t *)probe+off,blob.weight(j),blob.widths[j],cudaMemcpyHostToDevice,stream));off+=blob.widths[j];}}
        CUDA_CHECK(cudaEventRecord(b,stream));CUDA_CHECK(cudaEventSynchronize(b));CUDA_CHECK(cudaEventElapsedTime(&ms,a,b));c.copy_us=ms*1000/4;
        CUDA_CHECK(cudaEventRecord(a,stream));
        for(int i=0;i<4;++i){CUDA_CHECK(cudaMemcpyAsync(input,probe,size_t(work.columns)*4,cudaMemcpyDeviceToHost,stream));
            CUDA_CHECK(cudaLaunchHostFunc(stream,noop,nullptr));CUDA_CHECK(cudaMemcpyAsync(probe,output,size_t(work.columns)*4,cudaMemcpyHostToDevice,stream));}
        CUDA_CHECK(cudaEventRecord(b,stream));CUDA_CHECK(cudaEventSynchronize(b));CUDA_CHECK(cudaEventElapsedTime(&ms,a,b));c.overhead_us=ms*1000/4;
        CUDA_CHECK(cudaEventDestroy(a));CUDA_CHECK(cudaEventDestroy(b));CUDA_CHECK(cudaFree(probe));
        costs[key]=c;
        GGML_LOG_INFO("STRATA_CPU_CALIB tokens=%d types=%s/%s/%s cpu_us=%.1f h2d_us=%.1f boundary_us=%.1f pack_us=%.1f bytes=%zu\n",work.tokens,
            ggml_type_name(work.gate_type),ggml_type_name(work.up_type),ggml_type_name(work.down_type),c.compute_us,c.copy_us,c.overhead_us,c.pack_us,c.bytes);
        work.tokens=saved_tokens;return c;
    }
    void begin(cudaStream_t owner);
    void merge(ggml_tensor *dst,cudaStream_t owner);
    // I/O service, after the GPU has published all token placements. The CPU
    // service reads only selected rows; never touch their leases here. The
    // owner drains both services before reusing this table or these counters.
    void queue_release_unused(const bool *selected) {
        for(int id=0;id<queue_table.experts;++id)if(!selected[id])for(auto &lease:queue_leases[id])if(lease) {
            if(const auto bytes=ram->release_cached(lease)){++snapshot_released;snapshot_release_bytes+=bytes;}
            else ++snapshot_release_deferred;
        }
    }
    // CPU service, only after every pool worker has stopped reading this phase.
    // The I/O service touches only UNSELECTED CPU leases; mixed GPU occurrences
    // have separate fetch leases. Cache ownership prevents CUDA free here.
    void queue_release_weights(const strata_ds4::CpuQueueRequest &job,strata_ds4::CpuWeightPhase phase) {
        const auto start=now();uint64_t count=0,bytes=0,deferred=0;bool seen[256]={};
        const bool gate_up=phase==strata_ds4::CpuWeightPhase::gate_up;
        for(int i=0;i<job.rows;++i) {
            const int id=job.row[i].expert;if(seen[id])continue;seen[id]=true;
            for(int m=gate_up?0:2;m<(gate_up?2:3);++m) {
                auto &lease=queue_leases[id][m];
                if(const auto n=ram->release_cached(lease,true)){++count;bytes+=n;}else ++deferred;
            }
        }
        (gate_up?weight_release_gate_up:weight_release_down).fetch_add(count,std::memory_order_relaxed);
        weight_release_bytes.fetch_add(bytes,std::memory_order_relaxed);
        weight_release_deferred.fetch_add(deferred,std::memory_order_relaxed);
        weight_release_us.fetch_add(now()-start,std::memory_order_relaxed);
    }
    void queue_execute(strata_ds4::CpuQueueSlot &slot) {
        const auto &job=slot.job;
        if(job.backend!=queue_table.backend || job.epoch!=queue_table.epoch || job.layer!=queue_table.layer ||
           job.columns!=work.columns || job.tokens!=work.tokens || job.topk!=work.topk || job.rows<1 || job.rows>64) {
            slot.error=strata_ds4::cpu_bad_request;return;
        }
        // Validate the retained snapshot before dereferencing any host address.
        for(int i=0;i<job.rows;++i) {
            const auto &r=job.row[i];
            if(r.expert<0 || r.expert>=queue_table.experts || r.token<0 || r.token>=job.tokens ||
               r.slot<r.token*job.topk || r.slot>=(r.token+1)*job.topk || job.packet.slots[r.slot]!=i) {
                slot.error=strata_ds4::cpu_bad_request;return;
            }
            for(int m=0;m<3;++m) {
                const auto &lease=queue_leases[r.expert][m];
                if(!lease || r.matrix[m].address!=uint64_t(uintptr_t(lease.data())) || r.matrix[m].generation!=lease.generation()) {
                    slot.error=strata_ds4::cpu_bad_request;return;
                }
            }
        }
        work.input=slot.input;work.output=slot.output;work.experts.clear();
        if(trace.active && StrataCpuLaneTrace::sink().capture_limit>0) {
            // Capture audit: compare the retained RAM payload with the original
            // GGUF bytes, including matrices delivered by GPU demotion. This
            // uses native I/O and ordinary host storage; no CUDA on the service.
            strata_expert_file::Request reader;std::atomic<bool> cancel{false};bool seen[256]={};
            for(int i=0;i<job.rows;++i)if(!seen[job.row[i].expert]) {
                const auto &row=job.row[i];seen[row.expert]=true;
                for(int m=0;m<3;++m) {
                    const auto *w=branch.weights[m];const auto n=w->nb[2];
                    const auto *key=(const uint8_t *)w->data+size_t(row.expert)*n;
                    auto source=strata_expert_file::find(key,n);std::vector<uint8_t> expected(n);
                    if(!source || !reader.read_at(*source,source->offset_of(key,n),expected.data(),n,cancel,1500) ||
                       std::memcmp(expected.data(),(const void *)uintptr_t(row.matrix[m].address),n)) {
                        GGML_LOG_ERROR("STRATA_CPU_RAM_VERIFY failed layer=%s expert=%d matrix=%d\n",w->name,row.expert,m);
                        slot.error=strata_ds4::cpu_bad_request;return;
                    }
                    ++verified_matrices;verified_bytes+=n;
                }
            }
        }
        for(int i=0;i<job.rows;++i) {
            const auto &r=job.row[i];work.experts.push_back({(const void *)uintptr_t(r.matrix[1].address),
                (const void *)uintptr_t(r.matrix[0].address),(const void *)uintptr_t(r.matrix[2].address),r.token,r.slot});
        }
        strata_ds4::CpuWeightRetire retire;
        if(release_weights) {
            // trace.finish may serialize the actual weights after the GPU join.
            // Hold them for a configured capture even after its quota is used.
            if(trace.active && StrataCpuLaneTrace::sink().capture_limit>0)
                weight_capture_jobs.fetch_add(1,std::memory_order_relaxed);
            else {
                retire=[this,&job](auto phase){queue_release_weights(job,phase);};
                weight_release_jobs.fetch_add(1,std::memory_order_relaxed);
            }
        }
        if(trace.enabled)trace.callback_us=now();const auto t=now();
        work.compute(*pool,trace.enabled?&trace.cpu:nullptr,retire);compute_us+=now()-t;
        if(trace.enabled)trace.callback_end_us=now();
    }
    void queue_publish(const ggml_tensor *route,cudaStream_t owner) {
        GGML_ASSERT(gpu_queue);
        if(handoff_pending){GGML_ASSERT(owner==queue_owner);}
        else finish_consumer();
        if(trace.gpu_events)GGML_ABORT("CPU GPU queue currently supports host trace only");
        queue_owner=owner;gpu_queue_active=started=true;++queue_branches;
        // Logical snapshot lifetime outlives the service's individual reads.
        // A later context must not gain victims according to CPU completion
        // timing. FIFO owner retirement releases these non-copying references.
        if(context_overlap)epoch_leases=queue_leases;
        // The captured down graph always contains both the external event and
        // the queue join. This empty event makes that same graph valid for a
        // queue job, a callback fallback, and a short GPU-only prefill.
        CUDA_CHECK(cudaEventRecord(ready,stream));
        // The previous route's copy fence covers its table upload. A pinned
        // source avoids pageable H2D staging that may wait for the owner stream.
        const auto *table=&queue_table;
        if(upload_queue_table){std::memcpy(upload_queue_table,table,sizeof(queue_table));table=upload_queue_table;}
        CUDA_CHECK(cudaMemcpyAsync(device_queue_table,table,sizeof(queue_table),cudaMemcpyHostToDevice,owner));
        if(trace.enabled)trace.queued_us=now();gpu_queue->arm();
        strata_ds4::cpu_queue_bind<<<1,1,0,owner>>>(graph_storage->binding,gpu_queue->mapped,gpu_queue->device);
        strata_ds4::cpu_gpu_route<<<1,256,0,owner>>>(gpu_queue->mapped,gpu_queue->device,device_queue_table,
            (const int32_t *)route->data,int(route->nb[1]/4),(const float *)branch.nodes[0]->src[1]->data,device_packet);
        CUDA_CHECK(cudaGetLastError());
    }
    // Called after the scheduler's existing route fence, solely for the old
    // transfer planner. CPU compute has already been published independently.
    void queue_adopt(uint8_t *selected) {
        if(queue_adopted){std::memcpy(selected,packet.mask,sizeof(packet.mask));return;}queue_adopted=true;
        std::memset(packet.mask,0,sizeof(packet.mask));std::fill(std::begin(packet.slots),std::end(packet.slots),-1);
        for(int i=0;i<strata_ds4::cpu_queue_depth;++i) {
            auto &slot=gpu_queue->host[i];if(!slot.request.load(cuda::memory_order_acquire) || slot.job.epoch!=queue_epoch)continue;
            packet=slot.job.packet;++queue_jobs;queue_rows+=slot.job.rows;++operations;experts+=slot.job.rows;
            if(trace.enabled)trace.route.assign(slot.job.route,slot.job.route+slot.job.topk*slot.job.tokens);
            bool touched[256]={};
            for(int r=0;r<slot.job.rows;++r) {
                const int id=slot.job.row[r].expert;if(touched[id])continue;touched[id]=true;++ram_hits;
                for(auto *w:branch.weights) {
                    const void *key=(const uint8_t *)w->data+size_t(id)*w->nb[2];
                    auto file=strata_expert_file::find(key,w->nb[2]);
                    if(file) {
                        const auto offset=file->offset_of(key,w->nb[2]);
                        // This CPU demand was served by a validated RAM lease.
                        // It may already have been recycled after its last read.
                        if(release_weights)ram->observe_ram(file,offset,w->nb[2]);
                        else ram->acquire(file,offset,w->nb[2]);
                    }
                }
            }
            const size_t in=size_t(work.columns)*work.tokens*4,out=size_t(work.columns)*slot.job.rows*4;
            input_bytes+=in;output_bytes+=out;
            if(trace.enabled){trace.input_bytes=in;trace.output_bytes=out;}
            break;
        }
        std::memcpy(selected,packet.mask,sizeof(packet.mask));
    }
};

static __global__ void strata_cpu_publish(StrataCpuExpertLane::Packet *dst,StrataCpuExpertLane::Packet value) {
    const int i=threadIdx.x;dst->mask[i]=value.mask[i];if(i<64)dst->slots[i]=value.slots[i];
}
static __global__ void strata_cpu_scatter(float *dst,const float *source,const StrataCpuExpertLane::Packet *p,int columns,int rows,bool down) {
    const int slot=blockIdx.y,index=p->slots[slot];if(slot>=rows || index<0)return;
    for(int r=blockIdx.x*blockDim.x+threadIdx.x;r<columns;r+=blockDim.x*gridDim.x)
        dst[size_t(slot)*columns+r]=down?source[size_t(index)*columns+r]:0.0f;
}
inline void StrataCpuExpertLane::begin(cudaStream_t owner) {
    if(!configured || started)return;
    finish_consumer();
    if(gpu_queue) {
        strata_ds4::cpu_queue_bind<<<1,1,0,owner>>>(graph_storage->binding,gpu_queue->mapped,gpu_queue->device);
        strata_ds4::cpu_queue_reset<<<1,1,0,owner>>>(gpu_queue->mapped,gpu_queue->device);CUDA_CHECK(cudaGetLastError());
    }
    strata_cpu_publish<<<1,256,0,owner>>>(device_packet,packet);CUDA_CHECK(cudaGetLastError());
    if(!work.experts.empty()) {
        CUDA_CHECK(cudaEventRecord(input_ready,owner));CUDA_CHECK(cudaStreamWaitEvent(stream,input_ready,0));
        const size_t in=size_t(work.tokens)*work.columns*4,out=work.experts.size()*work.columns*4;
        if(trace.enabled){trace.queued_us=now();trace.input_bytes=in;trace.output_bytes=out;}
        trace.record(StrataCpuLaneTrace::input_begin,stream);
        CUDA_CHECK(cudaMemcpyAsync(input,branch.nodes[0]->src[1]->data,in,cudaMemcpyDeviceToHost,stream));
        trace.record(StrataCpuLaneTrace::input_end,stream);
        CUDA_CHECK(cudaEventRecord(input_copied,stream));
        CUDA_CHECK(cudaLaunchHostFunc(stream,execute,this));
        trace.record(StrataCpuLaneTrace::callback_end,stream);
        CUDA_CHECK(cudaMemcpyAsync(device_output,output,out,cudaMemcpyHostToDevice,stream));
        trace.record(StrataCpuLaneTrace::output_end,stream);
        CUDA_CHECK(cudaEventRecord(ready,stream));
        // Prevent the allocator from reusing x before its D2H completes. No
        // host synchronization, and no wait for CPU compute at this boundary.
        CUDA_CHECK(cudaStreamWaitEvent(owner,input_copied,0));
        input_bytes+=in;output_bytes+=out;++operations;experts+=work.experts.size();
    } else CUDA_CHECK(cudaEventRecord(ready,stream));
    started=true;
}
inline void StrataCpuExpertLane::merge(ggml_tensor *dst,cudaStream_t owner) {
    const bool down=dst==branch.nodes[2];
    if(down) {
        cudaStreamCaptureStatus capture;CUDA_CHECK(cudaStreamIsCapturing(owner,&capture));
        trace.record(StrataCpuLaneTrace::join_begin,owner);
        CUDA_CHECK(cudaStreamWaitEvent(owner,ready,capture==cudaStreamCaptureStatusActive?cudaEventWaitExternal:0));
        if(gpu_queue) {
            strata_ds4::cpu_queue_join_bound<<<1,256,0,owner>>>(graph_storage->binding,device_output,500000000ULL);
            CUDA_CHECK(cudaGetLastError());
        }
        trace.record(StrataCpuLaneTrace::join_end,owner);
    }
    const int cols=int(dst->ne[0]),rows=int(dst->ne[1]*dst->ne[2]);
    strata_cpu_scatter<<<dim3((cols+255)/256,rows),256,0,owner>>>((float *)dst->data,device_output,device_packet,cols,rows,down);
    CUDA_CHECK(cudaGetLastError());
}
