// MiniMax owns plan boundaries. The shared four-slot transport is unchanged.
#include "../common/expert_pipeline.hpp"
#include "pipeline_trace.hpp"
#include <fstream>
static thread_local std::shared_ptr<minimax_m2::PipelineTrace> mm27_pipeline_trace;
static thread_local int mm27_pipeline_readers=0,mm27_pipeline_chunk_mib=8;
static thread_local bool mm27_pipeline_active=false;
static thread_local bool mm27_pipeline_lookahead=false;
static thread_local bool mm27_pipeline_d2d_batch=false;
static thread_local int mm27_pipeline_events=0;
using mm27_ranges=std::vector<std::pair<size_t,size_t>>;
struct mm27_plan_tensor {const ggml_tensor *source;mm27_ranges ranges;};
static thread_local std::vector<mm27_plan_tensor> mm27_plan;
static thread_local size_t mm27_plan_next=0;
static thread_local std::unique_ptr<step35::ExpertCache::PlanPins> mm27_plan_pins;
// New expert groups also survive until all three planned matrices complete.
static thread_local std::vector<std::unique_ptr<step35::ExpertCache::PlanPins>> mm27_plan_fill_pins;
static thread_local std::unique_ptr<StrataExpertPipeline> mm27_pipeline_queue;
static thread_local cudaStream_t mm27_pipeline_stream=nullptr;
static thread_local cudaEvent_t mm27_scratch_released=nullptr,mm27_copy_ready=nullptr;
// At most one matrix is awaiting retirement. Its slots remain pinned until
// the copy event completes, before the next matrix's cache decisions.
static thread_local bool mm27_pending_copy=false;
static thread_local std::vector<std::unique_ptr<step35::ExpertCache::PlanPins>> mm27_pending_fills;
struct mm27_pending_commit {
    step35::MatrixKey key;size_t bytes;std::shared_ptr<strata_expert_file::Source> file;uint64_t offset;
};
static thread_local std::vector<mm27_pending_commit> mm27_pending_commits;
static thread_local std::vector<ggml_backend_t> mm27_async_backends;
static void mm27_pipeline_retire_copy() {
    if(!mm27_pending_copy)return;
    ++mm27_stats.pipeline_retire_checks;
    const auto status=cudaEventQuery(mm27_copy_ready);
    if(status==cudaErrorNotReady) {
        ++mm27_stats.pipeline_retire_waits;step35::cuda_check(cudaEventSynchronize(mm27_copy_ready));
    } else step35::cuda_check(status);
    for(const auto &c:mm27_pending_commits) {
        mm27_cache->commit(c.key,c.bytes);
        if(mm27_ram_cache)mm27_ram_cache->drop_gpu(c.file,c.offset,c.bytes);
    }
    mm27_pending_commits.clear();mm27_pending_fills.clear();mm27_pending_copy=false;
}
static thread_local StrataExpertPipeline::Counters mm27_pipeline_accounted;
struct mm27_pipeline_job_state {
    const std::atomic<bool> *cancel=nullptr;
    bool read_failure=false,copy_failure=false;
};
// Written only between drained plans, before start's publication mutex.
static thread_local std::shared_ptr<mm27_pipeline_job_state> mm27_pipeline_job;
static void mm27_pipeline_account() {
    if(!mm27_pipeline_queue)return;
    const auto c=mm27_pipeline_queue->counters();auto &a=mm27_pipeline_accounted;auto &s=mm27_stats;
    s.pipeline_groups+=c.groups-a.groups;s.pipeline_chunks+=c.chunks-a.chunks;
    s.chunks+=c.chunks-a.chunks;s.pipeline_h2d_bytes+=c.h2d_bytes-a.h2d_bytes;
    s.pipeline_d2d_bytes+=c.d2d_bytes-a.d2d_bytes;s.pipeline_unused_bytes+=c.unused_bytes-a.unused_bytes;
    s.pipeline_wait_us+=c.consumer_wait_us-a.consumer_wait_us;s.pipeline_slot_wait_us+=c.slot_wait_us-a.slot_wait_us;
    s.pipeline_read_us+=c.read_us-a.read_us;s.pipeline_submit_us+=c.submit_us-a.submit_us;
    s.staging_bytes=c.pinned_bytes;s.pipeline_device_bytes=c.device_ring_bytes;
    s.pipeline_queued_bytes=c.queued_bytes;s.pipeline_reader_owned_bytes=c.reader_owned_bytes;
    s.pipeline_read_peak=std::max(s.pipeline_read_peak,c.read_peak);a=c;
}
static void mm27_pipeline_dispose() {
    if(mm27_pipeline_queue) {
        mm27_pipeline_queue->finish(false);
        if(mm27_pipeline_stream)cudaStreamSynchronize(mm27_pipeline_stream);
        mm27_pipeline_account();
        // Destructor joins every reader, drains H2D and all consumer events.
        mm27_pipeline_queue.reset();
    }
    if(mm27_pipeline_stream)cudaStreamDestroy(mm27_pipeline_stream);
    if(mm27_scratch_released)cudaEventDestroy(mm27_scratch_released);
    if(mm27_copy_ready)cudaEventDestroy(mm27_copy_ready);
    mm27_scratch_released=mm27_copy_ready=nullptr;
    // Stream was drained above, including copies without a published event.
    mm27_pending_commits.clear();mm27_pending_fills.clear();mm27_pending_copy=false;
    mm27_pipeline_stream=nullptr;mm27_pipeline_job.reset();mm27_pipeline_accounted={};mm27_pipeline_active=false;
    mm27_plan_fill_pins.clear();mm27_plan_pins.reset();mm27_plan.clear();mm27_plan_next=0;
    if(mm27_cache)mm27_cache->end_plan();
    mm27_stats.staging_bytes=mm27_staging?mm27_staging_limit:0;
    mm27_stats.pipeline_device_bytes=mm27_stats.pipeline_queued_bytes=mm27_stats.pipeline_reader_owned_bytes=0;
}
void strata_mm27_pipeline(int readers,int chunk_mib,bool lookahead,bool d2d_batch,int events) {
    if(!readers && mm27_ram_cache)throw std::invalid_argument("disable MiniMax RAM cache before pipeline");
    if(readers<0 || readers>2 || (chunk_mib!=4 && chunk_mib!=8 && chunk_mib!=16) ||
        (readers && (mm27_mode!=2 || mm27_reader)) || ((lookahead || d2d_batch) && !readers) ||
        events<0 || events>2 || (events && (!readers || !d2d_batch)))
        throw std::invalid_argument("MiniMax pipeline requires mode 2/file, 0..2 readers and 4/8/16 MiB chunks");
    if(mm27_pipeline_active)throw std::logic_error("cannot configure an active MiniMax pipeline");
    mm27_pipeline_dispose();mm27_pipeline_trace.reset();mm27_pipeline_readers=readers;mm27_pipeline_chunk_mib=chunk_mib;
    mm27_pipeline_lookahead=lookahead;
    mm27_pipeline_d2d_batch=d2d_batch;
    mm27_pipeline_events=events;
    if(readers && mm27_staging) {cudaFreeHost(mm27_staging);mm27_staging=nullptr;mm27_stats.staging_bytes=0;}
}
void strata_mm27_ram_cache_configure(uint64_t bytes) {
    if(mm27_pipeline_active)throw std::logic_error("cannot configure active MiniMax RAM cache");
    if(bytes && (mm27_mode!=2 || mm27_reader || !mm27_pipeline_readers || bytes>(131072ull<<20)))
        throw std::invalid_argument("MiniMax RAM cache requires file pipeline and cap <=131072 MiB");
    mm27_pipeline_dispose();mm27_ram_cache.reset();
    if(bytes) {
        auto cache=std::make_shared<minimax_m2::HostCache>(bytes);
        cache->set_gpu_partition(bool(mm27_cache));
        if(!mm27_host_budget)mm27_host_budget=std::make_unique<strata_glm::HostWorkingSetBudget>();
        mm27_stats.host_working_set_limit=mm27_host_budget->apply(94);
        ++mm27_stats.host_budget_updates;mm27_host_refresh=mm27_clock::now();
        mm27_ram_cache=std::move(cache);mm27_ram_cache->set_prefill(!mm27_cache_admit);
    } else if(!mm27_reader) {mm27_host_budget.reset();mm27_stats.host_working_set_limit=0;}
}
static void mm27_pipeline_create() {
    if(mm27_pipeline_queue)return;
    const size_t chunk=size_t(mm27_pipeline_chunk_mib)<<20,capacity=chunk*StrataExpertPipeline::slots;
    strata_mm27_memory(capacity+(256u<<20),capacity+(64u<<20));
    MEMORYSTATUSEX ram{};ram.dwLength=sizeof(ram);
    if(!GlobalMemoryStatusEx(&ram) || ram.ullAvailPageFile<(1ull<<30)+2*capacity)
        throw std::runtime_error("MiniMax pipeline host commit reserve unavailable");
    int device=0;step35::cuda_check(cudaGetDevice(&device));
    try {
        step35::cuda_check(cudaStreamCreateWithFlags(&mm27_pipeline_stream,cudaStreamNonBlocking));
        if(mm27_pipeline_events) {
            step35::cuda_check(cudaEventCreateWithFlags(&mm27_scratch_released,cudaEventDisableTiming));
            step35::cuda_check(cudaEventCreateWithFlags(&mm27_copy_ready,cudaEventDisableTiming));
        }
        mm27_pipeline_job=std::make_shared<mm27_pipeline_job_state>();const auto job=mm27_pipeline_job;
        const auto trace=mm27_pipeline_trace;const auto ram_cache=mm27_ram_cache;
        mm27_pipeline_queue=std::make_unique<StrataExpertPipeline>(device,chunk,false,mm27_pipeline_readers,1,
            [job,trace](cudaStream_t stream,bool begin,size_t) {
                if(begin && job->cancel && job->cancel->load())throw std::runtime_error("MiniMax request cancelled");
                if(begin && job->copy_failure)throw std::runtime_error("injected MiniMax pipeline copy failure");
                if(trace)trace->record(stream,true,begin);
            },mm27_pipeline_readers,true,StrataExpertPipeline::HostCopy{},
            [job,ram_cache](const std::shared_ptr<strata_expert_file::Source> &file,const void *src,void *dst,size_t bytes,
                const std::atomic<bool> &cancel,strata_expert_file::Request &request) {
                if(job->read_failure)throw std::runtime_error("injected MiniMax pipeline read failure");
                if(ram_cache)return ram_cache->read(file,file->offset_of(src,bytes),dst,bytes,cancel,request);
                return request.read(*file,src,dst,bytes,cancel);
            });
        mm27_pipeline_account();
    }catch(...) {mm27_pipeline_dispose();throw;}
}
void strata_mm27_pipeline_trace(bool enabled) {
    if(mm27_pipeline_active || (enabled && !mm27_pipeline_readers))throw std::invalid_argument("MiniMax trace needs an idle configured pipeline");
    mm27_pipeline_dispose();mm27_pipeline_trace=enabled?std::make_shared<minimax_m2::PipelineTrace>():nullptr;
}
void strata_mm27_pipeline_trace_write(const char *path) {
    if(!mm27_pipeline_trace || !path || !*path)throw std::invalid_argument("MiniMax trace not configured");
    std::ofstream out(path);if(!out)throw std::runtime_error("cannot open MiniMax pipeline trace");
    out<<mm27_pipeline_trace->snapshot().dump(2)<<'\n';if(!out)throw std::runtime_error("cannot write MiniMax pipeline trace");
}
