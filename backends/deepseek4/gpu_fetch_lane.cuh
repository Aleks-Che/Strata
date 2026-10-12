#pragma once
#include "gpu_fetch_queue.cuh"
#include "host_cache.hpp"
#include "expert_file.hpp"

// Diagnostic transport: immutable GPU/RAM snapshot for one MoE branch. Cache
// admission stays on the old path; this service only reads retained sources.
struct StrataGpuFetchLane {
    bool enabled=false,active=false;
    bool submitted[3]={};
    uint64_t epoch=0,branches=0,matrices=0,ram_bytes=0,file_bytes=0,h2d_bytes=0;
    strata_ds4::GpuFetchPlan plan[3]{};
    std::shared_ptr<strata_expert_file::Source> files[3];
    uint64_t offsets[3]={};
    std::array<std::array<strata_ds4::HostCache::Lease,256>,3> leases;
    std::unique_ptr<strata_ds4::GpuFetchQueue> queue;
    StrataGpuFetchLane() {
        const char *v=std::getenv("STRATA_DS4_GPU_FETCH_QUEUE");enabled=v && std::strcmp(v,"1")==0;
        // The shadow route recorder requires every original host route. Keep
        // its existing semantics instead of silently writing a partial trace.
        if(std::getenv("STRATA_DS4_ROUTE_TRACE"))enabled=false;
    }
    void allocate() {
        if(!queue)queue=std::make_unique<strata_ds4::GpuFetchQueue>([this](int m,auto &slot,auto &service){execute(m,slot,service);});
    }
    void execute(int m,strata_ds4::GpuFetchSlot &slot,strata_ds4::GpuFetchQueue &service) {
        const auto &job=slot.job;const auto &p=plan[m];
        if(!active || job.epoch!=epoch || job.epoch!=p.epoch || job.count<0 || job.count>64) {
            slot.error=strata_ds4::cpu_bad_request;return;
        }
        bool used[256]={};
        for(int i=0;i<job.count;++i) {
            const int id=job.ids[i];
            if(id<0 || id>=p.experts || used[id] || p.cached[id] || (p.compact && i>=p.compact)) {
                slot.error=strata_ds4::cpu_bad_request;return;
            }
            used[id]=true;
            const auto offset=p.compact?size_t(i)*(p.width+512):size_t(id)*p.width;
            if(job.destinations[i]!=(uint8_t *)p.scratch+offset) {slot.error=strata_ds4::cpu_bad_request;return;}
        }
        strata_expert_file::Request reader;
        for(int i=0;i<job.count;++i) {
            const int id=job.ids[i];const size_t n=p.width+(id<p.experts-1?std::min<size_t>(p.width,512):0);
            if(service.cancelled.load()) {slot.error=strata_ds4::cpu_cancelled;return;}
            if(leases[m][id]) {
                service.upload(job.destinations[i],leases[m][id].data(),n);ram_bytes+=n;
            } else {
                // Explicit offsets and retained file handles; no mmap faults,
                // new pinned allocations or cache mutation on the I/O service.
                const auto offset=offsets[m]+size_t(id)*p.width;
                service.read_upload(job.destinations[i],n,[&](size_t off,void *dest,size_t bytes) {
                    return reader.read_at(*files[m],offset+off,dest,bytes,service.cancelled);
                });file_bytes+=n;
            }
            h2d_bytes+=n;++matrices;
        }
    }
    void finish(cudaStream_t owner) {
        if(!active)return;
        CUDA_CHECK(cudaStreamSynchronize(owner));queue->drain();
        for(int m=0;m<3;++m)if(submitted[m]) {
            const auto &slot=queue->host[m];
            const auto failure=slot.failure.load(cuda::memory_order_acquire);
            if(failure || slot.job.epoch!=epoch ||
               slot.consumed.load(cuda::memory_order_acquire)!=slot.request.load(cuda::memory_order_acquire))
                GGML_ABORT("GPU fetch queue failed: matrix=%d epoch=%llu status=%u",m,(unsigned long long)epoch,failure);
        }
        active=false;
        for(auto &matrix:leases)for(auto &lease:matrix)lease={};
        for(auto &file:files)file.reset();
    }
    ~StrataGpuFetchLane() {
        if(enabled)GGML_LOG_INFO("STRATA_GPU_FETCH_QUEUE branches=%llu matrices=%llu ram_bytes=%llu file_bytes=%llu h2d_bytes=%llu stage_bytes=%zu copy_blocks=%d admission=0 transport=mapped\n",
            (unsigned long long)branches,(unsigned long long)matrices,(unsigned long long)ram_bytes,
            (unsigned long long)file_bytes,(unsigned long long)h2d_bytes,queue?2*strata_ds4::GpuFetchQueue::stage_bytes:0,queue?queue->copy_blocks:0);
    }
};
