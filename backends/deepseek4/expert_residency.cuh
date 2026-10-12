#pragma once
#include "host_cache.hpp"
#include <cuda_runtime.h>
#include <list>

// Host-side lifetime protocol for the optional RAM tier. This is not yet a
// GPU router table. Entries own publication tokens, while the transfer cache's
// existing direct-reader leases and stream fences own the actual GPU address.
class StrataExpertResidency {
    static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
public:
    struct Upload {
        std::shared_ptr<strata_ds4::HostCache> cache;
        strata_ds4::HostCache::GpuToken token;
        cudaEvent_t event=nullptr;
        ~Upload(){cache->end_gpu(token);if(event)cudaEventDestroy(event);}
        void poll() {
            if(!event)return;
            auto status=cudaEventQuery(event);
            if(status==cudaErrorNotReady)return;
            check(status);cache->ready_gpu(token);check(cudaEventDestroy(event));event=nullptr;
        }
    };
private:
    struct Download {
        std::unique_ptr<strata_ds4::HostCache::Demotion> fill;
        cudaEvent_t event=nullptr;
        // Event synchronization protects the pinned destination even during
        // exception unwinding. Successful publication happens in poll only.
        ~Download(){if(event){cudaEventSynchronize(event);cudaEventDestroy(event);}}
    };
    std::list<std::weak_ptr<Upload>> uploads;
    std::list<Download> downloads;
public:
    std::shared_ptr<strata_ds4::HostCache> cache;
    uint64_t d2h_bytes=0,d2h_jobs=0;
    std::shared_ptr<Upload> uploaded(const std::shared_ptr<strata_expert_file::Source> &source,
            uint64_t offset,size_t bytes,bool drop,cudaStream_t stream) {
        auto item=std::make_shared<Upload>();item->cache=cache;
        item->token=cache->begin_gpu(source,offset,bytes,drop);
        check(cudaEventCreateWithFlags(&item->event,cudaEventDisableTiming));
        check(cudaEventRecord(item->event,stream));uploads.push_back(item);
        return item;
    }
    void poll(bool wait=false) {
        for(auto it=uploads.begin();it!=uploads.end();) {
            auto item=it->lock();
            if(item){if(wait && item->event)check(cudaEventSynchronize(item->event));item->poll();}
            if(!item || !item->event)it=uploads.erase(it);else ++it;
        }
        for(auto it=downloads.begin();it!=downloads.end();) {
            auto status=wait?cudaEventSynchronize(it->event):cudaEventQuery(it->event);
            if(status==cudaErrorNotReady){++it;continue;}
            check(status);it->fill->complete();it=downloads.erase(it);
        }
    }
    // Must be called after enqueuing old readers and BEFORE the reuse fence
    // or free. D2H is on the owning stream, so a new owner cannot overwrite the
    // source until DMA completes. No extra GPU allocation is retained.
    void evict(const std::shared_ptr<Upload> &item,const void *address,cudaStream_t stream,bool prefill) {
        if(!item)return;
        poll();
        auto fill=cache->begin_demotion(item->token,prefill);
        if(!fill)return;
        downloads.emplace_back();auto &job=downloads.back();job.fill=std::move(fill);
        try {
            check(cudaEventCreateWithFlags(&job.event,cudaEventDisableTiming));
            check(cudaMemcpyAsync(job.fill->data(),address,item->token.bytes,cudaMemcpyDeviceToHost,stream));
            check(cudaEventRecord(job.event,stream));
            d2h_bytes+=item->token.bytes;++d2h_jobs;
        } catch(...) {
            // A failed event record must not release an in-flight destination.
            cudaStreamSynchronize(stream);downloads.pop_back();throw;
        }
    }
};
