#pragma once
#include "expert_cache.hpp"
#include "ggml-backend.h"
#include <cstring>
#include <algorithm>
#include "ggml-backend-impl.h"
#include "nlohmann/json.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <mutex>

namespace step35 {
// Opt-in diagnostic only. CUDA events on the actual copy and compute streams;
// collect/drain at graph boundaries. Timed runs must keep this disabled.
class GpuTrace {
    static void trace_require(bool ok,const char * message) {if(!ok)throw std::runtime_error(message);}
    struct Event {
        cudaEvent_t event=nullptr;
        Event() {check(cudaEventCreate(&event));}
        ~Event() {if(event)cudaEventDestroy(event);}
        Event(Event && e) noexcept:event(e.event) {e.event=nullptr;}
        Event(const Event &)=delete;
    };
    struct Span {Event start,end;bool complete=false;};
    Event origin;
    bool started=false;
    std::atomic<int> remaining;
    std::mutex mutex;
    std::vector<Span> copies,computes;
    nlohmann::json graphs=nlohmann::json::array();
    static void check(cudaError_t e) {trace_require(e==cudaSuccess,cudaGetErrorString(e));}
    using Intervals=std::vector<std::pair<double,double>>;
    Intervals intervals(std::vector<Span> & spans) {
        Intervals result;
        for (const auto & s:spans) {
            if (!s.complete) continue;
            check(cudaEventSynchronize(s.end.event));
            float a=0,b=0;check(cudaEventElapsedTime(&a,origin.event,s.start.event));
            check(cudaEventElapsedTime(&b,origin.event,s.end.event));result.push_back({a,b});
        }
        std::sort(result.begin(),result.end());Intervals merged;
        for (auto s:result) {
            if (!merged.empty() && s.first<=merged.back().second) merged.back().second=std::max(merged.back().second,s.second);
            else merged.push_back(s);
        }
        return merged;
    }
    static double duration(const Intervals & intervals) {double t=0;for(auto s:intervals)t+=s.second-s.first;return t;}
public:
    explicit GpuTrace(int count):remaining(count) {trace_require(count>=1 && count<=128,"Step trace graphs must be 1..128");}
    void record(cudaStream_t stream,ggml_backend_t backend,bool begin) {
        if (remaining.load()==0) return;
        std::lock_guard<std::mutex> lock(mutex);
        if (!started) {check(cudaEventRecord(origin.event,nullptr));check(cudaEventSynchronize(origin.event));started=true;}
        auto & spans=backend?computes:copies;
        if (begin) spans.emplace_back();
        trace_require(!spans.empty(),"unpaired CUDA trace marker");
        auto & s=spans.back();auto event=begin?s.start.event:s.end.event;
        if (backend) {
            // Audited CUDA backend stores cudaEvent_t in this event's context.
            // The public recorder uses the backend's real compute stream.
            trace_require(std::strstr(ggml_backend_name(backend),"CUDA")!=nullptr,"trace requires CUDA backend");
            ggml_backend_event wrapper{ggml_backend_get_device(backend),event};
            ggml_backend_event_record(&wrapper,backend);
        } else check(cudaEventRecord(event,stream));
        if (!begin) s.complete=true;
    }
    void finish(bool cancelled) {
        if (remaining.load()==0) return;
        std::lock_guard<std::mutex> lock(mutex);
        if (!started) return;
        const auto h=intervals(copies),c=intervals(computes);double overlap=0;
        size_t i=0,j=0;
        while (i<h.size() && j<c.size()) {
            overlap+=std::max(0.0,std::min(h[i].second,c[j].second)-std::max(h[i].first,c[j].first));
            if (h[i].second<c[j].second) ++i;else ++j;
        }
        graphs.push_back({{"h2d_ms",duration(h)},{"compute_ms",duration(c)},{"overlap_ms",overlap},
            {"h2d_submissions",copies.size()},{"compute_splits",computes.size()},{"cancelled",cancelled},
            {"h2d_intervals_ms",h},{"compute_intervals_ms",c}});
        copies.clear();computes.clear();started=false;--remaining;
    }
    nlohmann::json snapshot() const {return {{"kind","CUDA events on actual streams"},{"graphs",graphs}};}
};
}
