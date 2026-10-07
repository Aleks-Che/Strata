#pragma once
#include "ggml-backend.h"
#include <cstring>
#include <algorithm>
#include "ggml-backend-impl.h"
#include "nlohmann/json.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <mutex>

namespace mimo2 {
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
    std::atomic<int> remaining,skip;
    int graph_index=0;
    std::mutex mutex;
    struct Pool {std::vector<Span> spans;size_t used=0;};
    Pool pools[5]; // H2D, compute, cached delivery, ring delivery, cache fill
    nlohmann::json graphs=nlohmann::json::array();
    static void check(cudaError_t e) {trace_require(e==cudaSuccess,cudaGetErrorString(e));}
    using Intervals=std::vector<std::pair<double,double>>;
    Intervals intervals(Pool & pool) {
        Intervals result;
        for (size_t i=0;i<pool.used;++i) {
            const auto &s=pool.spans[i];
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
    explicit GpuTrace(int count,int skip_count=0):remaining(count),skip(skip_count) {trace_require(count>=1 && count<=128 && skip_count>=0,"invalid MiMo trace range");}
    void record(cudaStream_t stream,ggml_backend_t backend,bool begin,int kind=0) {
        if (remaining.load()==0 || skip.load()>0) return;
        std::lock_guard<std::mutex> lock(mutex);
        if (!started) {check(cudaEventRecord(origin.event,nullptr));check(cudaEventSynchronize(origin.event));started=true;}
        trace_require(kind>=0 && kind<5,"invalid CUDA trace category");
        auto & pool=pools[backend?1:kind];
        if (begin) {
            if(pool.used==pool.spans.size())pool.spans.emplace_back();
            pool.spans[pool.used++].complete=false;
        }
        trace_require(pool.used>0,"unpaired CUDA trace marker");
        auto & s=pool.spans[pool.used-1];auto event=begin?s.start.event:s.end.event;
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
        ++graph_index;
        if(skip.load()>0) {--skip;return;}
        if (!started) return;
        const auto h=intervals(pools[0]),c=intervals(pools[1]);double overlap=0;
        size_t i=0,j=0;
        while (i<h.size() && j<c.size()) {
            overlap+=std::max(0.0,std::min(h[i].second,c[j].second)-std::max(h[i].first,c[j].first));
            if (h[i].second<c[j].second) ++i;else ++j;
        }
        graphs.push_back({{"h2d_ms",duration(h)},{"compute_ms",duration(c)},{"overlap_ms",overlap},
            {"graph_index",graph_index},{"h2d_submissions",pools[0].used},{"compute_splits",pools[1].used},{"cancelled",cancelled},
            {"h2d_intervals_ms",h},{"compute_intervals_ms",c}});
        const char *names[]={"cached_d2d","ring_d2d","fill_d2d"};
        double end=0;
        for(int k=0;k<5;++k) {
            const auto v=intervals(pools[k]);if(!v.empty())end=std::max(end,v.back().second);
            if(k>=2) {
                graphs.back()[std::string(names[k-2])+"_ms"]=duration(v);
                graphs.back()[std::string(names[k-2])+"_submissions"]=pools[k].used;
                graphs.back()[std::string(names[k-2])+"_intervals_ms"]=v;
            }
            pools[k].used=0;
        }
        graphs.back()["span_ms"]=end;started=false;--remaining;
    }
    nlohmann::json snapshot() const {return {{"kind","CUDA events on actual streams; pooled events; diagnostic overhead included"},{"graphs",graphs}};}
};
}
