#pragma once
#include <cuda_runtime.h>
#include "nlohmann/json.hpp"
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "ggml-backend-impl.h"
namespace minimax_m2 {
// Diagnostic only, disabled in timing runs. Events are recorded on the actual
// H2D, ring-to-scratch and actual CUDA compute streams.
class PipelineTrace {
    static void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
    struct Event {
        cudaEvent_t value=nullptr;
        Event() {check(cudaEventCreate(&value));}
        ~Event() {if(value)cudaEventDestroy(value);}
    };
    struct Span {Event begin,end;bool complete=false;};
    using Spans=std::vector<std::unique_ptr<Span>>;
    using Intervals=std::vector<std::pair<double,double>>;
    std::mutex mutex;Event origin;Spans uploads,deliveries,computes;int remaining=4;bool active=false;
    nlohmann::ordered_json groups=nlohmann::ordered_json::array();
    Intervals intervals(Spans &spans) {
        Intervals result;
        for(const auto &s:spans)if(s->complete) {
            check(cudaEventSynchronize(s->end.value));float a=0,b=0;
            check(cudaEventElapsedTime(&a,origin.value,s->begin.value));
            check(cudaEventElapsedTime(&b,origin.value,s->end.value));result.emplace_back(a,b);
        }
        std::sort(result.begin(),result.end());return result;
    }
public:
    void begin() {
        std::lock_guard<std::mutex> lock(mutex);if(!remaining)return;
        check(cudaEventRecord(origin.value,nullptr));check(cudaEventSynchronize(origin.value));active=true;
    }
    void record(cudaStream_t stream,bool upload,bool begin) {
        std::lock_guard<std::mutex> lock(mutex);if(!active)return;
        auto &spans=upload?uploads:deliveries;
        if(begin) {if(spans.size()>=4096)throw std::runtime_error("MiniMax trace span bound");spans.emplace_back(std::make_unique<Span>());}
        if(spans.empty())throw std::runtime_error("unpaired MiniMax pipeline trace");
        auto &s=*spans.back();check(cudaEventRecord(begin?s.begin.value:s.end.value,stream));if(!begin)s.complete=true;
    }
    void end(bool cancelled) {
        std::lock_guard<std::mutex> lock(mutex);if(!active)return;
        auto h=intervals(uploads),d=intervals(deliveries),c=intervals(computes);double overlap=0,compute_overlap=0;
        for(const auto &a:h)for(const auto &b:d)overlap+=std::max(0.,std::min(a.second,b.second)-std::max(a.first,b.first));
        for(const auto &a:h)for(const auto &b:c)compute_overlap+=std::max(0.,std::min(a.second,b.second)-std::max(a.first,b.first));
        groups.push_back({{"h2d_intervals_ms",h},{"delivery_intervals_ms",d},{"overlap_ms",overlap},
            {"compute_intervals_ms",c},{"h2d_compute_overlap_ms",compute_overlap},{"cancelled",cancelled}});
        uploads.clear();deliveries.clear();computes.clear();active=false;--remaining;
    }
    void record_compute(ggml_backend_t backend,bool begin) {
        std::lock_guard<std::mutex> lock(mutex);if(!active)return;
        if(std::string(ggml_backend_name(backend)).find("CUDA")==std::string::npos)
            throw std::runtime_error("MiniMax trace requires CUDA backend");
        if(begin) {if(computes.size()>=4096)throw std::runtime_error("MiniMax compute trace bound");computes.emplace_back(std::make_unique<Span>());}
        if(computes.empty())throw std::runtime_error("unpaired MiniMax compute trace");
        auto &s=*computes.back();
        // The pinned CUDA backend's event context is cudaEvent_t. Public event
        // recording uses its actual compute stream, including nonblocking mode.
        ggml_backend_event wrapper{ggml_backend_get_device(backend),begin?s.begin.value:s.end.value};
        ggml_backend_event_record(&wrapper,backend);if(!begin)s.complete=true;
    }
    nlohmann::ordered_json snapshot() {
        std::lock_guard<std::mutex> lock(mutex);
        return {{"kind","CUDA events on actual H2D and ring-to-scratch streams"},
            {"scope","first four matrix plans; compute intervals use the actual CUDA backend stream"},{"groups",groups}};
    }
};
}
