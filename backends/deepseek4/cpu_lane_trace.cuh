#pragma once
#include "cpu_lane.hpp"
#include "cpu_expert_fixture.hpp"
#include "cpu_route_profile.hpp"
#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

// Opt-in diagnostics. Host timestamps and CUDA spans are separate clocks.
// Events are queried only after the lane's existing retirement barrier, never
// in its CUDA host callback. CUDA timed events require a separate no-graphs
// diagnostic run; the host trace and fixture capture preserve graph replay.
struct StrataCpuLaneTrace {
    struct Sink {
        FILE *file=nullptr;
        std::mutex mutex;
        uint64_t sequence=0;
        std::filesystem::path capture_dir;
        int captures=0,capture_limit=0,capture_tokens=0;
        Sink() {
            if(const char *dir=std::getenv("STRATA_DS4_CPU_CAPTURE_DIR"))if(*dir) {
                capture_dir=dir;capture_limit=1;
                if(const char *v=std::getenv("STRATA_DS4_CPU_CAPTURE_JOBS"))capture_limit=std::max(0,std::min(8,std::atoi(v)));
                if(const char *v=std::getenv("STRATA_DS4_CPU_CAPTURE_TOKENS"))capture_tokens=std::atoi(v);
                std::filesystem::create_directories(capture_dir);
            }
            const char *path=std::getenv("STRATA_DS4_CPU_TRACE");
            if(!path || !*path)return;
            file=std::fopen(path,"w");if(!file)GGML_ABORT("cannot open CPU trace");
            std::setvbuf(file,nullptr,_IOFBF,1<<20);
            std::fputs("phase\towner\tjob\tlayer\ttokens\trows\tunique_experts\tinput_bytes\toutput_bytes\tselect_us\tlease_us\tlease_calls\tlease_failures\tqueued_us\tcallback_us\tcallback_end_us\tpack_us\tprepare_us\tgate_up_us\thidden_quant_us\tdown_us\tgu_dispatch_us\tgu_first_worker_us\tgu_last_worker_us\tgu_workers\tdown_dispatch_us\tdown_first_worker_us\tdown_last_worker_us\tdown_workers\tretire_us\tinput_ms\tcallback_stream_ms\toutput_ms\tjoin_ms\tcompleted\tgpu_events\troute_get_us\troute_wait_us\troute_reused\troute_measured\tpredicted_cpu_us\tpredicted_gpu_transfer_us\tpredicted_boundary_us\tpredicted_saved_bytes\tpredicted_gpu_compute_us\tpredicted_gpu_branch_us\tcost_generation\n",file);
        }
        ~Sink(){if(file)std::fclose(file);}
    };
    static Sink &sink(){static Sink s;return s;}
    bool enabled=sink().file!=nullptr;
    bool gpu_events=false;
    bool active=false;
    std::vector<int32_t> route;
    enum Event { input_begin,input_end,callback_end,output_end,join_begin,join_end,count };
    cudaEvent_t events[count]={};
    uint64_t job=0,select_us=0,lease_us=0,lease_calls=0,lease_failures=0;
    uint64_t queued_us=0,callback_us=0,callback_end_us=0,pack_us=0;
    size_t input_bytes=0,output_bytes=0;
    float predicted_cpu_us=0,predicted_gpu_us=0,predicted_boundary_us=0;
    float predicted_gpu_compute_us=0,predicted_gpu_branch_us=0;
    uint64_t predicted_saved_bytes=0,cost_generation=0;
    strata_ds4::CpuExpertTiming cpu;
    std::string phase,layer;
    strata_ds4::CpuRouteProfile route_timing;
    bool route_measured=false;
    void allocate() {
        const char *v=std::getenv("STRATA_DS4_CPU_TRACE_GPU");
        gpu_events=enabled && v && std::atoi(v)!=0;
        if(gpu_events) {
            const char *graphs=std::getenv("GGML_CUDA_DISABLE_GRAPHS");
            if(!graphs || std::strcmp(graphs,"1"))GGML_ABORT("CPU trace GPU events require GGML_CUDA_DISABLE_GRAPHS=1; host trace supports graphs");
            if(!events[0])for(auto &e:events)CUDA_CHECK(cudaEventCreate(&e));
        }
    }
    ~StrataCpuLaneTrace(){for(auto e:events)if(e)CUDA_CHECK(cudaEventDestroy(e));}
    void begin(const char *name,const void *route_tensor) {
        if(!enabled)return;
        active=true;layer=name;
        route_timing=strata_ds4::CpuRouteProfile::current();
        route_measured=route_timing.tensor==route_tensor;
        if(!route_measured)route_timing={};
        const char *p=std::getenv("STRATA_DS4_COST_PHASE");phase=p?p:"init";
        {std::lock_guard<std::mutex> lock(sink().mutex);job=++sink().sequence;}
        select_us=lease_us=lease_calls=lease_failures=queued_us=callback_us=callback_end_us=pack_us=0;
        input_bytes=output_bytes=0;cpu={};
        predicted_cpu_us=predicted_gpu_us=predicted_boundary_us=0;predicted_saved_bytes=0;
        predicted_gpu_compute_us=predicted_gpu_branch_us=0;
        cost_generation=0;
    }
    void record(Event e,cudaStream_t stream) {
        if(!gpu_events)return;
        CUDA_CHECK(cudaEventRecord(events[e],stream));
    }
    float ms(Event a,Event b) {
        float value=0;CUDA_CHECK(cudaEventElapsedTime(&value,events[a],events[b]));return value;
    }
    void finish(const void *owner,const strata_ds4::CpuExpertWork &work,uint64_t retire,bool completed) {
        if(!active)return;
        size_t unique=0;for(size_t i=0;i<work.experts.size();++i) {
            bool seen=false;for(size_t j=0;j<i;++j)seen|=work.experts[i].gate==work.experts[j].gate;
            unique+=!seen;
        }
        float in=0,callback=0,out=0,join=0;
        if(gpu_events && completed && !work.experts.empty()) {
            in=ms(input_begin,input_end);callback=ms(input_end,callback_end);
            out=ms(callback_end,output_end);join=ms(join_begin,join_end);
        }
        auto &s=sink();std::lock_guard<std::mutex> lock(s.mutex);
        std::fprintf(s.file,"%s\t%p\t%llu\t%s\t%d\t%zu\t%zu\t%zu\t%zu",
            phase.c_str(),owner,(unsigned long long)job,layer.c_str(),work.tokens,work.experts.size(),unique,input_bytes,output_bytes);
        const uint64_t fields[]={select_us,lease_us,lease_calls,lease_failures,queued_us,callback_us,callback_end_us,pack_us,
            cpu.prepare_us,cpu.gate_up_us,cpu.hidden_quant_us,cpu.down_us,cpu.gate_up_pool.dispatch_us,
            cpu.gate_up_pool.first_worker_us,cpu.gate_up_pool.last_worker_us,uint64_t(cpu.gate_up_pool.workers),
            cpu.down_pool.dispatch_us,cpu.down_pool.first_worker_us,cpu.down_pool.last_worker_us,uint64_t(cpu.down_pool.workers),retire};
        for(auto v:fields)std::fprintf(s.file,"\t%llu",(unsigned long long)v);
        std::fprintf(s.file,"\t%.6f\t%.6f\t%.6f\t%.6f\t%d\t%d\t%llu\t%llu\t%d\t%d\t%.3f\t%.3f\t%.3f\t%llu\t%.3f\t%.3f\t%llu\n",in,callback,out,join,int(completed),int(gpu_events),
            (unsigned long long)route_timing.get_us,(unsigned long long)route_timing.wait_us,int(route_timing.reused),int(route_measured),
            predicted_cpu_us,predicted_gpu_us,predicted_boundary_us,(unsigned long long)predicted_saved_bytes,predicted_gpu_compute_us,predicted_gpu_branch_us,(unsigned long long)cost_generation);
        if(std::ferror(s.file))GGML_ABORT("cannot write CPU trace");active=false;
        if(completed && !work.experts.empty() && s.captures<s.capture_limit &&
           (!s.capture_tokens || work.tokens==s.capture_tokens) && strata_ds4::CpuExpertFixture::bytes(work)<=strata_ds4::CpuExpertFixture::limit) {
            strata_ds4::CpuExpertFixture::write(s.capture_dir/("cpu-job-"+std::to_string(job)+".bin"),work,route.data());++s.captures;
        }
    }
};
