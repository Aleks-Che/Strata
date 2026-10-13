// Offline calibration only: real captured GGUF weights, no model inference or
// adaptation while a user's request is being decoded.
#include "cpu_expert_fixture.hpp"
#include "cpu_cost_profile.hpp"
#include "cpu_cost_device.cuh"
#include "cpu_gpu_queue.cuh"
#include "cpu_mmvq_fast.hpp"
#include "ggml-quants.h"
#include "ggml-cpp.h"
#include "ggml-cuda.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <cmath>
using namespace strata_ds4;
using json=nlohmann::ordered_json;
using Clock=std::chrono::steady_clock;
static double us(Clock::time_point start){return std::chrono::duration<double,std::micro>(Clock::now()-start).count();}
static double median(std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];}
static void env(const char *key,const char *value) {
#ifdef _WIN32
    _putenv_s(key,value);
#else
    setenv(key,value,1);
#endif
}
struct Bank {
    uint8_t *p=nullptr;size_t bytes=0;std::array<size_t,3> offsets{},sizes{};
    ~Bank(){if(p)cudaFreeHost(p);}
    void allocate(const CpuExpertWork &w) {
        for(int m=0;m<3;++m){offsets[m]=bytes;sizes[m]=CpuExpertFixture::weight_bytes(w,m);bytes+=sizes[m];}
        cpu_queue_check(cudaMallocHost((void **)&p,bytes));
    }
    void *weight(int m){return p+offsets[m];}
};
static __global__ void publish_probe(CpuQueueSlot *ring,CpuQueueDevice *state,const float *input,int columns,int tokens,int rows) {
    __shared__ unsigned seq;
    if(!threadIdx.x) {
        seq=cpu_queue_reserve(ring,state);if(seq){auto &j=ring[seq%cpu_queue_depth].job;j={};j.epoch=1;j.columns=columns;j.tokens=tokens;j.rows=rows;}
    }
    __syncthreads();if(!seq)return;auto &s=ring[seq%cpu_queue_depth];
    for(int i=threadIdx.x;i<tokens*columns;i+=blockDim.x)s.input[i]=input[i];
    __threadfence_system();__syncthreads();if(!threadIdx.x)s.request.store(seq,cuda::memory_order_release);
}
static __global__ void scatter_probe(float *dst,const float *src,int size) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=blockDim.x*gridDim.x)dst[i]=src[i];
}
static __global__ void warm_gpu(float *data,uint64_t duration=250000000) {
    uint64_t start,now;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(start));
    float x=float(threadIdx.x+1);
    do {
        for(int i=0;i<128;++i)x=__fmaf_rn(x,0.999999f,0.00001f);
        asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now));
    }while(now-start<duration);
    data[blockIdx.x*blockDim.x+threadIdx.x]=x;
}
#include "cpu_calibrate_gpu.cuh"
#include "cpu_pool_probe.cuh"
static json boundaries(int columns) {
    CpuGpuQueue queue([](CpuQueueSlot &){});cudaStream_t stream;cpu_queue_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    float *input,*out,*dst;cpu_queue_check(cudaMalloc((void **)&input,columns*4*4));cpu_queue_check(cudaMemset(input,0,columns*4*4));
    // The fixed warmup grid writes 4096 floats, even for smaller test shapes.
    cpu_queue_check(cudaMalloc((void **)&out,std::max(columns*8,4096)*sizeof(float)));cpu_queue_check(cudaMalloc((void **)&dst,columns*8*4));
    warm_gpu<<<16,256,0,stream>>>(out);cpu_queue_check(cudaStreamSynchronize(stream));
    for(int i=0;i<cpu_queue_depth;++i)std::fill(queue.host[i].output,queue.host[i].output+columns*8,0.0f);
    cudaEvent_t a,b;cpu_queue_check(cudaEventCreate(&a));cpu_queue_check(cudaEventCreate(&b));queue.arm();json result=json::array();
    for(int n=1;n<=4;++n)for(int rows=1;rows<=8;++rows) {
        std::vector<double> samples;
        for(int i=0;i<17;++i) {
            cpu_queue_check(cudaEventRecord(a,stream));publish_probe<<<1,256,0,stream>>>(queue.mapped,queue.device,input,columns,n,rows);
            cpu_queue_join<<<1,256,0,stream>>>(queue.mapped,queue.device,out,500000000);
            scatter_probe<<<16,256,0,stream>>>(dst,out,columns*rows);cpu_queue_check(cudaGetLastError());
            cpu_queue_check(cudaEventRecord(b,stream));cpu_queue_check(cudaEventSynchronize(b));float ms;cpu_queue_check(cudaEventElapsedTime(&ms,a,b));
            CpuQueueDevice state;cpu_queue_check(cudaMemcpy(&state,queue.device,sizeof(state),cudaMemcpyDeviceToHost));
            if(state.error || !state.active)throw std::runtime_error("queue boundary probe failed");
            if(i)samples.push_back(ms*1000);
        }
        result.push_back({{"tokens",n},{"rows",rows},{"us",median(samples)},{"samples_us",samples}});
    }
    queue.drain();cudaEventDestroy(a);cudaEventDestroy(b);cudaStreamDestroy(stream);cudaFree(input);cudaFree(out);cudaFree(dst);return result;
}
static void partitions(int left,int max,std::vector<int> &current,std::vector<std::vector<int>> &all) {
    if(!left){all.push_back(current);return;}
    for(int i=std::min(left,max);i>=1;--i){current.push_back(i);partitions(left-i,i,current,all);current.pop_back();}
}
static json calibrate(const std::vector<CpuExpertFixture *> &fixtures,int threads,bool q8,bool probe=false) {
    auto w=fixtures[0]->work;w.experts.clear();if(q8)w.gate_type=w.up_type=w.down_type=GGML_TYPE_Q8_0;
    if(w.columns>16384 || w.hidden>16384)throw std::runtime_error("calibration shape too large");
    std::array<Bank,8> banks;std::vector<float> decoded(std::max(w.columns,w.hidden));
    for(int b=0;b<8;++b) {
        banks[b].allocate(w);const auto &f=*fixtures[b%fixtures.size()];const auto &e=f.work.experts[0];
        const void *src[]={e.gate,e.up,e.down};const ggml_type types[]={f.work.gate_type,f.work.up_type,f.work.down_type};
        for(int m=0;m<3;++m) {
            if(!q8)std::memcpy(banks[b].weight(m),src[m],banks[b].sizes[m]);
            else {
                const int columns=m<2?w.columns:w.hidden,rows=m<2?w.hidden:w.columns;
                for(int r=0;r<rows;++r) {
                    ggml_get_type_traits(types[m])->to_float(static_cast<const uint8_t *>(src[m])+r*ggml_row_size(types[m],columns),decoded.data(),columns);
                    quantize_row_q8_0_ref(decoded.data(),static_cast<block_q8_0 *>(banks[b].weight(m))+r*(columns/32),columns);
                }
            }
        }
    }
    const size_t resident=banks[0].bytes*8;if(resident<(65ULL<<20))throw std::runtime_error("need a calibration set larger than aggregate 64 MiB L3");
    json result={{"gate",int(w.gate_type)},{"up",int(w.up_type)},{"down",int(w.down_type)},
        {"columns",w.columns},{"hidden",w.hidden},{"clamp",w.clamp},{"resident_weight_bytes",resident},
        {"source",q8?"captured weights converted to Q8_0 for timing only":"captured GGUF weights"},{"cases",json::array()}};
    std::vector<float> input(4*w.columns),output(8*w.columns);
    for(size_t i=0;i<input.size();++i)input[i]=fixtures[0]->input[i%fixtures[0]->input.size()];
    w.input=input.data();w.output=output.data();w.topk=8;
    if(probe)return pool_probe(w,banks,input,threads);
    CpuLanePool pool(threads);
    std::exception_ptr error;
    // Same calling role as the persistent service; pool creation is not timed.
    auto measure_cpu=[&](bool loaded,int gap_us=0) {
    size_t case_index=0;std::thread service([&]{try {
        for(int n=1;n<=4;++n)for(int rows=1;rows<=4;++rows) {
            std::vector<std::vector<int>> shapes;std::vector<int> scratch;partitions(rows,n,scratch,shapes);
            for(const auto &shape:shapes) {
                w.tokens=n;std::vector<double> rounds,samples,gaps;CpuExpertTiming total{};
                for(int pass=0;pass<(gap_us?2:4);++pass) {
                    double sum=0;
                    for(int b=0;b<8;++b) {
                        w.experts.clear();for(size_t g=0;g<shape.size();++g)for(int t=0;t<shape[g];++t) {
                            auto &bank=banks[(b+g)%8];w.experts.push_back({bank.weight(0),bank.weight(1),bank.weight(2),t,int(t*8+g)});
                        }
                        const auto pause=Clock::now();if(gap_us)std::this_thread::sleep_for(std::chrono::microseconds(gap_us));
                        if(pass && gap_us)gaps.push_back(us(pause));
                        CpuExpertTiming timing;auto start=Clock::now();w.compute(pool,&timing);const double duration=us(start);
                        if(pass){sum+=duration;total.prepare_us+=timing.prepare_us;total.gate_up_us+=timing.gate_up_us;
                            total.hidden_quant_us+=timing.hidden_quant_us;total.down_us+=timing.down_us;samples.push_back(duration);}
                    }
                    if(pass)rounds.push_back(sum/8);
                }
                if(!loaded && !gap_us)result["cases"].push_back({{"tokens",n},{"counts",shape},{"compute_us",median(rounds)},{"rounds_us",rounds},
                    {"prepare_us",total.prepare_us/24.},{"gate_up_us",total.gate_up_us/24.},
                    {"hidden_quant_us",total.hidden_quant_us/24.},{"down_us",total.down_us/24.}});
                else {
                    auto &row=result["cases"][case_index];const std::string prefix=gap_us?(loaded?"burst_loaded_":"burst_"):"loaded_";
                    row[prefix+"compute_us"]=gap_us?median(samples):median(rounds);row[prefix+"rounds_us"]=rounds;
                    row[prefix+"samples_us"]=samples;
                    if(gap_us)row[prefix+"gaps_us"]=gaps;
                    const double count=samples.size();row[prefix+"prepare_us"]=total.prepare_us/count;
                    row[prefix+"gate_up_us"]=total.gate_up_us/count;row[prefix+"hidden_quant_us"]=total.hidden_quant_us/count;
                    row[prefix+"down_us"]=total.down_us/count;
                }
                ++case_index;
            }
        }
    }catch(...){error=std::current_exception();}});service.join();if(error)std::rethrow_exception(error);
    };
    measure_cpu(false);
    constexpr int burst_gap_us=20000;result["burst_gap_requested_us"]=burst_gap_us;
    measure_cpu(false,burst_gap_us);
    {
        CalibrationCopyLoad load(banks);warm_gpu<<<16,256,0,load.stream>>>(static_cast<float *>(load.device));
        cpu_queue_check(cudaStreamSynchronize(load.stream));load.start();measure_cpu(true);load.finish();
        result["loaded_copy_us_per_byte"]=median(load.samples)/banks[0].bytes;
        result["loaded_copy_samples"]=load.samples.size();result["loaded_copy_min_us"]=*std::min_element(load.samples.begin(),load.samples.end());
        result["loaded_copy_max_us"]=*std::max_element(load.samples.begin(),load.samples.end());
    }
    {
        CalibrationCopyLoad load(banks);warm_gpu<<<16,256,0,load.stream>>>(static_cast<float *>(load.device));
        cpu_queue_check(cudaStreamSynchronize(load.stream));load.start();measure_cpu(true,burst_gap_us);load.finish();
        // Pauses dominate this stream's samples. Keep them separate from the
        // continuously loaded DMA price used by the placement model.
        result["burst_copy_us_per_byte"]=median(load.samples)/banks[0].bytes;
        result["burst_copy_samples"]=load.samples.size();
    }
    cudaStream_t stream;cpu_queue_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    void *device;cpu_queue_check(cudaMalloc(&device,banks[0].bytes));cudaEvent_t a,b;
    cpu_queue_check(cudaEventCreate(&a));cpu_queue_check(cudaEventCreate(&b));std::vector<double> copy;
    // PCIe measurements taken at idle P8 overprice GPU work. Warm the SMs
    // immediately before the copy probe without changing system power policy.
    warm_gpu<<<16,256,0,stream>>>(static_cast<float *>(device));cpu_queue_check(cudaStreamSynchronize(stream));
    for(int pass=0;pass<4;++pass)for(auto &bank:banks) {
        cpu_queue_check(cudaEventRecord(a,stream));for(int m=0;m<3;++m)
            cpu_queue_check(cudaMemcpyAsync(static_cast<uint8_t *>(device)+bank.offsets[m],bank.weight(m),bank.sizes[m],cudaMemcpyHostToDevice,stream));
        cpu_queue_check(cudaEventRecord(b,stream));cpu_queue_check(cudaEventSynchronize(b));float ms;cpu_queue_check(cudaEventElapsedTime(&ms,a,b));if(pass)copy.push_back(ms*1000);
    }
    result["gpu_warmup_ms"]=250;result["copy_samples_us"]=copy;result["copy_us_per_byte"]=median(copy)/banks[0].bytes;
    cudaEventDestroy(a);cudaEventDestroy(b);cudaFree(device);cudaStreamDestroy(stream);
    result["boundary"]=boundaries(w.columns);
    {CalibrationGpu gpu(w,banks,input,fixtures[0]->work.topk);result["gpu"]=gpu.measure(input,pool);}
    std::cerr<<"calibrated "<<ggml_type_name(w.gate_type)<<"/"<<ggml_type_name(w.down_type)<<" cases="<<result["cases"].size()<<"\n";
    return result;
}
int main(int argc,char **argv)try {
    if(argc<4)throw std::runtime_error("usage: deepseek4_cpu_calibrate output.json threads fixture.bin ...");
    const std::filesystem::path path=argv[1];if(std::filesystem::exists(path))throw std::runtime_error("output already exists");
    char *end=nullptr;const long parsed=std::strtol(argv[2],&end,10);
    if(!end || *end || parsed<1 || parsed>16)throw std::runtime_error("threads must be 1..16");
    const int threads=int(parsed);CpuLanePool initialize(1);env("STRATA_DS4_CPU_MMVQ_FAST","1");env("STRATA_DS4_CPU_MMVQ_SINGLE","0");
    if(cpu_cost_mode()=="unsupported")throw std::runtime_error("calibration needs the fast AVX2 kernels");
    if(cpu_cost_machine().rfind("/threads=",0)==0)throw std::runtime_error("set STRATA_DS4_CPU_MACHINE to a stable CPU identity on this platform");
    cudaDeviceProp prop{};cpu_queue_check(cudaGetDeviceProperties(&prop,0));
    if(prop.major<8 || (prop.major==12 && prop.minor==1))throw std::runtime_error("calibration needs the generic NVIDIA MMVQ dispatch");
    ggml_backend_ptr backend(ggml_backend_cuda_init(0));if(!backend)throw std::runtime_error("cannot initialize CUDA backend");
    std::vector<std::unique_ptr<CpuExpertFixture>> owned;
    std::map<std::tuple<int,int,int,int,int,float>,std::vector<CpuExpertFixture *>> groups;
    for(int i=3;i<argc;++i){owned.push_back(std::make_unique<CpuExpertFixture>(argv[i]));auto &f=*owned.back();auto &w=f.work;
        groups[{int(w.gate_type),int(w.up_type),int(w.down_type),w.columns,w.hidden,w.clamp}].push_back(&f);}
    json result={{"version",3},{"build",cpu_cost_build_id()},{"machine",cpu_cost_machine()},
        {"gpu",strata_cpu_cost_gpu(0)},{"mode",cpu_cost_mode()},{"threads",threads},{"profiles",json::array()}};
    const char *probe_env=std::getenv("STRATA_DS4_CPU_POOL_PROBE");const bool probe=probe_env && std::string(probe_env)=="1";
    if(probe){result["version"]=0;result["purpose"]="offline pool sweep, not a placement profile";}
    for(auto &g:groups)result["profiles"].push_back(calibrate(g.second,threads,false,probe));
    if(!probe)result["profiles"].push_back(calibrate(groups.begin()->second,threads,true));
    std::ofstream file(path);file<<result.dump(2)<<'\n';file.close();if(!file)throw std::runtime_error("cannot save calibration");
    std::cout<<path.string()<<"\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}
