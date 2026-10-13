#pragma once
#include "cpu_cost_gpu_probe.hpp"
#include <future>

// Independent pinned H2D stream competes for DRAM with the CPU pool. It never
// writes host weights or inference data. The same eight resident banks rotate.
struct CalibrationCopyLoad {
    std::array<Bank,8> &banks;cudaStream_t stream=nullptr;void *device=nullptr;
    cudaEvent_t a=nullptr,b=nullptr;std::thread worker;std::atomic<bool> stop{false};
    std::exception_ptr error;std::vector<double> samples;
    explicit CalibrationCopyLoad(std::array<Bank,8> &data):banks(data) {
        cpu_queue_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        cpu_queue_check(cudaMalloc(&device,banks[0].bytes));
        cpu_queue_check(cudaEventCreate(&a));cpu_queue_check(cudaEventCreate(&b));
    }
    void start() {
        std::promise<void> ready;auto future=ready.get_future();int dev;cpu_queue_check(cudaGetDevice(&dev));
        worker=std::thread([this,dev,&ready]{bool announced=false;try {
            cpu_queue_check(cudaSetDevice(dev));size_t index=0;
            do {
                auto &bank=banks[index++%8];
                // Keep the graphics clock active as in inference; outside the copy interval.
                warm_gpu<<<16,256,0,stream>>>(static_cast<float *>(device),1000);
                cpu_queue_check(cudaEventRecord(a,stream));
                for(int m=0;m<3;++m)cpu_queue_check(cudaMemcpyAsync(static_cast<uint8_t *>(device)+bank.offsets[m],
                    bank.weight(m),bank.sizes[m],cudaMemcpyHostToDevice,stream));
                cpu_queue_check(cudaEventRecord(b,stream));cpu_queue_check(cudaEventSynchronize(b));float ms;
                cpu_queue_check(cudaEventElapsedTime(&ms,a,b));samples.push_back(ms*1000);
                if(!announced){announced=true;ready.set_value();}
            }while(!stop.load());
        }catch(...){error=std::current_exception();if(!announced)ready.set_exception(error);}});
        future.get();
    }
    void finish(){stop=true;if(worker.joinable())worker.join();if(error)std::rethrow_exception(error);}
    ~CalibrationCopyLoad(){stop=true;if(worker.joinable())worker.join();cudaEventDestroy(a);cudaEventDestroy(b);cudaFree(device);cudaStreamDestroy(stream);}
};

struct CalibrationGpu {
    const CpuExpertWork &shape;std::array<Bank,8> &banks;
    std::vector<void *> allocations;cudaStream_t stream=nullptr;cudaEvent_t a=nullptr,b=nullptr;
    void *weights[3],*q;float *x,*g,*u,*h,*out;int32_t *ids;int *slots;uint8_t *mask,*flush;
    static constexpr int experts=32;int topk;
    template<class T>T *allocate(size_t bytes){void *p;cpu_queue_check(cudaMalloc(&p,bytes));allocations.push_back(p);cpu_queue_check(cudaMemset(p,0,bytes));return static_cast<T *>(p);}
    CalibrationGpu(const CpuExpertWork &w,std::array<Bank,8> &data,const std::vector<float> &input,int selected):shape(w),banks(data),topk(selected) {
        if(topk<1 || topk>8)throw std::runtime_error("invalid GPU calibration top-k");
        cpu_queue_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        cpu_queue_check(cudaEventCreate(&a));cpu_queue_check(cudaEventCreate(&b));
        for(int m=0;m<3;++m){weights[m]=allocate<void>(banks[0].sizes[m]*experts);
            for(int e=0;e<experts;++e)cpu_queue_check(cudaMemcpy(static_cast<uint8_t *>(weights[m])+e*banks[0].sizes[m],
                banks[e%8].weight(m),banks[0].sizes[m],cudaMemcpyHostToDevice));}
        x=allocate<float>(input.size()*4);cpu_queue_check(cudaMemcpy(x,input.data(),input.size()*4,cudaMemcpyHostToDevice));
        g=allocate<float>(32*w.hidden*4);u=allocate<float>(32*w.hidden*4);h=allocate<float>(32*w.hidden*4);
        out=allocate<float>(32*w.columns*4);
        // Padded Q8_1 is smaller than twice the F32 input even at 256 columns.
        q=allocate<void>(32*size_t(std::max(w.columns,w.hidden)+512)*4);
        ids=allocate<int32_t>(32*4);slots=allocate<int>(32*4);mask=allocate<uint8_t>(experts);
        flush=allocate<uint8_t>(256ULL<<20);
    }
    ~CalibrationGpu(){cudaStreamSynchronize(stream);cudaEventDestroy(a);cudaEventDestroy(b);for(auto p:allocations)cudaFree(p);cudaStreamDestroy(stream);}
    void launch(int n) {
        const auto &w=shape;
        strata_cpu_cost_matvec(w.gate_type,weights[0],x,q,g,ids,slots,mask,w.columns,w.hidden,experts,topk,n,1,stream);
        strata_cpu_cost_matvec(w.up_type,weights[1],x,q,u,ids,slots,mask,w.columns,w.hidden,experts,topk,n,1,stream);
        strata_cpu_cost_swiglu(g,u,h,w.hidden,n*topk*w.hidden,w.clamp,stream);
        strata_cpu_cost_matvec(w.down_type,weights[2],h,q,out,ids,slots,mask,w.hidden,w.columns,experts,topk,n,topk,stream);
        cpu_queue_check(cudaGetLastError());
    }
    json overlap() {
        // Same origin event for both streams. Copy into independent staging:
        // this measures possible overlap, not a claim that dependent weights
        // in a real MoE layer are available before their transfer completes.
        std::vector<int> skipped(3*topk,-1);for(int t=0;t<3;++t)skipped[t*topk+topk-1]=t;
        cpu_queue_check(cudaMemcpy(slots,skipped.data(),skipped.size()*4,cudaMemcpyHostToDevice));
        void *destination=allocate<void>(banks[0].bytes);cudaStream_t copy;cudaEvent_t done;
        cpu_queue_check(cudaStreamCreateWithFlags(&copy,cudaStreamNonBlocking));cpu_queue_check(cudaEventCreate(&done));
        cudaGraph_t graph;cudaGraphExec_t executable;
        cpu_queue_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch(3);
        cpu_queue_check(cudaStreamEndCapture(stream,&graph));cpu_queue_check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        json result={{"tokens",3},{"cpu_rows",3},{"copied_bytes",banks[0].bytes*8}};
        std::vector<double> samples[2];
        for(int round=0;round<5;++round)for(bool parallel:{false,true,true,false}) {
                cpu_queue_check(cudaMemsetAsync(flush,0,256ULL<<20,stream));cpu_queue_check(cudaEventRecord(a,stream));
                cpu_queue_check(cudaGraphLaunch(executable,stream));const auto target=parallel?copy:stream;
                if(parallel)cpu_queue_check(cudaStreamWaitEvent(copy,a));
                for(auto &bank:banks)for(int m=0;m<3;++m)cpu_queue_check(cudaMemcpyAsync(static_cast<uint8_t *>(destination)+bank.offsets[m],
                    bank.weight(m),bank.sizes[m],cudaMemcpyHostToDevice,target));
                if(parallel){cpu_queue_check(cudaEventRecord(done,copy));cpu_queue_check(cudaStreamWaitEvent(stream,done));}
                cpu_queue_check(cudaEventRecord(b,stream));cpu_queue_check(cudaEventSynchronize(b));float ms;
                cpu_queue_check(cudaEventElapsedTime(&ms,a,b));if(round)samples[int(parallel)].push_back(ms*1000);
        }
        for(bool parallel:{false,true}) {
            result[parallel?"parallel_us":"serial_us"]=median(samples[int(parallel)]);
            result[parallel?"parallel_samples_us":"serial_samples_us"]=samples[int(parallel)];
        }
        cudaGraphExecDestroy(executable);cudaGraphDestroy(graph);cudaEventDestroy(done);cudaStreamDestroy(copy);return result;
    }
    json measure(const std::vector<float> &input,CpuLanePool &pool) {
        json cases=json::array();std::vector<float> actual(32*shape.columns),reference(actual.size());
        int verified=0;
        for(int n=1;n<=4;++n)for(int layout=0;layout<2;++layout) {
            std::vector<int32_t> route(n*topk);auto w=shape;w.tokens=n;w.topk=topk;w.input=input.data();w.output=reference.data();w.experts.clear();
            for(int t=0;t<n;++t)for(int j=0;j<topk;++j) {
                const int e=layout?t*topk+j:j;route[t*topk+j]=e;auto &bank=banks[e%8];
                w.experts.push_back({bank.weight(0),bank.weight(1),bank.weight(2),t,t*topk+j});
            }
            w.compute(pool);cpu_queue_check(cudaMemcpy(ids,route.data(),route.size()*4,cudaMemcpyHostToDevice));
            for(int rows=0;rows<=std::min(4,n*topk);++rows)for(int spread=0;spread<2;++spread) {
                std::vector<int> skipped(n*topk,-1);
                for(int r=0;r<rows;++r)skipped[spread?(r%n)*topk+(topk-1-r/n):n*topk-1-r]=r;
                cpu_queue_check(cudaMemcpy(slots,skipped.data(),skipped.size()*4,cudaMemcpyHostToDevice));
                cpu_queue_check(cudaMemset(out,0xff,actual.size()*4));launch(n);cpu_queue_check(cudaStreamSynchronize(stream));
                cpu_queue_check(cudaMemcpy(actual.data(),out,actual.size()*4,cudaMemcpyDeviceToHost));
                for(int r=0;r<n*topk;++r) {
                    if(skipped[r]<0){if(std::memcmp(actual.data()+r*w.columns,reference.data()+r*w.columns,w.columns*4))
                        throw std::runtime_error("GPU calibration differs from compatible CPU output");}
                    else for(int j=0;j<w.columns;++j)if(!std::isnan(actual[r*w.columns+j]))throw std::runtime_error("GPU calibration skipped row was overwritten");
                }
                ++verified;
                cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
                cpu_queue_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch(n);
                cpu_queue_check(cudaStreamEndCapture(stream,&graph));cpu_queue_check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
                std::vector<double> samples;
                for(int i=0;i<10;++i) {
                    // Match revisiting an expert after other layers, not an L2-only loop.
                    cpu_queue_check(cudaMemsetAsync(flush,0,256ULL<<20,stream));
                    cpu_queue_check(cudaEventRecord(a,stream));cpu_queue_check(cudaGraphLaunch(executable,stream));
                    cpu_queue_check(cudaEventRecord(b,stream));cpu_queue_check(cudaEventSynchronize(b));float ms;
                    cpu_queue_check(cudaEventElapsedTime(&ms,a,b));if(i)samples.push_back(ms*1000);
                }
                cudaGraphExecDestroy(executable);cudaGraphDestroy(graph);
                cases.push_back({{"tokens",n},{"cpu_rows",rows},{"layout",layout?"distinct":"repeated"},
                    {"mask",spread?"spread":"packed"},{"us",median(samples)},{"samples_us",samples}});
            }
        }
        return {{"topk",topk},{"cases",cases},{"exact_cases",verified},{"l2_evict_bytes",256ULL<<20},{"overlap",overlap()}};
    }
};
