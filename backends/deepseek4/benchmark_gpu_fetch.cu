// Standalone transport diagnostic. It never loads a model or alters its caches.
// Compare the actual mapped queue with DMA and a plain mapped-memory kernel.
#include "gpu_fetch_queue.cuh"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
using namespace strata_ds4;
using Clock=std::chrono::steady_clock;

static __global__ void mapped_copy(uint8_t *dst,const uint8_t *src,size_t n) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n/16;i+=gridDim.x*blockDim.x)
        reinterpret_cast<uint4 *>(dst)[i]=reinterpret_cast<const uint4 *>(src)[i];
}
static double us(Clock::time_point start) {
    return std::chrono::duration<double,std::micro>(Clock::now()-start).count();
}
static void report(const char *mode,size_t bytes,int matrices,int blocks,const std::vector<double> &times,int depth=0,uint64_t peak=0,size_t stage=0,int slots=0) {
    auto sorted=times;std::sort(sorted.begin(),sorted.end());
    const double med=(sorted[(sorted.size()-1)/2]+sorted[sorted.size()/2])*0.5;
    std::printf("%s,%zu,%d,%d,%.3f,%.3f,%.3f,%.3f,%d,%llu,%zu,%d\n",mode,bytes,matrices,blocks,
        med,sorted.front(),sorted.back(),bytes*matrices/med/1000.0,depth,(unsigned long long)peak,stage,slots);std::fflush(stdout);
}
int main() {try {
    int device=0;cudaDeviceProp prop{};cpu_queue_check(cudaGetDevice(&device));
    cpu_queue_check(cudaGetDeviceProperties(&prop,device));
    std::fprintf(stderr,"device=%s sms=%d\n",prop.name,prop.multiProcessorCount);
    cudaStream_t stream;cpu_queue_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    constexpr size_t capacity=64ULL<<20;uint8_t *src,*dst;
    cpu_queue_check(cudaMallocHost((void **)&src,capacity));cpu_queue_check(cudaMalloc((void **)&dst,capacity));
    for(size_t i=0;i<capacity;++i)src[i]=uint8_t((i*37+i/256)%251);
    int32_t *ids;const void **table;
    cpu_queue_check(cudaMalloc((void **)&ids,64*sizeof(int32_t)));
    cpu_queue_check(cudaMalloc((void **)&table,256*sizeof(void *)));
    int32_t route[64];std::iota(route,route+64,0);
    cpu_queue_check(cudaMemcpy(ids,route,sizeof(route),cudaMemcpyHostToDevice));
    std::puts("mode,matrix_bytes,matrices,blocks,median_us,min_us,max_us,GB_per_s,depth,peak_inflight,stage_bytes,stage_slots");
    for(size_t bytes:{size_t(2ULL<<20),size_t(4ULL<<20),size_t(8ULL<<20)})for(int matrices:{1,6}) {
        const size_t total=bytes*matrices;
        for(const char *mode:{"dma","mapped_plain"})for(int blocks:{16,64,128}) {
            if(blocks>prop.multiProcessorCount || (mode[0]=='d' && blocks!=16))continue;
            std::vector<double> times;
            for(int r=0;r<15;++r) {
                cpu_queue_check(cudaMemsetAsync(dst,0xff,total,stream));
                cpu_queue_check(cudaStreamSynchronize(stream));
                const auto start=Clock::now();
                for(int m=0;m<matrices;++m) {
                    if(mode[0]=='d')cpu_queue_check(cudaMemcpyAsync(dst+m*bytes,src+m*bytes,bytes,cudaMemcpyHostToDevice,stream));
                    else mapped_copy<<<blocks,256,0,stream>>>(dst+m*bytes,src+m*bytes,bytes);
                }
                cpu_queue_check(cudaStreamSynchronize(stream));if(r>=3)times.push_back(us(start));
            }
            std::vector<uint8_t> output(total);cpu_queue_check(cudaMemcpy(output.data(),dst,total,cudaMemcpyDeviceToHost));
            if(std::memcmp(output.data(),src,total))throw std::runtime_error("plain/DMA payload mismatch");
            report(mode,bytes,matrices,blocks,times);
        }
        for(const char *mode:{"queue_direct","queue_staged","queue_hit","owner_dma_direct","owner_dma_staged","owner_dma_hit",
                             "owner_dma2_direct","owner_dma2_staged","owner_dma4_direct","owner_dma4_staged","owner_dma8_direct","owner_dma8_staged",
                             "owner_dma4s4_staged","owner_dma8s4_staged","owner_dma8s8_staged","owner_dma16s16_staged"})for(int blocks:{16}) {
            if(blocks>prop.multiProcessorCount)continue;
            const bool dma=std::strstr(mode,"owner_dma")!=nullptr;
            const int depth=std::strstr(mode,"dma16")?16:std::strstr(mode,"dma8")?8:std::strstr(mode,"dma4")?4:std::strstr(mode,"dma2")?2:1;
            const bool staged=std::strstr(mode,"staged")!=nullptr,hit=std::strstr(mode,"hit")!=nullptr;
            GpuFetchQueue q([&](int,GpuFetchSlot &slot,GpuFetchQueue &service) {
                if(slot.job.count!=(hit?0:matrices))throw std::runtime_error("unexpected queue misses");
                for(int i=0;i<slot.job.count;++i) {
                    const int id=slot.job.ids[i];
                    if(staged)service.read_upload(slot.job.destinations[i],bytes,[&](size_t offset,void *out,size_t n){
                        std::memcpy(out,src+id*bytes+offset,n);return true;});
                    else service.upload(slot.job.destinations[i],src+id*bytes,bytes);
                }
            });
            q.owner_dma=dma;q.dma_depth=depth;q.dma_stage_slots=std::strstr(mode,"s16")?16:std::strstr(mode,"s8")?8:std::strstr(mode,"s4")?4:2;
            q.ordered_sources=false;q.copy_blocks=blocks;GpuFetchPlan plan{};plan.epoch=1;plan.experts=matrices;
            plan.topk=matrices;plan.tokens=1;plan.width=bytes;plan.scratch=dst;
            if(hit)for(int i=0;i<matrices;++i)plan.cached[i]=dst+i*bytes;
            // Avoid the optional +512 padding: the callback transfers exactly bytes;
            // no tensor consumer needs padding in this isolated transport probe.
            std::vector<double> times;
            for(int r=0;r<15;++r) {
                if(!hit) {
                    cpu_queue_check(cudaMemsetAsync(dst,0xff,total,stream));
                    cpu_queue_check(cudaStreamSynchronize(stream));
                }
                const auto start=Clock::now();q.arm();
                gpu_fetch_publish<<<1,256,0,stream>>>(q.mapped,plan,ids,matrices,nullptr,table);
                q.wait(0,stream);cpu_queue_check(cudaStreamSynchronize(stream));q.drain();
                if(q.host[0].failure.load(cuda::memory_order_acquire))throw std::runtime_error("queue failure");
                if(r>=3)times.push_back(us(start));
            }
            if(!hit) {
                std::vector<uint8_t> output(total);cpu_queue_check(cudaMemcpy(output.data(),dst,total,cudaMemcpyDeviceToHost));
                if(std::memcmp(output.data(),src,total))throw std::runtime_error("queue payload mismatch");
            }
            report(mode,hit?0:bytes,matrices,blocks,times,depth,q.dma_peak_inflight,q.staging_bytes(),q.dma_stage_slots);
        }
    }
    cudaFree(table);cudaFree(ids);cudaFree(dst);cudaFreeHost(src);cudaStreamDestroy(stream);
    return 0;
}catch(const std::exception &e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}}
