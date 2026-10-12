#pragma once
#include "cpu_lane.hpp"
#include "host_cache.hpp"
#include <array>
#include <chrono>
#include <exception>
#include <tuple>
#include <cmath>

// The CPU callback only touches CPU memory and the CPU pool. All CUDA calls
// are enqueued by the engine before it starts. This permits an external event
// wait inside a captured down graph without a host join or device spin loop.
struct StrataCpuExpertLane {
    static constexpr size_t scratch_bytes=2ULL<<20;
    // mask skips transfers only when every occurrence is on CPU. slots skips
    // compute per token, including experts shared with GPU rows in this window.
    struct Packet { uint8_t mask[256]={}; int slots[64]; } packet{};
    struct Blob {
        const void *key[3]={}; size_t widths[3]={};
        strata_ds4::HostCache::Lease shared[3];
        const void *weight(int i)const {
            if(shared[i])return shared[i].data();
            size_t off=0;for(int j=0;j<i;++j)off+=widths[j];return data+off;
        }
        uint8_t *data=nullptr;size_t capacity=0;uint64_t stamp=0;
        bool filled=false,held=false;
        ~Blob(){if(data)CUDA_CHECK(cudaFreeHost(data));}
        size_t bytes()const{return widths[0]+widths[1]+widths[2];}
        void fill(){if(!filled){size_t off=0;for(int i=0;i<3;++i){std::memcpy(data+off,key[i],widths[i]);off+=widths[i];}filled=true;}}
    };
    struct Cost {double compute_us=0,copy_us=0,overhead_us=0,pack_us=0;size_t bytes=0;};
    std::map<std::tuple<int,int,int,int,int,int>,Cost> costs;
    std::vector<std::unique_ptr<Blob>> cache;
    std::vector<Blob *> leases;
    std::shared_ptr<strata_ds4::HostCache> ram;
    bool shared_ram=false;
    size_t cache_bytes=0,cache_limit=256ULL<<20;
    uint64_t clock=0;
    int threads=0,cap=1;
    bool force=false,fixed=false,legacy=false,pending=false,configured=false,started=false,completion=false;
    int device=0;
    StrataCpuExpertBranch branch{};
    strata_ds4::CpuExpertWork work;
    std::unique_ptr<strata_ds4::CpuLanePool> pool;
    float *input=nullptr,*output=nullptr,*device_output=nullptr;
    Packet *device_packet=nullptr;
    cudaStream_t stream=nullptr;
    cudaEvent_t input_ready=nullptr,input_copied=nullptr,ready=nullptr,consumed=nullptr;
    std::exception_ptr error;
    uint64_t operations=0,experts=0,saved_bytes=0,compute_us=0,pack_us=0,retire_us=0,select_us=0;
    uint64_t input_bytes=0,output_bytes=0,ram_hits=0,ram_misses=0,declined=0,branches=0;
    static int64_t now(){return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
    StrataCpuExpertLane() {
        if(const char *v=std::getenv("STRATA_DS4_CPU_THREADS"))threads=std::atoi(v);
        if(const char *v=std::getenv("STRATA_DS4_CPU_SHARED_RAM"))shared_ram=std::atoi(v)!=0;
        if(const char *v=std::getenv("STRATA_DS4_CPU_MAX_EXPERTS"))cap=std::atoi(v);
        if(const char *v=std::getenv("STRATA_DS4_CPU_POLICY"))force=std::strcmp(v,"force")==0;
        // Diagnostic only: stable device placement separates numerical changes
        // caused by adaptive routing from KV/session lifecycle defects.
        if(const char *v=std::getenv("STRATA_DS4_CPU_TEST_FIXED"))fixed=std::atoi(v)!=0;
        force|=fixed;
        if(const char *v=std::getenv("STRATA_DS4_CPU_MATRIX_LEGACY"))legacy=std::atoi(v)!=0;
        if(const char *v=std::getenv("STRATA_DS4_CPU_RAM_MIB"))cache_limit=size_t(std::max(0,std::min(4096,std::atoi(v))))<<20;
    }
    ~StrataCpuExpertLane() {
        clear();cache.clear();
        if(stream)CUDA_CHECK(cudaStreamDestroy(stream));
        for(auto e:{input_ready,input_copied,ready,consumed})if(e)CUDA_CHECK(cudaEventDestroy(e));
        if(input)CUDA_CHECK(cudaFreeHost(input));if(output)CUDA_CHECK(cudaFreeHost(output));
        if(device_output)CUDA_CHECK(cudaFree(device_output));if(device_packet)CUDA_CHECK(cudaFree(device_packet));
    }
    void allocate(int dev) {
        if(stream)return;device=dev;
        pool=std::make_unique<strata_ds4::CpuLanePool>(threads);
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        for(auto p:{&input_ready,&input_copied,&ready,&consumed})CUDA_CHECK(cudaEventCreateWithFlags(p,cudaEventDisableTiming));
        CUDA_CHECK(cudaMallocHost((void **)&input,scratch_bytes));CUDA_CHECK(cudaMallocHost((void **)&output,scratch_bytes));
        CUDA_CHECK(cudaMalloc((void **)&device_output,scratch_bytes));CUDA_CHECK(cudaMalloc((void **)&device_packet,sizeof(Packet)));
    }
    Blob *find(const StrataCpuExpertBranch &b,int id) {
        for(auto &p:cache){bool same=true;for(int j=0;j<3;++j)
            same&=p->key[j]==(const uint8_t *)b.weights[j]->data+size_t(id)*b.weights[j]->nb[2];
            if(same)return p.get();}
        return nullptr;
    }
    Blob *lease(const StrataCpuExpertBranch &b,int id) {
        if(auto *p=find(b,id)){p->held=true;p->stamp=++clock;++ram_hits;return p;}
        if(shared_ram) {
            if(!ram)return nullptr;
            auto p=std::make_unique<Blob>();
            for(int j=0;j<3;++j) {
                auto *w=b.weights[j];p->key[j]=(const uint8_t *)w->data+size_t(id)*w->nb[2];p->widths[j]=w->nb[2];
                auto file=strata_expert_file::find(p->key[j],p->widths[j]);
                if(!file){++ram_misses;return nullptr;}
                p->shared[j]=ram->acquire(file,file->offset_of(p->key[j],p->widths[j]),p->widths[j]);
                if(!p->shared[j]){++ram_misses;return nullptr;}
            }
            p->filled=p->held=true;++ram_hits;auto *result=p.get();cache.push_back(std::move(p));return result;
        }
        size_t bytes=0;for(auto *w:b.weights)bytes+=w->nb[2];
        if(bytes>cache_limit)return nullptr;
        Blob *p=nullptr;
        for(auto &v:cache)if(!v->held && v->capacity>=bytes && (!p || v->stamp<p->stamp))p=v.get();
        // Grow the bounded cache before replacing an old expert. No CUDA
        // allocation/free occurs on the callback thread.
        if(cache_bytes+bytes<=cache_limit) {
            auto v=std::make_unique<Blob>();CUDA_CHECK(cudaMallocHost((void **)&v->data,bytes));
            v->capacity=bytes;cache_bytes+=bytes;p=v.get();cache.push_back(std::move(v));
        }
        if(!p)return nullptr;
        for(int j=0;j<3;++j){p->key[j]=(const uint8_t *)b.weights[j]->data+size_t(id)*b.weights[j]->nb[2];p->widths[j]=b.weights[j]->nb[2];}
        p->filled=false;p->held=true;p->stamp=++clock;++ram_misses;return p;
    }
    void clear() {
        if(started){auto t=now();CUDA_CHECK(cudaStreamSynchronize(stream));
            if(completion)CUDA_CHECK(cudaEventSynchronize(consumed));retire_us+=now()-t;}
        if(error){try{std::rethrow_exception(error);}catch(const std::exception &e){GGML_ABORT("CPU expert failed: %s",e.what());}catch(...){GGML_ABORT("CPU expert failed");}}
        for(auto &p:cache)p->held=false;
        leases.clear();if(shared_ram)cache.clear();work.experts.clear();configured=pending=started=completion=false;branch={};
    }
    bool matches(const ggml_tensor *w)const {
        if(!configured)return false;
        for(auto *n:branch.nodes)if(n->src[0]==w)return true;
        return false;
    }
    bool selected(const ggml_tensor *w,int id)const{return matches(w) && packet.mask[id];}
    void prepare_work() {
        auto *up=branch.weights[0],*gate=branch.weights[1],*down=branch.weights[2];
        work.up_type=up->type;work.gate_type=gate->type;work.down_type=down->type;
        work.columns=int(up->ne[0]);work.hidden=int(up->ne[1]);
        work.tokens=int(branch.nodes[0]->ne[2]);work.topk=int(branch.nodes[0]->ne[1]);work.clamp=branch.clamp;
        work.input=input;work.output=output;
    }
    static void CUDART_CB execute(void *arg) noexcept {
        auto &c=*(StrataCpuExpertLane *)arg;
        try {
            auto t=now();if(!c.shared_ram){for(auto *p:c.leases)p->fill();c.pack_us+=now()-t;}
            t=now();c.work.compute(*c.pool);c.compute_us+=now()-t;
        } catch(...) {c.error=std::current_exception();std::fill(c.output,c.output+scratch_bytes/sizeof(float),NAN);}
    }
    static void CUDART_CB noop(void *) {}
    Cost calibrate(Blob &blob) {
        auto key=std::make_tuple(int(work.gate_type),int(work.up_type),int(work.down_type),work.columns,work.hidden,1);
        auto found=costs.find(key);if(found!=costs.end())return found->second;
        const int saved_tokens=work.tokens;work.tokens=1;
        Cost c;c.bytes=blob.bytes();auto t=now();blob.fill();c.pack_us=shared_ram?0:now()-t;
        std::fill(input,input+size_t(work.tokens)*work.columns,0.125f);
        const auto saved=work.experts;work.experts.clear();
        for(int i=0;i<work.tokens;++i)work.experts.push_back({blob.weight(1),blob.weight(0),blob.weight(2),i,i});
        work.compute(*pool);
        for(int i=0;i<3;++i){t=now();work.compute(*pool);c.compute_us+=now()-t;}c.compute_us/=3;
        work.experts=saved;
        void *probe=nullptr;cudaEvent_t a,b;CUDA_CHECK(cudaMalloc(&probe,c.bytes));
        CUDA_CHECK(cudaEventCreate(&a));CUDA_CHECK(cudaEventCreate(&b));float ms;
        CUDA_CHECK(cudaEventRecord(a,stream));
        for(int i=0;i<4;++i){size_t off=0;for(int j=0;j<3;++j){
            CUDA_CHECK(cudaMemcpyAsync((uint8_t *)probe+off,blob.weight(j),blob.widths[j],cudaMemcpyHostToDevice,stream));off+=blob.widths[j];}}
        CUDA_CHECK(cudaEventRecord(b,stream));CUDA_CHECK(cudaEventSynchronize(b));CUDA_CHECK(cudaEventElapsedTime(&ms,a,b));c.copy_us=ms*1000/4;
        CUDA_CHECK(cudaEventRecord(a,stream));
        for(int i=0;i<4;++i){CUDA_CHECK(cudaMemcpyAsync(input,probe,size_t(work.columns)*4,cudaMemcpyDeviceToHost,stream));
            CUDA_CHECK(cudaLaunchHostFunc(stream,noop,nullptr));CUDA_CHECK(cudaMemcpyAsync(probe,output,size_t(work.columns)*4,cudaMemcpyHostToDevice,stream));}
        CUDA_CHECK(cudaEventRecord(b,stream));CUDA_CHECK(cudaEventSynchronize(b));CUDA_CHECK(cudaEventElapsedTime(&ms,a,b));c.overhead_us=ms*1000/4;
        CUDA_CHECK(cudaEventDestroy(a));CUDA_CHECK(cudaEventDestroy(b));CUDA_CHECK(cudaFree(probe));
        costs[key]=c;
        GGML_LOG_INFO("STRATA_CPU_CALIB tokens=%d types=%s/%s/%s cpu_us=%.1f h2d_us=%.1f boundary_us=%.1f pack_us=%.1f bytes=%zu\n",work.tokens,
            ggml_type_name(work.gate_type),ggml_type_name(work.up_type),ggml_type_name(work.down_type),c.compute_us,c.copy_us,c.overhead_us,c.pack_us,c.bytes);
        work.tokens=saved_tokens;return c;
    }
    void begin(cudaStream_t owner);
    void merge(ggml_tensor *dst,cudaStream_t owner);
};

static __global__ void strata_cpu_publish(StrataCpuExpertLane::Packet *dst,StrataCpuExpertLane::Packet value) {
    const int i=threadIdx.x;dst->mask[i]=value.mask[i];if(i<64)dst->slots[i]=value.slots[i];
}
static __global__ void strata_cpu_scatter(float *dst,const float *source,const StrataCpuExpertLane::Packet *p,int columns,int rows,bool down) {
    const int slot=blockIdx.y,index=p->slots[slot];if(slot>=rows || index<0)return;
    for(int r=blockIdx.x*blockDim.x+threadIdx.x;r<columns;r+=blockDim.x*gridDim.x)
        dst[size_t(slot)*columns+r]=down?source[size_t(index)*columns+r]:0.0f;
}
inline void StrataCpuExpertLane::begin(cudaStream_t owner) {
    if(!configured || started)return;
    strata_cpu_publish<<<1,256,0,owner>>>(device_packet,packet);CUDA_CHECK(cudaGetLastError());
    if(!work.experts.empty()) {
        CUDA_CHECK(cudaEventRecord(input_ready,owner));CUDA_CHECK(cudaStreamWaitEvent(stream,input_ready,0));
        const size_t in=size_t(work.tokens)*work.columns*4,out=work.experts.size()*work.columns*4;
        CUDA_CHECK(cudaMemcpyAsync(input,branch.nodes[0]->src[1]->data,in,cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaEventRecord(input_copied,stream));
        CUDA_CHECK(cudaLaunchHostFunc(stream,execute,this));
        CUDA_CHECK(cudaMemcpyAsync(device_output,output,out,cudaMemcpyHostToDevice,stream));
        CUDA_CHECK(cudaEventRecord(ready,stream));
        // Prevent the allocator from reusing x before its D2H completes. No
        // host synchronization, and no wait for CPU compute at this boundary.
        CUDA_CHECK(cudaStreamWaitEvent(owner,input_copied,0));
        input_bytes+=in;output_bytes+=out;++operations;experts+=work.experts.size();
    } else CUDA_CHECK(cudaEventRecord(ready,stream));
    started=true;
}
inline void StrataCpuExpertLane::merge(ggml_tensor *dst,cudaStream_t owner) {
    const bool down=dst==branch.nodes[2];
    if(down) {
        cudaStreamCaptureStatus capture;CUDA_CHECK(cudaStreamIsCapturing(owner,&capture));
        CUDA_CHECK(cudaStreamWaitEvent(owner,ready,capture==cudaStreamCaptureStatusActive?cudaEventWaitExternal:0));
    }
    const int cols=int(dst->ne[0]),rows=int(dst->ne[1]*dst->ne[2]);
    strata_cpu_scatter<<<dim3((cols+255)/256,rows),256,0,owner>>>((float *)dst->data,device_output,device_packet,cols,rows,down);
    CUDA_CHECK(cudaGetLastError());
}
