#include "expert_transfer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <thread>
#include <exception>

static void stage_size(const char *value) {
#ifdef _WIN32
    _putenv_s("STRATA_EXPERT_STAGE_MIB",value);
#else
    setenv("STRATA_EXPERT_STAGE_MIB",value,1);
#endif
}

static bool mixed_arena(ggml_backend_t cpu,StrataExpertCopy copy,StrataExpertPlan plan,
                        StrataExpertFinish finish,StrataExpertStats stats,bool match,bool protect) {
#ifdef _WIN32
    _putenv_s("STRATA_EXPERT_CACHE_MATCH_SIZE",match?"1":"0");
#else
    setenv("STRATA_EXPERT_CACHE_MATCH_SIZE",match?"1":"0",1);
#endif
    auto *gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
    auto *host_ctx=ggml_init({ggml_tensor_overhead()*4,nullptr,true});
    auto *dev_ctx=ggml_init({ggml_tensor_overhead()*4,nullptr,true});
    const size_t widths[]={1<<20,3<<19,3<<18,3<<19};
    ggml_tensor *source[4],*dest[4];
    for(int i=0;i<4;++i) {
        source[i]=ggml_new_tensor_3d(host_ctx,GGML_TYPE_I8,widths[i],1,2);
        dest[i]=ggml_new_tensor_3d(dev_ctx,GGML_TYPE_I8,widths[i],1,2);
    }
    auto *host=ggml_backend_alloc_ctx_tensors(host_ctx,cpu);
    auto *device=ggml_backend_alloc_ctx_tensors(dev_ctx,gpu);
    std::vector<uint8_t> expected[4];
    for(int i=0;i<4;++i) {
        expected[i].resize(widths[i]*2);
        for(size_t j=0;j<expected[i].size();++j)expected[i][j]=uint8_t(i*37+j*3+(j>>11));
        ggml_backend_tensor_set(source[i],expected[i].data(),0,expected[i].size());
    }
    bool ok=true;
    auto transfer=[&](int i,bool preserve=false) {
        StrataExpertSlice slices[2]={{source[i]->data,widths[i]+512,true},
                                     {source[1]->data,widths[1]+512,true}};
        plan(gpu,slices,preserve?2:1);
        copy(gpu,source[i],dest[i],0,0,1);finish(gpu);ggml_backend_synchronize(gpu);
        std::vector<uint8_t> actual(widths[i]+512);
        ggml_backend_tensor_get(dest[i],actual.data(),0,actual.size());
        ok &= std::memcmp(actual.data(),expected[i].data(),actual.size())==0;
    };
    // A 4 MiB arena: 1 MiB + 1.5 MiB + 0.75 MiB, plus expert-edge padding.
    // The new 1.5 MiB matrix fits with one same-size eviction. Strict LRU
    // needs two. A protected matching matrix must never be evicted; with it
    // retained, fragmentation requires a bypass even after freeing the others.
    transfer(0);transfer(1);transfer(2);
    StrataExpertCounters before,after;stats(gpu,&before);
    transfer(3,protect);stats(gpu,&after);
    ok &= after.evictions-before.evictions==uint64_t(match&&!protect?1:2);
    ok &= after.bypass-before.bypass==uint64_t(protect?1:0);
    before=after;
    transfer(protect?1:2);stats(gpu,&after);
    ok &= after.hits==before.hits+1 && after.h2d_bytes==before.h2d_bytes;
    if(match&&!protect) {
        before=after;transfer(0);stats(gpu,&after);
        ok &= after.hits==before.hits+1 && after.h2d_bytes==before.h2d_bytes;
    }
    ggml_backend_buffer_free(device);ggml_backend_buffer_free(host);
    ggml_free(dev_ctx);ggml_free(host_ctx);ggml_backend_free(gpu);
    return ok;
}

int main() {
#ifdef _WIN32
    _putenv_s("STRATA_EXPERT_CACHE_MIB","4");
    _putenv_s("STRATA_EXPERT_STAGE_MIB","1");
#else
    setenv("STRATA_EXPERT_CACHE_MIB","4",1);
    setenv("STRATA_EXPERT_STAGE_MIB","1",1);
#endif
    ggml_backend_load_all();
    auto *cpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
    auto *gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
    if(!gpu || !cpu) {std::puts("SKIP: CPU and CUDA backends required");return 77;}
    auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu));
    auto copy=(StrataExpertCopy)ggml_backend_reg_get_proc_address(reg,"strata_expert_copy");
    auto stats=(StrataExpertStats)ggml_backend_reg_get_proc_address(reg,"strata_expert_stats");
    auto budget=(StrataExpertBudget)ggml_backend_reg_get_proc_address(reg,"strata_expert_budget");
    auto control=(StrataExpertControl)ggml_backend_reg_get_proc_address(reg,"strata_expert_control");
    auto plan=(StrataExpertPlan)ggml_backend_reg_get_proc_address(reg,"strata_expert_plan");
    auto finish=(StrataExpertFinish)ggml_backend_reg_get_proc_address(reg,"strata_expert_finish");
    if(!copy || !stats || !budget || !control || !plan || !finish)return 1;
    budget(4);
    constexpr int count=10;
    constexpr size_t width=(1<<20)+(256<<10); // crosses the 1 MiB staging boundary
    auto *host_ctx=ggml_init({ggml_tensor_overhead()*2,nullptr,true});
    auto *dev_ctx=ggml_init({ggml_tensor_overhead()*2,nullptr,true});
    auto *source=ggml_new_tensor_3d(host_ctx,GGML_TYPE_I8,width,1,count);
    auto *dest=ggml_new_tensor_3d(dev_ctx,GGML_TYPE_I8,width,1,count);
    auto *host=ggml_backend_alloc_ctx_tensors(host_ctx,cpu);
    auto *device=ggml_backend_alloc_ctx_tensors(dev_ctx,gpu);
    std::vector<uint8_t> expected(width*count),actual(width*count);
    for(size_t i=0;i<expected.size();++i)expected[i]=uint8_t((i*17+(i>>11)*23)&255);
    ggml_backend_tensor_set(source,expected.data(),0,expected.size());
    bool ok=true;
    if(const char *p=std::getenv("STRATA_EXPERT_PIPELINE");p && std::strcmp(p,"1")==0) {
        // Force construction to fail after allocating the legacy cache. The
        // registry must remain empty and allow a clean retry on this backend.
        stage_size("0");bool rejected=false;
        try {plan(gpu,nullptr,0);}catch(const std::exception &) {rejected=true;}
        stage_size("1");
        StrataExpertCounters empty;stats(gpu,&empty);
        StrataVramStatus live;control(0,nullptr,&live);
        ok &= rejected && empty.pipeline_groups==0 && live.cache_bytes==0 && live.matrices==0;
    }
    auto begin=[&](ggml_backend_t backend,int first,int last,int tokens) {
        std::vector<StrataExpertSlice> slices;
        for(int i=first;i<=last;++i)slices.push_back({(uint8_t *)source->data+size_t(i)*width,width+(i<count-1?512:0),tokens<=8});
        plan(backend,slices.data(),slices.size());
    };
    auto check=[&](int first,int last,int tokens) {
        ggml_backend_buffer_clear(device,0xA5);
        begin(gpu,first,last,tokens);
        copy(gpu,source,dest,first,last,tokens);
        finish(gpu);
        ggml_backend_synchronize(gpu);
        ggml_backend_tensor_get(dest,actual.data(),0,actual.size());
        size_t begin=size_t(first)*width,end=(size_t(last)+1)*width+(last<count-1?512:0);
        for(size_t i=0;i<actual.size();++i)if(actual[i]!=(i>=begin&&i<end?expected[i]:0xA5)) {
            std::fprintf(stderr,"Mismatch at byte %zu for experts %d..%d\n",i,first,last);ok=false;break;
        }
    };
    check(2,3,1);
    StrataExpertCounters before,after;
    stats(gpu,&before);
    check(2,3,1);stats(gpu,&after);
    ok &= after.hits==before.hits+2 && after.h2d_bytes==before.h2d_bytes;
    // Force arena fragmentation, eviction, and reuse while CUDA copies are queued.
    for(int round=0;round<3;++round) {
        begin(gpu,0,count-1,1);
        for(int i=0;i<count;++i)copy(gpu,source,dest,i,i,1);
        finish(gpu);
    }
    ggml_backend_synchronize(gpu);
    ggml_backend_tensor_get(dest,actual.data(),0,actual.size());
    ok &= expected==actual;
    stats(gpu,&before);ok &= before.evictions>0;
    check(0,9,32);stats(gpu,&after); // prefill bypass must not populate/evict cache
    ok &= before.hits==after.hits && before.misses==after.misses && before.evictions==after.evictions;
    check(9,9,1); // final expert: no padding beyond tensor end
    // A second context gets a separate smaller arena. It must neither inherit
    // the target's 4 MiB nor clear its entries when the draft context is freed.
    auto *draft_gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
    budget(1); // too small for a 1.25 MiB expert => bypass, no cached entries
    begin(draft_gpu,2,2,1);
    copy(draft_gpu,source,dest,2,2,1);
    finish(draft_gpu);
    ggml_backend_synchronize(draft_gpu);
    StrataExpertCounters draft_stats;
    stats(draft_gpu,&draft_stats);
    ok &= draft_stats.bypass==1 && draft_stats.misses==0;
    ggml_backend_tensor_get(dest,actual.data(),2*width,width);
    ok &= std::memcmp(actual.data(),expected.data()+2*width,width)==0;
    ggml_backend_free(draft_gpu);
    budget(4);stats(gpu,&before);check(9,9,1);stats(gpu,&after);
    ok &= after.hits==before.hits+1 && after.h2d_bytes==before.h2d_bytes;
    ok &= mixed_arena(cpu,copy,plan,finish,stats,false,false);
    ok &= mixed_arena(cpu,copy,plan,finish,stats,true,false);
    // Only the pipeline retains a multi-matrix plan; the synchronous path
    // consumes one matrix at a time and does not need a protected set.
    if(const char *p=std::getenv("STRATA_EXPERT_PIPELINE");p && std::strcmp(p,"1")==0)
        ok &= mixed_arena(cpu,copy,plan,finish,stats,true,true);
    // Changing policy converts the fixed arena once. Thereafter individual
    // matrices can be freed without discarding the remaining hot entries.
    StrataVramPolicy policy;policy.mode=1;policy.matrices=3;policy.reserve_mib=128;
    StrataVramStatus live;
    ok &= control(0,&policy,&live);
    check(2,3,1);check(4,4,1);
    control(0,nullptr,&live);ok &= live.matrices==3 && live.cache_bytes>3*width;
    policy.matrices=2;control(0,&policy,&live);ok &= live.matrices==2;
    stats(gpu,&before);check(3,4,1);stats(gpu,&after);
    ok &= after.hits==before.hits+2; // only the oldest was released
    // The matrix cap is shared with a second (DSpark) CUDA context.
    draft_gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
    budget(1);begin(draft_gpu,5,5,1);copy(draft_gpu,source,dest,5,5,1);finish(draft_gpu);ggml_backend_synchronize(draft_gpu);
    control(0,nullptr,&live);ok &= live.matrices==2;
    ggml_backend_free(draft_gpu);budget(4);
    policy.matrices=0;control(0,&policy,&live);ok &= live.matrices==0 && live.cache_bytes==0;
    check(0,9,1);control(0,nullptr,&live);ok &= live.matrices==0;
    policy.matrices=6;control(0,&policy,&live);check(2,5,1);
    control(0,nullptr,&live);ok &= live.matrices==4;
    // A separate allocation stands in for another application taking VRAM.
    policy.mode=2;policy.target_mib=((live.total_bytes-live.free_bytes)>>20)+128;
    control(0,&policy,&live);
    auto *pressure=ggml_backend_alloc_buffer(gpu,256ULL<<20);
    if(!pressure)ok=false;
    else {
        // WDDM residency is visible globally after the allocation is touched.
        ggml_backend_buffer_clear(pressure,0);ggml_backend_synchronize(gpu);
        for(int i=0;i<30;++i) {
            control(0,nullptr,&live);
            if(!live.matrices)break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        control(0,nullptr,&live);ok &= live.matrices==0 && live.target_unreachable;
        ggml_backend_buffer_free(pressure);
        for(int i=0;i<30;++i) {
            control(0,nullptr,&live);
            if(!live.target_unreachable && live.limit_bytes>(64ULL<<20))break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        control(0,nullptr,&live);check(2,3,1);
        control(0,nullptr,&live);ok &= live.matrices==2 && !live.target_unreachable;
    }
    // Configured budgets can be restored after dynamic use, with byte parity.
    policy.mode=0;control(0,&policy,&live);check(0,9,1);
    control(0,nullptr,&live);ok &= live.cache_bytes<=(4ULL<<20);
    stats(gpu,&after);ok &= after.ordered_reuses>0;
    // Growth headroom must not freeze admission below the actual byte limit.
    // A same-size replacement needs no extra VRAM, even with <64 MiB spare.
    policy.mode=2;policy.target_mib=((live.total_bytes-live.free_bytes)>>20)+8;
    control(0,&policy,&live);
    auto resident_bytes=live.cache_bytes;stats(gpu,&before);
    check(1,1,1);stats(gpu,&after);control(0,nullptr,&live);
    ok &= after.ordered_reuses>before.ordered_reuses && live.cache_bytes==resident_bytes;
    // Cancel a partially consumed plan and change residency before a new plan.
    begin(gpu,0,count-1,1);copy(gpu,source,dest,0,0,1);finish(gpu);
    policy.mode=1;policy.matrices=0;control(0,&policy,&live);
    check(8,9,1);
    stats(gpu,&after);
    if(const char *p=std::getenv("STRATA_EXPERT_PIPELINE");p && std::strcmp(p,"1")==0)
        ok &= after.pipeline_groups>0 && after.pipeline_chunks>0 && after.pipeline_fallbacks==0;
    // Repeated hot matrices survive a stream of one-use weights. Rejected
    // admissions must still arrive byte-for-byte in the working tensor, and a
    // newly repeated matrix must eventually enter the cache.
#ifdef _WIN32
    _putenv_s("STRATA_EXPERT_CACHE_POLICY","frequency");
#else
    setenv("STRATA_EXPERT_CACHE_POLICY","frequency",1);
#endif
    auto *frequency_gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
    budget(4);policy.mode=1;policy.matrices=3;control(0,&policy,&live);
    auto frequency_copy=[&](int id) {
        begin(frequency_gpu,id,id,1);copy(frequency_gpu,source,dest,id,id,1);finish(frequency_gpu);
        ggml_backend_synchronize(frequency_gpu);
        ggml_backend_tensor_get(dest,actual.data(),id*width,width);
        ok &= std::memcmp(actual.data(),expected.data()+id*width,width)==0;
    };
    for(int repeat=0;repeat<8;++repeat)for(int id=0;id<3;++id)frequency_copy(id);
    stats(frequency_gpu,&before);
    for(int id=4;id<10;++id)frequency_copy(id);
    stats(frequency_gpu,&after);
    ok &= after.admission_rejects==before.admission_rejects+6 && after.evictions==before.evictions;
    before=after;
    for(int id=0;id<3;++id)frequency_copy(id);
    stats(frequency_gpu,&after);ok &= after.hits==before.hits+3;
    for(int repeat=0;repeat<12;++repeat)frequency_copy(4);
    stats(frequency_gpu,&before);frequency_copy(4);stats(frequency_gpu,&after);
    ok &= after.hits==before.hits+1 && after.h2d_bytes==before.h2d_bytes;
    // Live cache limits remain authoritative even for hot entries.
    policy.matrices=0;control(0,&policy,&live);ok &= live.cache_bytes==0 && live.matrices==0;
    ggml_backend_free(frequency_gpu);
    ggml_backend_buffer_free(device);ggml_backend_buffer_free(host);
    ggml_free(dev_ctx);ggml_free(host_ctx);
    ggml_backend_free(gpu);ggml_backend_free(cpu);
    std::puts(ok?"Expert cache/staging byte parity passed":"Expert cache/staging byte parity FAILED");
    return ok?0:1;
}
