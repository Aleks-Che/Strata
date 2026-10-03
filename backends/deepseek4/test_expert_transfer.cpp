#include "expert_transfer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
    if(!copy || !stats || !budget)return 1;
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
    auto check=[&](int first,int last,int tokens) {
        ggml_backend_buffer_clear(device,0xA5);
        copy(gpu,source,dest,first,last,tokens);
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
    for(int round=0;round<3;++round) for(int i=0;i<count;++i)copy(gpu,source,dest,i,i,1);
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
    copy(draft_gpu,source,dest,2,2,1);
    ggml_backend_synchronize(draft_gpu);
    StrataExpertCounters draft_stats;
    stats(draft_gpu,&draft_stats);
    ok &= draft_stats.bypass==1 && draft_stats.misses==0;
    ggml_backend_tensor_get(dest,actual.data(),2*width,width);
    ok &= std::memcmp(actual.data(),expected.data()+2*width,width)==0;
    ggml_backend_free(draft_gpu);
    budget(4);stats(gpu,&before);check(9,9,1);stats(gpu,&after);
    ok &= after.hits==before.hits+1 && after.h2d_bytes==before.h2d_bytes;
    ggml_backend_buffer_free(device);ggml_backend_buffer_free(host);
    ggml_free(dev_ctx);ggml_free(host_ctx);
    ggml_backend_free(gpu);ggml_backend_free(cpu);
    std::puts(ok?"Expert cache/staging byte parity passed":"Expert cache/staging byte parity FAILED");
    return ok?0:1;
}
