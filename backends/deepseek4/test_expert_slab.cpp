#include "expert_slab.hpp"
#include <cstdio>
#include <cstring>

#define REQUIRE(x) do {if(!(x)) {std::fprintf(stderr,"Failed line %d: %s\n",__LINE__,#x);return 1;}} while(0)
#define CUDA(x) REQUIRE((x)==cudaSuccess)

int main() {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    CUDA(cudaSetDevice(0));
    strata_ds4::ExpertSlabAllocator pool(4<<20);
    constexpr size_t bytes=(1<<20)+17,stride=(bytes+255)&~size_t(255);
    void *a=nullptr,*b=nullptr,*c=nullptr,*d=nullptr;
    // Two different requested lengths share a stride, with bounded partial growth.
    CUDA(pool.allocate(&a,bytes,2*stride));
    CUDA(pool.allocate(&b,bytes+1,0));
    REQUIRE(pool.status().reserved==2*stride && pool.status().blocks==1);
    bool denied=false;
    REQUIRE(pool.allocate(&c,bytes,0,&denied)==cudaErrorMemoryAllocation && denied && !c);
    CUDA(cudaMemset(a,0x31,bytes));CUDA(cudaMemset(b,0x72,bytes+1));
    REQUIRE(pool.release(static_cast<uint8_t *>(a)+1)==cudaErrorInvalidValue);
    CUDA(pool.release(a));REQUIRE(pool.release(a)==cudaErrorInvalidValue);
    REQUIRE(pool.status().reserved==2*stride && pool.status().requested==bytes+1);
    CUDA(pool.allocate(&c,bytes,0));REQUIRE(c==a);
    CUDA(cudaMemset(c,0x55,bytes));
    std::vector<uint8_t> actual(bytes+1);
    CUDA(cudaMemcpy(actual.data(),b,bytes+1,cudaMemcpyDeviceToHost));
    for(auto x:actual)REQUIRE(x==0x72); // Neighbor survived hole reuse.
    // Different strides require their own block; one oversized matrix gets one slot.
    CUDA(pool.allocate(&d,5<<20,5<<20));REQUIRE(pool.status().blocks==2);
    CUDA(pool.release(d));CUDA(pool.release(b));
    REQUIRE(pool.status().reserved==2*stride && pool.status().requested==bytes);
    CUDA(cudaMemcpy(actual.data(),c,bytes,cudaMemcpyDeviceToHost));
    for(size_t i=0;i<bytes;++i)REQUIRE(actual[i]==0x55);
    CUDA(pool.release(c));REQUIRE(pool.status().reserved==0 && pool.status().blocks==0);
    REQUIRE(pool.status().block_allocations==2);
    REQUIRE(pool.allocate(&a,0,1<<20)==cudaErrorInvalidValue);
    REQUIRE(pool.allocate(&a,SIZE_MAX,SIZE_MAX)==cudaErrorInvalidValue);
    std::puts("Slab ownership, partial growth, padding and release passed");
    return 0;
}
