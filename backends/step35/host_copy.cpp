#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif
namespace step35 {
// Pinned buffers are normally aligned; the prefix also admits arbitrary test
// destinations. Every vector load/store is within the requested byte range.
// Temporal stores intentionally retain the hot ring buffers in the CPU cache.
#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__)
__attribute__((noinline))
#endif
void host_copy_avx2_cached(void * destination,const void * source,size_t bytes) {
#if defined(_M_X64) || defined(__x86_64__)
    auto * dst=static_cast<uint8_t *>(destination);
    auto * src=static_cast<const uint8_t *>(source);
    const size_t head=std::min(bytes,(64-(reinterpret_cast<uintptr_t>(dst)&63))&63);
    if(head){std::memcpy(dst,src,head);dst+=head;src+=head;bytes-=head;}
    while(bytes>=128) {
        const auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
        const auto b=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(src+32));
        const auto c=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(src+64));
        const auto d=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(src+96));
        _mm256_store_si256(reinterpret_cast<__m256i *>(dst),a);
        _mm256_store_si256(reinterpret_cast<__m256i *>(dst+32),b);
        _mm256_store_si256(reinterpret_cast<__m256i *>(dst+64),c);
        _mm256_store_si256(reinterpret_cast<__m256i *>(dst+96),d);
        dst+=128;src+=128;bytes-=128;
    }
    _mm256_zeroupper();
    if(bytes)std::memcpy(dst,src,bytes);
#else
    throw std::runtime_error("Step AVX2 host copy is unavailable on this architecture");
#endif
}
}
