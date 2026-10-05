#pragma once
// Diagnostic copy variants. Production expert transport still uses std::memcpy.
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace strata_host_copy {
#if defined(_M_X64) || defined(__x86_64__)
constexpr bool supported=true;
#else
constexpr bool supported=false;
#endif
constexpr size_t threshold=64*1024;
// Non-overlapping source/destination, like memcpy. Stream complete cache lines
// only; prefix/tail never share a line with a streaming store. The producer's
// SFENCE must complete before publishing this buffer or submitting CUDA DMA.
// Return whether streaming stores were used (small copies retain libc memcpy).
inline bool copy(void *destination,const void *source,size_t bytes,bool stream) {
#if defined(_M_X64) || defined(__x86_64__)
    if(stream && bytes>=threshold) {
        auto *d=static_cast<uint8_t *>(destination);
        auto *s=static_cast<const uint8_t *>(source);
        const size_t prefix=(64-(uintptr_t(d)&63))&63;
        if(prefix) {std::memcpy(d,s,prefix);d+=prefix;s+=prefix;bytes-=prefix;}
        while(bytes>=64) {
#ifdef STRATA_HOST_COPY_AVX2
            const auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(s));
            const auto b=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(s+32));
            _mm256_stream_si256(reinterpret_cast<__m256i *>(d),a);
            _mm256_stream_si256(reinterpret_cast<__m256i *>(d+32),b);
#else
            const auto a=_mm_loadu_si128(reinterpret_cast<const __m128i *>(s));
            const auto b=_mm_loadu_si128(reinterpret_cast<const __m128i *>(s+16));
            const auto c=_mm_loadu_si128(reinterpret_cast<const __m128i *>(s+32));
            const auto e=_mm_loadu_si128(reinterpret_cast<const __m128i *>(s+48));
            _mm_stream_si128(reinterpret_cast<__m128i *>(d),a);
            _mm_stream_si128(reinterpret_cast<__m128i *>(d+16),b);
            _mm_stream_si128(reinterpret_cast<__m128i *>(d+32),c);
            _mm_stream_si128(reinterpret_cast<__m128i *>(d+48),e);
#endif
            d+=64;s+=64;bytes-=64;
        }
        if(bytes)std::memcpy(d,s,bytes);
        _mm_sfence();
        return true;
    }
#else
    (void)stream;
#endif
    if(bytes)std::memcpy(destination,source,bytes);
    return false;
}
}
