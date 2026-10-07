#include "scatter_copy.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

namespace {
constexpr int capacity=32;
struct Range {unsigned char *dst;const unsigned char *src;size_t bytes;uint32_t end_tile;};
struct Packet {Range ranges[capacity];int count;};
static_assert(sizeof(Packet)<4096,"Keep launch metadata below the legacy parameter limit");

template<size_t TileBytes>
__global__ void scatter(Packet packet) {
    // Every block takes exactly one tile; tiny guards occupy one block rather
    // than a rectangle sized for the largest matrix. Search is warp-uniform.
    int lo=0,hi=packet.count-1;
    while(lo<hi) {
        const int mid=(lo+hi)/2;
        if(blockIdx.x<packet.ranges[mid].end_tile)hi=mid;else lo=mid+1;
    }
    const auto r=packet.ranges[lo];
    const uint32_t first=lo?packet.ranges[lo-1].end_tile:0;
    const size_t offset=size_t(blockIdx.x-first)*TileBytes;
    const size_t bytes=r.bytes-offset<TileBytes?r.bytes-offset:TileBytes;
    auto *dst=r.dst+offset;const auto *src=r.src+offset;
    if(((reinterpret_cast<uintptr_t>(dst)|reinterpret_cast<uintptr_t>(src))&15)==0) {
        const auto *from=reinterpret_cast<const uint4 *>(src);
        auto *to=reinterpret_cast<uint4 *>(dst);
        for(size_t i=threadIdx.x;i<bytes/16;i+=blockDim.x)to[i]=from[i];
        for(size_t i=(bytes&~size_t(15))+threadIdx.x;i<bytes;i+=blockDim.x)dst[i]=src[i];
    } else {
        for(size_t i=threadIdx.x;i<bytes;i+=blockDim.x)dst[i]=src[i];
    }
}

template<size_t TileBytes>
cudaError_t launch(void *const *dsts,const void *const *srcs,const size_t *sizes,size_t count,
                   cudaStream_t stream,size_t *launches) {
    Packet packet{};uint32_t tiles=0;
    for(size_t i=0;i<count;++i) {
        if(!sizes[i])continue;
        tiles+=uint32_t(1+(sizes[i]-1)/TileBytes);
        packet.ranges[packet.count++]={static_cast<unsigned char *>(dsts[i]),
            static_cast<const unsigned char *>(srcs[i]),sizes[i],tiles};
        if(packet.count==capacity || i+1==count) {
            scatter<TileBytes><<<tiles,256,0,stream>>>(packet);
            if(launches)++*launches;
            const auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
            packet.count=0;tiles=0;
        }
    }
    // A trailing zero-size range must not drop an unfinished packet.
    if(packet.count) {
        scatter<TileBytes><<<tiles,256,0,stream>>>(packet);
        if(launches)++*launches;
        return cudaGetLastError();
    }
    return cudaSuccess;
}
}

cudaError_t mimo_scatter_copy(void *const *dsts,const void *const *srcs,const size_t *sizes,
                            size_t count,cudaStream_t stream,size_t *launches,int tile_kib) {
    if(launches)*launches=0;
    if(tile_kib!=16 && tile_kib!=64 && tile_kib!=256)return cudaErrorInvalidValue;
    if(!count)return cudaSuccess;
    if(count>8192 || !dsts || !srcs || !sizes)return cudaErrorInvalidValue;
    // Validate the whole descriptor list before launching any work. Restrict
    // each range so even a full packet fits the CUDA grid.x limit at tile16K.
    for(size_t i=0;i<count;++i) {
        if(!sizes[i])continue;
        const auto dst=reinterpret_cast<uintptr_t>(dsts[i]),src=reinterpret_cast<uintptr_t>(srcs[i]);
        if(!dst || !src || sizes[i]>(size_t(1)<<30) ||
           sizes[i]>std::numeric_limits<uintptr_t>::max()-dst ||
           sizes[i]>std::numeric_limits<uintptr_t>::max()-src)return cudaErrorInvalidValue;
    }
    if(tile_kib==16)return launch<16384>(dsts,srcs,sizes,count,stream,launches);
    if(tile_kib==64)return launch<65536>(dsts,srcs,sizes,count,stream,launches);
    return launch<262144>(dsts,srcs,sizes,count,stream,launches);
}
