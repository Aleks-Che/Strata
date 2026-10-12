#pragma once
#include "cpu_lane.hpp"
#include <array>

struct StrataCpuLane {
    int threads=0, max_experts=1;
    std::unique_ptr<strata_ds4::CpuLanePool> pool;
    std::unordered_set<const void *> planned;
    const ggml_tensor *weight=nullptr;
    std::array<uint8_t,256> mask{};
    uint8_t *device_mask=nullptr;
    float *input=nullptr,*output=nullptr;
    size_t input_capacity=0,output_capacity=0;
    cudaEvent_t uploaded=nullptr;
    bool uploading=false;
    struct Row {const void *weight;size_t input_offset,output_offset;};
    std::vector<Row> rows;
    std::vector<uint8_t> quantized;
    int columns=0,nrows=0;
    ggml_type type=GGML_TYPE_F32;
    uint64_t operations=0,experts=0,saved_bytes=0,wait_us=0,input_us=0;
    StrataCpuLane() {
        const char *v=std::getenv("STRATA_DS4_CPU_THREADS");threads=v?std::atoi(v):0;
        const char *m=std::getenv("STRATA_DS4_CPU_MAX_EXPERTS");max_experts=m?std::atoi(m):1;
        GGML_ASSERT(threads>=0 && threads<=16 && max_experts>=1 && max_experts<=6);
#ifndef STRATA_DS4_CPU_LANE_COMPILED
        if(threads)GGML_ABORT("CPU lane requires STRATA_DEEPSEEK_DIRECT_CACHE_EXPERIMENT=ON");
#endif
    }
    void wait_upload() {
        if(uploading){CUDA_CHECK(cudaEventSynchronize(uploaded));uploading=false;}
    }
    void clear() {
        if(pool)pool->wait();wait_upload();weight=nullptr;mask.fill(0);rows.clear();
    }
    ~StrataCpuLane() {
        clear();if(uploaded)CUDA_CHECK(cudaEventDestroy(uploaded));
        if(device_mask)CUDA_CHECK(cudaFree(device_mask));
        if(input)CUDA_CHECK(cudaFreeHost(input));if(output)CUDA_CHECK(cudaFreeHost(output));
    }
    void allocate(size_t in,size_t out) {
        wait_upload();
        if(!pool)pool=std::make_unique<strata_ds4::CpuLanePool>(threads);
        if(!device_mask)CUDA_CHECK(cudaMalloc((void **)&device_mask,mask.size()));
        if(!uploaded)CUDA_CHECK(cudaEventCreateWithFlags(&uploaded,cudaEventDisableTiming));
        if(in>input_capacity){if(input)CUDA_CHECK(cudaFreeHost(input));CUDA_CHECK(cudaMallocHost((void **)&input,in));input_capacity=in;}
        if(out>output_capacity){if(output)CUDA_CHECK(cudaFreeHost(output));CUDA_CHECK(cudaMallocHost((void **)&output,out));output_capacity=out;}
    }
    bool selected(const ggml_tensor *w,int id)const{return weight==w && mask[size_t(id)];}
    void launch() {
        const size_t qbytes=strata_ds4::cpu_lane_activation_bytes(type,columns);
        quantized.resize(rows.size()*qbytes);
        for(size_t i=0;i<rows.size();++i)
            strata_ds4::cpu_lane_quantize(type,columns,(float *)((uint8_t *)input+rows[i].input_offset),quantized.data()+i*qbytes);
        const int blocks=(nrows+63)/64;
        pool->submit(int(rows.size())*blocks,[this,qbytes,blocks](int job) {
            size_t i=size_t(job/blocks);int first=(job%blocks)*64;
            strata_ds4::cpu_lane_rows(type,columns,rows[i].weight,quantized.data()+i*qbytes,
                (float *)((uint8_t *)output+rows[i].output_offset),first,std::min(nrows,first+64));
        });
        ++operations;
    }
    void join(ggml_tensor *dst,cudaStream_t stream) {
        if(!weight)return;
        const auto t=ggml_time_us();pool->wait();wait_us+=ggml_time_us()-t;
        for(auto &r:rows)CUDA_CHECK(cudaMemcpyAsync((uint8_t *)dst->data+r.output_offset,
            (uint8_t *)output+r.output_offset,size_t(nrows)*sizeof(float),cudaMemcpyHostToDevice,stream));
        CUDA_CHECK(cudaEventRecord(uploaded,stream));uploading=true;
        weight=nullptr;
    }
};
