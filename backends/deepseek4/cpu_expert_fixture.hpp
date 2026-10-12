#pragma once
#include "cpu_lane.hpp"
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace strata_ds4 {
// Local diagnostic fixture, little-endian x86 format. It stores original
// quantized rows, route IDs, input activations and CPU output; no requantization.
struct CpuExpertFixture {
    static constexpr size_t limit=256ULL<<20;
    static size_t weight_bytes(const CpuExpertWork &w,int matrix) {
        return matrix==0?ggml_row_size(w.gate_type,w.columns)*w.hidden:
            matrix==1?ggml_row_size(w.up_type,w.columns)*w.hidden:ggml_row_size(w.down_type,w.hidden)*w.columns;
    }
    static size_t bytes(const CpuExpertWork &w) {
        return 8+9*4+size_t(w.tokens)*(w.topk+w.columns)*4+
            w.experts.size()*(3*4+weight_bytes(w,0)+weight_bytes(w,1)+weight_bytes(w,2)+size_t(w.columns)*4);
    }
    static void write(const std::filesystem::path &path,const CpuExpertWork &w,const int32_t *ids) {
        if(bytes(w)>limit)throw std::runtime_error("CPU fixture exceeds 256 MiB");
        if(std::filesystem::exists(path))throw std::runtime_error("CPU fixture already exists");
        FILE *f=std::fopen(path.string().c_str(),"wb");if(!f)throw std::runtime_error("cannot open CPU fixture");
        auto put=[&](const void *p,size_t n){if(std::fwrite(p,1,n,f)!=n)throw std::runtime_error("cannot write CPU fixture");};
        try {
            put("SCPU0001",8);
            uint32_t clamp;std::memcpy(&clamp,&w.clamp,4);
            uint32_t h[]={uint32_t(w.columns),uint32_t(w.hidden),uint32_t(w.tokens),uint32_t(w.topk),uint32_t(w.experts.size()),
                uint32_t(w.gate_type),uint32_t(w.up_type),uint32_t(w.down_type),clamp};put(h,sizeof(h));
            put(ids,size_t(w.tokens)*w.topk*4);put(w.input,size_t(w.tokens)*w.columns*4);
            for(size_t i=0;i<w.experts.size();++i) {
                const auto &e=w.experts[i];int32_t row[]={e.token,e.slot,ids[e.slot]};put(row,sizeof(row));
                put(e.gate,weight_bytes(w,0));put(e.up,weight_bytes(w,1));put(e.down,weight_bytes(w,2));
                put(w.output+i*w.columns,size_t(w.columns)*4);
            }
        } catch(...) {std::fclose(f);throw;}
        if(std::fclose(f))throw std::runtime_error("cannot close CPU fixture");
    }
    CpuExpertWork work;
    std::vector<int32_t> ids;
    std::vector<float> input,expected,output;
    std::vector<std::array<std::vector<uint8_t>,3>> weights;
    explicit CpuExpertFixture(const std::filesystem::path &path) {
        const auto size=std::filesystem::file_size(path);
        if(size>limit || size<44)throw std::runtime_error("invalid CPU fixture size");
        FILE *f=std::fopen(path.string().c_str(),"rb");if(!f)throw std::runtime_error("cannot read CPU fixture");
        auto get=[&](void *p,size_t n){if(std::fread(p,1,n,f)!=n)throw std::runtime_error("truncated CPU fixture");};
        try {
            char magic[8];get(magic,8);if(std::memcmp(magic,"SCPU0001",8))throw std::runtime_error("invalid CPU fixture version");
            uint32_t h[9];get(h,sizeof(h));
            if(!h[0] || h[0]>16384 || !h[1] || h[1]>65536 || !h[2] || h[2]>8 || !h[3] || h[3]>8 || !h[4] || h[4]>h[2]*h[3])
                throw std::runtime_error("invalid CPU fixture shape");
            work.columns=h[0];work.hidden=h[1];work.tokens=h[2];work.topk=h[3];std::memcpy(&work.clamp,&h[8],4);
            work.gate_type=ggml_type(h[5]);work.up_type=ggml_type(h[6]);work.down_type=ggml_type(h[7]);
            if(!cpu_lane_format(work.gate_type,work.columns) || !cpu_lane_format(work.up_type,work.columns) || !cpu_lane_format(work.down_type,work.hidden))
                throw std::runtime_error("unsupported CPU fixture quantization");
            work.experts.resize(h[4]);if(bytes(work)!=size)throw std::runtime_error("CPU fixture length mismatch");
            ids.resize(size_t(work.tokens)*work.topk);input.resize(size_t(work.tokens)*work.columns);
            expected.resize(size_t(h[4])*work.columns);output.resize(expected.size());weights.resize(h[4]);
            get(ids.data(),ids.size()*4);get(input.data(),input.size()*4);
            for(size_t i=0;i<weights.size();++i) {
                int32_t row[3];get(row,sizeof(row));
                if(row[0]<0 || row[0]>=work.tokens || row[1]<0 || size_t(row[1])>=ids.size() || row[1]/work.topk!=row[0] || ids[row[1]]!=row[2])
                    throw std::runtime_error("invalid CPU fixture route");
                for(int j=0;j<3;++j){weights[i][j].resize(weight_bytes(work,j));get(weights[i][j].data(),weights[i][j].size());}
                work.experts[i]={weights[i][0].data(),weights[i][1].data(),weights[i][2].data(),row[0],row[1]};
                // Preserve pointer grouping of repeated experts, as in the engine.
                for(size_t k=0;k<i;++k)if(ids[work.experts[k].slot]==row[2]) {
                    if(weights[k]!=weights[i])throw std::runtime_error("CPU fixture repeated expert differs");
                    auto &e=work.experts[i];e.gate=work.experts[k].gate;e.up=work.experts[k].up;e.down=work.experts[k].down;break;
                }
                get(expected.data()+i*work.columns,size_t(work.columns)*4);
            }
            work.input=input.data();work.output=output.data();
        } catch(...) {std::fclose(f);throw;}
        std::fclose(f);
    }
};
}
