#pragma once
#include <cuda_runtime.h>
#include <string>
inline std::string strata_cpu_cost_gpu(int device) {
    cudaDeviceProp p{};if(cudaGetDeviceProperties(&p,device)!=cudaSuccess)return "unavailable";
    int driver=0;cudaDriverGetVersion(&driver);
    std::string result=std::string(p.name)+"/sm="+std::to_string(p.major)+std::to_string(p.minor)+"/driver="+std::to_string(driver)+"/uuid=";
    const char *hex="0123456789abcdef";for(unsigned char b:p.uuid.bytes){result+=hex[b>>4];result+=hex[b&15];}return result;
}
