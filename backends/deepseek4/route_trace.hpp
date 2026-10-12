#pragma once
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <mutex>

// Shadow observations only: no prediction changes a route or starts I/O.
// The scheduler has already synchronized these IDs for its ordinary copies.
inline void strata_route_trace(ggml_backend_t owner,const ggml_tensor *weight,
        const ggml_tensor *route,const int32_t *ids) {
    struct Writer {
        FILE *file=nullptr;
        std::mutex mutex;
        size_t rows=0;
        Writer() {
            if(const char *path=std::getenv("STRATA_DS4_ROUTE_TRACE")) {
                file=std::fopen(path,"wb");
                if(!file)GGML_ABORT("Cannot open STRATA_DS4_ROUTE_TRACE");
            }
        }
        ~Writer(){if(file)std::fclose(file);}
    };
    static Writer writer;
    if(!writer.file)return;
    std::lock_guard<std::mutex> lock(writer.mutex);
    if(writer.rows++>=200000)return;
    // Model tensor names contain no user text. Avoid escaping assumptions.
    char name[GGML_MAX_NAME];size_t i=0;
    for(;i<sizeof(name)-1 && weight->name[i];++i) {
        const char c=weight->name[i];name[i]=(c=='"' || c=='\\' || c<32)?'_':c;
    }
    name[i]=0;
    const bool prefill=route->ne[1]>8;
    const int64_t first=prefill?route->ne[1]-1:0;
    std::fprintf(writer.file,"{\"owner\":\"%p\",\"matrix\":\"%s\",\"hash\":%s,\"tokens\":%lld,\"prefill\":%s,\"ids\":[",
        (void *)owner,name,route->op==GGML_OP_GET_ROWS?"true":"false",(long long)route->ne[1],prefill?"true":"false");
    for(int64_t t=first;t<route->ne[1];++t) {
        std::fputs(t==first?"[":",[",writer.file);
        for(int64_t k=0;k<route->ne[0];++k)
            std::fprintf(writer.file,k?",%d":"%d",ids[t*route->nb[1]/4+k*route->nb[0]/4]);
        std::fputs("]",writer.file);
    }
    std::fputs("]}\n",writer.file);
}
