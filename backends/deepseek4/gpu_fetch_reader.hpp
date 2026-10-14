#pragma once
#include "expert_file.hpp"
#include <cstdlib>
#include <cstring>

namespace strata_ds4 {
// One instance belongs to the fetch service. A borrowed view is only valid
// while its model and the immutable branch snapshot are alive. finish_plan()
// drains that service before context/model teardown. Source alone owns only
// the file handle and is NOT a lifetime token for the view.
class GpuFetchReader {
    strata_expert_file::Request reused;
public:
    enum Mode { native, reuse, resident, mapped };
    Mode mode=native;
    uint64_t file_bytes=0,mmap_bytes=0,resident_checks=0,resident_misses=0;
    GpuFetchReader() {
        const char *v=std::getenv("STRATA_DS4_GPU_FETCH_READER");
        if(v && std::strcmp(v,"reuse")==0)mode=reuse;
        if(v && std::strcmp(v,"resident")==0)mode=resident;
        if(v && std::strcmp(v,"mapped")==0)mode=mapped;
    }
    const char *name() const {return mode==reuse?"reuse":mode==resident?"resident":mode==mapped?"mapped":"native";}
    bool read(strata_expert_file::Request &local,const strata_expert_file::Source &source,
              uint64_t offset,const void *view,void *dest,size_t bytes,const std::atomic<bool> &cancel) {
        if(!source.contains(offset,bytes))throw std::runtime_error("GPU fetch read outside source");
        if(cancel.load())return false;
        if(!bytes)return true;
        if(!dest)throw std::runtime_error("null GPU fetch destination");
        bool use_view=false;
        if(view && (mode==resident || mode==mapped)) {
            if(source.offset_of(view,bytes)!=offset)throw std::runtime_error("GPU fetch view offset mismatch");
            if(mode==resident) {
                ++resident_checks;use_view=strata_expert_file::resident(view,bytes);
                if(!use_view)++resident_misses;
            } else use_view=true;
        }
        if(use_view) {
            // Residency is a hint, not a lock. Windows can evict a checked page.
            // Unlike native reads, page faults inside memcpy cannot be cancelled
            // or given a timeout. Both mmap modes are explicit experiments.
            std::memcpy(dest,view,bytes);mmap_bytes+=bytes;
            return !cancel.load();
        }
        const bool ok=(mode==native?local:reused).read_at(source,offset,dest,bytes,cancel,1500);
        if(ok)file_bytes+=bytes;
        return ok;
    }
};
}
