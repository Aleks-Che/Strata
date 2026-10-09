#pragma once
#include "sessions.hpp"
#include "runtime.hpp"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace minimax_m2 {
// Lifetime and identity are structural: one immutable model/quant, context,
// batch and F32 KV configuration; no disk format, import, or cross-engine use.
class SessionRuntime {
    llama_context *const ctx;
    uint64_t reserve=0;
    uint64_t available() {
        MEMORYSTATUSEX m{};m.dwLength=sizeof(m);
        require(GlobalMemoryStatusEx(&m)!=0,"cannot sample session archive RAM");
        reserve=m.ullTotalPhys/20+(256ull<<20);
        // A physical-memory check alone is insufficient when commit is full.
        return std::min<uint64_t>(m.ullAvailPhys,m.ullAvailPageFile);
    }
public:
    SessionArchive archive;
    size_t saved_bytes=0, restored_bytes=0;
    SessionRuntime(llama_context *context,size_t cap,size_t count):ctx(context),archive(cap,count) {}
    void trim() {
        if(!archive.cap)return;
        available();archive.trim(reserve,[&]{return available();});
    }
    void reset_transfer() {saved_bytes=restored_bytes=0;}
    bool switch_to(ResidentPrefix &resident,const std::string &session,const std::vector<int32_t> &tokens) {
        reset_transfer();trim();
        if(!archive.cap || resident.key==session)return false;
        const auto *target=archive.find(session);
        const bool eligible=target && target->prefix.reusable(session,tokens,llama_n_batch(ctx))>0;
        const std::string protect=eligible?session:std::string();
        if(!resident.key.empty() && !resident.prompt.empty()) {
            llama_synchronize(ctx);
            require(llama_memory_seq_pos_min(llama_get_memory(ctx),0)==0 &&
                    llama_memory_seq_pos_max(llama_get_memory(ctx),0)==int(resident.computed)-1,"snapshot KV positions mismatch");
            const auto bytes=llama_state_seq_get_size(ctx,0);
            require(bytes>0,"cannot size session snapshot");
            available();
            if(archive.room(SessionArchive::charge_for(resident,bytes),reserve,protect,[&]{return available();}) &&
               archive.save(resident,bytes,[&](uint8_t *data,size_t size){return llama_state_seq_get_data(ctx,data,size,0)==size;}))
                saved_bytes=bytes;
        }
        // Re-find after possible evictions. Restored entries are consumed;
        // cancelled/failed requests must not leave a checkpoint for this key.
        target=archive.find(session);
        if(!target || !eligible){archive.erase(session);return false;}
        require(!session.empty(),"anonymous archive restore");
        clear(ctx);
        const bool ok=llama_state_seq_set_data(ctx,target->data.get(),target->size,0)==target->size;
        if(ok){resident=target->prefix;restored_bytes=target->size;++archive.restores;}
        archive.erase(session);
        require(ok,"cannot restore session snapshot");
        return true;
    }
};
} // namespace minimax_m2
