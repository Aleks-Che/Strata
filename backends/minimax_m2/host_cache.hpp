#pragma once
#include "../common/expert_file.hpp"
#include <algorithm>
#include <cstring>
#include <functional>
#include <list>
#include <tuple>

namespace minimax_m2 {
// Read-only file views, not private copies or locked pages. A hit means view
// reuse; Windows can page its contents out. Charged bytes include map alignment.
class HostCache {
public:
    struct Memory {uint64_t total,available,commit_available;};
    using Probe=std::function<Memory()>;
    struct Stats {
        uint64_t bytes=0,budget=0,cap=0,entries=0,readers=0,peak_bytes=0;
        uint64_t hits=0,misses=0,mapped_bytes=0,file_bytes=0,admissions=0;
        uint64_t evictions=0,gpu_drops=0,gpu_drop_bytes=0,prefill_bypasses=0,rejected=0;
        uint64_t gpu_waits=0,history_entries=0;
    };
    static Memory memory() {
        MEMORYSTATUSEX m{};m.dwLength=sizeof(m);
        if(!GlobalMemoryStatusEx(&m))throw std::runtime_error("MiniMax RAM cache memory sample failed");
        return {m.ullTotalPhys,m.ullAvailPhys,m.ullAvailPageFile};
    }
    static uint64_t allowance(Memory m,uint64_t owned,uint64_t cap) {
        if(!m.total || m.available>m.total)throw std::runtime_error("invalid MiniMax RAM cache memory sample");
        // 94% target plus margin; the runtime independently enforces global95.
        const uint64_t reserve=m.total/100*6+(256ull<<20);
        if(m.commit_available<(256ull<<20))return 0; // mappings still need metadata
        if(m.available>=reserve)return std::min(cap,owned+std::min(cap,m.available-reserve));
        return std::min(cap,owned-std::min(owned,reserve-m.available));
    }
private:
    using Key=std::tuple<uintptr_t,uint64_t,size_t>;
    using Queue=std::list<Key>;
    struct Mapping {
        std::shared_ptr<strata_expert_file::Source> source;
        HANDLE handle=nullptr;
        explicit Mapping(std::shared_ptr<strata_expert_file::Source> s):source(std::move(s)) {
            handle=CreateFileMappingW(source->file,nullptr,PAGE_READONLY,0,0,nullptr);
            if(!handle)throw std::system_error(GetLastError(),std::system_category(),"MiniMax RAM file mapping");
        }
        ~Mapping(){if(handle)CloseHandle(handle);}
    };
    struct View {
        std::shared_ptr<Mapping> mapping;
        void *base=nullptr;
        size_t delta,charged;
        View(std::shared_ptr<Mapping> m,uint64_t offset,size_t bytes,size_t prefix,size_t charge)
            :mapping(std::move(m)),delta(prefix),charged(charge) {
            const auto start=offset-delta;
            base=MapViewOfFile(mapping->handle,FILE_MAP_READ,DWORD(start>>32),DWORD(start),delta+bytes);
            if(!base)throw std::system_error(GetLastError(),std::system_category(),"MiniMax RAM file view");
        }
        ~View(){if(base)UnmapViewOfFile(base);}
    };
    struct Entry {std::shared_ptr<View> view;Queue::iterator position;bool retired=false;};
    std::mutex mutex;
    Probe probe;
    uint64_t cap;
    size_t granularity,page;
    Stats stats;
    bool prefill=true;
    bool gpu_partition=false;
    std::map<Key,Entry> entries;
    Queue recent;
    std::map<uintptr_t,std::weak_ptr<Mapping>> mappings;
    std::map<Key,std::weak_ptr<strata_expert_file::Source>> gpu_bypassed;
    static constexpr size_t max_chunk=16u<<20,max_entries=65536;

    void erase(typename std::map<Key,Entry>::iterator it) {
        stats.bytes-=it->second.view->charged;recent.erase(it->second.position);entries.erase(it);++stats.evictions;
    }
    void trim(uint64_t target) {
        auto it=recent.end();
        while(it!=recent.begin() && stats.bytes>target) {
            --it;auto found=entries.find(*it);
            if(found->second.view.use_count()!=1)continue;
            auto next=std::next(it);erase(found);it=next;
        }
    }
    void refresh_locked() {stats.budget=allowance(probe(),stats.bytes,cap);trim(stats.budget);}
    bool eligible(const Key &key) const {
        if(!gpu_partition)return true;
        const auto id=std::get<0>(key),offset=std::get<1>(key),bytes=std::get<2>(key);
        auto it=gpu_bypassed.upper_bound(Key{id,offset,SIZE_MAX});
        if(it==gpu_bypassed.begin())return false;
        --it;const auto start=std::get<1>(it->first),length=std::get<2>(it->first);
        return std::get<0>(it->first)==id && !it->second.expired() && offset>=start && offset-start<=length && bytes<=length-(offset-start);
    }
    void unpin(const Key &key,std::shared_ptr<View> &view) {
        std::lock_guard<std::mutex> lock(mutex);view.reset();--stats.readers;
        auto it=entries.find(key);
        if(it!=entries.end() && it->second.retired && it->second.view.use_count()==1)erase(it);
        trim(stats.budget);
    }
    // Keep SEH outside C++ unwinding scopes. An in-page I/O error must become a
    // runtime error so the owner drains its pipeline before releasing views.
    static bool copy(void *dst,const void *src,size_t n) {
        __try {std::memcpy(dst,src,n);return true;}
        __except(GetExceptionCode()==EXCEPTION_IN_PAGE_ERROR?EXCEPTION_EXECUTE_HANDLER:EXCEPTION_CONTINUE_SEARCH) {return false;}
    }
public:
    explicit HostCache(uint64_t limit,Probe sample=memory):probe(std::move(sample)),cap(limit) {
        SYSTEM_INFO info;GetSystemInfo(&info);granularity=info.dwAllocationGranularity;page=info.dwPageSize;
        // Paging existing views out must never cause the configured capacity
        // to grow beyond the initial physical headroom.
        cap=allowance(probe(),0,cap);refresh_locked();stats.cap=cap;
    }
    HostCache(const HostCache&)=delete;
    void set_prefill(bool value) {std::lock_guard<std::mutex> lock(mutex);prefill=value;}
    void set_gpu_partition(bool value) {std::lock_guard<std::mutex> lock(mutex);gpu_partition=value;gpu_bypassed.clear();}
    // Record actual GPU admission refusal. Mapping on the next miss avoids
    // allocating then immediately retiring views for every successful GPU fill.
    void gpu_bypass(const std::shared_ptr<strata_expert_file::Source> &source,uint64_t offset,size_t bytes) {
        if(!source || !source->contains(offset,bytes) || bytes>max_chunk)
            throw std::runtime_error("invalid MiniMax GPU bypass range");
        std::lock_guard<std::mutex> lock(mutex);
        const Key key{uintptr_t(source.get()),offset,bytes};
        if(!gpu_bypassed.count(key) && gpu_bypassed.size()>=max_entries)gpu_bypassed.clear();
        gpu_bypassed[key]=source;
    }
    void refresh() {std::lock_guard<std::mutex> lock(mutex);refresh_locked();}
    Stats snapshot() {std::lock_guard<std::mutex> lock(mutex);auto s=stats;s.entries=entries.size();s.history_entries=gpu_bypassed.size();return s;}
    void reset_counters() {
        std::lock_guard<std::mutex> lock(mutex);const auto s=stats;stats={};
        stats.bytes=s.bytes;stats.budget=s.budget;stats.cap=cap;stats.readers=s.readers;stats.peak_bytes=s.bytes;
    }
    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        for(auto &e:entries)e.second.retired=true;
        trim(0);mappings.clear();gpu_bypassed.clear();
    }
    // Called only after the GPU copy is complete. A producer holding a view
    // keeps it alive until its last memcpy ends; retired entries cannot hit.
    void drop_gpu(const std::shared_ptr<strata_expert_file::Source> &source,uint64_t offset,size_t bytes) {
        if(!source || !source->contains(offset,bytes))throw std::runtime_error("invalid MiniMax GPU/RAM partition range");
        std::lock_guard<std::mutex> lock(mutex);const auto id=uintptr_t(source.get());
        auto history=gpu_bypassed.lower_bound(Key{id,offset-std::min<uint64_t>(offset,max_chunk),0});
        while(history!=gpu_bypassed.end() && std::get<0>(history->first)==id && std::get<1>(history->first)<offset+bytes) {
            if(std::get<1>(history->first)+std::get<2>(history->first)>offset)history=gpu_bypassed.erase(history);
            else ++history;
        }
        auto it=entries.lower_bound(Key{id,offset-std::min<uint64_t>(offset,max_chunk),0});
        while(it!=entries.end() && std::get<0>(it->first)==id && std::get<1>(it->first)<offset+bytes) {
            auto next=std::next(it);const auto start=std::get<1>(it->first),n=std::get<2>(it->first);
            if(start+n>offset && !it->second.retired) {
                it->second.retired=true;++stats.gpu_drops;stats.gpu_drop_bytes+=it->second.view->charged;
                if(it->second.view.use_count()==1)erase(it);
            }
            it=next;
        }
    }
    bool read(const std::shared_ptr<strata_expert_file::Source> &source,uint64_t offset,void *destination,
              size_t bytes,const std::atomic<bool> &cancel,strata_expert_file::Request &request) {
        if(!source || !source->contains(offset,bytes) || bytes>max_chunk || (bytes && !destination))
            throw std::runtime_error("invalid MiniMax RAM cache read");
        if(cancel.load())return false;
        if(!bytes)return true;
        const Key key{uintptr_t(source.get()),offset,bytes};std::shared_ptr<View> view;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto found=entries.find(key);
            if(found!=entries.end() && !found->second.retired) {
                view=found->second.view;++stats.hits;
                if(!prefill)recent.splice(recent.begin(),recent,found->second.position);
            } else {
                ++stats.misses;
                if(prefill)++stats.prefill_bypasses;
                else if(!eligible(key))++stats.gpu_waits;
                else if(found==entries.end()) {
                    refresh_locked();
                    const auto prefix=size_t(offset%granularity),charge=(prefix+bytes+page-1)/page*page;
                    if(charge<=stats.budget) {
                        trim(stats.budget-charge);
                        if(stats.bytes<=stats.budget-charge && entries.size()<max_entries) {
                            auto &weak=mappings[uintptr_t(source.get())];auto mapping=weak.lock();
                            if(!mapping){mapping=std::make_shared<Mapping>(source);weak=mapping;}
                            view=std::make_shared<View>(mapping,offset,bytes,prefix,charge);
                            recent.push_front(key);
                            try {entries.emplace(key,Entry{view,recent.begin()});}
                            catch(...) {recent.pop_front();throw;}
                            stats.bytes+=charge;++stats.admissions;stats.peak_bytes=std::max(stats.peak_bytes,stats.bytes);
                        }
                    }
                }
                if(!view)++stats.rejected;
            }
            if(view)++stats.readers;
        }
        if(!view) {
            const bool ok=request.read_at(*source,offset,destination,bytes,cancel);
            if(ok){std::lock_guard<std::mutex> lock(mutex);stats.file_bytes+=bytes;}
            return ok;
        }
        const bool ok=copy(destination,static_cast<const char *>(view->base)+view->delta,bytes);
        unpin(key,view);
        if(!ok) {clear();throw std::runtime_error("MiniMax mapped expert read failed (in-page I/O)");}
        {std::lock_guard<std::mutex> lock(mutex);stats.mapped_bytes+=bytes;}
        return !cancel.load();
    }
};
} // namespace minimax_m2
