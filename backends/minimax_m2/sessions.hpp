#pragma once
#include "prefix.hpp"
#include <list>
#include <memory>
#include <limits>
#include <stdexcept>

namespace minimax_m2 {
// Private, in-process CPU blobs. Never import a state or share this store
// between contexts/models. The resident GPU sequence is outside these limits.
class SessionArchive {
public:
    struct Entry {
        ResidentPrefix prefix;
        std::unique_ptr<uint8_t[]> data;
        size_t size=0, charge=0;
    };
    const size_t cap, max_entries;
    size_t used=0, evictions=0, rejected=0, saves=0, restores=0;
    SessionArchive(size_t bytes,size_t count):cap(bytes),max_entries(count) {}
    SessionArchive(const SessionArchive &)=delete;
    SessionArchive &operator=(const SessionArchive &)=delete;
    size_t count() const {return entries.size();}
    static size_t charge_for(const ResidentPrefix &prefix,size_t size) {
        // Include token metadata and a conservative allowance for key/list/
        // allocator overhead; the blob itself has exactly size bytes.
        const size_t overhead=sizeof(Entry)+256+prefix.prompt.size()*sizeof(int32_t);
        return size>std::numeric_limits<size_t>::max()-overhead?std::numeric_limits<size_t>::max():size+overhead;
    }
    const Entry *find(const std::string &key) const {
        for(const auto &e:entries)if(e.prefix.key==key)return &e;
        return nullptr;
    }
    void erase(const std::string &key) {
        for(auto i=entries.begin();i!=entries.end();++i)if(i->prefix.key==key){used-=i->charge;entries.erase(i);return;}
    }
    bool evict(const std::string &protect={}) {
        for(auto i=entries.begin();i!=entries.end();++i)if(i->prefix.key!=protect){used-=i->charge;entries.erase(i);++evictions;return true;}
        return false;
    }
    // read_available is queried after every eviction: released virtual bytes
    // need not equal newly available physical RAM (a blob may be paged out).
    template<class Available> bool room(size_t charge,size_t reserve,const std::string &protect,Available available) {
        if(!max_entries || charge>cap){++rejected;return false;}
        while(used>cap-charge || count()>=max_entries)if(!evict(protect)){++rejected;return false;}
        if(charge>std::numeric_limits<size_t>::max()-reserve){++rejected;return false;}
        while(available()<charge+reserve)if(!evict(protect)){++rejected;return false;}
        return true;
    }
    template<class Available> void trim(size_t reserve,Available available) {
        while(available()<reserve && evict()) {}
    }
    template<class Write> bool save(const ResidentPrefix &prefix,size_t size,Write write) {
        const auto charge=charge_for(prefix,size);
        if(!size || prefix.key.empty() || find(prefix.key) || !max_entries || count()>=max_entries || charge>cap || used>cap-charge)
            throw std::logic_error("session archive admission mismatch");
        try {
            Entry e;
            e.prefix.key=prefix.key;e.prefix.batch=prefix.batch;e.prefix.computed=prefix.computed;
            e.prefix.prompt.assign(prefix.prompt.begin(),prefix.prompt.end());
            e.data=std::make_unique<uint8_t[]>(size);e.size=size;e.charge=charge;
            if(!write(e.data.get(),size)){++rejected;return false;}
            entries.push_back(std::move(e));used+=charge;++saves;return true;
        } catch(const std::bad_alloc &) {++rejected;return false;}
    }
private:
    std::list<Entry> entries; // oldest inactive checkpoint first
};
} // namespace minimax_m2
