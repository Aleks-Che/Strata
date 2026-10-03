#pragma once
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <io.h>
#endif

// Only mappings created by the private llama_mmap hook are registered. Their
// file offset is zero; tensor pointers can then be translated without guessing
// paths or offsets from virtual-memory regions. Requests retain the file handle.
namespace strata_expert_file {
struct Source {
    uintptr_t base;
    size_t bytes;
    Source(const Source&)=delete;
    Source &operator=(const Source&)=delete;
#ifdef _WIN32
    HANDLE file=INVALID_HANDLE_VALUE;
    Source(const void *ptr,size_t size,int fd):base(uintptr_t(ptr)),bytes(size) {
        file=ReOpenFile((HANDLE)_get_osfhandle(fd),GENERIC_READ,
            FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,FILE_FLAG_OVERLAPPED|FILE_FLAG_RANDOM_ACCESS);
        if(file==INVALID_HANDLE_VALUE)throw std::system_error(GetLastError(),std::system_category(),"expert ReOpenFile");
    }
    ~Source() {if(file!=INVALID_HANDLE_VALUE)CloseHandle(file);}
#endif
};
struct Registry {std::mutex mutex;std::map<uintptr_t,std::shared_ptr<Source>> files;};
inline Registry &registry() {static Registry value;return value;}
inline void add(const void *ptr,size_t bytes,int fd) {
#ifdef _WIN32
    auto source=std::make_shared<Source>(ptr,bytes,fd);
    auto &r=registry();std::lock_guard<std::mutex> lock(r.mutex);
    r.files.emplace(uintptr_t(ptr),std::move(source));
#else
    (void)ptr;(void)bytes;(void)fd;
#endif
}
inline void remove(const void *ptr) {
    auto &r=registry();std::lock_guard<std::mutex> lock(r.mutex);r.files.erase(uintptr_t(ptr));
}
inline std::shared_ptr<Source> find(const void *ptr,size_t bytes) {
    auto &r=registry();std::lock_guard<std::mutex> lock(r.mutex);
    auto it=r.files.upper_bound(uintptr_t(ptr));
    if(it==r.files.begin())return {};
    auto source=(--it)->second;
    size_t offset=uintptr_t(ptr)-source->base;
    return offset<=source->bytes && bytes<=source->bytes-offset?source:nullptr;
}
inline bool resident(const void *ptr,size_t bytes) {
#ifdef _WIN32
    static const size_t page=[] {SYSTEM_INFO info;GetSystemInfo(&info);return size_t(info.dwPageSize);}();
    uintptr_t first=uintptr_t(ptr)&~(uintptr_t(page)-1);
    size_t count=(uintptr_t(ptr)+bytes-first+page-1)/page;
    thread_local std::vector<PSAPI_WORKING_SET_EX_INFORMATION> pages;
    pages.resize(count);
    for(size_t i=0;i<count;++i)pages[i].VirtualAddress=(void *)(first+i*page);
    if(!QueryWorkingSetEx(GetCurrentProcess(),pages.data(),DWORD(count*sizeof(pages[0]))))return false;
    for(const auto &p:pages)if(!p.VirtualAttributes.Valid)return false;
    return true;
#else
    (void)ptr;(void)bytes;return true;
#endif
}

// One request object per reader. Each owns a distinct OVERLAPPED and event.
// Cancellation is followed by completion before reusing the pinned buffer.
class Request {
#ifdef _WIN32
    HANDLE event=nullptr;
#endif
public:
    Request()=default;
    Request(const Request&)=delete;
    Request &operator=(const Request&)=delete;
    ~Request() {
#ifdef _WIN32
        if(event)CloseHandle(event);
#endif
    }
    bool read(const Source &source,const void *ptr,void *dest,size_t bytes,const std::atomic<bool> &cancel) {
#ifdef _WIN32
        if(cancel.load())return false;
        if(!event)event=CreateEventW(nullptr,TRUE,FALSE,nullptr);
        if(!event)throw std::system_error(GetLastError(),std::system_category(),"expert read event");
        ResetEvent(event);
        uint64_t offset=uintptr_t(ptr)-source.base;
        OVERLAPPED op{};op.hEvent=event;op.Offset=DWORD(offset);op.OffsetHigh=DWORD(offset>>32);
        if(bytes>MAXDWORD || offset>source.bytes || bytes>source.bytes-offset)
            throw std::runtime_error("expert read outside registered mapping");
        DWORD done=0;
        if(!ReadFile(source.file,dest,DWORD(bytes),nullptr,&op)) {
            DWORD error=GetLastError();
            if(error!=ERROR_IO_PENDING)throw std::system_error(error,std::system_category(),"expert ReadFile");
        }
        while(WaitForSingleObject(event,20)==WAIT_TIMEOUT) {
            if(cancel.load()) {CancelIoEx(source.file,&op);break;}
        }
        if(!GetOverlappedResult(source.file,&op,&done,TRUE)) {
            DWORD error=GetLastError();
            if(error==ERROR_OPERATION_ABORTED && cancel.load())return false;
            throw std::system_error(error,std::system_category(),"expert read completion");
        }
        if(done!=bytes)throw std::runtime_error("short expert file read");
        return true;
#else
        (void)source;(void)ptr;(void)dest;(void)bytes;(void)cancel;
        throw std::runtime_error("native expert file queue requires Windows");
#endif
    }
};
}
