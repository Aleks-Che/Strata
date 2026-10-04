#pragma once
#include "nlohmann/json.hpp"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <stdexcept>
#include <tuple>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace strata_glm {
struct WarmExpert {size_t tensor;int expert;unsigned score;};
struct WarmProfile {
    std::string status="missing";
    bool writable=true;
    std::vector<WarmExpert> entries;
};
// Metadata only: no weights, addresses, generation ids, tokens or model state.
// Geometry and file identity must match before any hint reaches the runtime.
inline WarmProfile read_warm_profile(const std::filesystem::path &path,const nlohmann::json &identity,
                                     const std::vector<size_t> &expert_counts) {
    WarmProfile result;
    try {
        if(!std::filesystem::exists(path))return result;
        result.writable=false;
        if(std::filesystem::file_size(path)>8*1024*1024)throw std::runtime_error("profile too large");
        std::ifstream input(path,std::ios::binary);
        auto data=nlohmann::json::parse(input);
        if(data.at("version")!=1 || data.at("identity")!=identity) {result.status="mismatch";return result;}
        const auto &rows=data.at("entries");
        if(!rows.is_array() || rows.size()>131072)throw std::runtime_error("invalid profile rows");
        std::set<std::pair<size_t,int>> seen;
        for(const auto &row:rows) {
            if(!row.is_array() || row.size()!=3)throw std::runtime_error("invalid profile row");
            for(const auto &number:row)if(!number.is_number_integer())throw std::runtime_error("invalid profile integer");
            const auto t=row[0].get<int64_t>(),e=row[1].get<int64_t>(),s=row[2].get<int64_t>();
            if(t<0 || size_t(t)>=expert_counts.size() || e<0 || uint64_t(e)>=expert_counts[size_t(t)] ||
               e>INT32_MAX || s<1 || s>255 || !seen.insert({size_t(t),int(e)}).second)
                throw std::runtime_error("profile bounds/duplicate");
            result.entries.push_back({size_t(t),int(e),unsigned(s)});
        }
        std::sort(result.entries.begin(),result.entries.end(),[](const auto &a,const auto &b) {
            if(a.score!=b.score)return a.score>b.score;
            return std::tie(a.tensor,a.expert)<std::tie(b.tensor,b.expert);
        });
        result.status="loaded";result.writable=true;
    }catch(const std::exception &) {result.status="invalid";result.writable=false;result.entries.clear();}
    return result;
}
inline void write_warm_profile(const std::filesystem::path &path,const nlohmann::json &identity,
                               const std::vector<WarmExpert> &entries) {
    if(entries.size()>131072)throw std::runtime_error("too many warm entries");
    nlohmann::json rows=nlohmann::json::array();
    for(const auto &item:entries)rows.push_back({item.tensor,item.expert,item.score});
    const auto text=nlohmann::json({{"version",1},{"identity",identity},{"entries",rows}}).dump()+"\n";
    if(text.size()>8*1024*1024)throw std::runtime_error("warm profile too large");
    static std::atomic<uint64_t> sequence{0};
#ifdef _WIN32
    const auto pid=GetCurrentProcessId();
#else
    const auto pid=getpid();
#endif
    auto temporary=path;temporary+="."+std::to_string(pid)+"."+std::to_string(++sequence)+".tmp";
    try {
        {std::ofstream output(temporary,std::ios::binary|std::ios::trunc);output<<text;output.flush();
         if(!output)throw std::runtime_error("write warm profile failed");}
#ifdef _WIN32
        if(!MoveFileExW(temporary.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("replace warm profile failed");
#else
        std::filesystem::rename(temporary,path);
#endif
    }catch(...) {std::error_code ignored;std::filesystem::remove(temporary,ignored);throw;}
}
}
