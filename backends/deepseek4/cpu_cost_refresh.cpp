#include "cpu_cost_refresh.hpp"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace strata_ds4 {
CpuCostModel CpuCostRefresh::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);return current_;
}
void CpuCostRefresh::begin(bool eligible) {
    std::lock_guard<std::mutex> lock(mutex_);
    // A missing end, excluded request or abort breaks the drift streak.
    for(auto &entry:cells_) {auto &s=entry.second;if(active_ || !eligible)s.direction=0;s.count=0;}
    active_=eligible && current_.enabled;
}
void CpuCostRefresh::observe(uint64_t generation,int tokens,int pattern,bool copying,double us) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!active_ || generation!=current_.generation || tokens<1 || tokens>4 || pattern<1 || pattern>=cpu_cost_patterns ||
       !std::isfinite(us) || us<=0)return;
    const bool loaded=current_.contention && copying;
    const float idle=current_.compute[tokens-1][pattern],busy=current_.loaded_compute[tokens-1][pattern];
    const float predicted=loaded?std::max(idle,busy):idle;
    if(idle<=0 || (loaded && busy<=0) || predicted<=0)return; // Never invent unmeasured cells.
    auto &s=cells_[{tokens-1,pattern,loaded}];s.values[s.count++%s.values.size()]=float(us/predicted);
}
CpuCostRefresh::Result CpuCostRefresh::finish(bool success,const Writer &write) {
    std::lock_guard<std::mutex> lock(mutex_);
    Result result;result.generation=current_.generation;
    CpuCostModel next=current_;
    for(auto &entry:cells_) {
        const auto [t,k,loaded]=entry.first;auto &s=entry.second;
        result.samples+=s.count;
        if(!active_ || !success || s.count<8) {s.direction=0;s.count=0;continue;}
        auto values=s.values;const auto n=size_t(std::min<uint64_t>(s.count,values.size()));
        std::sort(values.begin(),values.begin()+n);
        const float ratio=values[n/2];
        // Require >=75% of the recent samples to agree, in two consecutive
        // successful requests. A single slow callback cannot trigger a rebuild.
        const int direction=values[n/4]>1.2f?1:values[(3*n)/4]<0.8f?-1:0;
        if(direction && direction==s.direction) {
            const float old=current_.compute[t][k];
            const float predicted=loaded?std::max(old,current_.loaded_compute[t][k]):old;
            const float baseline=loaded?std::max(base_.compute[t][k],base_.loaded_compute[t][k]):base_.compute[t][k];
            float value=std::clamp(predicted*std::clamp(ratio,0.8f,1.25f),baseline*0.5f,baseline*2.f);
            if(loaded)value=std::max(value,old); // Loaded data cannot lower the idle cost.
            if(std::abs(value-predicted)>predicted*0.001f) {
                (loaded?next.loaded_compute[t][k]:next.compute[t][k])=value;++result.changed;
            }
            s.direction=0;
        } else s.direction=direction;
        s.count=0;
    }
    active_=false;
    if(result.changed) {
        next.generation=current_.generation+1;
        try {if(write)write(next);current_=next;result.generation=next.generation;}
        catch(const std::exception &e){result.error=e.what();result.changed=0;}
        catch(...){result.error="snapshot write failed";result.changed=0;}
    }
    return result;
}
bool cpu_cost_auto_enabled() {
    static const bool enabled=[] {const char *s=std::getenv("STRATA_DS4_CPU_COST_AUTO");return s && std::string(s)=="1";}();
    return enabled;
}
namespace {
using json=nlohmann::json;
struct Entry {std::shared_ptr<CpuCostRefresh> profile;json identity;size_t id;};
struct Registry {
    std::mutex mutex;std::map<std::string,Entry> profiles;
    bool active=false,eligible=false;uint64_t request=0;
    const std::string run=std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
};
Registry &registry(){static Registry r;return r;}
void save(const Entry &e,const CpuCostModel &m,const Registry &r) {
    const char *dir=std::getenv("STRATA_DS4_CPU_COST_AUTO_DIR");if(!dir || !*dir)return;
    // Session-local effective tables, for inspection only. A subsequent process
    // must validate its original hardware calibration and learn its own timing.
    json j={{"kind","strata-cpu-runtime-cost"},{"version",1},{"identity",e.identity},
        {"generation",m.generation},{"request",r.request},{"compute",m.compute},{"loaded_compute",m.loaded_compute},
        {"boundary",m.boundary},{"gpu_compute",m.gpu_compute},{"copy_us_per_byte",m.copy_us_per_byte},
        {"loaded_copy_us_per_byte",m.loaded_copy_us_per_byte},{"gpu_overlap",m.gpu_overlap},
        {"budget_us_per_token",m.budget_us_per_token},{"margin",m.margin}};
    const auto folder=std::filesystem::path(dir);std::filesystem::create_directories(folder);
    const auto path=folder/(r.run+"-"+std::to_string(e.id)+"-g"+std::to_string(m.generation)+".json");
    auto temp=path;temp+=".tmp";
    try {
        std::ofstream f(temp,std::ios::binary|std::ios::trunc);f<<j.dump(2)<<'\n';f.close();
        if(!f)throw std::runtime_error("cannot write runtime cost snapshot");
        std::filesystem::rename(temp,path);
    } catch(...) {std::error_code ec;std::filesystem::remove(temp,ec);throw;}
}
}
std::shared_ptr<CpuCostRefresh> cpu_cost_refresh_for(const std::string &path,const CpuExpertWork &w,
        int threads,const std::string &gpu,const CpuCostModel &base) {
    if(!cpu_cost_auto_enabled() || !base.enabled)return {};
    json identity={{"source",path},{"build",cpu_cost_build_id()},{"machine",cpu_cost_machine()},
        {"gpu",gpu},{"threads",threads},{"mode",cpu_cost_mode()},{"gate",int(w.gate_type)},
        {"up",int(w.up_type)},{"down",int(w.down_type)},{"columns",w.columns},{"hidden",w.hidden},
        {"clamp",w.clamp},{"topk",w.topk},{"burst",base.burst},{"contention",base.contention},
        {"gpu_enabled",base.gpu_enabled},{"budget",base.budget_us_per_token}};
    auto &r=registry();std::lock_guard<std::mutex> lock(r.mutex);const auto key=identity.dump();
    auto it=r.profiles.find(key);
    if(it==r.profiles.end()) {
        auto profile=std::make_shared<CpuCostRefresh>(base);profile->begin(r.active && r.eligible);
        it=r.profiles.emplace(key,Entry{profile,std::move(identity),r.profiles.size()}).first;
    }
    return it->second.profile;
}
void cpu_cost_request_boundary(int operation) {
    if(!cpu_cost_auto_enabled())return;
    auto &r=registry();std::lock_guard<std::mutex> lock(r.mutex);
    if(operation==0 || operation==1) {
        ++r.request;r.active=true;r.eligible=operation==0;
        for(auto &e:r.profiles)e.second.profile->begin(r.eligible);
        std::fprintf(stderr,"STRATA_CPU_COST_REQUEST begin=%llu eligible=%d\n",(unsigned long long)r.request,int(r.eligible));
    } else if(r.active) {
        const auto start=std::chrono::steady_clock::now();
        for(auto &e:r.profiles) {
            const auto result=e.second.profile->finish(operation==2 && r.eligible,[&](const auto &m){save(e.second,m,r);});
            const auto current=e.second.profile->snapshot();
            std::fprintf(stderr,"STRATA_CPU_COST_REFRESH request=%llu profile=%zu success=%d generation=%llu changed=%d samples=%llu error=%s\n",
                (unsigned long long)r.request,e.second.id,int(operation==2 && r.eligible),(unsigned long long)current.generation,
                result.changed,(unsigned long long)result.samples,result.error.empty()?"none":result.error.c_str());
        }
        r.active=false;r.eligible=false;
        std::fprintf(stderr,"STRATA_CPU_COST_REQUEST end=%llu maintenance_us=%lld\n",(unsigned long long)r.request,
            (long long)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count());
    }
}
}
