#pragma once
#include <cstdlib>
#include <cstring>
#include <set>
#include <tuple>
static int strata_projection_splits() {
    const char *e=std::getenv("STRATA_DS4_PROJECTION_SPLITS");
    return e && !std::strcmp(e,"4")?4:1;
}
static int strata_projection_tile() {
    const char *e=std::getenv("STRATA_DS4_PROJECTION_TILE");
    if(e && !std::strcmp(e,"32"))return 32;
    if(e && !std::strcmp(e,"64"))return 64;
    return 8;
}
static void strata_projection_grid_trace(const mmq_args &a,int j,int i,int blocks,int tiles,bool fixed) {
    const char *e=std::getenv("STRATA_DS4_PROJECTION_TRACE");
    if(!e || std::strcmp(e,"1") || a.type_x!=GGML_TYPE_Q8_0 ||
       !((a.ncols_x==4096 && (a.nrows_x==512 || a.nrows_x==1024)) || (a.ncols_x==1024 && a.nrows_x==32768) ||
         (a.ncols_x==8192 && a.nrows_x==4096)) ||
       a.ids_dst || (a.nchannels_y!=1 && a.nchannels_y!=8) || a.nsamples_y!=1)return;
    using Key=std::tuple<int64_t,int64_t,int64_t,int,int,int,int,bool>;
    static thread_local std::set<Key> seen;
    if(seen.size()>=128 || !seen.emplace(a.nrows_x,a.ncols_dst,a.nchannels_y,j,i,blocks,tiles,fixed).second)return;
    GGML_LOG_INFO("strata_projection_mmq_grid m=%lld tokens=%lld channels=%lld J=%d I=%d blocks=%d tiles=%d fixup=%d fixed=%d\n",
        (long long)a.nrows_x,(long long)a.ncols_dst,(long long)a.nchannels_y,j,i,blocks,tiles,int(tiles%blocks!=0),int(fixed));
}
