#pragma once
#include <algorithm>
#include <cstdint>

namespace strata_ds4 {
// Mirror the pinned stream-K choice for one query. Replication is exact only
// when every output tile receives the same integer number of CUDA blocks.
// Return zero for irregular partitions: the caller must use separate queries.
inline int stable_stream_k_blocks(int queries,int tiles_per_query,int kv_tiles,int max_blocks,int query_limit=4) {
    if(queries<1 || queries>query_limit || query_limit>4096 || tiles_per_query<1 || kv_tiles<1 || max_blocks<1)return 0;
    const int64_t raw=std::min<int64_t>(max_blocks,int64_t(kv_tiles)*tiles_per_query);
    const int64_t rounded=raw/tiles_per_query*tiles_per_query;
    const int loss=rounded>0?int(100*(raw-rounded)/raw):100;
    const int64_t one=loss<=5?rounded:raw;
    if(one%tiles_per_query!=0 || one*queries>INT32_MAX)return 0;
    return int(one*queries);
}
}
