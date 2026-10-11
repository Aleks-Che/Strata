#include "stable_stream_k.hpp"
#include <iostream>
#include <stdexcept>
static void require(bool x){if(!x)throw std::runtime_error("stream-K partition mismatch");}
int main(){
    try {
        int cases=0,fallback=0;
        for(int q=1;q<=4;++q)for(int tiles:{1,8,16})for(int kv=1;kv<=64;++kv)for(int capacity=1;capacity<=256;++capacity) {
            // Independent transcription of the pinned one-query launch rule.
            int raw=capacity<kv*tiles?capacity:kv*tiles,rounded=raw-raw%tiles;
            int one=rounded && 100*(raw-rounded)/raw<=5?rounded:raw;
            int actual=strata_ds4::stable_stream_k_blocks(q,tiles,kv,capacity);++cases;
            if(one%tiles){require(actual==0);++fallback;continue;}
            require(actual==q*one);
            int parts=one/tiles;
            for(int tile=0;tile<q*tiles;++tile)for(int part=0;part<parts;++part) {
                int begin=(tile*parts+part)*(kv*q*tiles)/actual-tile*kv;
                int end=(tile*parts+part+1)*(kv*q*tiles)/actual-tile*kv;
                require(begin==part*kv/parts && end==(part+1)*kv/parts);
            }
        }
        require(strata_ds4::stable_stream_k_blocks(0,8,16,170)==0);
        require(strata_ds4::stable_stream_k_blocks(5,8,16,170)==0);
        require(strata_ds4::stable_stream_k_blocks(4096,8,4,170,4096)==4096*32);
        require(strata_ds4::stable_stream_k_blocks(4097,8,4,170,4096)==0);
        for(int queries:{5,371,409,4096})for(int kv:{2,4,16,64})for(int capacity:{32,170,256}) {
            int raw=std::min(capacity,kv*8),rounded=raw/8*8;
            int one=rounded && 100*(raw-rounded)/raw<=5?rounded:raw;
            int blocks=strata_ds4::stable_stream_k_blocks(queries,8,kv,capacity,4096);
            if(one%8){require(blocks==0);continue;}
            require(blocks==queries*one);
            for(int tile:{0,7,queries*8-1})for(int part=0;part<one/8;++part) {
                auto start=int64_t(tile*(one/8)+part)*(kv*queries*8)/blocks-int64_t(tile)*kv;
                auto stop=int64_t(tile*(one/8)+part+1)*(kv*queries*8)/blocks-int64_t(tile)*kv;
                require(start==part*kv/(one/8) && stop==(part+1)*kv/(one/8));
            }
        }
        require(strata_ds4::stable_stream_k_blocks(3,8,16,0)==0);
        require(strata_ds4::stable_stream_k_blocks(4,INT32_MAX,INT32_MAX,INT32_MAX)==0);
        std::cout<<"PASS "<<cases<<" partitions, "<<fallback<<" irregular fallbacks\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
