#include "cpu_lane.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <array>
using namespace strata_ds4;
static void check(bool b,const char *s){if(!b)throw std::runtime_error(s);}
int main()try {
    int batches=0;
    for(int threads:{1,4,12,16})for(int spin:{0,50,200,2000}) {
        CpuLanePool pool(threads,{spin,2});
        for(int iteration=0;iteration<160;++iteration) {
            const int n=iteration%5==0?0:iteration%5==1?1:iteration%5==2?3:iteration%5==3?32:67;
            std::array<std::atomic<int>,67> seen{};
            auto job=[&](int i){check(i>=0 && i<n,"invalid job index");++seen[i];if(i%13==0)std::this_thread::yield();};
            if(iteration%2){pool.submit(n,job);pool.wait();}else pool.run(n,job);
            for(int i=0;i<n;++i)check(seen[i]==1,"lost or repeated job");
            if(iteration%19==0)std::this_thread::sleep_for(std::chrono::milliseconds(3));
            ++batches;
        }
        // Error propagation and teardown while workers may still be spinning.
        bool caught=false;try{pool.run(32,[](int i){if(i==7)throw std::runtime_error("worker error");});}
        catch(const std::runtime_error &){caught=true;}check(caught,"worker exception lost");
    }
    std::cout<<"CPU pool: "<<batches<<" alternating batches, sleeping/spinning wakeup, exact-once and exception teardown passed\n";
    return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
