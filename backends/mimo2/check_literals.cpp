// CPU-only allocation/latency comparison of the old and literal guard overloads.
#include "contract.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
static size_t allocations=0;
static bool tracking=false;
void *operator new(size_t n) {if(tracking)++allocations;if(auto p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void *p) noexcept {std::free(p);}
void operator delete(void *p,size_t) noexcept {std::free(p);}
int main() {
    try {
        size_t checks=0;
        for(bool fast:{false,true}) {
            for(float value:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()}) {
                bool rejected=false;
                try {if(fast)mimo2::require(std::isfinite(value),"non-finite logits");else mimo2::require(std::isfinite(value),std::string("non-finite logits"));}
                catch(const std::runtime_error &e) {rejected=std::string(e.what())=="non-finite logits";}
                mimo2::require(rejected,"guard lost non-finite rejection");++checks;
            }
        }
        bool rejected=false;try {mimo2::require(false,std::string("dynamic ")+"error");}catch(const std::runtime_error &e){rejected=std::string(e.what())=="dynamic error";}
        mimo2::require(rejected,"dynamic message changed");++checks;
        std::vector<float> values(152576);
        for(size_t i=0;i<values.size();++i)values[i]=float(int(i%199)-99)*.01f;
        values[0]=std::numeric_limits<float>::max();values[1]=std::numeric_limits<float>::denorm_min();
        for(bool fast:{false,true,true,false}) {
            allocations=0;tracking=true;const auto start=std::chrono::steady_clock::now();
            for(int row=0;row<100;++row)for(float value:values) {
                if(fast)mimo2::require(std::isfinite(value),"non-finite logits");
                else mimo2::require(std::isfinite(value),std::string("non-finite logits"));
            }
            const auto end=std::chrono::steady_clock::now();tracking=false;
            // Other standard libraries may keep this string inline; the fast
            // path must allocate nothing on every library.
            mimo2::require(allocations<=100*values.size() && (!fast || !allocations),"unexpected guard allocations");++checks;
            std::cout<<"MODE "<<fast<<" allocations "<<allocations<<" ms_per_row "<<std::chrono::duration<double,std::milli>(end-start).count()/100<<'\n';
        }
        std::cout<<"PASS "<<checks<<" literal guard cases\n";return 0;
    }catch(const std::exception &e) {tracking=false;std::cerr<<e.what()<<'\n';return 1;}
}
