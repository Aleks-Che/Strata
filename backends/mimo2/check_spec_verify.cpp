#include "spec_verify.hpp"
#include <iostream>
using namespace mimo2;
int main() {
    int checks=0;
    auto check=[&](std::vector<int> proposals,std::vector<int> target,int budget,
                   std::vector<int> expected,int accepted,int keep) {
        int reads=0;
        auto result=verify_greedy(proposals,budget,[&](int row) {++reads;return target.at(row);},[](int t) {return t==99;});
        if(result.tokens!=expected || result.accepted!=accepted || result.keep!=keep || reads!=int(expected.size()))
            throw std::runtime_error("wrong verified prefix");
        ++checks;
    };
    check({}, {4}, 5, {4}, 0, 1);
    check({1,2,3}, {8}, 8, {8}, 0, 1); // no unverified suffix reads
    check({1,2,3}, {1,8}, 8, {1,8}, 1, 2);
    check({1,2,3}, {1,2,8}, 8, {1,2,8}, 2, 3);
    check({1,2,3}, {1,2,3,4}, 8, {1,2,3,4}, 3, 4);
    check({1,99,3}, {1,99}, 8, {1,99}, 2, 3);
    check({1,2,3}, {1,99}, 8, {1,99}, 1, 2);
    check({1,2,3}, {1}, 1, {1}, 1, 2);
    check({1,2,3}, {1,2}, 2, {1,2}, 2, 3);
    bool rejected=false;
    try {verify_greedy({},0,[](int) {return 0;},[](int) {return false;});}
    catch(const std::invalid_argument &) {rejected=true;}
    if(!rejected)throw std::runtime_error("zero budget accepted");
    ++checks;
    for(int pos: {0,127,248,249,254,255,256,511}) {
        for(int depth: {0,1,7}) {
            for(int remaining: {1,2,8,32}) {
                const int n=bounded_proposal_count(depth,remaining,pos,true);
                if(n<0 || n>depth || n>=remaining || pos/256!=(pos+n)/256 ||
                    (n<depth && n<remaining-1 && (pos+n)%256!=255) ||
                    bounded_proposal_count(depth,remaining,pos,false)!=std::min(depth,remaining-1))
                    throw std::runtime_error("wrong KV boundary limit");
                ++checks;
            }
        }
    }
    std::cout << checks << " verified-prefix/boundary checks PASS\n";
}
