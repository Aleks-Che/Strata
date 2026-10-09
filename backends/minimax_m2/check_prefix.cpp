#include "prefix.hpp"
#include <iostream>
#include <numeric>
#include <stdexcept>
using namespace minimax_m2;
int main() {
    size_t checks=0;
    auto expect=[&](bool ok) {++checks;if(!ok)throw std::runtime_error("prefix check "+std::to_string(checks));};
    ResidentPrefix p;
    for(size_t batch:{size_t(1),size_t(8),size_t(16)}) {
        for(size_t n=1;n<=65;++n) {
            std::vector<int32_t> ids(n);std::iota(ids.begin(),ids.end(),1);
            p.invalidate();expect(p.reusable("a",ids,batch)==0);
            p.commit("a",ids,batch,n+7);
            expect(p.computed==n+7 && p.prompt.size()==n/batch*batch);
            expect(p.reusable("a",ids,batch)==(n-1)/batch*batch);
            expect(p.reusable("b",ids,batch)==0 && p.reusable("",ids,batch)==0);
            expect(p.reusable("a",ids,batch+1)==0);
            auto longer=ids;longer.insert(longer.end(),32,999);
            expect(p.reusable("a",longer,batch)==n/batch*batch);
            for(size_t cut=0;cut<n;++cut) {
                auto branch=ids;branch[cut]=999;
                expect(p.reusable("a",branch,batch)==cut/batch*batch);
                branch.resize(cut+1);
                expect(p.reusable("a",branch,batch)==cut/batch*batch);
            }
            p.invalidate();expect(p.reusable("a",ids,batch)==0);
            p.commit("",ids,batch,n);expect(p.prompt.empty() && p.computed==0);
        }
    }
    std::cout<<"{\"pass\":true,\"checks\":"<<checks<<"}\n";
}
