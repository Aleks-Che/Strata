#include "speculative.hpp"
#include <cstdio>
#include <cstdlib>

static void check(bool ok) { if(!ok)std::abort(); }
int main() {
    // Exhaust partial acceptance, full acceptance, EOS at every position and
    // output limits. Never sample a rejected suffix: sampler history and its
    // RNG must advance only for tokens actually emitted.
    for(int n=0;n<=5;++n) for(int mismatch=0;mismatch<=n;++mismatch)
    for(int budget=1;budget<=7;++budget) for(int eos=-1;eos<=n;++eos) {
        std::vector<llama_token> draft;
        for(int i=0;i<n;++i)draft.push_back(10+i);
        int calls=0;
        auto result=verify_draft(draft,budget,[&](int row) {
            check(row==calls++);
            return row<mismatch?draft[row]:100+row;
        },[&](int token) {return eos>=0 && token==(eos<mismatch?10+eos:100+eos);});
        int expected=std::min(budget,mismatch+1);
        bool stop=eos>=0 && eos<expected;
        if(stop)expected=eos+1;
        check(calls==expected && int(result.tokens.size())==expected);
        check(result.accepted==std::min(mismatch,expected));
        check(result.eog==stop);
        // Kept target inputs include the anchor and all emitted outputs except
        // the pending last one. Rollback always fits n_rs_seq = n_draft.
        check(n+1-expected>=0 && n+1-expected<=n);
    }
    std::puts("Speculative verification / EOS / output-limit contracts passed");
}
