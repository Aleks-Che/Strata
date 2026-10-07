#pragma once
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace mimo2 {
// This offline context uses one contiguous sequence and 256-cell KV padding.
// Crossing a padded KV extent inside one verification batch changes FA's
// reduction geometry for its early queries compared with single-token decode.
inline int bounded_proposal_count(int depth, int remaining, int pos, bool tokenwise) {
    if(depth<0 || remaining<=0 || pos<0) throw std::invalid_argument("invalid verification range");
    const int limit=std::min(depth,remaining-1);
    return tokenwise ? std::min(limit,255-pos%256) : limit;
}
struct VerifiedPrefix {
    std::vector<int> tokens;
    int accepted = 0;
    int keep = 0;
};

// Only target samples may reach output. After the first mismatch, discard the
// rest of the speculative batch. Keep anchor + accepted inputs in target KV.
template<class Sample, class Stop>
VerifiedPrefix verify_greedy(const std::vector<int> &proposals, int remaining, Sample sample, Stop stop) {
    if (remaining <= 0) throw std::invalid_argument("no output budget");
    VerifiedPrefix result;
    for (int row=0; row<=int(proposals.size()) && int(result.tokens.size())<remaining; ++row) {
        const int token=sample(row);
        result.tokens.push_back(token);
        const bool match=row<int(proposals.size()) && token==proposals[row];
        result.accepted+=match;
        if (!match || stop(token)) break;
    }
    result.keep=std::min(int(proposals.size())+1,result.accepted+1);
    return result;
}
}
