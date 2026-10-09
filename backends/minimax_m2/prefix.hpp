#pragma once
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace minimax_m2 {
// One resident sequence, never an archive. Only full prefill batches are
// reusable: serially decoded outputs and a partial final batch may have used
// different arithmetic from a fresh batched prefill. Always re-evaluate the
// final prompt batch so its logits and graph shape match the fresh request.
struct ResidentPrefix {
    std::string key;
    std::vector<int32_t> prompt;
    size_t computed = 0;
    size_t batch = 0;

    void invalidate() {key.clear();prompt.clear();computed=0;batch=0;}
    size_t reusable(const std::string &session,const std::vector<int32_t> &tokens,size_t width) const {
        if(session.empty() || session!=key || width==0 || width!=batch || tokens.empty())return 0;
        size_t common=0;
        while(common<std::min(prompt.size(),tokens.size()) && prompt[common]==tokens[common])++common;
        return std::min(common/width,(tokens.size()-1)/width)*width;
    }
    void commit(const std::string &session,const std::vector<int32_t> &tokens,size_t width,size_t kv_tokens) {
        invalidate();
        if(session.empty())return;
        key=session;batch=width;computed=kv_tokens;
        prompt.assign(tokens.begin(),tokens.begin()+(tokens.size()/width)*width);
    }
};
} // namespace minimax_m2
