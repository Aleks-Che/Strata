#pragma once
#include "runtime.hpp"
#include <filesystem>
#include <map>

namespace step35 {
// Register actual split-file offsets, not assumptions about layer-to-shard order.
// The model must remain alive and immutable until cache release/begin.
inline void register_cache(llama_model * model,const std::string & path,size_t cap) {
    strata_step_cache_begin(std::filesystem::weakly_canonical(path).string().c_str(),cap);
    std::map<std::string,const ggml_tensor *> tensors;
    for (auto & item:model->tensors_by_name) if (expert(item.first)) tensors.emplace(item.first,item.second);
    using Header=std::unique_ptr<gguf_context,decltype(&gguf_free)>;
    Header first(gguf_init_from_file(path.c_str(),{true,nullptr}),gguf_free);
    require(bool(first),"cache registry cannot open first GGUF");
    const auto split=gguf_find_key(first.get(),"split.count");
    const int count=split>=0?gguf_get_val_u16(first.get(),split):1;
    require(count>0 && count<=1024,"invalid Step shard count");
    std::vector<char> prefix(path.size()+64),name(path.size()+64);
    if (count>1) require(llama_split_prefix(prefix.data(),prefix.size(),path.c_str(),0,count)>0,"cache registry requires first shard");
    for (int i=0;i<count;++i) {
        std::string shard=path;
        if (count>1) {
            require(llama_split_path(name.data(),name.size(),prefix.data(),i,count)>0,"cannot resolve Step shard");
            shard=name.data();
        }
        Header header(gguf_init_from_file(shard.c_str(),{true,nullptr}),gguf_free);
        require(bool(header),"cache registry cannot open shard");
        const auto file_bytes=std::filesystem::file_size(shard);
        for (int64_t j=0;j<gguf_get_n_tensors(header.get());++j) {
            const std::string tensor_name=gguf_get_tensor_name(header.get(),j);
            if (!expert(tensor_name)) continue;
            auto found=tensors.find(tensor_name);
            require(found!=tensors.end(),"duplicate or unexpected Step expert in shards");
            const auto offset=gguf_get_data_offset(header.get())+gguf_get_tensor_offset(header.get(),j);
            require(offset<=file_bytes && ggml_nbytes(found->second)<=file_bytes-offset,"Step expert outside shard");
            strata_step_cache_register(found->second,std::filesystem::weakly_canonical(shard).string().c_str(),offset);
            tensors.erase(found);
        }
    }
    require(tensors.empty(),"Step expert missing from shard registry");
}
} // namespace step35
