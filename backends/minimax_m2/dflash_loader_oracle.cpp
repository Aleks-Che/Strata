#include "dflash_contract.hpp"
#include "nlohmann/json.hpp"
#include <iostream>
using namespace minimax_m2;
using json=nlohmann::ordered_json;
int main(int argc,char **argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--version") {
            std::cout<<"MiniMax DFlash no-allocation loader "<<STRATA_MM27_SOURCE_SHA<<'\n';return 0;
        }
        require(argc==3,"usage: dflash-loader TARGET.gguf DRAFT.gguf");
        inspect(argv[1]);const auto bytes=inspect_dflash(argv[1],argv[2]);
        auto mp=llama_model_default_params();mp.no_alloc=true;mp.n_gpu_layers=0;
        mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_OFF;mp.use_extra_bufts=false;mp.load_mtp=false;
        std::unique_ptr<llama_model,decltype(&llama_model_free)> m(llama_model_load_from_file(argv[2],mp),llama_model_free);
        require(bool(m),"draft registration failed");registered_dflash(*m);
        uint64_t registered=0;json tensors=json::array();
        for(const auto &[name,t]:m->tensors_by_name) {
            require(!t->data && (!t->buffer || ggml_backend_buffer_get_size(t->buffer)==0),"unexpected payload allocation");
            registered+=ggml_nbytes(t);tensors.push_back({{"name",name},{"shape",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}},
                {"type_id",int(t->type)},{"bytes",ggml_nbytes(t)}});
        }
        require(tensors.size()==58 && registered==bytes,"registration payload count differs");
        std::cout<<json({{"pass",true},{"source_revision",STRATA_MM27_SOURCE_SHA},{"target",argv[1]},{"draft",argv[2]},
            {"scope","header contract and actual no-allocation loader; no inference"},{"allocated_weight_bytes",0},
            {"logical_bytes",bytes},{"target_layers",m->target_layer_ids},{"rope_dim",m->hparams.n_rot()},
            {"rope_freq_scale",m->hparams.rope_freq_scale_train},{"causal_default",m->hparams.causal_attn},
            {"draft_vocab",dflash_vocab},{"target_vocab",target_vocab},{"tensors",tensors}}).dump(2)<<'\n';return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
