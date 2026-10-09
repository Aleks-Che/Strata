#include "state_bulk.hpp"
#include "runtime.hpp"
#include "synthetic_minimax_m2.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <random>
using namespace minimax_m2;
using json=nlohmann::ordered_json;
struct Info {std::vector<uint8_t> * tensor;uint8_t * ptr;size_t size,offset;};

int main(int argc,char **argv) {
    json report={{"pass",false},{"cpu_cases",0},{"gpu_checks",json::array()}};
    std::filesystem::path dir;
    try {
        require(argc==2,"usage: state-bulk-check NEW_DIRECTORY");dir=argv[1];
        require(!std::filesystem::exists(dir),"directory exists");std::filesystem::create_directories(dir);
        std::mt19937 rng(270029);
        auto get=[](auto *t,void *p,size_t o,size_t n){require(o<=t->size() && n<=t->size()-o,"get bounds");std::memcpy(p,t->data()+o,n);};
        auto set=[](auto *t,const void *p,size_t o,size_t n){require(o<=t->size() && n<=t->size()-o,"set bounds");std::memcpy(t->data()+o,p,n);};
        auto size=[](auto *t){return t->size();};
        size_t grouped=0;
        for(size_t trial=0;trial<1200;++trial) {
            std::vector<uint8_t> tensor(8192), serialized(16384,0xcc), expected=serialized;
            for(auto &b:tensor)b=uint8_t(rng());
            std::vector<Info> infos;size_t o=trial%17,p=3;
            for(size_t j=0;j<64;++j) {
                const size_t n=1+rng()%64;if(o+n>tensor.size())break;
                infos.push_back({&tensor,serialized.data()+p,n,o});p+=n;o+=n+rng()%65;
                if(trial%7==0 && j%9==0)o=0; // overlapping/reordered ranges
                if(trial%11==0)p+=2; // serialized metadata between descriptors
            }
            auto legacy=infos;for(auto &i:legacy)i.ptr=expected.data()+(i.ptr-serialized.data());
            minimax_m2_state::flush<false>(legacy,false,get,set,size);
            auto stats=minimax_m2_state::flush<false>(infos,true,get,set,size,trial%4?1024:0);
            require(serialized==expected,"gather byte parity");grouped+=stats.groups;
            for(auto &b:serialized)b=uint8_t(rng());expected=serialized;
            auto original=tensor;
            minimax_m2_state::flush<true>(legacy,false,get,set,size);auto restored=tensor;tensor=original;
            stats=minimax_m2_state::flush<true>(infos,true,get,set,size,trial%4?1024:0);
            require(tensor==restored && stats.scratch_bytes<=1024,"scatter/gap/overlap parity or cap");
            report["cpu_cases"]=trial+1;
        }
        require(grouped>0,"no grouped transfers exercised");report["cpu_groups"]=grouped;
        minimax_m2_state::totals={};
        // Full serialized KV and logits on the real CUDA graph, with holes and a
        // protected second sequence. Native parsing/truncation semantics stay intact.
        environment();ggml_backend_load_all();strata_mm27_mode(1);
        struct Release {~Release(){strata_mm27_release();}} release;
        const auto path=(dir/"fixture.gguf").string();write_synthetic_minimax_m2(path,true);
        auto model=load(path,false,true);
        auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=16;cp.n_seq_max=2;
        cp.n_threads=cp.n_threads_batch=4;cp.type_k=cp.type_v=GGML_TYPE_F32;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=cp.op_offload=true;
        cp.kv_unified=true;
        Context ctx(llama_init_from_model(model.get(),cp),llama_free);require(bool(ctx),"context");clear(ctx.get());
        auto capture=[&](int seq,bool bulk) {
            _putenv_s("STRATA_MM27_STATE_BULK",bulk?"1":"0");
            std::vector<uint8_t> data(llama_state_seq_get_size(ctx.get(),seq));
            require(llama_state_seq_get_data(ctx.get(),data.data(),data.size(),seq)==data.size(),"state get");return data;
        };
        auto check=[&](const std::string &name,bool ok){report["gpu_checks"].push_back({{"name",name},{"pass",ok}});require(ok,name);};
        std::vector<llama_token> tokens(240);for(size_t i=0;i<tokens.size();++i)tokens[i]=11+i%32;
        for(size_t i=0;i<tokens.size();i+=16)decode(ctx.get(),tokens,i,int(std::min(size_t(16),tokens.size()-i)),int(i));
        const auto dense=capture(0,false);check("dense-state-bytes",dense==capture(0,true));
        decode(ctx.get(),{17},0,1,240);std::vector<float> logits(llama_get_logits_ith(ctx.get(),-1),llama_get_logits_ith(ctx.get(),-1)+64);
        clear(ctx.get());_putenv_s("STRATA_MM27_STATE_BULK","1");
        check("dense-restore-size",llama_state_seq_set_data(ctx.get(),dense.data(),dense.size(),0)==dense.size());
        check("dense-restored-bytes",capture(0,false)==dense);
        decode(ctx.get(),{17},0,1,240);
        check("continuation-logits",std::memcmp(logits.data(),llama_get_logits_ith(ctx.get(),-1),64*sizeof(float))==0);
        auto *mem=llama_get_memory(ctx.get());llama_memory_seq_cp(mem,0,1,0,-1);
        for(int p=5;p<240;p+=9)require(llama_memory_seq_rm(mem,0,p,p+3),"remove holes");
        const auto protected_seq=capture(1,false),fragmented=capture(0,false);
        check("fragmented-state-bytes",fragmented==capture(0,true));
        require(llama_memory_seq_rm(mem,0,0,-1),"remove seq0");_putenv_s("STRATA_MM27_STATE_BULK","1");
        check("fragmented-restore-size",llama_state_seq_set_data(ctx.get(),fragmented.data(),fragmented.size(),0)==fragmented.size());
        check("fragmented-restored-bytes",capture(0,false)==fragmented);
        check("other-sequence-preserved",capture(1,false)==protected_seq);
        clear(ctx.get());_putenv_s("STRATA_MM27_STATE_BULK","1");
        check("truncated-state-rejected",llama_state_seq_set_data(ctx.get(),dense.data(),dense.size()-1,0)==0);
        clear(ctx.get());check("after-truncation-restore",llama_state_seq_set_data(ctx.get(),dense.data(),dense.size(),0)==dense.size());
        check("after-truncation-bytes",capture(0,false)==dense);
        const auto stats=minimax_m2_state::totals;
        check("bulk-gpu-path-covered",stats.groups>0 && stats.scratch_bytes==minimax_m2_state::scratch_limit);
        report["gpu_io"]={{"gets",stats.gets},{"sets",stats.sets},{"bulk_groups",stats.groups},{"max_scratch_bytes",stats.scratch_bytes}};
        report["state_bytes"]=dense.size();report["pass"]=true;
    } catch(const std::exception &e) {report["error"]=e.what();}
    if(!dir.empty() && std::filesystem::exists(dir))std::ofstream(dir/"report.json")<<report.dump(2)<<'\n';
    std::cout<<report.dump(2)<<'\n';return report["pass"].get<bool>()?0:2;
}
