#include "draft_probe.hpp"
#include "spec_verify.hpp"
#include "nlohmann/json.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
using namespace mimo2;
using json=nlohmann::ordered_json;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t) {return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
int main(int argc,char **argv) {
    try {
        std::string model_path,draft_path,kind="none";bool shared_target=false,head_columns=false,memory_stages=false;size_t cache_mib=14336;
        for(int i=1;i<argc;++i) {std::string k=argv[i];require(i+1<argc,"missing option");std::string v=argv[++i];
            if(k=="--model")model_path=v;else if(k=="--draft")draft_path=v;else if(k=="--kind")kind=v;
            else if(k=="--expert-cache-mib") {size_t end=0;cache_mib=std::stoull(v,&end);require(end==v.size() && cache_mib>0 && cache_mib<=14336,"invalid cache limit");}
            else if(k=="--target-head-columns" && (v=="0" || v=="1"))head_columns=v=="1";
            else if(k=="--memory-stages" && (v=="0" || v=="1"))memory_stages=v=="1";
            else if(k=="--share-target" && (v=="0" || v=="1"))shared_target=v=="1";else throw std::runtime_error("unknown option "+k);}
        require(!model_path.empty() && (kind=="none" || kind=="mtp" || kind=="dflash") && ((kind=="none")==draft_path.empty()),"invalid probe inputs");
        require(!shared_target || kind=="dflash","shared target is a DFlash experiment");
        _putenv_s("STRATA_MIMO_DFLASH_SHARE_TARGET",shared_target?"1":"0");
        _putenv_s("STRATA_MIMO_TARGET_HEAD_COLUMNS",head_columns?"1":"0");
        environment();ggml_backend_load_all();strata_mimo_mode(2);
        {
            json stages=json::array();
            auto stage=[&](const char *name) {if(memory_stages) {
                strata_mimo_memory();const auto s=strata_mimo_snapshot();
                stages.push_back({{"name",name},{"gpu_free",s.gpu_free},{"gpu_total",s.gpu_total},
                    {"ram_free",s.ram_free},{"ram_total",s.ram_total},{"cache_bytes",s.cache_bytes}});
            }};
            stage("before_load");auto model=load(model_path);stage("target_weights");
            auto ctx=probe_context(model.get());stage("target_context");
            std::unique_ptr<DraftMask> mask;
            Model dm(nullptr,llama_model_free);Context dc(nullptr,llama_free);std::unique_ptr<DraftProbe> draft;
            if(kind!="none") {
                dm=probe_load_draft(draft_path,kind=="mtp");
                stage("draft_weights");
                if(shared_target) {require(!dm->tok_embd && !dm->output,"shared weights were not skipped");mask=std::make_unique<DraftMask>(draft_path);dm->tok_embd=mask->tensor;}
                dc=probe_context(dm.get(),kind=="mtp",shared_target?ctx.get():nullptr);draft=std::make_unique<DraftProbe>(ctx.get(),dc.get(),kind=="mtp");
                stage("draft_context");
            }
            // Lazy CUDA pools for batched verification differ from last-logit
            // prefill. Warm all-logit batch8 BEFORE the expert cache can fill.
            // Discard both KV states afterwards; this is load time, not throughput.
            strata_mimo_reader(true);strata_mimo_phase(true);
            std::vector<llama_token> warm(8,11);
            bool verify_warmed=bool(draft);
            if(draft) {
                if(memory_stages) {
                    decode(ctx.get(),warm,0,8,0);stage("target_last_logit_prefill8");clear(ctx.get());
                    decode(ctx.get(),warm,0,2,0,true);stage("target_all_logit_verify2");clear(ctx.get());
                }
                decode(ctx.get(),warm,0,8,0,true);
                stage("target_all_logit_warmup8");
                draft->process(warm,0);draft->propose(11,8,kind=="mtp"?1:7,0);draft->reset();
                stage("draft_warmup");
            }
            clear(ctx.get());strata_mimo_cache(cache_mib<<20);strata_mimo_cache_prefill(false);strata_mimo_pipeline_config(1,8);
            std::cout<<"READY\n"<<std::flush;
            std::string line;
            while(std::getline(std::cin,line) && line!="QUIT") {
                auto req=json::parse(line);auto prompt=req.at("tokens").get<std::vector<llama_token>>();
                // Diagnostic upper bound: known baseline proposals, no draft graph or features.
                // Still verify every token with the target; never emit this array directly.
                auto oracle=req.value("oracle_ids",std::vector<llama_token>{});
                const int predict=req.value("predict",32),depth=req.value("depth",0);const float pmin=req.value("p_min",0.0f);
                require(!prompt.empty() && predict>0 && predict<=128 && prompt.size()+predict+8<=480 && depth>=0 && depth<=7 &&
                    (!depth || draft || !oracle.empty()) && (kind!="mtp" || depth<=1) && pmin>=0 && pmin<=1 &&
                    (oracle.empty() || (kind=="none" && int(oracle.size())==predict)),"invalid probe request");
                const int nv=llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
                for(auto id:prompt)require(id>=0 && id<nv,"invalid prompt ID");
                for(auto id:oracle)require(id>=0 && id<nv,"invalid oracle ID");
                if(!oracle.empty() && !verify_warmed) {
                    // Oracle diagnostics also need batched logits. Normal no-draft
                    // requests must retain the memory saved by never using them.
                    require(strata_mimo_snapshot().cache_bytes==0,"oracle diagnostic must be the first request");
                    strata_mimo_phase(true);decode(ctx.get(),warm,0,8,0,true);verify_warmed=true;
                }
                clear(ctx.get());if(draft)draft->reset();strata_mimo_reset();strata_mimo_phase(true);
                double target_ms=0,draft_ms=0,catchup_ms=0;auto start=Clock::now();
                for(size_t i=0;i<prompt.size();i+=8) {
                    std::vector<llama_token> ids(prompt.begin()+i,prompt.begin()+std::min(prompt.size(),i+8));
                    decode(ctx.get(),ids,0,int(ids.size()),int(i));
                    if(draft)draft->process(ids,int(i));
                }
                const double prefill_ms=ms(start);const auto prefill_stats=strata_mimo_snapshot();strata_mimo_phase(false);
                std::ofstream logits;
                const std::string path=req.value("logits",std::string());
                if(!path.empty()) {require(std::filesystem::path(path).extension()==".f32" && !std::filesystem::exists(path),"fresh .f32 required");logits.open(path,std::ios::binary);require(bool(logits),"cannot write logits");}
                auto sample=[&](int row) {auto token=draft_greedy(ctx.get(),row);
                    if(logits.is_open()) {logits.write(reinterpret_cast<const char *>(llama_get_logits_ith(ctx.get(),row)),size_t(nv)*4);require(bool(logits),"logits write failed");}return token;};
                auto generation=Clock::now();std::vector<llama_token> out{sample(-1)};int pos=int(prompt.size()),proposed=0,accepted=0,cycles=0;
                std::vector<int> proposed_counts,accepted_counts;
                while(int(out.size())<predict && !is_stop(out.back())) {
                    auto tick=Clock::now();const int n=std::min(depth,predict-int(out.size())-1);
                    auto proposals=draft?draft->propose(out.back(),pos,n,pmin):std::vector<llama_token>{};draft_ms+=ms(tick);
                    if(!oracle.empty())proposals.assign(oracle.begin()+out.size(),oracle.begin()+out.size()+n);
                    std::vector<llama_token> input{out.back()};input.insert(input.end(),proposals.begin(),proposals.end());
                    tick=Clock::now();decode(ctx.get(),input,0,int(input.size()),pos,true);target_ms+=ms(tick);
                    auto verified=verify_greedy(proposals,predict-int(out.size()),sample,is_stop);
                    out.insert(out.end(),verified.tokens.begin(),verified.tokens.end());
                    const int matched=verified.accepted,keep=verified.keep;input.resize(keep);
                    tick=Clock::now();require(llama_memory_seq_rm(llama_get_memory(ctx.get()),0,pos+keep,-1),"target rollback failed");
                    if(draft)draft->process(input,pos,kind=="mtp" && n>0);catchup_ms+=ms(tick);
                    pos+=keep;proposed+=int(proposals.size());accepted+=matched;++cycles;
                    proposed_counts.push_back(int(proposals.size()));accepted_counts.push_back(matched);
                }
                const double generation_ms=ms(generation);auto s=strata_mimo_snapshot();
                require(!s.rejected_cpu_nodes && !s.rejected_full_copies && !s.pipeline_reader_owned && !s.pipeline_queued,"probe GPU/drain audit failed");
                std::cout<<json({{"name",req.value("name",std::string())},{"kind",oracle.empty()?kind:"oracle"},{"shared_target",shared_target},{"depth",depth},{"p_min",pmin},{"ids",out},
                    {"prefill_ms",prefill_ms},{"generation_ms",generation_ms},{"tokens_per_second",1000.0*(out.size()-1)/generation_ms},
                    {"target_ms",target_ms},{"draft_ms",draft_ms},{"catchup_ms",catchup_ms},{"proposed",proposed},{"accepted",accepted},
                    {"cycles",cycles},{"proposed_counts",proposed_counts},{"accepted_counts",accepted_counts},{"cache_bytes",s.cache_bytes},
                    {"target_head_columns",head_columns},{"memory_stages",stages},
                    {"cache_payload_bytes",s.cache_payload_bytes},{"cache_limit",s.cache_limit},{"decode_h2d_bytes",s.h2d_bytes-prefill_stats.h2d_bytes},
                    {"cache_request_mib",cache_mib},{"cache_slab_mib",s.cache_slab_mib},{"cache_decay",s.cache_decay},
                    {"pipeline_batch",s.pipeline_batch},{"pipeline_packed_guards",s.pipeline_packed_guards},
                    {"pipeline_d2d_batch",s.pipeline_d2d_batch},{"pipeline_d2d_batches",s.pipeline_d2d_batches},
                    {"pipeline_d2d_kernel_launches",s.pipeline_d2d_kernel_launches},{"cache_fill_batch",s.cache_fill_batch},
                    {"pipeline_early_host_refill",s.pipeline_early_host_refill},{"verify_warmup_tokens",verify_warmed?8:0},
                    {"pipeline_consumer_wait_us",s.pipeline_consumer_wait_us},{"pipeline_submit_us",s.pipeline_submit_us},
                    {"gpu_free",s.gpu_free},{"gpu_total",s.gpu_total},{"ram_free",s.ram_free},{"ram_total",s.ram_total},
                    {"gpu_nodes",s.gpu_nodes},{"rejected_cpu_nodes",s.rejected_cpu_nodes},{"pipeline_queued",s.pipeline_queued}}).dump()<<'\n'<<std::flush;
            }
            strata_mimo_release();
        }
        llama_backend_free();return 0;
    } catch(const std::exception &e) {strata_mimo_release();std::cerr<<"MiMo spec probe: "<<e.what()<<'\n';return 1;}
}
