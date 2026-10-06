#include "mtp.hpp"
#include "synthetic_hy3.hpp"
#include "llama-context.h"
#include "llama-kv-cache.h"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
using namespace hy3;
using json=nlohmann::ordered_json;
struct Output {std::vector<llama_token> ids;std::vector<float> logits;Mtp::Counters counts;bool cancelled=false;int first_accepted=-1;};
// Full-model target-only oracle, with no MTP weights or draft state. The fixed
// English continuation is verified in batches 1..4 to separate batch rounding
// from errors in speculative token/hidden alignment or KV repair.
static int teacher_forced(const std::string & file,const std::string & reference,const std::filesystem::path & directory) {
    json report={{"status","error"},{"runs",json::array()}};
    try {
        require(!std::filesystem::exists(directory),"output directory already exists");std::filesystem::create_directories(directory);
        json ref;std::ifstream(reference)>>ref;require(ref["status"]=="pass","passing reference required");
        std::vector<llama_token> prompt,ids;
        for(const auto & p:ref["prompts"]) if(p["name"]=="en") prompt=p["ids"].get<std::vector<llama_token>>();
        for(const auto & r:ref["runs"][1]["requests"]) if(r["name"]=="en") ids=r["ids"].get<std::vector<llama_token>>();
        require(!prompt.empty() && ids.size()>4,"English prompt/reference missing");
        environment();ggml_backend_load_all();strata_hy3_sync_mode(2);
        {
            auto model=load(file);auto ctx=context(model.get(),2048,17);
            const int vocab=llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
            report["prompt_ids"]=prompt;report["expected_ids"]=ids;
            for(int batch:{1,2,3,4}) {
                clear(ctx.get());configure_cache(model.get(),size_t(8192)<<20);strata_hy3_pipeline_config(2,4);
                std::ofstream out(directory/("batch"+std::to_string(batch)+".f32"),std::ios::binary);
                std::vector<llama_token> predicted;
                auto capture=[&](int row) {
                    const auto * l=llama_get_logits_ith(ctx.get(),row);require(l,"missing teacher logits");
                    out.write(reinterpret_cast<const char *>(l),vocab*sizeof(float));require(bool(out),"teacher logits write failed");
                    predicted.push_back(llama_token(std::max_element(l,l+vocab)-l));
                };
                for(size_t i=0;i<prompt.size();i+=17) decode(ctx.get(),prompt,i,int(std::min(size_t(17),prompt.size()-i)),int(i));
                capture(-1);
                for(size_t i=0;i+1<ids.size();i+=batch) {
                    int n=int(std::min(size_t(batch),ids.size()-1-i));
                    decode(ctx.get(),ids,i,n,int(prompt.size()+i),true);for(int j=0;j<n;++j) capture(j);
                }
                report["runs"].push_back({{"batch",batch},{"ids",predicted},{"ids_exact",predicted==ids}});
                require(predicted==ids,"target-only batching changed greedy IDs");
            }
            strata_hy3_cache_begin(0);
        }
        strata_hy3_sync_release();llama_backend_free();report["status"]="pass";
    } catch(const std::exception & e) {report["error"]=e.what();strata_hy3_sync_release();llama_backend_free();}
    std::ofstream(directory/"teacher-report.json")<<report.dump(2)<<'\n';
    return report["status"]=="pass"?0:1;
}
int main(int argc,char ** argv) {
    if(argc==5 && std::string(argv[1])=="--teacher-forced") return teacher_forced(argv[2],argv[3],argv[4]);
    json report={{"status","error"},{"checks",json::array()}};
    std::filesystem::path directory=argc>1 ? argv[1] : "";
    try {
        require(!directory.empty(),"fixture directory required");std::filesystem::create_directories(directory);
        environment();ggml_backend_load_all();strata_hy3_sync_mode(2);
        strata_hy3_gpu_cache_allocator(true);
        std::map<std::string,Output> resident_results;
        for(int variant=0;variant<3;++variant) {
          for(int placement:{0,1,2}) {
            const bool streamed=placement!=0,share_scratch=placement==2;
            const auto file=(directory/("mtp-"+std::to_string(variant)+".gguf")).string();
            write_synthetic_hy3(file,variant>0,variant==2);
            std::vector<llama_token> prompt(39);
            for(size_t i=0;i<prompt.size();++i) prompt[i]=(7*int(i)+11)%64;
            // Reference actually omits the MTP weights and hidden extraction.
            auto off=load(file,false,false,true);
            auto off_ctx=context(off.get(),128,17);
            auto model=load(file,false,false,true,true,streamed);
            auto ctx=context(model.get(),128,17);
            Mtp mtp(model.get(),ctx.get(),128,17,3,share_scratch);
            auto * kv=dynamic_cast<llama_kv_cache *>(ctx->get_memory());
            require(kv && kv->get_layer_ids()==std::vector<uint32_t>{0,1},"main KV includes draft block");
            auto run=[&](bool original,int depth,int count,const std::vector<llama_token> & prefix,
                         const std::vector<llama_token> * forced=nullptr,int cancel_poll=0,int stop_id=-1,bool penalties=false) {
                auto * target=original ? off_ctx.get() : ctx.get();
                clear(target);mtp.reset();if(depth) mtp.set_depth(depth);
                configure_cache(original ? off.get() : model.get(),8*1024*1024);
                strata_hy3_pipeline_config(2,4,0,true);strata_hy3_sync_reset();
                std::unique_ptr<llama_sampler,decltype(&llama_sampler_free)> sampler(
                    llama_sampler_chain_init(llama_sampler_chain_default_params()),llama_sampler_free);
                llama_sampler_chain_add(sampler.get(),llama_sampler_init_penalties(64,32,penalties?1.1f:1,penalties?.1f:0,0));
                llama_sampler_chain_add(sampler.get(),llama_sampler_init_greedy());
                for(auto token:prefix) llama_sampler_accept(sampler.get(),token);
                for(size_t i=0;i<prefix.size();i+=17) {
                    int n=int(std::min(size_t(17),prefix.size()-i));
                    decode(target,prefix,i,n,int(i));
                    if(depth) mtp.prefill(prefix,int(i),n,int(i));
                }
                Output out;
                auto sample=[&](int row) {
                    auto * l=llama_get_logits_ith(target,row);require(l,"missing target logits");
                    for(int i=0;i<64;++i) require(std::isfinite(l[i]),"non-finite logits");
                    out.logits.insert(out.logits.end(),l,l+64);
                    return llama_sampler_sample(sampler.get(),target,row);
                };
                auto stop=[&](llama_token t){return t==stop_id;};
                out.ids.push_back(sample(-1));int position=int(prefix.size()),polls=0;
                while(int(out.ids.size())<count && !stop(out.ids.back())) {
                    if(!depth) {
                        decode(target,out.ids,out.ids.size()-1,1,position++);out.ids.push_back(sample(-1));
                    } else {
                        auto r=mtp.advance(out.ids.back(),position,count-int(out.ids.size()),sample,stop,
                            [&]{return cancel_poll && ++polls>=cancel_poll;},out.ids.size()==1 ? forced : nullptr);
                        if(r.cancelled) {out.cancelled=true;break;}
                        if(out.first_accepted<0) out.first_accepted=r.accepted;
                        out.ids.insert(out.ids.end(),r.tokens.begin(),r.tokens.end());position=r.next_position;
                    }
                    require(position<=128,"MTP wrote past context");
                }
                out.counts=mtp.counters;
                require(!share_scratch || mtp.scratch_shared(),"shared scratch detached during MTP check");
                auto stats=strata_hy3_sync_snapshot();
                require(!stats.rejected_cpu_nodes && !stats.rejected_full_copies,"invalid compute/copy fallback");
                if(depth && out.counts.rounds && !out.cancelled) {
                    require((stats.mtp_cache_bytes>0)==streamed,"MTP cache placement was not exercised");
                    require(stats.mtp_cache_bytes<=stats.cache_bytes,"invalid MTP cache accounting");
                }
                clear(target);mtp.reset();return out;
            };
            auto compare=[&](std::string name,const Output & got,const Output & ref) {
                bool same=got.ids.size()<=ref.ids.size() && std::equal(got.ids.begin(),got.ids.end(),ref.ids.begin());
                require(got.logits.size()<=ref.logits.size(),"too many samples");
                double max_abs=0,energy=0,error=0;
                for(size_t i=0;i<got.logits.size();++i) {
                    double d=double(got.logits[i])-ref.logits[i];max_abs=std::max(max_abs,std::abs(d));error+=d*d;energy+=double(ref.logits[i])*ref.logits[i];
                }
                double nmse=error/std::max(energy,1e-30);
                // Same bounds as Hy3's existing serial-vs-batch graph oracle.
                // Quantized batch dispatch can differ from single-token CUDA.
                const double abs_limit=variant ? .03 : 5e-4, nmse_limit=variant ? 2e-3 : 1e-7;
                const auto key=std::to_string(variant)+":"+name;
                if(!streamed) resident_results[key]=got;
                else {
                    const auto & resident=resident_results.at(key);
                    require(got.ids==resident.ids && got.logits==resident.logits &&
                        got.counts.proposed==resident.counts.proposed && got.counts.accepted==resident.counts.accepted &&
                        got.counts.reject_first==resident.counts.reject_first && got.counts.reject_middle==resident.counts.reject_middle,
                        "resident/streamed MTP mismatch: "+name);
                }
                report["checks"].push_back({{"variant",variant},{"mtp_experts",streamed ? "streamed" : "resident"},{"shared_scratch",share_scratch},{"scratch_saved_bytes",mtp.scratch_saved()},{"resident_exact",streamed},{"name",name},{"ids_exact",same},
                    {"tokens",got.ids},{"max_abs",max_abs},{"nmse",nmse},{"max_abs_limit",abs_limit},{"nmse_limit",nmse_limit},
                    {"proposed",got.counts.proposed},{"accepted",got.counts.accepted},
                    {"reject_first",got.counts.reject_first},{"reject_middle",got.counts.reject_middle},{"accept_all",got.counts.accept_all}});
                require(same && max_abs<=abs_limit && nmse<=nmse_limit,"MTP parity failed: "+name);
            };
            auto baseline=run(true,0,24,prompt);
            compare("loaded MTP, target only",run(false,0,24,prompt),baseline);
            for(int depth=1;depth<=3;++depth) compare("native depth "+std::to_string(depth),run(false,depth,24,prompt),baseline);
            for(int accept=0;accept<=3;++accept) {
                std::vector<llama_token> forced(baseline.ids.begin()+1,baseline.ids.begin()+4);
                if(accept<3) forced[accept]=(forced[accept]+1)%64;
                auto got=run(false,3,24,prompt,&forced);
                require(got.first_accepted==accept,"forced rejection position not exercised");
                compare("forced accepted "+std::to_string(accept),got,baseline);
            }
            compare("output limit one",run(false,3,1,prompt),baseline);
            compare("output limit two",run(false,3,2,prompt),baseline);
            auto stopped=run(false,3,24,prompt,nullptr,0,baseline.ids[0]);
            require(stopped.ids.size()==1 && !stopped.counts.rounds,"first EOG drafted");compare("first stop",stopped,baseline);
            if(baseline.ids[1]!=baseline.ids[0]) {
                std::vector<llama_token> forced(baseline.ids.begin()+1,baseline.ids.begin()+4);
                auto eos=run(false,3,24,prompt,&forced,0,baseline.ids[1]);
                require(eos.ids.size()==2 && eos.counts.accepted==1,"accepted stop branch");compare("accepted stop",eos,baseline);
            }
            {
                llama_token rejected_stop=0;
                while(std::find(baseline.ids.begin(),baseline.ids.end(),rejected_stop)!=baseline.ids.end()) ++rejected_stop;
                std::vector<llama_token> forced{rejected_stop};
                auto rejected=run(false,3,24,prompt,&forced,0,rejected_stop);
                require(rejected.first_accepted==0 && rejected.counts.reject_first>0,"rejected draft stop branch");
                compare("rejected draft stop",rejected,baseline);
            }
            for(int poll:{1,4,5}) {
                auto cancelled=run(false,3,24,prompt,nullptr,poll);
                require(cancelled.cancelled,"cancellation was not exercised");
                compare("recovery after cancel poll "+std::to_string(poll),run(false,3,24,prompt),baseline);
            }
            auto penalized=run(true,0,24,prompt,nullptr,0,-1,true);
            compare("greedy penalties",run(false,3,24,prompt,nullptr,0,-1,true),penalized);
            prompt.resize(112);for(size_t i=0;i<prompt.size();++i) prompt[i]=(7*int(i)+11)%64;
            auto near=run(true,0,16,prompt);
            compare("near limit loaded target only",run(false,0,16,prompt),near);
            // Independent teacher-forced target batching: no draft graph or KV rollback.
            clear(off_ctx.get());configure_cache(off.get(),8*1024*1024);strata_hy3_pipeline_config(2,4);
            for(size_t i=0;i<prompt.size();i+=17) decode(off_ctx.get(),prompt,i,int(std::min(size_t(17),prompt.size()-i)),int(i));
            Output batched;batched.ids.push_back(near.ids[0]);
            const auto * first=llama_get_logits_ith(off_ctx.get(),-1);batched.logits.assign(first,first+64);
            for(int i=0;i<15;i+=4) {
                int n=std::min(4,15-i);decode(off_ctx.get(),near.ids,i,n,int(prompt.size())+i,true);
                for(int j=0;j<n;++j) {
                    const auto * l=llama_get_logits_ith(off_ctx.get(),j);batched.logits.insert(batched.logits.end(),l,l+64);
                    batched.ids.push_back(llama_token(std::max_element(l,l+64)-l));
                }
            }
            compare("near limit target-only batch4 control",batched,near);
            compare("near context limit",run(false,3,16,prompt),near);
            strata_hy3_cache_begin(0);
          }
        }
        strata_hy3_sync_release();llama_backend_free();report["status"]="pass";
    } catch(const std::exception & e) {report["error"]=e.what();strata_hy3_sync_release();llama_backend_free();}
    std::ofstream(directory/"mtp-report.json")<<report.dump(2)<<'\n';
    std::cout<<report["status"]<<" "<<report.value("error","")<<'\n';return report["status"]=="pass"?0:1;
}
