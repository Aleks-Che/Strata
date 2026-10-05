#include "mtp.hpp"
#include "runtime_memory.hpp"
#include "host_pages.hpp"
#include "synthetic_glm.hpp"
#include "llama-context.h"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace strata_glm;
using json=nlohmann::json;
using Clock=std::chrono::steady_clock;
static double seconds(Clock::time_point s) {return std::chrono::duration<double>(Clock::now()-s).count();}
static json counts(const Mtp::Counters & c) {
    return {{"proposed",c.proposed},{"accepted",c.accepted},{"rounds",c.rounds},{"reject_first",c.reject_first},
        {"reject_middle",c.reject_middle},{"accept_all",c.accept_all},{"draft_ms",c.draft_ms},{"verify_ms",c.verify_ms},{"repair_ms",c.repair_ms}};
}
struct Output {std::vector<llama_token> tokens;std::vector<float> logits;json report;};
struct Probe {
    bool active=false;
    std::map<std::string,std::vector<float>> values;
    static bool callback(ggml_tensor * t,bool ask,void * data) {
        const std::string name=t->name;
        const bool wanted=static_cast<Probe *>(data)->active && t->type==GGML_TYPE_F32 && (name.find("kda_")==0 || name.find("l_last")==0 ||
            name.find("hc_attn_pre")==0 || name.find("ffn_")==0 || name.find("attn_out")==0 || name.find("result_")==0);
        if (ask) return wanted;
        if (wanted) {auto & v=static_cast<Probe *>(data)->values[name];v.resize(size_t(t->ne[0]));ggml_backend_tensor_get(t,v.data(),0,v.size()*sizeof(float));}
        return true;
    }
};
int main(int argc,char ** argv) {
    json report={{"status","error"},{"checks",json::array()}};std::string file,output,reference,fixture,reference_logits;int percent=0,tokens=64,diagnostic=0,baseline_only=0;
    try {
        for (int i=1;i<argc;++i) {
            const std::string k=argv[i];require(i+1<argc,"missing argument");const std::string v=argv[++i];
            if (k=="--model") file=v;else if (k=="--report") output=v;else if (k=="--reference") reference=v;
            else if (k=="--fixture") fixture=v;else if (k=="--memory-percent") percent=std::stoi(v);
            else if (k=="--tokens") tokens=std::stoi(v);
            else if (k=="--batch-diagnostic") diagnostic=std::stoi(v);
            else if (k=="--reference-logits") reference_logits=v;
            else if (k=="--baseline-only") baseline_only=std::stoi(v);
            else throw std::runtime_error("unknown option "+k);
        }
        const bool synthetic=!fixture.empty();
        require(tokens>=16 && tokens<=64,"test token count must be 16..64");
        if (synthetic) {std::filesystem::create_directories(fixture);file=fixture+"/mtp.gguf";write_synthetic_glm(file,8,true);}
        require(!file.empty(),"model or fixture required");
        std::vector<llama_token> prompt,reference_ids;
        if (!reference.empty()) {std::ifstream in(reference);json ref;in>>ref;prompt=ref["prompt_ids"].get<std::vector<llama_token>>();reference_ids=ref["generated_ids"].get<std::vector<llama_token>>();}
        else {require(synthetic,"real model requires saved reference prompt/IDs");for (int i=0;i<39;++i) prompt.push_back((7*i+11)%64);}
        environment();ggml_backend_load_all();llama_backend_init();strata_glm_sync_enable(true);
        {
            HostWorkingSetBudget load_budget;
            if(ram_experts())load_budget.apply(percent);
            auto model=load(file,false,false,true);const auto * vocab=llama_model_get_vocab(model.get());const int nv=llama_vocab_n_tokens(vocab);
            const int context_size=synthetic?512:2048;
            Probe probe;
            auto ctx=context(model.get(),context_size,16,4,3,false,diagnostic?Probe::callback:nullptr,diagnostic?&probe:nullptr);
            Mtp mtp(model.get(),ctx.get(),context_size,16,4,3);
            RuntimeMemory memory(model,file,percent,percent,true,4,512);memory.warm();
            if(synthetic && ram_experts() && ram_expert_layers()==2) {
                const auto storage=memory.snapshot();
                require(storage["private_expert_bytes"].get<uint64_t>()>0 && storage["mapped_expert_bytes"].get<uint64_t>()>0,
                    "hybrid fixture must exercise both private and mapped expert buffers");
            }
            report["memory_after_warm"]=memory.snapshot();report["configuration"]={{"model",file},{"synthetic",synthetic},{"n_ctx",context_size},
                {"batch",16},{"n_rs_seq",3},{"kv","F16"},{"pipeline",true},{"ram_target_percent",percent},{"vram_target_percent",percent},{"prompt_ids",prompt},
                {"shared_scratch_saved_bytes",mtp.shared_scratch_saved_bytes()}};
            if (diagnostic) {
                require(reference_ids.size()>=2,"diagnostic needs reference IDs");Probe single,batched;
                for (int n:{1,2}) {
                    clear(ctx.get());
                    for(size_t i=0;i<prompt.size();i+=16) decode(ctx.get(),prompt,int(i),int(std::min(size_t(16),prompt.size()-i)),int(i));
                    probe.active=true;
                    decode(ctx.get(),reference_ids,0,n,int(prompt.size()),true);
                    probe.active=false;(n==1?single:batched).values=std::move(probe.values);probe.values.clear();
                }
                require(!single.values.empty(),"no diagnostic probe tensors captured");
                for(const auto & [name,a]:single.values) {
                    const auto & b=batched.values.at(name);require(a.size()==b.size(),"probe shape mismatch");
                    double max_abs=0,error=0,energy=0;
                    for(size_t i=0;i<a.size();++i) {const double d=double(a[i])-b[i];max_abs=std::max(max_abs,std::abs(d));error+=d*d;energy+=double(a[i])*a[i];}
                    report["checks"].push_back({{"name",name},{"max_abs",max_abs},{"nmse",error/std::max(energy,1e-30)}});
                }
                report["status"]="diagnostic";
                std::ofstream out(output,std::ios::binary);out<<report.dump(2)<<"\n";return 0;
            }
            auto run=[&](int depth,int n,bool stochastic,int forced_accept=-1,bool cancel_round=false,int stop_id=-1) {
                clear(ctx.get());mtp.reset();if (depth) mtp.set_depth(depth);memory.refresh();strata_glm_sync_reset();
                using Sampler=std::unique_ptr<llama_sampler,decltype(&llama_sampler_free)>;
                Sampler sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()),llama_sampler_free);
                llama_sampler_chain_add(sampler.get(),llama_sampler_init_penalties(nv,64,stochastic?1.1f:1,0,0));
                if (stochastic) {
                    llama_sampler_chain_add(sampler.get(),llama_sampler_init_top_k(32));
                    llama_sampler_chain_add(sampler.get(),llama_sampler_init_temp(.7f));
                    llama_sampler_chain_add(sampler.get(),llama_sampler_init_dist(42));
                } else llama_sampler_chain_add(sampler.get(),llama_sampler_init_greedy());
                for (auto t:prompt) llama_sampler_accept(sampler.get(),t);
                const auto prefill_start=Clock::now();
                for (size_t i=0;i<prompt.size();i+=16) {
                    const int count=int(std::min(size_t(16),prompt.size()-i));decode(ctx.get(),prompt,int(i),count,int(i));
                    if (depth) mtp.prefill(prompt,int(i),count,int(i));
                }
                Output out;out.report["prefill_seconds"]=seconds(prefill_start);out.report["depth"]=depth;out.report["stochastic"]=stochastic;
                auto sample=[&](int row) {
                    const auto * l=llama_get_logits_ith(ctx.get(),row);require(l!=nullptr,"missing verification logits");
                    for (int j=0;j<nv;++j) require(std::isfinite(l[j]),"non-finite target logits");
                    out.logits.insert(out.logits.end(),l,l+nv);return llama_sampler_sample(sampler.get(),ctx.get(),row);
                };
                auto stop=[&](llama_token t){return t==stop_id;};
                out.tokens.push_back(sample(-1));int position=int(prompt.size()),polls=0;
                const auto start=Clock::now();
                while (int(out.tokens.size())<n && !stop(out.tokens.back())) {
                    if (!depth) {
                        decode(ctx.get(),out.tokens,int(out.tokens.size())-1,1,position++);out.tokens.push_back(sample(-1));
                    } else {
                        std::vector<llama_token> forced;
                        if (forced_accept>=0 && out.tokens.size()==1) {
                            require(reference_ids.size()>=4,"forced test needs reference continuation");
                            forced={reference_ids[1],reference_ids[2],reference_ids[3]};
                            if (forced_accept<3) forced[forced_accept]=(forced[forced_accept]+1)%nv;
                        }
                        auto r=mtp.advance(out.tokens.back(),position,n-int(out.tokens.size()),sample,stop,
                            [&]{return cancel_round && ++polls>2;},forced.empty()?nullptr:&forced);
                        if (r.cancelled) {out.report["cancelled"]=true;break;}
                        if (!forced.empty()) require(r.accepted==std::min(forced_accept,r.proposed),"forced rejection branch not exercised");
                        out.tokens.insert(out.tokens.end(),r.tokens.begin(),r.tokens.end());position=r.next_position;
                    }
                }
                out.report["decode_seconds"]=seconds(start);out.report["tokens"]=out.tokens;
                out.report["tokens_per_second"]=(out.tokens.size()-1)/out.report["decode_seconds"].get<double>();
                out.report["mtp"]=counts(mtp.counters);out.report["memory"]=memory.snapshot();
                out.report["shared_scratch_active"]=mtp.shared_scratch_active();
                require(!mtp.shared_scratch_saved_bytes() || mtp.shared_scratch_active(),"shared scratch detached during numerical check");
                std::cerr<<"GLM_MTP_CHECK "<<json({{"depth",depth},{"tps",out.report["tokens_per_second"]},{"mtp",out.report["mtp"]}}).dump()<<"\n";
                auto s=strata_glm_sync_snapshot();out.report["cpu_compute_nodes"]=s.rejected_cpu_nodes;
                require(s.rejected_cpu_nodes==0,"CPU compute encountered");clear(ctx.get());mtp.reset();return out;
            };
            auto baseline=run(0,tokens,false);
            if (!reference_ids.empty()) require(reference_ids.size()>=size_t(tokens) && baseline.tokens==std::vector<llama_token>(reference_ids.begin(),reference_ids.begin()+tokens),"target baseline changed when loading MTP weights/rollback slots");
            else reference_ids=baseline.tokens;
            report["checks"].push_back({{"name","baseline"},{"status","pass"},{"result",baseline.report}});
            if (!reference_logits.empty()) {
                std::ifstream in(reference_logits,std::ios::binary);std::vector<float> saved(baseline.logits.size());
                in.read(reinterpret_cast<char *>(saved.data()),saved.size()*sizeof(float));
                require(in.gcount()==std::streamsize(saved.size()*sizeof(float)),"saved reference logits truncated");
                require(!std::memcmp(saved.data(),baseline.logits.data(),saved.size()*sizeof(float)),"baseline with MTP/rollback state differs from saved logits");
                report["baseline_vs_saved_logits"]={{"path",reference_logits},{"elements",saved.size()},{"bit_exact",true}};
            }
            if (baseline_only) {
                report["status"]="pass";std::ofstream out(output,std::ios::binary);out<<report.dump(2)<<"\n";return 0;
            }
            auto compare=[&](const char * name,const Output & candidate,const Output & expected) {
                report["checks"].push_back({{"name",name},{"status","error"},{"result",candidate.report}});
                auto & entry=report["checks"].back();
                const bool same=candidate.tokens==std::vector<llama_token>(expected.tokens.begin(),expected.tokens.begin()+candidate.tokens.size());
                size_t common=0;while(common<candidate.tokens.size() && candidate.tokens[common]==expected.tokens[common])++common;
                entry["equal_prefix_tokens"]=common;
                entry["row_max_abs"]=json::array();
                for(size_t row=0;row<std::min(common+1,candidate.tokens.size());++row) {
                    double d=0;for(int j=0;j<nv;++j) d=std::max(d,std::abs(double(candidate.logits[row*nv+j])-expected.logits[row*nv+j]));
                    entry["row_max_abs"].push_back(d);
                }
                require(candidate.logits.size()<=expected.logits.size(),"extra target sampling calls");
                double energy=0,error=0,max_abs=0;
                const size_t size=std::min(candidate.logits.size(),(common+1)*nv);
                for (size_t i=0;i<size;++i) {const double d=double(candidate.logits[i])-expected.logits[i];error+=d*d;energy+=double(expected.logits[i])*expected.logits[i];max_abs=std::max(max_abs,std::abs(d));}
                const auto nmse=error/std::max(energy,1e-30);
                entry["nmse"]=nmse;entry["max_abs"]=max_abs;entry["nmse_limit"]=1e-6;entry["max_abs_limit"]=.05;
                require(same,std::string(name)+": token mismatch at "+std::to_string(common));
                require(nmse<=1e-6 && max_abs<=.05,"MTP target logits numerical mismatch");entry["status"]="pass";
            };
            for (int depth:{1,2,3}) {auto candidate=run(depth,tokens,false);compare(("native depth "+std::to_string(depth)).c_str(),candidate,baseline);}
            for (int accept:{0,1,2,3}) {auto candidate=run(3,16,false,accept);compare(("forced first round accepted "+std::to_string(accept)).c_str(),candidate,baseline);}
            auto sampled=run(0,std::min(tokens,32),true),spec_sampled=run(3,std::min(tokens,32),true);compare("seeded target sample-and-match",spec_sampled,sampled);
            auto cancelled=run(3,16,false,-1,true);require(cancelled.report.value("cancelled",false),"draft cancellation did not fire");
            report["checks"].push_back({{"name","cancel during draft"},{"status","pass"},{"result",cancelled.report}});
            auto clean=run(3,16,false);compare("clean after cancel",clean,baseline);
            if (synthetic) {
                auto first_stop=run(3,16,false,-1,false,reference_ids[0]);
                require(first_stop.tokens.size()==1 && first_stop.report["mtp"]["rounds"]==0,"first sampled stop must not draft");
                compare("stop on first sample",first_stop,baseline);
                auto accepted_stop=run(3,16,false,3,false,reference_ids[1]);
                require(accepted_stop.tokens.size()==2 && accepted_stop.report["mtp"]["accepted"]==1,"accepted stop must end the round");
                compare("accepted draft stop",accepted_stop,baseline);
                auto rejected_stop=run(3,16,false,0,false,(reference_ids[1]+1)%nv);
                require(rejected_stop.report["mtp"]["reject_first"].get<int>()>=1,"rejected draft stop was not tested");
                compare("rejected draft stop",rejected_stop,baseline);
            }
            report["memory_final"]=memory.snapshot();report["status"]="pass";
        }
        strata_glm_sync_release();llama_backend_free();
    } catch (const std::exception & e) {report["error"]=e.what();std::cerr<<"GLM_MTP_ERROR "<<e.what()<<"\n";strata_glm_sync_release();llama_backend_free();}
    const auto text=report.dump(2)+"\n";
    if (output.empty()) std::cout<<text;else {std::ofstream out(output,std::ios::binary);out<<text;}
    return report["status"]=="pass"?0:1;
}
