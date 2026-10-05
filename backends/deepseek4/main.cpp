// DeepSeek V4 execution through the pinned llama.cpp graph. This executable speaks
// Strata's pipe protocol; Qwen continues to use the existing specialized engine.
#include "llama.h"
#include "ggml-backend.h"
#include "expert_transfer.h"
#include "vram_control.hpp"
#include "speculative.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/conversation_memory.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now()-t).count(); }
static int64_t epoch_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }
static int integer(const std::string &s) {
    size_t end = 0; int n = std::stoi(s, &end);
    if (end != s.size()) throw std::runtime_error("invalid integer");
    return n;
}
struct Options {
    std::string model, draft_model;
    int context=8192, batch=512, threads=1, gpu_layers=99, gpu_expert_layers=0, slots=4;
    uint64_t budget=2048ULL<<20, floor=8192ULL<<20;
    uint64_t working_set=0;
    int expert_cache_mib=0, expert_stage_mib=0;
    std::string expert_cache_policy="lru";
    int expert_cache_match_size=0;
    int expert_pipeline=0;
    int expert_readers=2;
    std::string expert_read_mode="mmap";
    int draft_max=3, draft_expert_cache_mib=1024, draft_gpu_expert_layers=0;
    int draft_shared_scratch=0;
    float draft_min_confidence=0;
    bool vocab_only=false;
};
static Options options(int argc, char **argv) {
    Options o;
    for (int i=1;i<argc;++i) {
        std::string k=argv[i];
        if (k=="--serve") continue;
        if (k=="--tokenizer") { o.vocab_only=true; continue; }
        if (k=="--help") {
            std::cout << "Strata DeepSeek V4 backend (llama.cpp 3cf03257)\n"
                "--native FIRST.gguf --max-context 8192 --threads 1 --batch-size 512\n"
                "GPU-only forward; cold expert weights stream from mapped GGUF to GPU\n"
                "--gpu-expert-layers 0 (keep the last N layers' routed experts in VRAM)\n"
                "--expert-cache-mib 0 (GPU LRU of individual expert matrices)\n"
                "--expert-cache-policy lru|frequency (frequency gates LRU admission using decaying access counts)\n"
                "--expert-cache-match-size 0 (0/1; prefer same-size victims in the fixed GPU arena)\n"
                "--expert-stage-mib 0 (size of EACH of two pinned upload buffers)\n"
                "--expert-pipeline 0 (0/1; background staging + separate H2D stream; needs stage > 0)\n"
                "--expert-readers 2 (1..4 bounded readers); --expert-read-mode mmap|file|auto\n"
                "--draft-model FILE.gguf (0731 DSpark; omitted = speculation off)\n"
                "--draft-shared-scratch 0 (0/1; serialize target/draft on one GPU compute buffer)\n"
                "--draft-max 3 (1..5 target-verified draft tokens)\n"
                "--draft-min-confidence 0 (0..1; reject low-confidence draft suffix; 0 disables)\n"
                "--draft-expert-cache-mib 1024 --draft-gpu-expert-layers 0 (0..3)\n"
                "--working-set-mib 0 (Windows process RAM cap; 0 leaves paging to the OS)\n"
                "--conversation-cache-mib 2048 --conversation-cache-slots 4\n"
                "--conversation-cache-min-free-mib 8192; --tokenizer: vocab-only protocol\n";
            std::exit(0);
        }
        if (++i>=argc) throw std::runtime_error("missing value for "+k);
        std::string v=argv[i];
        if (k=="--native") o.model=v;
        else if (k=="--draft-model") o.draft_model=v;
        else if (k=="--draft-shared-scratch") o.draft_shared_scratch=integer(v);
        else if (k=="--draft-max") o.draft_max=integer(v);
        else if (k=="--draft-min-confidence") {
            size_t end=0;o.draft_min_confidence=std::stof(v,&end);
            if(end!=v.size() || !std::isfinite(o.draft_min_confidence) || o.draft_min_confidence<0 || o.draft_min_confidence>1)
                throw std::runtime_error("invalid DSpark confidence threshold");
        }
        else if (k=="--draft-expert-cache-mib") o.draft_expert_cache_mib=integer(v);
        else if (k=="--draft-gpu-expert-layers") o.draft_gpu_expert_layers=integer(v);
        else if (k=="--max-context") o.context=integer(v);
        else if (k=="--threads") o.threads=integer(v);
        else if (k=="--batch-size") o.batch=integer(v);
        else if (k=="--gpu-layers") o.gpu_layers=integer(v);
        else if (k=="--gpu-expert-layers") o.gpu_expert_layers=integer(v);
        else if (k=="--expert-cache-mib") o.expert_cache_mib=integer(v);
        else if (k=="--expert-cache-policy") o.expert_cache_policy=v;
        else if (k=="--expert-cache-match-size") o.expert_cache_match_size=integer(v);
        else if (k=="--expert-stage-mib") o.expert_stage_mib=integer(v);
        else if (k=="--expert-pipeline") o.expert_pipeline=integer(v);
        else if (k=="--expert-readers") o.expert_readers=integer(v);
        else if (k=="--expert-read-mode") o.expert_read_mode=v;
        else if (k=="--conversation-cache-slots") o.slots=integer(v);
        else if (k=="--conversation-cache-mib" || k=="--conversation-cache-min-free-mib" || k=="--working-set-mib") {
            int n=integer(v); if(n<0) throw std::runtime_error("negative memory limit");
            (k=="--working-set-mib" ? o.working_set : k=="--conversation-cache-mib" ? o.budget : o.floor)=uint64_t(n)<<20;
        } else throw std::runtime_error("unsupported DeepSeek option: "+k);
    }
    if(o.model.empty() || o.context<32 || o.context>1048576 || o.batch<1 || o.batch>4096 || o.threads<1 || o.threads>256 || o.slots<0 || o.slots>64 || o.gpu_expert_layers<0)
        throw std::runtime_error("invalid model/context/batch/thread/cache settings");
    if(o.expert_cache_mib<0 || o.expert_cache_mib>65536 || o.expert_stage_mib<0 || o.expert_stage_mib>256)
        throw std::runtime_error("invalid expert GPU cache or pinned stage size");
    if(o.expert_cache_policy!="lru" && o.expert_cache_policy!="frequency")
        throw std::runtime_error("expert-cache-policy must be lru/frequency");
    if(o.expert_cache_match_size<0 || o.expert_cache_match_size>1)
        throw std::runtime_error("expert-cache-match-size must be 0/1");
    if(o.expert_pipeline<0 || o.expert_pipeline>1 || (o.expert_pipeline && !o.expert_stage_mib))
        throw std::runtime_error("expert-pipeline must be 0/1 and requires expert-stage-mib > 0");
    if(o.expert_readers<1 || o.expert_readers>4 || (o.expert_read_mode!="mmap" && o.expert_read_mode!="file" && o.expert_read_mode!="auto"))
        throw std::runtime_error("expert-readers must be 1..4; expert-read-mode must be mmap/file/auto");
#ifndef _WIN32
    if(o.expert_pipeline && o.expert_read_mode=="file")throw std::runtime_error("native expert file reads require Windows");
#endif
    if(o.draft_max<1 || o.draft_max>5 || o.draft_expert_cache_mib<0 || o.draft_expert_cache_mib>65536 || o.draft_gpu_expert_layers<0 || o.draft_gpu_expert_layers>3)
        throw std::runtime_error("invalid DSpark draft size/cache/resident-layer setting");
    if(!o.draft_model.empty() && o.batch<o.draft_max+1)
        throw std::runtime_error("DSpark batch size must fit anchor plus draft tokens");
    if(o.draft_min_confidence>0 && o.draft_model.empty())
        throw std::runtime_error("DSpark confidence filtering requires --draft-model");
    if(o.draft_shared_scratch<0 || o.draft_shared_scratch>1 || (o.draft_shared_scratch && o.draft_model.empty()))
        throw std::runtime_error("draft-shared-scratch must be 0/1 and requires --draft-model");
    return o;
}
struct Request {
    int count=0;
    std::string session;
    std::map<std::string,std::string> sampling;
    std::vector<llama_token> tokens;
};
static Request request(const std::string &line, int context, int vocab) {
    std::istringstream in(line); std::string verb, word; Request r;
    in>>verb>>word;
    if(verb!="GEN") throw std::runtime_error("DeepSeek backend accepts text GEN only");
    r.count=integer(word);
    if(r.count<1 || r.count>context) throw std::runtime_error("invalid generation length");
    bool have_tokens=false;
    while(in>>word) {
        auto eq=word.find('=');
        if(eq!=std::string::npos && !have_tokens) {
            auto key=word.substr(0,eq),value=word.substr(eq+1);
            if(key=="session") {
                if(value.size()!=64 || value.find_first_not_of("0123456789abcdef")!=std::string::npos)
                    throw std::runtime_error("invalid session hash");
                r.session=value;
            } else r.sampling[key]=value;
        } else {
            if(have_tokens) throw std::runtime_error("extra request data");
            have_tokens=true; std::istringstream ids(word); std::string id;
            while(std::getline(ids,id,',')) {
                int n=integer(id); if(n<0 || n>=vocab) throw std::runtime_error("token outside vocabulary");
                r.tokens.push_back(n);
                if(r.tokens.size()>size_t(context)) throw std::runtime_error("prompt exceeds context");
            }
        }
    }
    if(r.tokens.empty() || r.tokens.size()+r.count+8>size_t(context)) throw std::runtime_error("prompt plus generation exceeds context");
    return r;
}
static llama_sampler *sampler(const Request &r, int vocab) {
    auto value=[&](const char *key,double def) {
        auto it=r.sampling.find(key); if(it==r.sampling.end()) return def;
        size_t n=0; double x=std::stod(it->second,&n);
        if(n!=it->second.size() || !std::isfinite(x)) throw std::runtime_error("invalid sampling value");
        return x;
    };
    double temp=value("temperature",0),tp=value("top_p",1),mp=value("min_p",0);
    double k=value("top_k",64),ln=value("penalty_last_n",64),seed=value("seed",LLAMA_DEFAULT_SEED);
    double repeat=value("penalty_repeat",1),freq=value("penalty_freq",0),present=value("penalty_present",0);
    if(k<0 || k>vocab || ln<0 || ln>1048576 || seed<0 || seed>UINT32_MAX || repeat<=0)
        throw std::runtime_error("invalid sampling range");
    int topk=int(k),lastn=int(ln);
    if(temp<0 || temp>100 || tp<=0 || tp>1 || mp<0 || mp>1 || topk<0 || lastn<0 || lastn>1048576)
        throw std::runtime_error("invalid sampling range");
    auto *s=llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(s,llama_sampler_init_penalties(vocab,lastn,float(repeat),float(freq),float(present)));
    if(temp==0) llama_sampler_chain_add(s,llama_sampler_init_greedy());
    else {
        llama_sampler_chain_add(s,llama_sampler_init_top_k(topk));
        llama_sampler_chain_add(s,llama_sampler_init_top_p(float(tp),1));
        llama_sampler_chain_add(s,llama_sampler_init_min_p(float(mp),1));
        llama_sampler_chain_add(s,llama_sampler_init_temp(float(temp)));
        llama_sampler_chain_add(s,llama_sampler_init_dist(uint32_t(seed)));
    }
    for(auto t:r.tokens) llama_sampler_accept(s,t);
    return s;
}
struct Snapshot {
    uint64_t id;
    std::string session;
    std::vector<llama_token> tokens;
    std::vector<uint8_t> state, draft_state;
    int64_t used;
    size_t bytes() const { return state.size()+draft_state.size()+tokens.size()*sizeof(llama_token); }
};
class Runner {
    llama_context *ctx;
    llama_context *draft_ctx;
    std::unique_ptr<DSpark> spec;
    StrataExpertBudget expert_budget;
    VramControl &vram;
    const llama_vocab *vocab;
    Options o;
    std::atomic<bool> &stop;
    std::deque<Snapshot> cache;
    std::vector<llama_token> active;
    std::string active_session;
    uint64_t serial=0,hits=0,misses=0,evictions=0,skips=0;
    const char *save_reason="not_saved";
    size_t used() const { size_t b=0;for(auto &s:cache)b+=s.bytes();return b; }
    static bool prefix(const std::vector<llama_token>&a,const std::vector<llama_token>&b) {
        return !a.empty() && a.size()<b.size() && std::equal(a.begin(),a.end(),b.begin());
    }
    void reset() {
        llama_memory_clear(llama_get_memory(ctx),true);
        if(draft_ctx)llama_memory_clear(llama_get_memory(draft_ctx),true);
        active.clear();active_session.clear();
    }
    void budget(bool draft) { vram.poll();if(expert_budget)expert_budget(draft?o.draft_expert_cache_mib:o.expert_cache_mib); }
    void inject(const llama_batch& b) { if(spec) {budget(true);spec->inject(b);} }
    void status(const char *phase,const char *source,double save=0,double restore=0,const char *reason=nullptr) {
        if(!reason)reason=save_reason;
        std::cout<<"CACHE phase="<<phase<<" source="<<source<<" reason="<<reason<<" bytes="<<used()<<" sessions="<<cache.size()
            <<" hits="<<hits<<" misses="<<misses<<" evictions="<<evictions<<" skipped_saves="<<skips<<" save_ms="<<save<<" restore_ms="<<restore<<"\n"<<std::flush;
    }
    bool decode(const std::vector<llama_token> &tokens,size_t begin,size_t end,bool prompt) {
        for(size_t i=begin;i<end;) {
            if(stop.load()) return false;
            size_t n=std::min(size_t(o.batch),end-i);
            Batch batch{int(n)};
            batch.positions(int(n),int(active.size()),false);batch.value.logits[n-1]=true;
            std::copy_n(tokens.data()+i,n,batch.value.token);
            budget(false);int rc=llama_decode(ctx,batch.value);
            if(rc) { if(stop.load()) return false; throw std::runtime_error("llama_decode failed: "+std::to_string(rc)); }
            inject(batch.value);
            active.insert(active.end(),tokens.begin()+i,tokens.begin()+i+n); i+=n;
            if(prompt) std::cout<<"PP "<<i<<" "<<tokens.size()<<"\n"<<std::flush;
        }
        return true;
    }
    double save(const Request&r) {
        if(!o.budget || !o.slots || active.empty()) {save_reason="disabled";return 0;}
        // An exact restore already has this snapshot. Keep it even when RAM is
        // now below the admission floor instead of deleting and reallocating it.
        for(const auto &entry:cache) {
            if(entry.session==r.session && entry.tokens==active) {save_reason="retained";return 0;}
        }
        size_t size=llama_state_get_size(ctx),draft_size=draft_ctx?llama_state_get_size(draft_ctx):0;
        size_t bytes=size+draft_size+active.size()*sizeof(llama_token);
        if(bytes>o.budget) {++skips;save_reason="budget";return 0;}
        for(auto i=cache.begin();i!=cache.end();) {
            if(i->session==r.session) i=cache.erase(i); else ++i;
        }
        while(!cache.empty() && (cache.size()>=size_t(o.slots) || used()+bytes>o.budget)) {cache.pop_front();++evictions;}
        if(auto *why=strata::core::conversation_memory_blocker(strata::core::conversation_memory_status(),bytes,o.floor)) {++skips;save_reason=why;return 0;}
        status("saving","prompt"); auto t=Clock::now();
        try {
            Snapshot s{++serial,r.session,active,{},{},epoch_ms()}; s.state.resize(size);s.draft_state.resize(draft_size);
            size_t written=llama_state_get_data(ctx,s.state.data(),size);
            if(written!=size) throw std::runtime_error("incomplete state snapshot");
            if(draft_ctx && llama_state_get_data(draft_ctx,s.draft_state.data(),draft_size)!=draft_size)
                throw std::runtime_error("incomplete DSpark state snapshot");
            cache.push_back(std::move(s));
            save_reason="ok";
        } catch(const std::bad_alloc&) {++skips;save_reason="allocation_failed";}
        return ms(t);
    }
public:
    Runner(llama_context*c,llama_context*d,const llama_vocab*v,Options opts,std::atomic<bool>&s,StrataExpertBudget b,VramControl &vc):
        ctx(c),draft_ctx(d),expert_budget(b),vram(vc),vocab(v),o(opts),stop(s) {
        if(draft_ctx)spec=std::make_unique<DSpark>(ctx,draft_ctx,o.draft_max,o.draft_min_confidence,o.draft_shared_scratch!=0);
        if(spec)std::cerr<<"STRATA_DSPARK_SCRATCH saved_bytes="<<spec->scratch_saved()<<" shared="<<spec->scratch_shared()<<"\n";
    }
    size_t scratch_saved() const {return spec?spec->scratch_saved():0;}
    bool scratch_shared() const {return spec && spec->scratch_shared();}
    void inventory() {
        std::cout<<"CACHE_ENTRIES {\"entries\":[";
        bool first=true;
        for(auto&s:cache) {
            if(!first)std::cout<<","; first=false;
            std::cout<<"{\"id\":\"session-"<<s.id<<"\",\"kind\":\"session\",\"deletable\":true,\"bytes\":"<<s.bytes()
                <<",\"tokens\":"<<s.tokens.size()<<",\"last_used_ms\":"<<s.used<<"}";
        }
        if(!active.empty()) {
            if(!first)std::cout<<",";
            std::cout<<"{\"id\":\"active\",\"kind\":\"active\",\"deletable\":false,\"bytes\":0,\"tokens\":"<<active.size()
                <<",\"last_used_ms\":"<<epoch_ms()<<"}";
        }
        std::cout<<"],\"updated_at_ms\":"<<epoch_ms()<<"}\n"<<std::flush;
    }
    void drop(const std::string&line) {
        std::istringstream in(line);std::string verb,nonce,id;in>>verb>>nonce>>id;
        auto it=std::find_if(cache.begin(),cache.end(),[&](auto&s){return id=="session-"+std::to_string(s.id);});
        bool found=it!=cache.end();if(found)cache.erase(it);
        inventory();status("idle","none");
        std::cout<<"CACHE_DROPPED "<<nonce<<" "<<(found?"removed":"missing")<<"\n"<<std::flush;
    }
    void run(const std::string&line) {
        Request r;
        int generated=0,offered=0,accepted=0;
        size_t reused=0;
        double prompt_ms=0;
        auto dt=Clock::now();
        try {
            r=request(line,o.context,llama_vocab_n_tokens(vocab));
            save_reason="not_saved";
            std::unique_ptr<llama_sampler,decltype(&llama_sampler_free)> smpl(sampler(r,llama_vocab_n_tokens(vocab)),llama_sampler_free);
            auto started=Clock::now();double restore=0,save_ms=0;const char*source="miss";
            if(active_session==r.session && prefix(active,r.tokens)) {source="active";++hits;}
            else {
                reset();
                auto best=cache.end();
                for(auto it=cache.begin();it!=cache.end();++it)
                    if(it->session==r.session && prefix(it->tokens,r.tokens) && (best==cache.end() || it->tokens.size()>best->tokens.size()))best=it;
                if(best!=cache.end()) {
                    status("restoring","ram");auto t=Clock::now();
                    if(llama_state_set_data(ctx,best->state.data(),best->state.size())!=best->state.size())
                        throw std::runtime_error("state restore failed");
                    if(draft_ctx && llama_state_set_data(draft_ctx,best->draft_state.data(),best->draft_state.size())!=best->draft_state.size())
                        throw std::runtime_error("DSpark state restore failed");
                    active=best->tokens;best->used=epoch_ms();restore=ms(t);source="ram";++hits;
                    Snapshot s=std::move(*best);cache.erase(best);cache.push_back(std::move(s));
                } else ++misses;
            }
            active_session=r.session;reused=active.size();
            std::cout<<"RESUME "<<reused<<"\n"<<std::flush;
            // Save BEFORE the last prompt token: an identical request can restore
            // this exact compressor state then recompute logits without a rewind.
            bool ok=decode(r.tokens,reused,r.tokens.size()-1,true);
            if(ok && !stop.load())save_ms=save(r);
            if(ok && !spec)ok=decode(r.tokens,active.size(),r.tokens.size(),true);
            prompt_ms=ms(started);dt=Clock::now();std::string finish="length";
            int rounds=0,proposed=0;double draft_ms=0,verify_ms=0;
            status("decoding",source,save_ms,restore);
            if(ok && spec) {
                llama_token pending=r.tokens.back();
                while(generated<r.count && !stop.load()) {
                    const int base=int(active.size());
                    auto t=Clock::now();budget(true);
                    auto draft=spec->propose(pending,base,r.count-generated-1);
                    double round_draft_ms=ms(t);
                    draft_ms+=round_draft_ms;proposed+=spec->last_proposed();offered+=int(draft.size());++rounds;
                    Batch batch(int(draft.size())+1);
                    batch.positions(int(draft.size())+1,base,true);
                    batch.value.token[0]=pending;
                    std::copy(draft.begin(),draft.end(),batch.value.token+1);
                    t=Clock::now();budget(false);
                    // Diagnostic causality probe: same anchor and batch shape,
                    // different future tokens. Never used in normal execution.
                    std::vector<uint8_t> probe_state;
                    if(std::getenv("STRATA_SPEC_CHECK_CAUSAL") && rounds<=3 && !draft.empty()) {
                        probe_state.resize(llama_state_get_size(ctx));
                        if(llama_state_get_data(ctx,probe_state.data(),probe_state.size())!=probe_state.size())
                            throw std::runtime_error("causality probe snapshot failed");
                    }
                    if(llama_decode(ctx,batch.value))throw std::runtime_error("DSpark target verification failed");
                    if(!probe_state.empty()) {
                        auto restore_probe=[&] {
                            if(llama_state_set_data(ctx,probe_state.data(),probe_state.size())!=probe_state.size())
                                throw std::runtime_error("causality probe restore failed");
                        };
                        const auto *logits=llama_get_logits_ith(ctx,0);
                        std::vector<float> expected(logits,logits+llama_vocab_n_tokens(vocab));
                        restore_probe();
                        for(size_t i=0;i<draft.size();++i)batch.value.token[i+1]=(draft[i]+17)%llama_vocab_n_tokens(vocab);
                        if(llama_decode(ctx,batch.value))throw std::runtime_error("causality probe decode failed");
                        logits=llama_get_logits_ith(ctx,0);float diff=0;
                        for(size_t i=0;i<expected.size();++i)diff=std::max(diff,std::abs(expected[i]-logits[i]));
                        std::cerr<<"STRATA_SPEC_CAUSAL pos="<<base<<" max_logit_diff="<<diff
                            <<" original="<<(std::max_element(expected.begin(),expected.end())-expected.begin())
                            <<" changed="<<(std::max_element(logits,logits+expected.size())-logits)<<"\n";
                        restore_probe();std::copy(draft.begin(),draft.end(),batch.value.token+1);
                        if(llama_decode(ctx,batch.value))throw std::runtime_error("causality probe replay failed");
                    }
                    if(stop.load()) {ok=false;break;}
                    auto verified=verify_draft(draft,r.count-generated,
                        [&](int row){return llama_sampler_sample(smpl.get(),ctx,row);},
                        [&](llama_token token){return llama_vocab_is_eog(vocab,token);});
                    if(std::getenv("STRATA_SPEC_TRACE") && rounds<=5) {
                        std::cerr<<"STRATA_SPEC_TRACE source="<<source<<" pos="<<base<<" draft=";
                        for(auto token:draft)std::cerr<<token<<",";
                        std::cerr<<" output=";
                        for(auto token:verified.tokens)std::cerr<<token<<",";
                        const float *logits=llama_get_logits_ith(ctx,0);
                        auto best=std::max_element(logits,logits+llama_vocab_n_tokens(vocab));
                        std::cerr<<" top="<<(best-logits)<<":"<<std::setprecision(9)<<*best<<"\n";
                    }
                    // Keep anchor + accepted inputs, leave the last sampled
                    // output pending. Recurrent compressor rollback is bounded
                    // by n_rs_seq and must succeed before emitting any tokens.
                    int keep=int(verified.tokens.size());
                    if(!llama_memory_seq_rm(llama_get_memory(ctx),0,base+keep,-1))
                        throw std::runtime_error("DeepSeek speculative rollback failed");
                    batch.value.n_tokens=keep;inject(batch.value);
                    active.insert(active.end(),batch.value.token,batch.value.token+keep);
                    double round_verify_ms=ms(t);
                    verify_ms+=round_verify_ms;accepted+=verified.accepted;
                    if(std::getenv("STRATA_SPEC_CONFIDENCE_TRACE")) {
                        std::cerr<<"STRATA_SPEC_ROUND round="<<rounds<<" proposed="<<spec->last_proposed()
                            <<" offered="<<draft.size()<<" accepted="<<verified.accepted
                            <<" draft_ms="<<round_draft_ms<<" verify_ms="<<round_verify_ms<<" confidence=";
                        for(float p:spec->last_confidence())std::cerr<<p<<",";
                        std::cerr<<"\n";
                    }
                    for(auto token:verified.tokens) {++generated;std::cout<<"T "<<token<<"\n"<<std::flush;}
                    pending=verified.tokens.back();
                    if(verified.eog) {finish="stop";break;}
                    if(generated==r.count)ok=decode(std::vector<llama_token>{pending},0,1,false);
                }
            } else if(ok) for(int i=0;i<r.count && !stop.load();++i) {
                auto token=llama_sampler_sample(smpl.get(),ctx,-1);
                ++generated;std::cout<<"T "<<token<<"\n"<<std::flush;
                if(llama_vocab_is_eog(vocab,token)) {finish="stop";break;}
                // Keep the active state and its token history in agreement, even
                // at the output limit. Cancellation invalidates partial graphs.
                if(!decode(std::vector<llama_token>{token},0,1,false)) {ok=false;break;}
            }
            if(!ok || stop.load()) {finish="cancel";reset();}
            if(spec)std::cerr<<"STRATA_SPEC type=dspark rounds="<<rounds<<" offered="<<offered<<" accepted="<<accepted
                <<" draft_ms="<<draft_ms<<" verify_ms="<<verify_ms<<" proposed="<<proposed<<" filtered="<<proposed-offered
                <<" shared_scratch="<<spec->scratch_shared()<<"\n";
            inventory();status("idle",source,save_ms,restore);
            std::cout<<"DONE "<<generated<<" "<<r.tokens.size()<<" "<<prompt_ms<<" "<<ms(dt)<<" "<<finish<<" "<<accepted<<" "<<offered<<" "<<reused<<"\n"<<std::flush;
        } catch(const std::exception&e) {
            reset();
            if(stop.load()) {
                inventory();status("idle","none",0,0,"cancelled");
                std::cout<<"DONE "<<generated<<" "<<r.tokens.size()<<" "<<prompt_ms<<" "<<ms(dt)<<" cancel "<<accepted<<" "<<offered<<" "<<reused<<"\n"<<std::flush;
            } else std::cout<<"ERR "<<e.what()<<"\n"<<std::flush;
        }
    }
};

static void tokenizer_loop(const llama_vocab*vocab) {
    std::cout<<"READY_TOKENIZER\n"<<std::flush;
    std::string line;
    while(std::getline(std::cin,line)) {
        if(line=="QUIT")break;
        try {
            std::istringstream in(line);std::string op,hex,text;int special;in>>op>>special>>hex;
            if(op!="ENC" || hex.size()%2)throw std::runtime_error("invalid ENC request");
            for(size_t i=0;i<hex.size();i+=2)text.push_back(char(std::stoi(hex.substr(i,2),nullptr,16)));
            int n=-llama_tokenize(vocab,text.data(),int(text.size()),nullptr,0,false,special!=0);
            std::vector<llama_token> ids(std::max(n,0));
            n=llama_tokenize(vocab,text.data(),int(text.size()),ids.data(),int(ids.size()),false,special!=0);
            if(n<0)throw std::runtime_error("tokenization failed");
            std::cout<<"IDS";for(int i=0;i<n;++i)std::cout<<" "<<ids[i];std::cout<<"\n"<<std::flush;
        }catch(const std::exception&e){std::cout<<"ERR "<<e.what()<<"\n"<<std::flush;}
    }
}
int main(int argc,char**argv) {
    try {
        Options o=options(argc,argv);
        int layers=0;
        {
            strata::GgufFile header(o.model);
            auto *architecture=header.get("general.architecture");
            auto *blocks=header.get("deepseek4.block_count");
            if(!architecture || architecture->s!="deepseek4" || !blocks || blocks->u<1 || blocks->u>512)
                throw std::runtime_error("this backend requires the first deepseek4 GGUF shard; use strata.exe for Qwen");
            layers=int(blocks->u);
        }
        if(!o.draft_model.empty()) {
            strata::GgufFile header(o.draft_model);
            auto require=[&](const char*key,uint64_t value) {
                auto *v=header.get(key);
                if(!v || v->u!=value)throw std::runtime_error(std::string("incompatible 0731 DSpark metadata: ")+key);
            };
            auto *a=header.get("general.architecture");
            if(!a || a->s!="dflash")throw std::runtime_error("--draft-model requires the 0731 DSpark GGUF (architecture=dflash)");
            require("dflash.block_count",3);require("dflash.embedding_length",4096);
            require("dflash.hyper_connection.count",4);require("dflash.block_size",5);
            // These optional keys default to true/false in pinned llama.cpp.
            if(auto*v=header.get("dflash.sample_from_anchor");v && !v->u)
                throw std::runtime_error("DSpark requires anchor-first sampling");
            if(auto*v=header.get("dflash.attention.causal");v && v->u)
                throw std::runtime_error("DSpark requires non-causal draft attention");
            bool markov=false,confidence=false;
            for(auto&t:header.tensors()) {
                if(t.name=="markov_w1.weight")markov=true;
                if(t.name=="conf_proj.weight")confidence=true;
            }
            if(!markov)throw std::runtime_error("DSpark Markov head is missing");
            if(o.draft_min_confidence>0 && !confidence)throw std::runtime_error("DSpark confidence head is missing");
            if(layers!=43)throw std::runtime_error("DSpark sidecar requires a 43-layer 0731 target");
        }
        if(o.gpu_expert_layers>layers || (!o.vocab_only && o.gpu_layers>=0 && o.gpu_layers<layers+1))
            throw std::runtime_error("GPU-only inference requires all dense layers offloaded");
        llama_log_set([](ggml_log_level level,const char*text,void*) {
            if(level!=GGML_LOG_LEVEL_DEBUG)std::fputs(text,stderr);
        },nullptr);
        // GGML normally offloads host-backed operations only for batches >=32.
        // Decode is batch=1: explicitly stream selected experts for that path too.
#ifdef _WIN32
        _putenv_s("GGML_OP_OFFLOAD_MIN_BATCH","1");
        _putenv_s("STRATA_EXPERT_CACHE_MIB",std::to_string(o.expert_cache_mib).c_str());
        _putenv_s("STRATA_EXPERT_CACHE_POLICY",o.expert_cache_policy.c_str());
        _putenv_s("STRATA_EXPERT_CACHE_MATCH_SIZE",std::to_string(o.expert_cache_match_size).c_str());
        _putenv_s("STRATA_EXPERT_STAGE_MIB",std::to_string(o.expert_stage_mib).c_str());
        _putenv_s("STRATA_EXPERT_PIPELINE",std::to_string(o.expert_pipeline).c_str());
        _putenv_s("STRATA_EXPERT_READERS",std::to_string(o.expert_readers).c_str());
        _putenv_s("STRATA_EXPERT_READ_MODE",o.expert_read_mode.c_str());
#else
        setenv("GGML_OP_OFFLOAD_MIN_BATCH","1",1);
        setenv("STRATA_EXPERT_CACHE_MIB",std::to_string(o.expert_cache_mib).c_str(),1);
        setenv("STRATA_EXPERT_CACHE_POLICY",o.expert_cache_policy.c_str(),1);
        setenv("STRATA_EXPERT_CACHE_MATCH_SIZE",std::to_string(o.expert_cache_match_size).c_str(),1);
        setenv("STRATA_EXPERT_STAGE_MIB",std::to_string(o.expert_stage_mib).c_str(),1);
        setenv("STRATA_EXPERT_PIPELINE",std::to_string(o.expert_pipeline).c_str(),1);
        setenv("STRATA_EXPERT_READERS",std::to_string(o.expert_readers).c_str(),1);
        setenv("STRATA_EXPERT_READ_MODE",o.expert_read_mode.c_str(),1);
#endif
        ggml_backend_load_all();llama_backend_init();
        auto gpu=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if(!o.vocab_only && !gpu)throw std::runtime_error("DeepSeek inference requires a GPU backend; CPU fallback is disabled");
        if(!o.vocab_only && (o.expert_cache_mib || o.expert_stage_mib)) {
            auto reg=ggml_backend_dev_backend_reg(gpu);
            if(!ggml_backend_reg_get_proc_address(reg,"strata_expert_copy"))
                throw std::runtime_error("this GPU backend does not support the requested expert cache/staging path");
        }
        auto budget_gpu=gpu?reinterpret_cast<StrataExpertBudget>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(gpu),"strata_expert_budget")):nullptr;
        auto control_gpu=gpu?reinterpret_cast<StrataExpertControl>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(gpu),"strata_expert_control")):nullptr;
        if(!o.vocab_only && !o.draft_model.empty() && o.draft_gpu_expert_layers<3 && !budget_gpu)
            throw std::runtime_error("streamed DSpark requires CUDA expert budget support");
        auto mp=llama_model_default_params();mp.n_gpu_layers=o.vocab_only?0:o.gpu_layers;
        mp.split_mode=LLAMA_SPLIT_MODE_NONE;mp.load_mode=LLAMA_LOAD_MODE_MMAP;
        mp.use_extra_bufts=false;mp.vocab_only=o.vocab_only;mp.load_mtp=false;
        // Host storage is separate from execution. The scheduler copies only the
        // selected expert slices into reusable GPU buffers for MUL_MAT_ID.
        auto cpu=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        std::vector<std::string> patterns;
        for(int layer=0;layer<layers-o.gpu_expert_layers;++layer)
            patterns.push_back("blk\\."+std::to_string(layer)+"\\.ffn_(gate|up|down)_exps\\.weight");
        std::vector<llama_model_tensor_buft_override> overrides;
        for(auto&pattern:patterns)overrides.push_back({pattern.c_str(),ggml_backend_dev_buffer_type(cpu)});
        if(!o.vocab_only)overrides.push_back({"token_embd\\.weight",ggml_backend_dev_buffer_type(gpu)});
        overrides.push_back({nullptr,nullptr});mp.tensor_buft_overrides=overrides.data();
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(o.model.c_str(),mp),llama_model_free);
        if(!model)throw std::runtime_error("could not load model (are all GGUF shards fully copied?)");
        char arch[64]={};llama_model_meta_val_str(model.get(),"general.architecture",arch,sizeof(arch));
        if(std::string(arch)!="deepseek4")throw std::runtime_error("this backend requires deepseek4; use strata.exe for Qwen");
        auto*vocab=llama_model_get_vocab(model.get());
        if(o.vocab_only){tokenizer_loop(vocab);return 0;}
        std::atomic<bool> stop{false};
        auto cp=llama_context_default_params();cp.n_ctx=o.context;cp.n_batch=o.batch;cp.n_ubatch=o.batch;
        cp.n_seq_max=1;cp.n_threads=o.threads;cp.n_threads_batch=o.threads;cp.no_perf=false;
        cp.n_rs_seq=o.draft_model.empty()?0:o.draft_max;
        if(!o.draft_model.empty())cp.n_outputs_max=cp.n_outputs_max_per_seq=o.draft_max+1;
        cp.op_offload=true;cp.offload_kqv=true;
        cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.abort_callback=[](void*p){return static_cast<std::atomic<bool>*>(p)->load();};cp.abort_callback_data=&stop;
        std::unique_ptr<llama_context,decltype(&llama_free)>ctx(llama_init_from_model(model.get(),cp),llama_free);
        if(!ctx)throw std::runtime_error("could not create DeepSeek context");
        // Declare the draft after the target: borrowed embeddings/output tensors
        // must outlive the draft model, context and speculative adapter.
        std::unique_ptr<llama_model,decltype(&llama_model_free)>draft_model(nullptr,llama_model_free);
        std::unique_ptr<llama_context,decltype(&llama_free)>draft_ctx(nullptr,llama_free);
        if(!o.draft_model.empty()) {
            auto dp=llama_model_default_params();dp.n_gpu_layers=99;dp.split_mode=LLAMA_SPLIT_MODE_NONE;
            dp.load_mode=LLAMA_LOAD_MODE_MMAP;dp.use_extra_bufts=false;
            std::vector<std::string> patterns;
            for(int layer=0;layer<3-o.draft_gpu_expert_layers;++layer)
                patterns.push_back("blk\\."+std::to_string(layer)+"\\.ffn_(gate|up|down)_exps\\.weight");
            std::vector<llama_model_tensor_buft_override> placement;
            for(auto&p:patterns)placement.push_back({p.c_str(),ggml_backend_dev_buffer_type(cpu)});
            placement.push_back({nullptr,nullptr});dp.tensor_buft_overrides=placement.data();
            draft_model.reset(llama_model_load_from_file(o.draft_model.c_str(),dp));
            if(!draft_model)throw std::runtime_error("could not load DSpark model");
            auto dc=cp;dc.n_rs_seq=0;dc.ctx_other=ctx.get();
            // Draft noise has at most five tokens. Feature injection is cheap
            // and chunked, so avoid reserving a 4096-token draft MoE graph.
            dc.n_batch=dc.n_ubatch=std::min(o.batch,256);
            dc.n_outputs_max=dc.n_outputs_max_per_seq=o.draft_max;
            draft_ctx.reset(llama_init_from_model(draft_model.get(),dc));
            if(!draft_ctx)throw std::runtime_error("could not create DSpark context");
        }
        const char *vram_policy=std::getenv("STRATA_VRAM_POLICY");
        // split_mode=NONE uses CUDA device 0 (after CUDA_VISIBLE_DEVICES).
        VramControl vram(control_gpu,0);
        if(vram_policy && *vram_policy) {
            vram.enqueue(std::string("VRAM_SET ")+vram_policy);vram.poll();
            if(!vram.configured())throw std::runtime_error("invalid or unsupported STRATA_VRAM_POLICY");
        }
        // Reserve final graph outputs and optionally share their temporary
        // buffer before checking space for the lazy expert caches.
        Runner runner(ctx.get(),draft_ctx.get(),vocab,o,stop,budget_gpu,vram);
        if((o.expert_cache_mib || draft_ctx || o.expert_pipeline) && !vram.configured()) {
            size_t free=0,total=0;ggml_backend_dev_memory(gpu,&free,&total);
            auto draft_cache=draft_ctx && o.draft_gpu_expert_layers<3?o.draft_expert_cache_mib:0;
            auto pipeline_mib=o.expert_pipeline*4*o.expert_stage_mib*((o.gpu_expert_layers<layers)+(draft_ctx && o.draft_gpu_expert_layers<3));
            if(uint64_t(free)<(uint64_t(o.expert_cache_mib)+draft_cache+pipeline_mib+512)*1048576)
                throw std::runtime_error("not enough free VRAM for expert caches, pipeline buffers and 512 MiB reserve; reduce cache budgets or resident expert layers");
        }
        if(o.working_set) {
#ifdef _WIN32
            if(o.working_set<(512ULL<<20) || !SetProcessWorkingSetSizeEx(GetCurrentProcess(),256ULL<<20,o.working_set,QUOTA_LIMITS_HARDWS_MAX_ENABLE))
                throw std::runtime_error("could not apply Windows process working-set cap: "+std::to_string(GetLastError()));
#else
            throw std::runtime_error("--working-set-mib is Windows-only; use OS memory controls on Linux");
#endif
        }
        llama_memory_breakdown_data draft_memory;
        if(draft_ctx)for(const auto &[buft,data]:llama_get_memory_breakdown(draft_ctx.get())) {
            auto *device=ggml_backend_buft_get_device(buft);
            if(device && !ggml_backend_buft_is_host(buft) && ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_GPU) {
                draft_memory.model+=data.model;draft_memory.context+=data.context;draft_memory.compute+=data.compute;
            }
        }
        // A shared physical compute buffer is attributed to the target once.
        // The raw per-context allocator sizes both include that same buffer.
        if(runner.scratch_shared())draft_memory.compute=0;
        std::cout<<"INFO engine=0.1.35-deepseek4 architecture=deepseek4 backend=llama.cpp mtp=0 spec="<<(draft_ctx?o.draft_max:0)
            <<" speculative="<<(draft_ctx?"dspark":"none")<<" expert_storage=mmap expert_compute=gpu gpu_only=1"
            <<" gpu_expert_layers="<<o.gpu_expert_layers<<" expert_cache_mib="<<o.expert_cache_mib<<" expert_stage_mib="<<o.expert_stage_mib
            <<" expert_cache_policy="<<o.expert_cache_policy
            <<" expert_cache_match_size="<<o.expert_cache_match_size
            <<" expert_pipeline="<<o.expert_pipeline<<" expert_pipeline_slots="<<(o.expert_pipeline?4:0)
            <<" expert_readers="<<(o.expert_pipeline?o.expert_readers:0)<<" expert_read_mode="<<o.expert_read_mode
            <<" draft_expert_cache_mib="<<(draft_ctx && o.draft_gpu_expert_layers<3?o.draft_expert_cache_mib:0)<<" draft_gpu_expert_layers="<<(draft_ctx?o.draft_gpu_expert_layers:0)
            <<" draft_shared_scratch="<<runner.scratch_shared()<<" draft_shared_scratch_saved_bytes="<<runner.scratch_saved()
            <<" draft_vram_weights_bytes="<<draft_memory.model<<" draft_vram_context_bytes="<<draft_memory.context<<" draft_vram_compute_bytes="<<draft_memory.compute
            <<" draft_vram_pipeline_bytes="<<(draft_ctx && o.draft_gpu_expert_layers<3?uint64_t(o.expert_pipeline)*4*o.expert_stage_mib*1048576:0)
            <<" draft_min_confidence="<<(draft_ctx?o.draft_min_confidence:0)
            <<" working_set_mib="<<(o.working_set>>20)<<" kv=fp16\n"
            <<"READY "<<o.context<<" stop session-id cache-admin"<<(control_gpu?" vram-control":"")<<"\n"<<std::flush;
        std::mutex mutex;std::condition_variable cv;std::deque<std::string> commands;bool ended=false;
        std::thread reader([&]{
            std::string line;while(std::getline(std::cin,line)) {
                if(line=="STOP") {stop.store(true);continue;}
                if(line=="QUIT")break;
                if(line.rfind("VRAM_SET ",0)==0) {vram.enqueue(line);cv.notify_one();continue;}
                {std::lock_guard lock(mutex);if(line.rfind("GEN ",0)==0)stop.store(false);commands.push_back(line);}cv.notify_one();
            }
            {std::lock_guard lock(mutex);ended=true;stop.store(true);}cv.notify_one();
        });
        while(true) {
            std::string line;
            {std::unique_lock lock(mutex);if(!ended && commands.empty())cv.wait_for(lock,std::chrono::seconds(1));
             if(ended)break;if(!commands.empty()){line=std::move(commands.front());commands.pop_front();}}
            vram.poll();
            if(line.empty())continue;
            if(line.rfind("CACHE_DROP ",0)==0)runner.drop(line);else runner.run(line);
        }
        reader.join();return 0;
    } catch(const std::exception&e) {std::cerr<<"strata-deepseek4: "<<e.what()<<"\n";return 1;}
}
