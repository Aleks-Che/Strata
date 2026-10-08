// Offline greedy DFlash benchmark; no server/default capability is enabled.
#include "dflash_decode.hpp"
#include "nlohmann/json.hpp"
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
using namespace minimax_m2;
using json=nlohmann::ordered_json;
using Clock=std::chrono::steady_clock;
static std::atomic<bool> cancelled{false};
static void interrupt(int) {cancelled.store(true);}
static double ms(Clock::time_point t) {return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct RoutingTrace {
    int wanted=-1,pos=-1,count=0;bool active=false;json nodes=json::array();
    static bool callback(ggml_tensor *t,bool ask,void *data) {
        auto &o=*static_cast<RoutingTrace *>(data);const std::string name=t->name;
        if(!o.active || o.wanted<o.pos || o.wanted>=o.pos+o.count || t->ne[2]!=1 || t->ne[3]!=1 ||
           !(name.rfind("ffn_moe_logits-",0)==0 || name.rfind("ffn_moe_probs_biased-",0)==0 || name.rfind("ffn_moe_topk-",0)==0 ||
             name.rfind("attn_norm-",0)==0 || name.rfind("ffn_norm-",0)==0 || name.rfind("ffn_moe_out-",0)==0 || name.rfind("l_out-",0)==0))return false;
        if(ask)return true;const int row=o.wanted-o.pos;require(row<t->ne[1],"trace row mismatch");
        json v;if(t->type==GGML_TYPE_F32) {std::vector<float> x(t->ne[0]);ggml_backend_tensor_get(t,x.data(),row*t->nb[1],x.size()*4);v=x;}
        else if(t->type==GGML_TYPE_I32) {std::vector<int> x(t->ne[0]);ggml_backend_tensor_get(t,x.data(),row*t->nb[1],x.size()*4);v=x;}
        else throw std::runtime_error("unexpected trace type");
        o.nodes.push_back({{"name",name},{"position",o.wanted},{"batch_position",o.pos},{"batch_count",o.count},{"values",v}});return true;
    }
};
static int self_test() {
    int cases=0;
    for(int depth=0;depth<=7;++depth)for(int mismatch=0;mismatch<=depth;++mismatch)
    for(int budget=1;budget<=9;++budget)for(int eos=-1;eos<=mismatch;++eos) {
        std::vector<llama_token> proposals(depth),target(depth+1);
        for(int i=0;i<=depth;++i) {target[i]=100+i;if(i<depth)proposals[i]=i<mismatch?target[i]:300+i;}
        if(eos>=0) {target[eos]=200020;if(eos<mismatch)proposals[eos]=200020;}
        int emitted=std::min(budget,mismatch+1);if(eos>=0)emitted=std::min(emitted,eos+1);
        std::vector<llama_token> expected(target.begin(),target.begin()+emitted);int calls=0;
        auto v=verify_greedy(proposals,budget,[&](int row){require(row==calls++,"nonsequential verification");return target[row];});
        const int accepted=std::min(emitted,mismatch);
        require(v.tokens==expected && v.accepted==accepted && v.keep==std::min(depth+1,accepted+1) && calls==emitted,"greedy prefix test failed");++cases;
    }
    std::cout<<json({{"pass",true},{"cases",cases},{"scope","greedy acceptance, mismatch/bonus, EOS and output budget"}}).dump()<<'\n';return 0;
}
static json snapshot() {
    const auto s=strata_mm27_snapshot();auto m=strata_mm27_memory();
    require(!s.rejected_cpu_nodes && !s.rejected_full_copies && !s.pipeline_queued_bytes && !s.pipeline_reader_owned_bytes,"GPU/drain audit failed");
    require(double(s.sampled_vram_used_peak)<=.95*m.vram_total && double(s.sampled_ram_used_peak)<=.95*m.ram_total,"sampled memory budget failed");
    return {{"gpu_nodes",s.gpu_nodes},{"expert_nodes",s.expert_nodes},{"rejected_cpu_nodes",s.rejected_cpu_nodes},{"rejected_full_copies",s.rejected_full_copies},
        {"pipeline_queued_bytes",s.pipeline_queued_bytes},{"pipeline_reader_owned_bytes",s.pipeline_reader_owned_bytes},
        {"h2d_bytes",s.h2d_bytes},{"selected_bytes",s.selected_bytes},{"cache_hit_bytes",s.cache_hit_bytes},{"cache_resident",s.cache_resident},
        {"cache_limit",s.cache_limit},{"arena_reserved",s.arena_reserved},{"memory_checks",s.memory_checks},{"pressure_rejections",s.pressure_rejections},
        {"ram_used_peak",s.sampled_ram_used_peak},{"ram_total",m.ram_total},{"vram_used_peak",s.sampled_vram_used_peak},{"vram_total",m.vram_total}};
}
int main(int argc,char **argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--self-test")return self_test();
        if(argc==2 && std::string(argv[1])=="--version") {std::cout<<"MiniMax offline greedy DFlash "<<STRATA_MM27_SOURCE_SHA<<'\n';return 0;}
        std::string path,draft_path,input,output;int depth=0,cache_mib=18432;RoutingTrace trace;std::string tokenwise="0";
        for(int i=1;i<argc;++i) {const std::string k=argv[i];require(i+1<argc,"missing argument");const std::string v=argv[++i];
            if(k=="--model")path=v;else if(k=="--draft")draft_path=v;else if(k=="--requests")input=v;else if(k=="--out")output=v;
            else if(k=="--trace-position")trace.wanted=std::stoi(v);
            else if(k=="--verify-tokenwise") {require(v.size()==1 && v[0]>='0' && v[0]<='3',"invalid tokenwise mode");tokenwise=v;}
            else if(k=="--depth" || k=="--cache-mib") {size_t end=0;const int n=std::stoi(v,&end);require(end==v.size(),"invalid integer");if(k=="--depth")depth=n;else cache_mib=n;}
            else throw std::runtime_error("unknown option: "+k);
        }
        require(!path.empty() && !input.empty() && !output.empty() && depth>=0 && depth<=7 &&
            ((depth==0)==draft_path.empty()) && cache_mib>=0 && cache_mib<=24576,"invalid benchmark options");
        const std::filesystem::path dir=output;require(!std::filesystem::exists(dir),"fresh output directory required");std::filesystem::create_directories(dir);
        json requests;{std::ifstream f(input);require(bool(f),"cannot read requests");f>>requests;}
        require(requests.is_array() && !requests.empty() && requests.size()<=32,"expected request array");
        auto set_tokenwise=[&](bool enabled) {
#ifdef _WIN32
            const int rc=_putenv_s("STRATA_MM27_TOKENWISE",enabled?tokenwise.c_str():"0");
#else
            const int rc=setenv("STRATA_MM27_TOKENWISE",enabled?tokenwise.c_str():"0",1);
#endif
            require(rc==0,"cannot set verification precision mode");
        };
        set_tokenwise(false);environment();ggml_backend_load_all();strata_mm27_mode(2);strata_mm27_pipeline(2,4,true,true);
        strata_mm27_cache_configure(uint64_t(cache_mib)<<20,true,64,0);strata_mm27_cancel(&cancelled);std::signal(SIGINT,interrupt);
        const auto loading=Clock::now();auto model=load(path);Context ctx(nullptr,llama_free);
        if(trace.wanted<0)ctx=context(model.get(),512,8);
        else {
            strata_mm27_cache_clear();auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=8;cp.n_seq_max=1;
            cp.n_threads=cp.n_threads_batch=4;cp.type_k=cp.type_v=GGML_TYPE_F32;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=cp.op_offload=true;
            cp.cb_eval=RoutingTrace::callback;cp.cb_eval_user_data=&trace;ctx.reset(llama_init_from_model(model.get(),cp));require(bool(ctx),"trace context failed");clear(ctx.get());strata_mm27_memory();
        }
        std::unique_ptr<DFlashDecode> draft;if(depth)draft=std::make_unique<DFlashDecode>(ctx.get(),path,draft_path);
        // Warm the required scratch before the expert cache fills. Both modes
        // perform the same eight-token prefill; speculation also needs all rows.
        strata_mm27_cache_decode(false);std::vector<llama_token> warm(8,11);set_tokenwise(depth>0);decode(ctx.get(),warm,0,8,0,depth>0);set_tokenwise(false);
        if(draft) {draft->process(8,0);draft->propose(11,8,depth);draft->reset();}
        clear(ctx.get());const double load_ms=ms(loading);
        json report={{"pass",false},{"source_revision",STRATA_MM27_SOURCE_SHA},{"patches",STRATA_MM27_PATCH_SET},
            {"model",path},{"draft",draft_path},{"depth",depth},{"cache_mib",cache_mib},{"context",512},{"batch",8},
            {"KV","F32"},{"flash_attention",false},{"graphs",false},{"target_strict_f32",true},{"pipeline_readers",2},{"pipeline_chunk_mib",4},
            {"load_and_warmup_ms",load_ms},{"trace_position",trace.wanted},{"verify_tokenwise",std::stoi(tokenwise)},{"results",json::array()}};
        for(size_t request_index=0;request_index<requests.size();++request_index) {
            const auto &req=requests[request_index];const std::string prompt=req.at("prompt");const int budget=req.at("max_tokens");
            require(!prompt.empty() && prompt.size()<=1u<<20 && budget>=1 && budget<=256,"invalid request");
            const auto *vocab=llama_model_get_vocab(model.get());const int nt=llama_tokenize(vocab,prompt.data(),int(prompt.size()),nullptr,0,false,true);
            require(nt<0 && -nt+budget+8<=512,"prompt/context budget");std::vector<llama_token> tokens(-nt);
            require(llama_tokenize(vocab,prompt.data(),int(prompt.size()),tokens.data(),int(tokens.size()),false,true)==-nt,"tokenization failed");
            trace.active=false;trace.nodes=json::array();clear(ctx.get());if(draft)draft->reset();strata_mm27_cache_decode(false);strata_mm27_reset();
            std::vector<float> saved;saved.reserve(size_t(budget)*target_vocab);std::vector<llama_token> out;out.reserve(budget);
            const auto start=Clock::now();
            for(size_t i=0;i<tokens.size();i+=8) {const int n=int(std::min<size_t>(8,tokens.size()-i));decode(ctx.get(),tokens,i,n,int(i));if(draft)draft->process(n,int(i));}
            const double prefill_ms=ms(start);auto prefill=snapshot();strata_mm27_reset();
            auto sample=[&](int row) {auto id=greedy(ctx.get(),row,target_vocab);const float *p=llama_get_logits_ith(ctx.get(),row);saved.insert(saved.end(),p,p+target_vocab);return id;};
            out.push_back(sample(-1));const double ttft_ms=ms(start);int pos=int(tokens.size()),proposed=0,accepted=0,forward=0,cycles=0;
            double draft_ms=0,verify_ms=0,catchup_ms=0;json accept_counts=json::array(),proposal_counts=json::array();
            const auto generation=Clock::now();
            while(int(out.size())<budget && !is_stop(out.back())) {
                require(!cancelled.load(),"cancelled");const int n=std::min(depth,budget-int(out.size())-1);
                auto tick=Clock::now();auto proposals=draft?draft->propose(out.back(),pos,n):std::vector<llama_token>{};draft_ms+=ms(tick);
                std::vector<llama_token> verify{out.back()};verify.insert(verify.end(),proposals.begin(),proposals.end());
                trace.pos=pos;trace.count=int(verify.size());trace.active=trace.wanted>=0;set_tokenwise(true);
                tick=Clock::now();decode(ctx.get(),verify,0,int(verify.size()),pos,true);verify_ms+=ms(tick);forward+=int(verify.size());
                set_tokenwise(false);
                auto v=verify_greedy(proposals,budget-int(out.size()),sample);out.insert(out.end(),v.tokens.begin(),v.tokens.end());
                tick=Clock::now();require(llama_memory_seq_rm(llama_get_memory(ctx.get()),0,pos+v.keep,-1),"target rollback failed");
                if(draft)draft->process(v.keep,pos);catchup_ms+=ms(tick);pos+=v.keep;
                require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==pos-1 && (!draft || draft->position()==pos-1),"target/draft history position mismatch");
                proposed+=int(proposals.size());accepted+=v.accepted;++cycles;accept_counts.push_back(v.accepted);proposal_counts.push_back(proposals.size());
                // The first decode can allocate serial/batched scratch. Admit
                // cache entries only after that workspace has been established.
                strata_mm27_cache_decode(true);
            }
            const double generation_ms=ms(generation),request_ms=ms(start);auto decode_stats=snapshot();
            if(trace.wanted>=0)std::ofstream(dir/(std::to_string(request_index)+"-trace.json"))<<trace.nodes.dump(2)<<'\n';
            const auto logit_file=std::to_string(request_index)+".f32";{std::ofstream f(dir/logit_file,std::ios::binary);f.write(reinterpret_cast<const char *>(saved.data()),saved.size()*sizeof(float));require(bool(f),"cannot save logits");}
            std::string text;for(auto id:out) {std::vector<char> piece(256);int n=llama_token_to_piece(vocab,id,piece.data(),int(piece.size()),0,true);if(n<0){piece.resize(-n);n=llama_token_to_piece(vocab,id,piece.data(),int(piece.size()),0,true);}require(n>=0,"token piece failed");text.append(piece.data(),n);}
            json result={{"name",req.value("name",std::to_string(request_index))},{"prompt_tokens",tokens.size()},{"generated_tokens",out.size()},
                {"ids",out},{"text",text},{"stop_reason",is_stop(out.back())?"eos":"length"},{"prefill_ms",prefill_ms},{"ttft_ms",ttft_ms},
                {"generation_ms",generation_ms},{"request_ms",request_ms},{"tokens_per_second",1000.0*(out.size()-1)/generation_ms},
                {"draft_ms",draft_ms},{"verify_ms",verify_ms},{"catchup_ms",catchup_ms},{"proposed",proposed},{"accepted",accepted},
                {"cycles",cycles},{"target_forward_tokens",forward},{"accepted_counts",accept_counts},{"proposal_counts",proposal_counts},
                {"logits_file",logit_file},{"prefill",prefill},{"decode",decode_stats}};
            report["results"].push_back(result);std::ofstream(dir/"report.json")<<report.dump(2)<<'\n';
            std::cout<<json({{"request",request_index},{"depth",depth},{"tokens_per_second",result["tokens_per_second"]},{"accepted",accepted},{"proposed",proposed}}).dump()<<'\n'<<std::flush;
        }
        draft.reset();ctx.reset();model.reset();strata_mm27_release();report["pass"]=true;std::ofstream(dir/"report.json")<<report.dump(2)<<'\n';return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';strata_mm27_release();return 1;}
}
