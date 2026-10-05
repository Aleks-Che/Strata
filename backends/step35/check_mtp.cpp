// Offline experiment only. Greedy text, one sequence, fresh KV per request.
// The short position limit deliberately excludes SWA eviction/rollback admission.
#include "runtime.hpp"
#include "cache_registry.hpp"
#include "llama-ext.h"
#include "speculative.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

using namespace step35;
using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }

static Model load_draft(const std::string & path, llama_model * target, bool shared_embedding, int active_heads) {
    auto mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_name("CUDA0"), nullptr};
    mp.devices = devices; mp.n_gpu_layers = -1; mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    mp.load_mtp = true; mp.no_host = true; mp.use_extra_bufts = false; mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    auto * cpu = ggml_backend_dev_by_name("CPU");
    const std::string inactive=active_heads==1 ? "^blk\\.(46|47)\\." : "^blk\\.47\\.";
    llama_model_tensor_buft_override overrides[] = {
        {shared_embedding ? "^(token_embd|output|output_norm)\\.weight$" : "^token_embd\\.weight$",
         ggml_backend_dev_buffer_type(shared_embedding ? cpu : devices[0])},
        {nullptr,nullptr}, {nullptr,nullptr}};
    if (active_heads<3) overrides[1]={inactive.c_str(),ggml_backend_dev_buffer_type(cpu)};
    mp.tensor_buft_overrides = overrides;
    Model model(llama_model_load_from_file(path.c_str(), mp), llama_model_free);
    require(bool(model), "draft load failed");
    require(model->arch == LLM_ARCH_STEP35 && model->hparams.n_layer_nextn == 3 &&
        model->hparams.n_layer() == 45 && llama_model_n_embd(model.get()) == llama_model_n_embd(target),
        "expected Step 45+3 MTP sidecar with matching embedding width");
    require(!model->layers[0].attn_norm, "expected MTP-only sidecar");
    const auto * a = llama_model_get_vocab(target), * b = llama_model_get_vocab(model.get());
    require(llama_vocab_n_tokens(a) == llama_vocab_n_tokens(b), "draft vocabulary size mismatch");
    for (llama_token i = 0; i < llama_vocab_n_tokens(a); ++i) {
        require(std::strcmp(llama_vocab_get_text(a,i), llama_vocab_get_text(b,i)) == 0 &&
            llama_vocab_get_attr(a,i) == llama_vocab_get_attr(b,i) &&
            llama_vocab_is_eog(a,i) == llama_vocab_is_eog(b,i), "draft vocabulary mismatch at " + std::to_string(i));
    }
    for (const auto & entry : model->tensors_by_name) {
        if (shared_embedding && (entry.first=="token_embd.weight" || entry.first=="output.weight" || entry.first=="output_norm.weight")) continue;
        if (entry.first.rfind("blk.",0)==0 && std::stoi(entry.first.substr(4))>=45+active_heads) continue;
        require(entry.second->data && entry.second->buffer && !ggml_backend_buffer_is_host(entry.second->buffer),
            "draft weight is not resident on GPU: " + entry.first);
    }
    for (int i = 45; i < 48; ++i) require(model->layers[i].nextn.eh_proj, "missing MTP head");
    if (shared_embedding) {
        for (int i=45;i<48;++i) require(model->layers[i].nextn.shared_head_head &&
            model->layers[i].nextn.shared_head_norm && !model->layers[i].nextn.embed_tokens,
            "shared placement requires explicit per-head outputs and the common embedding");
        auto * src=model->tok_embd, * dst=target->tok_embd;
        require(src && dst && src->type==dst->type && ggml_are_same_shape(src,dst) &&
            ggml_backend_buffer_is_host(src->buffer), "incompatible shared embedding layout");
        // Prove equality of every quantized byte before aliasing. Never infer
        // equality from model names. The target outlives the draft and both contexts.
        std::vector<uint8_t> chunk(8<<20);
        for (size_t offset=0;offset<ggml_nbytes(src);offset+=chunk.size()) {
            const size_t count=std::min(chunk.size(),ggml_nbytes(src)-offset);
            ggml_backend_tensor_get(dst,chunk.data(),offset,count);
            require(std::memcmp(chunk.data(),static_cast<const uint8_t *>(src->data)+offset,count)==0,
                "shared embedding values differ");
        }
        model->tok_embd=dst;
        // Global output/output_norm remain mapped in RAM; every admitted MTP
        // head uses its own shared_head_* weights instead (checked above).
    }
    return model;
}

static Context draft_context(llama_model * model) {
    auto cp = llama_context_default_params();
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cp.n_ctx = 2048; cp.n_batch = cp.n_ubatch = 17; cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv = cp.op_offload = true; cp.swa_full = false;
    Context ctx(llama_init_from_model(model, cp), llama_free);
    require(bool(ctx), "draft context creation failed");
    return ctx;
}

static json prefault_experts(llama_model * model, size_t reserve_mib) {
    uint64_t total=0, scanned=0, pages=0;
    for (const auto & entry:model->tensors_by_name) if (expert(entry.first)) total+=ggml_nbytes(entry.second);
    const auto start=Clock::now();
    bool limited=false; uint8_t checksum=0;
    for (const auto & entry:model->tensors_by_name) {
        if (!expert(entry.first)) continue;
        const auto * tensor=entry.second;
        require(ggml_backend_buffer_is_host(tensor->buffer),"prefault requires mapped host experts");
        const auto * bytes=static_cast<const volatile uint8_t *>(tensor->data);
        const size_t count=ggml_nbytes(tensor);
        for (size_t offset=0;offset<count;offset+=4096) {
            if ((offset % (16<<20))==0) {
                strata_step_cache_refresh();
                const auto memory=strata_step_sync_snapshot();
                // The monitor enforces 95%. Stop proactively, leaving another
                // reserve for later driver allocations and pages first used by
                // prefill/verification. This is in addition to the 5% OS margin.
                if (memory.ram_free<=memory.ram_total/20+(reserve_mib<<20)) {limited=true;break;}
            }
            checksum^=bytes[offset]; ++pages;
            scanned+=std::min<size_t>(4096,count-offset);
        }
        if (limited) break;
        if (count) checksum^=bytes[count-1];
    }
    strata_step_cache_refresh();
    const auto memory=strata_step_sync_snapshot();
    return json{{"requested_bytes",total},{"scanned_bytes",scanned},{"pages_touched",pages},
        {"limited_by_headroom",limited},{"milliseconds",ms(start)},{"checksum",checksum},
        {"extra_reserve_mib",reserve_mib},
        {"ram_free_after",memory.ram_free},{"ram_total",memory.ram_total}};
}

static llama_token greedy(llama_context * ctx, int row) {
    const float * logits = llama_get_logits_ith(ctx, row);
    require(logits != nullptr, "missing target logits");
    const int n = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    llama_token best = 0;
    for (int i = 0; i < n; ++i) {
        require(std::isfinite(logits[i]), "non-finite target logits");
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

static void evaluate(llama_context * ctx, llama_context * dft, common_speculative * spec,
                     const llama_tokens & ids, int pos, bool all, double & target_ms, double & catchup_ms) {
    auto batch = llama_batch_init(int(ids.size()),0,1);
    batch.n_tokens = int(ids.size());
    for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = ids[i]; batch.pos[i] = pos+i;
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
        batch.logits[i] = all || i+1 == batch.n_tokens;
    }
    try {
        const auto start = Clock::now();
        strata_step_cache_refresh();
        const int rc = llama_decode(ctx,batch);
        llama_synchronize(ctx); target_ms += ms(start);
        require(rc == 0,"target decode failed: " + std::to_string(rc));
        if (spec) {
            const auto catchup = Clock::now();
            require(common_speculative_process(spec,batch), "MTP catch-up failed");
            llama_synchronize(dft);
            catchup_ms += ms(catchup);
        }
    } catch (...) { llama_batch_free(batch); throw; }
    llama_batch_free(batch);
}

static json run(llama_context * ctx, llama_context * dft, const json & request, int active_heads) {
    const auto prompt = request.at("tokens").get<llama_tokens>();
    const int predict = request.value("predict",128), depth = request.value("depth",0);
    const float p_min = request.value("p_min",0.6f);
    require(!prompt.empty() && predict > 0 && predict <= 256 && prompt.size()+predict <= 480,
        "probe requires 1..256 generated tokens and <=480 total positions (no SWA rollover)");
    require(depth >= 0 && depth <= 3 && std::isfinite(p_min) && p_min >= 0 && p_min <= 1,
        "invalid depth/probability");
    require(depth == 0 || dft, "MTP requires --draft");
    require(depth<=active_heads,"requested depth exceeds resident heads");
    const auto * vocab = llama_model_get_vocab(llama_get_model(ctx));
    for (auto id : prompt) require(id >= 0 && id < llama_vocab_n_tokens(vocab), "invalid prompt token");
    clear(ctx); if (dft) clear(dft);
    llama_set_embeddings_nextn(ctx,depth > 0,false);
    common_speculative_ptr spec;
    if (depth) {
        common_params_speculative params;
        params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        params.draft.ctx_tgt = ctx; params.draft.ctx_dft = dft;
        params.draft.n_max = depth; params.draft.p_min = p_min;
        params.draft.backend_sampling = false;
        spec.reset(common_speculative_init(params,1));
        require(bool(spec), "MTP initialization failed");
    }
    strata_step_sync_reset(); RequestPhase phase(1);
    const auto start = Clock::now();
    double prefill_target_ms=0, prefill_draft_ms=0;
    for (size_t i = 0; i < prompt.size(); i += 17) {
        llama_tokens ids(prompt.begin()+i,prompt.begin()+std::min(prompt.size(),i+17));
        evaluate(ctx,dft,spec.get(),ids,int(i),false,prefill_target_ms,prefill_draft_ms);
    }
    if (spec) common_speculative_begin(spec.get(),0,prompt);
    const double prefill_ms = ms(start);
    phase.decode(); const auto generation = Clock::now();
    llama_tokens out{greedy(ctx,-1)}, history = prompt;
    double target_ms=0, draft_ms=0, catchup_ms=0;
    int proposed=0, accepted_total=0, cycles=0, rollbacks=0;
    std::vector<int> proposed_per_cycle, accepted_per_cycle;
    while (int(out.size()) < predict && !llama_vocab_is_eog(vocab,out.back())) {
        llama_tokens draft;
        if (spec) {
            auto & dp = common_speculative_get_draft_params(spec.get(),0);
            dp.drafting = true; dp.pos0 = int(history.size()); dp.id_last = out.back();
            dp.prompt = &history; dp.result = &draft;
            dp.n_max = std::min(depth,predict-int(out.size())-1);
            if (dp.n_max > 0) {
                const auto t = Clock::now(); common_speculative_draft(spec.get());
                llama_synchronize(dft); draft_ms += ms(t);
            } else dp.drafting = false;
        }
        llama_tokens verify{out.back()}; verify.insert(verify.end(),draft.begin(),draft.end());
        evaluate(ctx,dft,spec.get(),verify,int(history.size()),true,target_ms,catchup_ms);
        int accepted=0;
        for (size_t i = 0; i < verify.size() && int(out.size()) < predict; ++i) {
            auto token = greedy(ctx,int(i)); out.push_back(token);
            const bool matches = i < draft.size() && token == draft[i];
            if (matches) ++accepted;
            if (!matches || llama_vocab_is_eog(vocab,token)) break;
        }
        const size_t keep = std::min(verify.size(),size_t(accepted+1));
        history.insert(history.end(),verify.begin(),verify.begin()+keep);
        if (spec) common_speculative_accept(spec.get(),0,uint16_t(accepted));
        require(llama_memory_seq_rm(llama_get_memory(ctx),0,int(history.size()),-1), "target KV rollback failed");
        proposed += int(draft.size()); accepted_total += accepted; ++cycles;
        rollbacks += accepted < int(draft.size());
        proposed_per_cycle.push_back(int(draft.size())); accepted_per_cycle.push_back(accepted);
    }
    const double generation_ms = ms(generation);
    const auto stats = strata_step_sync_snapshot();
    return json{{"ids",out},{"depth",depth},{"p_min",p_min},{"prefill_ms",prefill_ms},
        {"generation_ms",generation_ms},{"request_ms",ms(start)},
        {"decode_tokens_per_second",out.size()*1000.0/generation_ms},
        {"prefill_target_ms",prefill_target_ms},{"prefill_draft_ms",prefill_draft_ms},
        {"target_verify_ms",target_ms},{"draft_ms",draft_ms},{"catchup_ms",catchup_ms},
        {"proposed",proposed},{"accepted",accepted_total},{"cycles",cycles},{"rollbacks",rollbacks},
        {"proposed_per_cycle",proposed_per_cycle},{"accepted_per_cycle",accepted_per_cycle},
        {"h2d_bytes",stats.h2d_bytes},{"source_bytes",stats.source_bytes},
        {"cache_bytes",stats.cache_bytes},{"cache_hits",stats.cache_hits},{"cache_misses",stats.cache_misses},
        {"finish",llama_vocab_is_eog(vocab,out.back()) ? "eog" : "length"}};
}

int main(int argc, char ** argv) {
    try {
        std::string path, draft_path, placement="resident"; int cache_mib=16384;
        bool prefault=false;
        int ram_reserve_mib=24576, active_heads=3;
        for (int i=1;i<argc;++i) {
            const std::string key=argv[i]; require(i+1<argc,"missing option value");
            const std::string val=argv[++i];
            if (key=="--model") path=val;
            else if (key=="--draft") draft_path=val;
            else if (key=="--cache-mib") cache_mib=std::stoi(val);
            else if (key=="--draft-placement") placement=val;
            else if (key=="--active-heads") active_heads=std::stoi(val);
            else if (key=="--prefault-experts") {require(val=="on" || val=="off","invalid prefault mode");prefault=val=="on";}
            else if (key=="--ram-prefault-reserve-mib") ram_reserve_mib=std::stoi(val);
            else throw std::runtime_error("unknown option: "+key);
        }
        require(!path.empty() && cache_mib>=0 && cache_mib<=16384,"invalid probe model/cache");
        require(placement=="resident" || placement=="shared-embedding" || placement=="active-heads","invalid draft placement");
        require(active_heads>=1 && active_heads<=3 && (placement=="active-heads" || active_heads==3),"invalid active heads");
        require(active_heads==3 || STRATA_STEP_MTP_ACTIVE_CATCHUP,"active heads placement requires the optional catch-up patch");
        require(ram_reserve_mib>=512 && ram_reserve_mib<=32768,"invalid RAM prefault reserve");
        environment(); ggml_backend_load_all(); strata_step_sync_mode(2);
        strata_step_cache_begin(path.c_str(),0);
        {
            auto model=load(path); auto ctx=context(model.get(),2048,17,GGML_TYPE_F32);
            Model draft(nullptr,llama_model_free); Context dft(nullptr,llama_free);
            if (!draft_path.empty()) { draft=load_draft(draft_path,model.get(),placement!="resident",active_heads); dft=draft_context(draft.get()); }
            register_cache(model.get(),path,size_t(cache_mib)<<20);
            strata_step_cache_prefill(false); strata_step_pipeline_config(1,8,0);
            const json prefault_result=prefault ? prefault_experts(model.get(),size_t(ram_reserve_mib)) : json(nullptr);
            std::cout << "READY " << json{{"draft",bool(dft)},{"draft_placement",placement},{"cache_mib",cache_mib},
                {"prefault_experts",prefault_result},
                {"active_heads",active_heads},{"catchup",STRATA_STEP_MTP_ACTIVE_CATCHUP ? "requested-heads-only" : "upstream-all-heads"},
                {"context",2048},{"batch",17},{"kv","f32"},{"pipeline_readers",1},
                {"max_total_positions",480},{"greedy_only",true}}.dump() << std::endl;
            std::string line;
            while (std::getline(std::cin,line) && line!="QUIT") {
                const auto result=run(ctx.get(),dft.get(),json::parse(line),active_heads);
                std::cout << "RESULT " << result.dump() << std::endl;
            }
            strata_step_sync_release();
        }
        llama_backend_free(); return 0;
    } catch (const std::exception & e) {
        std::cerr << "MTP_PROBE_ERROR " << e.what() << std::endl;
        strata_step_sync_release(); return 1;
    }
}
