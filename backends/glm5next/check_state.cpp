// Executes a tiny synthetic GLM through the real loader, graph and hybrid memory.
#include "synthetic_glm.hpp"
#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-memory-hybrid.h"
#include "llama-kv-cells.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>

using json = nlohmann::ordered_json;
using Floats = std::vector<float>;
using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
static void require(bool ok, const std::string & message) { if (!ok) throw std::runtime_error(message); }

static json compare(const std::string & name, const Floats & actual, const Floats & expected) {
    require(actual.size() == expected.size() && !actual.empty(), "comparison size: "+name);
    double error=0, energy=0, max_abs=0;
    bool finite=true;
    for (size_t i=0; i<actual.size(); ++i) {
        finite &= std::isfinite(actual[i]) && std::isfinite(expected[i]);
        const double d = double(actual[i])-expected[i];
        error += d*d; energy += double(expected[i])*expected[i];
        max_abs = std::max(max_abs, std::abs(d));
    }
    const double nmse = error/std::max(energy, 1e-30);
    return {{"name", name}, {"status", finite && max_abs<=5e-4 && nmse<=1e-7 ? "pass" : "fail"},
            {"elements", actual.size()}, {"finite", finite}, {"max_abs", max_abs}, {"nmse", nmse}};
}

struct Audit {
    llama_context * ctx = nullptr;
    bool gpu;
    std::set<std::string> cpu_nodes;
    std::map<std::string, size_t> compute_ops;
    explicit Audit(bool use_gpu) : gpu(use_gpu) {}
    static bool callback(ggml_tensor * t, bool ask, void * data) {
        auto & audit = *static_cast<Audit *>(data);
        if (!ask) return true;
        if (t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE ||
            t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_PERMUTE) return false;
        ++audit.compute_ops[ggml_op_desc(t)];
        auto * b = ggml_backend_sched_get_tensor_backend(audit.ctx->get_sched(), t);
        auto * dev = b ? ggml_backend_get_device(b) : nullptr;
        if (audit.gpu && (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU))
            audit.cpu_nodes.insert(std::string(t->name)+" / "+ggml_op_desc(t));
        return false;
    }
};

static Model load(const std::string & path, bool gpu) {
    auto mp = llama_model_default_params();
    auto * dev = ggml_backend_dev_by_name(gpu ? "CUDA0" : "CPU");
    require(dev != nullptr, "missing backend device");
    llama_model_tensor_buft_override overrides[] = {{"token_embd\\.weight", ggml_backend_dev_buffer_type(dev)}, {nullptr, nullptr}};
    ggml_backend_dev_t devices[] = {dev, nullptr};
    mp.devices = devices; mp.tensor_buft_overrides = overrides;
    mp.n_gpu_layers = gpu ? -1 : 0; mp.load_mtp = false;
    mp.use_extra_bufts = false; mp.load_mode = LLAMA_LOAD_MODE_NONE;
    Model model(llama_model_load_from_file(path.c_str(), mp), llama_model_free);
    require(bool(model), "synthetic model load failed");
    return model;
}

struct Run {
    Audit audit;
    Context ctx{nullptr, llama_free};
    Run(llama_model * model, bool gpu, int ubatch=32, bool fused=true) : audit(gpu) {
#ifdef _WIN32
        _putenv_s("LLAMA_FUSED_LID_DISABLE", fused ? "0" : "1");
#else
        setenv("LLAMA_FUSED_LID_DISABLE", fused ? "0" : "1", 1);
#endif
        auto cp = llama_context_default_params();
        cp.n_ctx=512; cp.n_batch=128; cp.n_ubatch=ubatch; cp.n_seq_max=1; cp.n_rs_seq=8;
        cp.n_threads=cp.n_threads_batch=4; cp.type_k=cp.type_v=GGML_TYPE_F32;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv=gpu; cp.op_offload=gpu;
        cp.cb_eval=Audit::callback; cp.cb_eval_user_data=&audit;
        ctx.reset(llama_init_from_model(model, cp));
        require(bool(ctx), "synthetic context failed"); audit.ctx=ctx.get();
    }
    llama_memory_hybrid * memory() {
        auto * m = dynamic_cast<llama_memory_hybrid *>(ctx->get_memory());
        require(m && m->get_mem_idx(), "hybrid/indexer memory required"); return m;
    }
    void clear() { llama_memory_clear(ctx->get_memory(), true); }
    Floats decode(int first, int count, int salt=0, int chunk=128) {
        Floats out;
        for (int pos=first; pos<first+count; pos+=chunk) {
            const int n=std::min(chunk, first+count-pos);
            auto batch=llama_batch_init(n, 0, 1);
            batch.n_tokens=n;
            for (int i=0; i<n; ++i) {
                batch.token[i]=(7*(pos+i)+11+salt)%64; batch.pos[i]=pos+i;
                batch.n_seq_id[i]=1; batch.seq_id[i][0]=0; batch.logits[i]=1;
            }
            const int status=llama_decode(ctx.get(), batch);
            llama_batch_free(batch);
            require(status==0, "llama_decode failed: "+std::to_string(status));
            for (int i=0; i<n; ++i) {
                const float * logits=llama_get_logits_ith(ctx.get(), i);
                require(logits!=nullptr, "missing logits"); out.insert(out.end(), logits, logits+64);
            }
            if (!audit.cpu_nodes.empty()) throw std::runtime_error("CPU compute in GPU fixture: "+json(audit.cpu_nodes).dump());
        }
        return out;
    }
    std::vector<uint8_t> save() {
        std::vector<uint8_t> bytes(llama_state_seq_get_size(ctx.get(), 0));
        require(llama_state_seq_get_data(ctx.get(), bytes.data(), bytes.size(), 0)==bytes.size(), "state save failed");
        return bytes;
    }
    void restore(const std::vector<uint8_t> & bytes) {
        require(llama_state_seq_set_data(ctx.get(), bytes.data(), bytes.size(), 0)==bytes.size(), "state restore failed");
    }
    void position(int expected) {
        auto * m=memory();
        require(m->get_mem_attn()->seq_pos_max(0)==expected && m->get_mem_idx()->seq_pos_max(0)==expected &&
                m->get_mem_recr()->seq_pos_max(0)==expected, "hybrid component positions diverged");
    }
};

static Floats tensor_values(ggml_tensor * t) {
    require(t->type==GGML_TYPE_F32 && ggml_is_contiguous(t), "expected contiguous F32 cache");
    Floats out(ggml_nelements(t)); ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t)); return out;
}

// Independent scalar pooling reference from resident raw keys/gates, joined by
// logical token positions rather than assuming contiguous physical cache cells.
static json pools(Run & run, const llama_model & model, const std::string & name) {
    auto * idx=run.memory()->get_mem_idx();
    auto * attn=run.memory()->get_mem_attn();
    const auto & cells=attn->get_cells(0);
    const auto values=tensor_values(idx->get_k_storage(1));
    const auto ape=tensor_values(model.layers[1].indexer_comp_ape);
    std::map<int, int> resident;
    for (uint32_t cell=0; cell<attn->get_size(); ++cell)
        if (!cells.is_empty(cell) && cells.seq_has(cell, 0)) resident[cells.pos_get(cell)]=int(cell);
    Floats actual, expected;
    for (const auto & item : resident) {
        const int end=item.first;
        if (end%4 != 3) continue;
        bool full=true; for (int j=0;j<4;++j) full &= resident.count(end-3+j)!=0;
        if (!full) continue;
        for (int d=0;d<128;++d) {
            double gates[4], largest=-INFINITY, sum=0, value=0;
            for (int j=0;j<4;++j) {
                gates[j]=values[resident.at(end-3+j)*384+128+d]+ape[j*128+d];
                largest=std::max(largest,gates[j]);
            }
            for (int j=0;j<4;++j) {
                const double w=std::exp(gates[j]-largest); sum+=w;
                value+=w*values[resident.at(end-3+j)*384+d];
            }
            expected.push_back(float(value/sum)); actual.push_back(values[item.second*384+256+d]);
        }
    }
    return compare(name, actual, expected);
}

int main(int argc, char ** argv) {
    json report={{"schema_version",1},{"status","error"},
        {"scope","synthetic two-layer GLM forward/state validation; not the 112 GiB model or throughput"},
        {"requested_revision",STRATA_GLM_SOURCE_SHA},{"archive_sha256",STRATA_GLM_ARCHIVE_SHA256},
        {"patch_set",STRATA_GLM_PATCH_SET},{"fixture_seed",0x53c001u},
        {"configuration",{{"n_ctx",512},{"n_batch",128},{"n_ubatch",32},{"split_n_ubatch",16},
            {"n_seq_max",1},{"n_rs_seq",8},{"kv_type","F32"},{"flash_attention","off"},
            {"NVIDIA_TF32_OVERRIDE","0"},{"max_abs_limit",5e-4},{"nmse_limit",1e-7}}},
        {"synthetic_model",{{"layers",2},{"kda_layers",1},{"dsa_layers",1},{"embedding",256},
            {"heads",2},{"kda_head_width",128},{"mla_width",128},{"indexer_heads",32},
            {"indexer_width",128},{"kpool",4},{"top_k_sparse",8},{"top_k_dense",512},
            {"experts",4},{"experts_used",2},{"vocabulary",64},{"weight_type","F32"}}},
        {"results",json::array()}};
    try {
        require(argc==2, "usage: strata-glm5next-state-check WORK_DIRECTORY");
        const char * tf32=std::getenv("NVIDIA_TF32_OVERRIDE");
        require(tf32 && std::string(tf32)=="0", "set NVIDIA_TF32_OVERRIDE=0");
        std::filesystem::create_directories(argv[1]);
        const auto base=std::filesystem::path(argv[1]);
        const auto sparse=(base/"synthetic-glm-sparse.gguf").string();
        const auto dense=(base/"synthetic-glm-dense.gguf").string();
        write_synthetic_glm(sparse,8); write_synthetic_glm(dense,512);
        ggml_backend_load_all();
        auto * device=ggml_backend_dev_by_name("CUDA0");
        require(device && ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_GPU,"CUDA0 required");
        report["device"]=ggml_backend_dev_description(device);
        auto model=load(sparse,true), dense_model=load(dense,true), cpu_model=load(sparse,false);
        Run gpu(model.get(),true), clean(model.get(),true), restore(model.get(),true),
            single(model.get(),true), split(model.get(),true,16), unfused(model.get(),true,32,false),
            cpu(cpu_model.get(),false,32,false), full(dense_model.get(),true);
        auto & results=report["results"];
        const auto baseline=gpu.decode(0,49);
        results.push_back(compare("hybrid/prefill-vs-tokenwise",single.decode(0,49,0,1),baseline));
        results.push_back(compare("hybrid/ubatch16-vs32",split.decode(0,49),baseline));
        results.push_back(compare("indexer/fused-vs-unfused",unfused.decode(0,49),baseline));
        results.push_back(compare("hybrid/cuda-vs-cpu",baseline,cpu.decode(0,49)));
        // At <= top_k+kpool-1, all complete pools and the current tail are kept.
        const auto dense_prefix=full.decode(0,11);
        results.push_back(compare("attention/sparse-vs-dense-prefix11",Floats(baseline.begin(),baseline.begin()+11*64),dense_prefix));
        results.push_back(pools(gpu,*model,"kpool/prefill-scalar"));
        results.push_back(pools(single,*model,"kpool/tokenwise-scalar"));
        for (int prefix : {7,8,9,15,16,17}) {
            for (int accepted=0;accepted<=4;++accepted) {
                gpu.clear(); clean.clear(); restore.clear();
                // Prefix and proposals are one batch, preserving the pre-draft snapshot too.
                gpu.decode(0,prefix+4); clean.decode(0,prefix+accepted);
                require(gpu.memory()->seq_rm(0,prefix+accepted,-1),"seq_rm rejected valid rollback");
                gpu.position(prefix+accepted-1);
                const auto saved=gpu.save(); restore.restore(saved); restore.position(prefix+accepted-1);
                const auto expected=clean.decode(prefix+accepted,9,23);
                const std::string tag="prefix="+std::to_string(prefix)+"/accepted="+std::to_string(accepted);
                results.push_back(compare("hybrid/rollback/"+tag,gpu.decode(prefix+accepted,9,23),expected));
                results.push_back(compare("hybrid/pending-save-restore/"+tag,restore.decode(prefix+accepted,9,23),expected));
                results.push_back(pools(gpu,*model,"kpool/rollback/"+tag));
            }
        }
        gpu.clear(); gpu.decode(0,25);
        const auto before=gpu.save();
        const bool rejected=!gpu.memory()->seq_rm(0,3,-1);
        results.push_back({{"name","hybrid/out-of-window-removal-atomic"},{"status",rejected && before==gpu.save()?"pass":"fail"}});
        // ordered_json object insertion can invalidate references to its values.
        // Aggregate before adding report keys; never iterate a dangling results reference.
        bool pass=true; for(const auto & r:results) pass = pass && r["status"]=="pass";
        report["gpu_compute_ops"]=gpu.audit.compute_ops;
        report["cpu_compute_nodes"]=gpu.audit.cpu_nodes;
        report["indexer_fused"]={{"cuda",gpu.ctx->get_cparams().fused_lid},
            {"cuda_unfused_reference",unfused.ctx->get_cparams().fused_lid},
            {"cpu_reference",cpu.ctx->get_cparams().fused_lid}};
        require(gpu.audit.compute_ops.count("LIGHTNING_INDEXER") && gpu.audit.compute_ops.count("GATED_DELTA_NET") &&
                gpu.audit.compute_ops.count("MUL_MAT_ID"),"required hybrid GPU operations did not execute");
        report["status"]=pass?"pass":"fail";
    } catch(const std::exception & e) { report["error"]=e.what(); }
    std::cout<<report.dump(2)<<'\n'; return report["status"]=="pass"?0:1;
}
