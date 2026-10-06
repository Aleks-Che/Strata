// Numerical/state admission only: synthetic Hy3 weights, no performance claims.
#include "synthetic_hy3.hpp"
#include "llama.h"
#include "llama-ext.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-kv-cache.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

using json = nlohmann::ordered_json;
using Floats = std::vector<float>;
using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
static void require(bool ok, const std::string & text) { if (!ok) throw std::runtime_error(text); }

static Floats read_f32(ggml_tensor * t) {
    require(t->type == GGML_TYPE_F32, "expected F32 capture");
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    Floats values;
    for (int64_t d = 0; d < t->ne[3]; ++d) for (int64_t c = 0; c < t->ne[2]; ++c)
        for (int64_t b = 0; b < t->ne[1]; ++b) for (int64_t a = 0; a < t->ne[0]; ++a) {
            float value;
            std::memcpy(&value, bytes.data() + a*t->nb[0] + b*t->nb[1] + c*t->nb[2] + d*t->nb[3], sizeof(value));
            values.push_back(value);
        }
    return values;
}

static json compare(const std::string & name, const Floats & actual, const Floats & expected,
                    double abs_limit = 5e-4, double nmse_limit = 1e-7, bool exact = false) {
    require(actual.size() == expected.size() && !actual.empty(), "comparison shape: " + name);
    double error = 0, energy = 0, max_abs = 0;
    bool finite = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        finite &= std::isfinite(actual[i]) && std::isfinite(expected[i]);
        const double delta = double(actual[i]) - expected[i];
        error += delta * delta; energy += double(expected[i]) * expected[i];
        max_abs = std::max(max_abs, std::abs(delta));
    }
    const double nmse = error / std::max(energy, 1e-30);
    const bool bits = std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0;
    return {{"name", name}, {"pass", finite && max_abs <= abs_limit && nmse <= nmse_limit && (!exact || bits)},
            {"elements", actual.size()}, {"finite", finite}, {"max_abs", max_abs}, {"nmse", nmse},
            {"abs_limit", abs_limit}, {"nmse_limit", nmse_limit}, {"exact_required", exact}, {"bit_exact", bits}};
}

struct Audit {
    llama_context * ctx = nullptr;
    bool gpu = false, capture = false;
    std::set<std::string> cpu_nodes;
    std::map<std::string,size_t> ops;
    std::map<std::string,Floats> values;
    static bool callback(ggml_tensor * t, bool ask, void * data) {
        auto & a = *static_cast<Audit *>(data);
        if (!ask) {
            a.values[t->name] = read_f32(t);
            if (t->op == GGML_OP_ROPE) {
                auto * normed=t->src[0];
                require(normed->op==GGML_OP_MUL && normed->src[0]->op==GGML_OP_RMS_NORM,
                        "Q/K must be RMS normalized before RoPE");
                const std::string key=normed->src[1]->name;
                // Copy the earlier callback captures: the scheduler may reuse
                // a source's storage after its last consumer has run.
                a.values[key+"/raw"]=a.values.at(normed->src[0]->src[0]->name);
                a.values[key+"/normed"]=a.values.at(normed->name);
                a.values[key+"/rope"]=read_f32(t);
            }
            return true;
        }
        if (t->op!=GGML_OP_NONE && t->op!=GGML_OP_VIEW && t->op!=GGML_OP_RESHAPE &&
            t->op!=GGML_OP_TRANSPOSE && t->op!=GGML_OP_PERMUTE) {
            ++a.ops[ggml_op_desc(t)];
            auto * backend=ggml_backend_sched_get_tensor_backend(a.ctx->get_sched(),t);
            auto * dev=backend ? ggml_backend_get_device(backend) : nullptr;
            if(a.gpu && (!dev || ggml_backend_dev_type(dev)!=GGML_BACKEND_DEVICE_TYPE_GPU))
                a.cpu_nodes.insert(std::string(t->name)+" / "+ggml_op_desc(t));
        }
        return a.capture && t->type==GGML_TYPE_F32 && t->op!=GGML_OP_NONE;
    }
};

static Model load(const std::string & path,bool gpu,bool mtp=true) {
    auto * dev=ggml_backend_dev_by_name(gpu ? "CUDA0" : "CPU");
    require(dev!=nullptr,"missing backend");
    ggml_backend_dev_t devices[]={dev,nullptr};
    llama_model_tensor_buft_override overrides[]={{"token_embd\\.weight",ggml_backend_dev_buffer_type(dev)},{nullptr,nullptr}};
    auto mp=llama_model_default_params();
    mp.devices=devices; mp.tensor_buft_overrides=overrides;
    mp.n_gpu_layers=gpu ? -1 : 0; mp.load_mtp=mtp;
    mp.load_mode=LLAMA_LOAD_MODE_NONE; mp.use_extra_bufts=false;
    Model model(llama_model_load_from_file(path.c_str(),mp),llama_model_free);
    require(bool(model),"Hy3 fixture load failed");
    return model;
}

struct Output { Floats logits,hidden; };
struct Run {
    Audit audit;
    bool mtp;
    Context ctx{nullptr,llama_free};
    Run(llama_model * model,bool gpu,int ubatch,bool draft=false):mtp(draft) {
        audit.gpu=gpu;
        auto cp=llama_context_default_params();
        cp.n_ctx=2048; cp.n_batch=64; cp.n_ubatch=ubatch; cp.n_seq_max=1;
        cp.n_threads=cp.n_threads_batch=4; cp.type_k=cp.type_v=GGML_TYPE_F32;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv=cp.op_offload=gpu;
        cp.ctx_type=draft ? LLAMA_CONTEXT_TYPE_MTP : LLAMA_CONTEXT_TYPE_DEFAULT;
        cp.cb_eval=Audit::callback; cp.cb_eval_user_data=&audit;
        ctx.reset(llama_init_from_model(model,cp));
        require(bool(ctx),"Hy3 context creation failed");
        audit.ctx=ctx.get();
        auto * kv=dynamic_cast<llama_kv_cache *>(ctx->get_memory());
        require(kv && kv->get_layer_ids()==(draft ? std::vector<uint32_t>{2} : std::vector<uint32_t>{0,1}),
                "incorrect main/MTP full-KV layer filter");
        llama_set_embeddings_nextn(ctx.get(),true,false);
    }
    void clear() { llama_memory_clear(ctx->get_memory(),true); }
    Output decode(int first,int count,int chunk,int salt=0,const Floats * hidden=nullptr) {
        require(!mtp || (hidden && hidden->size()==size_t(count*256)),"MTP input shape");
        Output out;
        for(int pos=first;pos<first+count;pos+=chunk) {
            const int n=std::min(chunk,first+count-pos);
            auto batch=llama_batch_init(n,mtp ? 256 : 0,1);
            if(mtp) {
                batch.token=static_cast<llama_token *>(std::malloc(sizeof(llama_token)*n));
                require(batch.token!=nullptr,"batch allocation");
                std::memcpy(batch.embd,hidden->data()+(pos-first)*256,n*256*sizeof(float));
            }
            batch.n_tokens=n;
            for(int i=0;i<n;++i) {
                batch.token[i]=(7*(pos+i)+11+salt)%64; batch.pos[i]=pos+i;
                batch.n_seq_id[i]=1; batch.seq_id[i][0]=0; batch.logits[i]=1;
            }
            const int status=llama_decode(ctx.get(),batch);
            llama_batch_free(batch);
            require(status==0,"decode failed at "+std::to_string(pos)+": "+std::to_string(status));
            for(int i=0;i<n;++i) {
                const float * logits=llama_get_logits_ith(ctx.get(),i);
                const float * h=llama_get_embeddings_nextn_ith(ctx.get(),i);
                require(logits && h,"missing logits/NextN hidden");
                out.logits.insert(out.logits.end(),logits,logits+64);
                out.hidden.insert(out.hidden.end(),h,h+256);
            }
            require(audit.cpu_nodes.empty(),"CPU math in GPU graph: "+json(audit.cpu_nodes).dump());
        }
        return out;
    }
    std::vector<uint8_t> save() {
        std::vector<uint8_t> data(llama_state_seq_get_size(ctx.get(),0));
        require(!data.empty() && llama_state_seq_get_data(ctx.get(),data.data(),data.size(),0)==data.size(),"KV save failed");
        return data;
    }
    void restore(const std::vector<uint8_t> & data) {
        clear();
        require(llama_state_seq_set_data(ctx.get(),data.data(),data.size(),0)==data.size(),"KV restore failed");
    }
};

static Floats norm(const Floats & input,const Floats & weights) {
    require(input.size()%weights.size()==0,"norm shape");
    Floats out(input.size());
    for(size_t off=0;off<input.size();off+=weights.size()) {
        double sq=0; for(size_t i=0;i<weights.size();++i) sq+=double(input[off+i])*input[off+i];
        const double factor=1/std::sqrt(sq/weights.size()+1e-5);
        for(size_t i=0;i<weights.size();++i) out[off+i]=float(input[off+i]*factor*weights[i]);
    }
    return out;
}
static Floats mm(ggml_tensor * tensor,const Floats & input,int expert=0) {
    const auto w=read_f32(tensor);
    const int k=int(tensor->ne[0]),m=int(tensor->ne[1]);
    require(input.size()==size_t(k),"scalar matmul shape");
    Floats out(m);
    for(int row=0;row<m;++row) {
        double sum=0;
        for(int col=0;col<k;++col) sum+=double(input[col])*w[(expert*m+row)*k+col];
        out[row]=float(sum);
    }
    return out;
}
static Floats ffn(ggml_tensor * gate,ggml_tensor * up,ggml_tensor * down,const Floats & x,int expert=0) {
    auto g=mm(gate,x,expert),u=mm(up,x,expert);
    for(size_t i=0;i<g.size();++i) g[i]=float(double(g[i])/(1+std::exp(-double(g[i])))*u[i]);
    return mm(down,g,expert);
}

// Independent scalar equations, fed with inputs captured from the native graph.
static void components(Run & run,llama_model & model,json & results,int position,const Floats * h=nullptr) {
    run.audit.values.clear(); run.audit.capture=true;
    auto output=run.decode(position,1,1,run.mtp ? 7 : 0,h);
    run.audit.capture=false;
    const auto get=[&](const std::string & key)->const Floats & {
        auto it=run.audit.values.find(key);
        require(it!=run.audit.values.end(),"missing capture: "+key);
        return it->second;
    };
    const int begin=run.mtp ? 2 : 0,end=run.mtp ? 3 : 2;
    for(int il=begin;il<end;++il) {
        auto & layer=model.layers[il];
        const std::string suffix="-"+std::to_string(il),prefix=run.mtp ? "mtp_" : "";
        for(const std::string kind:{"q","k"}) {
            auto * weight=kind=="q" ? layer.attn_q_norm : layer.attn_k_norm;
            const std::string key=weight->name;
            auto expected=norm(get(key+"/raw"),read_f32(weight));
            results.push_back(compare("scalar_"+kind+"_norm"+suffix,get(key+"/normed"),expected,2e-5,1e-10));
            const auto & n=get(key+"/normed"); expected=n;
            for(size_t off=0;off<n.size();off+=128) for(int pair=0;pair<64;++pair) {
                const double angle=position*std::pow(11158840.,-2.*pair/128);
                const double a=n[off+pair],b=n[off+pair+64];
                expected[off+pair]=float(a*std::cos(angle)-b*std::sin(angle));
                expected[off+pair+64]=float(a*std::sin(angle)+b*std::cos(angle));
            }
            results.push_back(compare("scalar_"+kind+"_neox_rope"+suffix,get(key+"/rope"),expected));
        }
        const auto & x=get(prefix+"ffn_norm"+suffix);
        if(il==0) results.push_back(compare("scalar_dense_swiglu",get("ffn_dense_out-0"),
            ffn(layer.ffn_gate,layer.ffn_up,layer.ffn_down,x),2e-5,1e-9));
        else {
            auto shared=ffn(layer.ffn_gate_shexp,layer.ffn_up_shexp,layer.ffn_down_shexp,x);
            results.push_back(compare("scalar_shared_swiglu"+suffix,get(prefix+"ffn_shared_out"+suffix),shared,2e-5,1e-9));
            auto scores=mm(layer.ffn_gate_inp,x),bias=read_f32(layer.ffn_exp_probs_b);
            std::vector<int> ids(16); for(int i=0;i<16;++i) {ids[i]=i; scores[i]=float(1/(1+std::exp(-double(scores[i]))));}
            std::sort(ids.begin(),ids.end(),[&](int a,int b){return scores[a]+bias[a]>scores[b]+bias[b];});
            double sum=0; for(int i=0;i<8;++i) sum+=scores[ids[i]];
            Floats routed(256);
            for(int i=0;i<8;++i) {
                auto e=ffn(layer.ffn_gate_exps,layer.ffn_up_exps,layer.ffn_down_exps,x,ids[i]);
                for(int j=0;j<256;++j) routed[j]+=float(e[j]*scores[ids[i]]/sum*2.826);
            }
            results.push_back(compare("scalar_sigmoid_bias_top8_normalized_scaled_moe"+suffix,
                get(prefix+"ffn_moe_out"+suffix),routed,2e-5,1e-9));
            for(int j=0;j<256;++j) shared[j]+=routed[j];
            results.push_back(compare("scalar_shared_plus_routed"+suffix,get(prefix+"ffn_out"+suffix),shared,2e-5,1e-9));
        }
    }
    const auto expected=norm(get(run.mtp ? "mtp_post_ffn-2" : "l_out-1"),
        read_f32(run.mtp ? model.layers[2].nextn.shared_head_norm : model.output_norm));
    results.push_back(compare(run.mtp ? "mtp_post_final_norm_hidden" : "main_post_final_norm_hidden",output.hidden,expected,2e-5,1e-10));
    results.push_back(compare(run.mtp ? "mtp_shared_output_head" : "main_output_head",output.logits,mm(model.output,expected),2e-5,1e-9));
    if(run.mtp) {
        auto & layer=model.layers[2];
        auto e=norm(get("mtp_tok_embd-2"),read_f32(layer.nextn.enorm));
        auto hidden=norm(*h,read_f32(layer.nextn.hnorm));
        results.push_back(compare("mtp_enorm",get("mtp_enorm-2"),e,2e-5,1e-10));
        results.push_back(compare("mtp_hnorm",get("mtp_hnorm-2"),hidden,2e-5,1e-10));
        e.insert(e.end(),hidden.begin(),hidden.end());
        results.push_back(compare("mtp_concat_e_then_h",get("mtp_concat-2"),e,2e-5,1e-10));
        results.push_back(compare("mtp_eh_projection",get("mtp_eh_proj-2"),mm(layer.nextn.eh_proj,e),2e-5,1e-9));
    }
}

static void attention_checks(Run & run,llama_model & model,json & results,const Floats * hidden=nullptr) {
    run.clear();
    std::map<int,std::vector<Floats>> keys,values;
    run.audit.capture=true;
    for(int pos=0;pos<5;++pos) {
        Floats h;
        if(hidden) h.assign(hidden->begin()+pos*256,hidden->begin()+(pos+1)*256);
        run.audit.values.clear(); run.decode(pos,1,1,run.mtp ? 7 : 0,hidden ? &h : nullptr);
        const auto & captures=run.audit.values;
        for(int il=run.mtp ? 2 : 0;il<(run.mtp ? 3 : 2);++il) {
            const std::string suffix="-"+std::to_string(il),prefix=run.mtp ? "mtp_" : "";
            const auto & layer=model.layers[il];
            const auto & q=captures.at(std::string(layer.attn_q_norm->name)+"/rope");
            keys[il].push_back(captures.at(std::string(layer.attn_k_norm->name)+"/rope"));
            values[il].push_back(captures.at("Vcur"+suffix));
            Floats joined(512);
            for(int head=0;head<4;++head) {
                std::vector<double> scores(pos+1); double top=-1e30,sum=0;
                for(int past=0;past<=pos;++past) {
                    double dot=0; for(int i=0;i<128;++i) dot+=double(q[head*128+i])*keys[il][past][i];
                    scores[past]=dot/std::sqrt(128.); top=std::max(top,scores[past]);
                }
                for(auto & s:scores) {s=std::exp(s-top);sum+=s;}
                for(int i=0;i<128;++i) {
                    double result=0; for(int past=0;past<=pos;++past) result+=scores[past]/sum*values[il][past][i];
                    joined[head*128+i]=float(result);
                }
            }
            results.push_back(compare("scalar_causal_full_gqa_pos"+std::to_string(pos)+suffix,
                captures.at(prefix+"attn_out"+suffix),mm(layer.wo,joined),2e-5,1e-9));
        }
    }
    run.audit.capture=false;
}

static void pair_results(json & out,const std::string & name,const Output & actual,const Output & expected,
                         bool mixed=false,bool exact=false) {
    // Quantized CPU and CUDA use different activation quantization kernels.
    out.push_back(compare(name+"/logits",actual.logits,expected.logits,mixed ? 0.03 : 5e-4,mixed ? 2e-3 : 1e-7,exact));
    out.push_back(compare(name+"/hidden",actual.hidden,expected.hidden,mixed ? 0.12 : 5e-4,mixed ? 2e-3 : 1e-7,exact));
}
static void state_checks(Run & run,json & results,const Floats * inputs=nullptr) {
    const std::string label=run.mtp ? "mtp/" : "main/";
    run.clear();
    Floats prefix,tail;
    if(inputs) {prefix.assign(inputs->begin(),inputs->begin()+17*256);tail.assign(inputs->begin()+17*256,inputs->end());}
    run.decode(0,17,17,0,inputs ? &prefix : nullptr);
    auto saved=run.save();
    auto a=run.decode(17,9,1,0,inputs ? &tail : nullptr);
    run.restore(saved);
    require(llama_memory_seq_pos_max(run.ctx->get_memory(),0)==16,"restored KV position");
    run.decode(17,9,1,13,inputs ? &tail : nullptr);
    run.restore(saved);
    auto again=run.decode(17,9,1,0,inputs ? &tail : nullptr);
    pair_results(results,label+"A_B_A_restore",again,a,false,true);
    require(llama_memory_seq_rm(run.ctx->get_memory(),0,21,-1),"KV rollback failed");
    require(llama_memory_seq_pos_max(run.ctx->get_memory(),0)==20,"rolled-back KV position");
    if(inputs) tail.erase(tail.begin(),tail.begin()+4*256);
    auto replay=run.decode(21,5,1,0,inputs ? &tail : nullptr);
    Output reference;
    reference.logits.assign(a.logits.begin()+4*64,a.logits.end());
    reference.hidden.assign(a.hidden.begin()+4*256,a.hidden.end());
    pair_results(results,label+"rollback_replay",replay,reference,false,true);
}

int main(int argc,char ** argv) {
    json report={{"status","fail"},{"source_sha",STRATA_HY3_SOURCE_SHA},{"patch_set",STRATA_HY3_PATCH_SET},
        {"scope","synthetic Hy3 graph/state admission; not full-model inference or MTP speed"},{"results",json::array()}};
    std::filesystem::path directory;
    bool created=false;
    try {
        require(argc==2,"usage: strata-hy3-graph-check NEW_OUTPUT_DIRECTORY");
        directory=std::filesystem::absolute(argv[1]);
        require(!std::filesystem::exists(directory),"output directory must be new");
        require(std::filesystem::create_directories(directory),"cannot create output directory");
        created=true;
        require(std::getenv("NVIDIA_TF32_OVERRIDE") && std::string(std::getenv("NVIDIA_TF32_OVERRIDE"))=="0","set NVIDIA_TF32_OVERRIDE=0");
        ggml_backend_load_all(); llama_backend_init();
        auto * gpu=ggml_backend_dev_by_name("CUDA0"); require(gpu,"CUDA0 required");
        report["device"]=ggml_backend_dev_description(gpu);
        for(bool mixed:{false,true}) {
            const auto path=(directory/(mixed ? "hy3-mixed.gguf" : "hy3-f32.gguf")).string();
            write_synthetic_hy3(path,mixed);
            auto cpu_model=load(path,false),gpu_model=load(path,true);
            Run cpu(cpu_model.get(),false,17),gpu_run(gpu_model.get(),true,17),micro(gpu_model.get(),true,4);
            Run cpu_draft(cpu_model.get(),false,17,true),gpu_draft(gpu_model.get(),true,17,true);
            json tests=json::array();
            auto c=cpu.decode(0,26,17),g=gpu_run.decode(0,26,17),m=micro.decode(0,26,17);
            pair_results(tests,"main_cpu_cuda",g,c,mixed);
            pair_results(tests,"main_microbatch4_vs17",m,g,mixed);
            gpu_run.clear(); pair_results(tests,"main_serial_vs_prefill",gpu_run.decode(0,26,1),g,mixed);
            auto cd=cpu_draft.decode(0,26,17,7,&c.hidden),gd=gpu_draft.decode(0,26,17,7,&c.hidden);
            pair_results(tests,"mtp_cpu_cuda_same_target_hidden",gd,cd,mixed);
            gpu_draft.clear(); pair_results(tests,"mtp_serial_vs_prefill",gpu_draft.decode(0,26,1,7,&c.hidden),gd,mixed);
            Floats seed(cd.hidden.end()-256,cd.hidden.end());
            pair_results(tests,"mtp_chained_hidden",gpu_draft.decode(26,1,1,7,&seed),cpu_draft.decode(26,1,1,7,&seed),mixed);
            if(!mixed) {
                components(gpu_run,*gpu_model,tests,26);
                components(gpu_draft,*gpu_model,tests,27,&seed);
                attention_checks(gpu_run,*gpu_model,tests);
                attention_checks(gpu_draft,*gpu_model,tests,&c.hidden);
            }
            state_checks(gpu_run,tests);
            state_checks(gpu_draft,tests,&c.hidden);
            auto off_model=load(path,true,false);
            for(const auto & entry:off_model->tensors_by_name) require(entry.first.rfind("blk.2.",0)!=0,"MTP-off allocated MTP tensor");
            Run off(off_model.get(),true,17);
            pair_results(tests,"main_mtp_loaded_vs_skipped",off.decode(0,26,17),g,false,true);
            report["results"].push_back({{"weights",mixed ? "mixed" : "F32"},{"checks",tests},
                {"fixture_dimensions",{{"main_blocks",2},{"mtp_blocks",1},{"width",256},{"q_heads",4},{"kv_heads",1},
                    {"head_dim",128},{"experts",16},{"top_k",8},{"routed_ff",256},{"shared_ff",256},{"vocab",64}}},
                {"kv_type","F32"},{"flash_attention",false},{"mtp_off_tensor_count",0},
                {"main_kv_layers",{0,1}},{"mtp_kv_layers",{2}},
                {"main_gpu_ops",gpu_run.audit.ops},{"mtp_gpu_ops",gpu_draft.audit.ops},
                {"cpu_fallback_nodes",gpu_run.audit.cpu_nodes.size()+gpu_draft.audit.cpu_nodes.size()+micro.audit.cpu_nodes.size()+off.audit.cpu_nodes.size()}});
        }
        bool pass=true; size_t count=0;
        for(const auto & fixture:report["results"]) for(const auto & check:fixture["checks"]) {pass&=check["pass"].get<bool>();++count;}
        report["check_count"]=count; report["status"]=pass ? "pass" : "fail";
    } catch(const std::exception & e) {report["error"]=e.what();}
    if(created) {
        std::ofstream f(directory/"graph-report.json"); f<<report.dump(2)<<'\n';
        if(!f) return 1;
    }
    std::cout<<report.dump(2)<<'\n';
    ggml_quantize_free(); llama_backend_free();
    return report["status"]=="pass" ? 0 : 1;
}

