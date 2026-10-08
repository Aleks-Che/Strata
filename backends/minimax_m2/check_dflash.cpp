// Offline correctness probe. Timings include diagnostics and are not a speed benchmark.
#include "runtime.hpp"
#include "dflash_contract.hpp"
#include "llama-ext.h"
#include "nlohmann/json.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
using namespace minimax_m2;
using json=nlohmann::ordered_json;
using Floats=std::vector<float>;
struct Batch {
    llama_batch b;
    Batch(int n,int pos,int width=0):b(llama_batch_init(n,width,1)) {
        b.n_tokens=n;for(int i=0;i<n;++i) {b.pos[i]=pos+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=width?0:1;}
    }
    ~Batch() {llama_batch_free(b);}
    Batch(const Batch &)=delete;
};
static void evaluate(llama_context *ctx,Batch &b) {
    strata_mm27_memory();int rc=llama_decode(ctx,b.b);llama_synchronize(ctx);
    require(rc==0,"draft decode failed: "+std::to_string(rc)+" "+strata_mm27_last_error());strata_mm27_memory();
}
static Floats logits(llama_context *ctx,int count,int vocab) {
    Floats out;out.reserve(size_t(count)*vocab);
    for(int i=0;i<count;++i) {auto *row=llama_get_logits_ith(ctx,i);require(row,"missing logits");out.insert(out.end(),row,row+vocab);}
    return out;
}
static json compare(const std::string &name,const Floats &a,const Floats &b,bool exact=true) {
    require(a.size()==b.size() && !a.empty(),"comparison size mismatch");
    double err=0,energy=0,max_abs=0;bool finite=true;size_t different=0;
    for(size_t i=0;i<a.size();++i) {
        finite&=std::isfinite(a[i]) && std::isfinite(b[i]);const double d=double(a[i])-b[i];
        err+=d*d;energy+=double(b[i])*b[i];max_abs=std::max(max_abs,std::abs(d));different+=a[i]!=b[i];
    }
    const double nmse=err/std::max(energy,1e-30);const bool bits=std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0;
    return {{"name",name},{"pass",finite && max_abs<=5e-4 && nmse<=1e-7 && (!exact || bits)},
        {"elements",a.size()},{"finite",finite},{"max_abs",max_abs},{"nmse",nmse},{"different",different},{"bit_exact",bits},{"exact_required",exact}};
}
static void save(const std::filesystem::path &path,const Floats &v) {
    std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char *>(v.data()),v.size()*sizeof(float));require(bool(out),"cannot save logits");
}
struct HeadObserver {
    Floats full,embedding;int rows=0,calls=0,embedding_calls=0;bool valid=true;
    static bool eval(ggml_tensor *t,bool ask,void *data) {
        auto &o=*static_cast<HeadObserver *>(data);
        const std::string name=t->name;
        if(name!="mm27_dflash_full_logits" && name!="inp_noise_embd")return false;
        if(ask)return true;
        if(name=="inp_noise_embd") {
            ++o.embedding_calls;o.valid&=t->type==GGML_TYPE_F32 && t->ne[0]==3072 && t->ne[2]==1 && t->ne[3]==1 && ggml_is_contiguous(t);
            if(o.valid) {o.embedding.resize(size_t(t->ne[0]*t->ne[1]));ggml_backend_tensor_get(t,o.embedding.data(),0,o.embedding.size()*sizeof(float));}
            return true;
        }
        ++o.calls;o.valid&=t->type==GGML_TYPE_F32 && t->ne[0]==target_vocab && t->ne[2]==1 && t->ne[3]==1 && ggml_is_contiguous(t);
        if(o.valid) {o.rows=int(t->ne[1]);o.full.resize(size_t(o.rows)*target_vocab);ggml_backend_tensor_get(t,o.full.data(),0,o.full.size()*sizeof(float));}
        return true;
    }
    Floats prefix() const {
        require(valid && calls==1 && rows>0,"full-head observer missing or wrong geometry");Floats v;
        for(int r=0;r<rows;++r)v.insert(v.end(),full.begin()+size_t(r)*target_vocab,full.begin()+size_t(r)*target_vocab+dflash_vocab);
        return v;
    }
};
int main(int argc,char **argv) {
    if(argc==2 && std::string(argv[1])=="--version") {std::cout<<"MiniMax DFlash correctness probe "<<STRATA_MM27_SOURCE_SHA<<'\n';return 0;}
    try {
        require(argc==4,"usage: dflash-check TARGET.gguf DRAFT.gguf FRESH_OUTPUT_DIR");
        const std::filesystem::path dir=argv[3];require(!std::filesystem::exists(dir),"output already exists");
        std::filesystem::create_directories(dir);environment();
        const auto draft_bytes=inspect_dflash(argv[1],argv[2]);ggml_backend_load_all();
        strata_mm27_mode(2);strata_mm27_pipeline(2,4,true,true);strata_mm27_cache_configure(18ull<<30,true,64,0);
        json tests=json::array(),blocks=json::array();
        auto target=load(argv[1]);auto tc=context(target.get(),512,8);
        // Clear cache before admitting the second model; dense draft never binds the expert cache.
        strata_mm27_cache_clear();strata_mm27_memory(draft_bytes+(2ull<<30),256ull<<20);
        auto *gpu=ggml_backend_dev_by_name("CUDA0");require(gpu,"CUDA0 missing");ggml_backend_dev_t devices[]={gpu,nullptr};
        auto mp=llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=-1;mp.split_mode=LLAMA_SPLIT_MODE_NONE;
        mp.load_mtp=false;mp.use_extra_bufts=false;mp.no_host=true;mp.load_mode=LLAMA_LOAD_MODE_MMAP;
        Model draft(llama_model_load_from_file(argv[2],mp),llama_model_free);require(bool(draft),"draft load failed");registered_dflash(*draft);
        for(const auto &v:draft->tensors_by_name)require(v.second->data && v.second->buffer && !ggml_backend_buffer_is_host(v.second->buffer),"draft weight not on GPU: "+v.first);
        HeadObserver observer;
        auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=8;cp.n_seq_max=1;cp.n_threads=cp.n_threads_batch=4;
        cp.type_k=cp.type_v=GGML_TYPE_F32;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=cp.op_offload=true;
        cp.ctx_other=tc.get();cp.cb_eval=HeadObserver::eval;cp.cb_eval_user_data=&observer;
        Context dc(llama_init_from_model(draft.get(),cp),llama_free);require(bool(dc),"draft context failed");
        clear(dc.get());llama_set_causal_attn(dc.get(),false);strata_mm27_memory();
        const std::string prompt="User: What is two plus two? Explain briefly.\nAssistant:";
        std::vector<llama_token> ids(128);const int nt=llama_tokenize(llama_model_get_vocab(target.get()),prompt.data(),int(prompt.size()),ids.data(),int(ids.size()),false,true);
        require(nt>=8,"probe prompt tokenization failed");ids.resize(8);
        strata_mm27_cache_decode(false);strata_mm27_reset();
        decode(tc.get(),ids,0,8,0,true);auto plain=logits(tc.get(),8,target_vocab);save(dir/"target-plain.f32",plain);
        clear(tc.get());for(int l:dflash_layers)llama_set_embeddings_layer_inp(tc.get(),l,true);
        decode(tc.get(),ids,0,8,0,true);auto captured=logits(tc.get(),8,target_vocab);save(dir/"target-captured.f32",captured);
        tests.push_back(compare("target_feature_capture_preserves_logits",captured,plain));
        Batch features(8,0,15360);json feature_norms=json::array();
        for(size_t k=0;k<dflash_layers.size();++k) {
            const auto *h=llama_get_embeddings_layer_inp(tc.get(),dflash_layers[k]);require(h,"missing target features");double norm=0;
            for(int t=0;t<8;++t)for(int d=0;d<3072;++d) {const auto v=h[t*3072+d];require(std::isfinite(v),"non-finite target feature");norm+=double(v)*v;features.b.embd[(t*5+k)*3072+d]=v;}
            require(norm>0,"zero target features");feature_norms.push_back({{"layer_input",dflash_layers[k]},{"sum_squared",norm}});
        }
        evaluate(dc.get(),features);
        require(llama_memory_seq_pos_max(llama_get_memory(dc.get()),0)==7,"feature injection position mismatch");
        const auto *last=captured.data()+7*target_vocab;const llama_token anchor=int(std::max_element(last,last+target_vocab)-last);
        require(anchor<dflash_vocab,"target anchor outside reviewed shared vocabulary");
        // Independent CPU reference for two constants, not model inference.
        // Read exactly the borrowed target rows; the sidecar has no embedding.
        Floats reference_embedding;
        for(int i=0;i<8;++i) {
            auto *w=target->tok_embd;const auto row_bytes=ggml_row_size(w->type,3072);
            std::vector<char> bytes(row_bytes);Floats row(3072);
            ggml_backend_tensor_get(w,bytes.data(),size_t(i?dflash_mask:anchor)*row_bytes,row_bytes);
            ggml_get_type_traits(w->type)->to_float(bytes.data(),row.data(),3072);
            reference_embedding.insert(reference_embedding.end(),row.begin(),row.end());
        }
        std::vector<llama_token> candidates;
        for(int depth:{1,2,4,7}) {
            Floats first;std::vector<llama_token> proposals;
            for(int repeat=0;repeat<2;++repeat) {
                observer=HeadObserver{};Batch noise(depth+1,8);noise.b.token[0]=anchor;
                for(int i=1;i<=depth;++i)noise.b.token[i]=dflash_mask;
                evaluate(dc.get(),noise);auto out=logits(dc.get(),depth+1,dflash_vocab);
                const auto tag="depth"+std::to_string(depth)+"_repeat"+std::to_string(repeat);
                save(dir/(tag+".f32"),out);save(dir/(tag+"_fullhead.f32"),observer.full);
                tests.push_back(compare(tag+"/all_logit_rows_match_full_head_prefix",out,observer.prefix()));
                if(depth==7 && !repeat) {
                    require(observer.embedding_calls==1,"missing noise embedding capture");
                    save(dir/"borrowed-embedding.f32",observer.embedding);save(dir/"reference-embedding.f32",reference_embedding);
                    tests.push_back(compare("borrowed_anchor_and_seven_MASK_embeddings",observer.embedding,reference_embedding,false));
                }
                if(!repeat) {
                    first=out;for(int i=1;i<=depth;++i) {const auto *r=out.data()+size_t(i)*dflash_vocab;proposals.push_back(int(std::max_element(r,r+dflash_vocab)-r));}
                } else tests.push_back(compare(tag+"/rollback_repeat",out,first));
                require(llama_memory_seq_rm(llama_get_memory(dc.get()),0,8,-1),"noise KV rollback failed");
                tests.push_back({{"name",tag+"/noise_KV_removed"},{"pass",llama_memory_seq_pos_max(llama_get_memory(dc.get()),0)==7}});
            }
            blocks.push_back({{"depth",depth},{"anchor",anchor},{"proposals",proposals}});if(depth==7)candidates=proposals;
        }
        std::vector<llama_token> verify{anchor};verify.insert(verify.end(),candidates.begin(),candidates.end());
        strata_mm27_cache_decode(true);decode(tc.get(),verify,0,8,8,true);auto batched=logits(tc.get(),8,target_vocab);save(dir/"target-verify-batch.f32",batched);
        int accepted=0;for(int i=0;i<7;++i) {const auto *r=batched.data()+size_t(i)*target_vocab;auto id=int(std::max_element(r,r+target_vocab)-r);if(candidates[i]!=id)break;++accepted;if(is_stop(id))break;}
        const int keep=1+accepted;Batch committed(keep,8,15360);
        for(size_t k=0;k<dflash_layers.size();++k) {
            const auto *h=llama_get_embeddings_layer_inp(tc.get(),dflash_layers[k]);require(h,"missing verified features");
            for(int i=0;i<keep;++i)std::copy_n(h+i*3072,3072,committed.b.embd+(i*5+k)*3072);
        }
        require(llama_memory_seq_rm(llama_get_memory(tc.get()),0,8,-1),"target rollback failed");
        Floats serial;for(int i=0;i<8;++i) {decode(tc.get(),verify,i,1,8+i,true);auto row=logits(tc.get(),1,target_vocab);serial.insert(serial.end(),row.begin(),row.end());}
        save(dir/"target-verify-serial.f32",serial);tests.push_back(compare("target_verification_batch_vs_serial",batched,serial,false));
        bool greedy_equal=true;for(int i=0;i<8;++i) {
            const auto *a=batched.data()+size_t(i)*target_vocab,*b=serial.data()+size_t(i)*target_vocab;
            greedy_equal&=(std::max_element(a,a+target_vocab)-a)==(std::max_element(b,b+target_vocab)-b);
        }
        tests.push_back({{"name","target_verification_greedy_batch_vs_serial"},{"pass",greedy_equal}});
        require(llama_memory_seq_rm(llama_get_memory(tc.get()),0,8+keep,-1),"target rejected suffix rollback failed");
        evaluate(dc.get(),committed);
        tests.push_back({{"name","target_and_draft_committed_prefix_positions"},{"pass",llama_memory_seq_pos_max(llama_get_memory(tc.get()),0)==7+keep && llama_memory_seq_pos_max(llama_get_memory(dc.get()),0)==7+keep}});
        const auto *bonus_row=batched.data()+size_t(keep-1)*target_vocab;
        const auto bonus=int(std::max_element(bonus_row,bonus_row+target_vocab)-bonus_row);
        require(bonus<dflash_vocab,"bonus outside shared vocabulary");Floats continued;
        for(int replay=0;replay<2;++replay) {
            if(replay) {clear(dc.get());evaluate(dc.get(),features);evaluate(dc.get(),committed);}
            observer=HeadObserver{};Batch noise(8,8+keep);noise.b.token[0]=bonus;for(int i=1;i<8;++i)noise.b.token[i]=dflash_mask;
            evaluate(dc.get(),noise);auto out=logits(dc.get(),8,dflash_vocab);
            const auto tag=replay?"fresh_replayed_prefix":"continued_prefix";save(dir/(std::string(tag)+".f32"),out);
            tests.push_back(compare(std::string(tag)+"/head_prefix",out,observer.prefix()));
            if(!replay)continued=out;else tests.push_back(compare("committed_prefix_vs_fresh_replay",out,continued));
            require(llama_memory_seq_rm(llama_get_memory(dc.get()),0,8+keep,-1),"continued draft rollback failed");
        }
        const auto s=strata_mm27_snapshot();const auto mem=strata_mm27_memory();
        tests.push_back({{"name","all_compute_on_GPU_no_full_expert_copy"},{"pass",s.gpu_nodes>0 && s.expert_nodes>0 && !s.rejected_cpu_nodes && !s.rejected_full_copies}});
        tests.push_back({{"name","sampled_global_RAM_VRAM_below_95_percent"},{"pass",s.memory_checks>0 &&
            double(s.sampled_ram_used_peak)<=double(mem.ram_total)*.95 && double(s.sampled_vram_used_peak)<=double(mem.vram_total)*.95}});
        dc.reset();draft.reset();tc.reset();target.reset();strata_mm27_release();
        int failures=0;for(const auto &t:tests)failures+=!t.at("pass").get<bool>();
        json report={{"pass",failures==0},{"failures",failures},{"cases",tests.size()},{"scope","short offline correctness probe; not end-to-end speculation or speed validation"},
            {"source_revision",STRATA_MM27_SOURCE_SHA},{"patches",STRATA_MM27_PATCH_SET},{"target",argv[1]},{"draft",argv[2]},
            {"prompt",prompt},{"prompt_truncated_to_tokens",ids},{"context",512},{"batch",8},{"KV","F32"},{"flash_attention",false},
            {"causal_draft",false},{"cache_gib",18},{"cache_block_mib",64},{"pipeline_readers",2},{"pipeline_chunk_mib",4},
            {"feature_norms",feature_norms},{"blocks",blocks},{"first_block_accepted",accepted},{"first_block_proposed",7},
            {"gpu_nodes",s.gpu_nodes},{"expert_nodes",s.expert_nodes},{"h2d_bytes",s.h2d_bytes},
            {"ram_used_peak",s.sampled_ram_used_peak},{"ram_total",mem.ram_total},{"vram_used_peak",s.sampled_vram_used_peak},{"vram_total",mem.vram_total},
            {"tests",tests}};
        std::ofstream(dir/"report.json")<<report.dump(2)<<'\n';std::cout<<json({{"pass",failures==0},{"cases",tests.size()},{"failures",failures},{"first_block_accepted",accepted}}).dump()<<'\n';return failures?1:0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';strata_mm27_release();return 2;}
}
