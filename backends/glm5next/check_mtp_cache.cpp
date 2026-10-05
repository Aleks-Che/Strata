#include "runtime.hpp"
#include "runtime_memory.hpp"
#include "synthetic_glm.hpp"
#include "llama-ext.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace strata_glm;
using json=nlohmann::json;

struct Probe {
    size_t expert_ops=0, cache_nodes=0;
    static bool callback(ggml_tensor * t,bool ask,void * data) {
        if (ask) return true;
        auto & p=*static_cast<Probe *>(data);
        p.expert_ops+=t->op==GGML_OP_MUL_MAT_ID;
        p.cache_nodes+=std::string(t->name).find("mtp_cache_kv")==0;
        return true;
    }
};

static void forward(llama_context * ctx,int pos,int count,bool outputs) {
    auto batch=llama_batch_init(count,256,1);
    batch.token=static_cast<llama_token *>(std::malloc(count*sizeof(llama_token)));
    if (!batch.token) {llama_batch_free(batch);throw std::bad_alloc();}
    batch.n_tokens=count;
    for (int i=0;i<count;++i) {
        batch.token[i]=(7*(pos+i)+11)%64;batch.pos[i]=pos+i;
        batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=outputs && i==count-1;
        for (int j=0;j<256;++j) batch.embd[i*256+j]=std::sin(float((pos+i+1)*(j+1))*.017f)*.1f;
    }
    const int rc=llama_decode(ctx,batch);llama_batch_free(batch);
    require(rc==0,"draft decode failed");llama_synchronize(ctx);
}

static std::vector<uint8_t> state(llama_context * ctx) {
    std::vector<uint8_t> result(llama_state_seq_get_size(ctx,0));
    require(!result.empty() && llama_state_seq_get_data(ctx,result.data(),result.size(),0)==result.size(),"state extraction failed");
    return result;
}

static void equal_floats(const float * a,const float * b,size_t n,const char * message) {
    require(a && b,"missing draft output");
    for (size_t i=0;i<n;++i) require(std::isfinite(a[i]) && std::isfinite(b[i]),"nonfinite draft output");
    require(std::memcmp(a,b,n*sizeof(float))==0,message);
}

int main(int argc,char ** argv) {
    json report={{"status","error"},{"checks",json::array()}};
    try {
        require(argc==2,"usage: mtp-cache-check fixture-directory");
        std::filesystem::create_directories(argv[1]);
        environment();ggml_backend_load_all();llama_backend_init();strata_glm_sync_enable(true);
        for (uint32_t top_k:{8u,512u}) {
            const auto file=(std::filesystem::path(argv[1])/("cache-"+std::to_string(top_k)+".gguf")).string();
            write_synthetic_glm(file,top_k,true);
            auto model=load(file,false,false,true);
            Probe full_probe,cache_probe;
            auto full=context(model.get(),512,16,4,0,true,Probe::callback,&full_probe);
            auto cached=context(model.get(),512,16,4,0,true,Probe::callback,&cache_probe);
            llama_set_embeddings_nextn(full.get(),true,true);
            llama_set_embeddings_nextn(cached.get(),true,true);
            RuntimeMemory memory(model,file,0,0,true,4,512);
            auto compare_state=[&](const std::string & name) {
                auto a=state(full.get()),b=state(cached.get());
                require(a==b,name+": MLA/indexer state differs");
                report["checks"].push_back({{"name",name},{"top_k",top_k},{"state_bytes",a.size()},{"bit_exact",true}});
            };
            auto next=[&](int & pos) {
                forward(full.get(),pos,1,true);forward(cached.get(),pos,1,true);++pos;
                equal_floats(llama_get_logits_ith(full.get(),-1),llama_get_logits_ith(cached.get(),-1),64,"next draft logits differ");
                equal_floats(llama_get_embeddings_nextn_ith(full.get(),0),llama_get_embeddings_nextn_ith(cached.get(),0),256,"next draft hidden differs");
                compare_state("continuation after cache-only");
            };
            auto catch_up=[&](int & pos,int count) {
                full_probe={};cache_probe={};
                forward(full.get(),pos,count,true);forward(cached.get(),pos,count,false);pos+=count;
                require(full_probe.expert_ops>0,"full control did not execute experts");
                require(cache_probe.expert_ops==0 && cache_probe.cache_nodes>0,"cache-only graph was not pruned");
                compare_state("catch-up "+std::to_string(count));
            };
            int pos=0;
            // Cross partial and completed kpool groups; repeated sizes alternate
            // full and zero-output graphs, exercising reuse in both directions.
            for (int count:{3,1,2,4,7,16,1,1,2,3}) {catch_up(pos,count);next(pos);}
            // Discard every possible prefix of a three-token draft, then repair
            // and continue. Includes the kpool dirty/rebuild path after seq_rm.
            for (int accepted:{0,1,2,3}) {
                const int start=pos;
                for (int i=0;i<3;++i) next(pos);
                pos=start+accepted;
                require(llama_memory_seq_rm(llama_get_memory(full.get()),0,pos,-1),"full rollback failed");
                require(llama_memory_seq_rm(llama_get_memory(cached.get()),0,pos,-1),"cached rollback failed");
                catch_up(pos,4);next(pos);
            }
            // Zero logits must NOT prune a caller's unmasked hidden extraction.
            llama_set_embeddings_nextn(full.get(),true,false);
            llama_set_embeddings_nextn(cached.get(),true,false);
            full_probe={};cache_probe={};
            forward(full.get(),pos,3,true);forward(cached.get(),pos,3,false);pos+=3;
            require(cache_probe.expert_ops>0 && cache_probe.cache_nodes==0,"unmasked hidden extraction was pruned");
            for (int i=0;i<3;++i) equal_floats(llama_get_embeddings_nextn_ith(full.get(),i),llama_get_embeddings_nextn_ith(cached.get(),i),256,"unmasked hidden differs");
            compare_state("unmasked hidden rows with zero logits");
            llama_set_embeddings_nextn(full.get(),true,true);
            llama_set_embeddings_nextn(cached.get(),true,true);
            catch_up(pos,1);next(pos);
            clear(full.get());clear(cached.get());pos=0;
            catch_up(pos,3);next(pos);
        }
        report["status"]="pass";
    } catch (const std::exception & e) {report["error"]=e.what();std::cerr<<e.what()<<"\n";}
    strata_glm_sync_release();llama_backend_free();
    if (argc==2) {std::ofstream out(std::filesystem::path(argv[1])/"report.json");out<<report.dump(2)<<"\n";}
    std::cout<<report.dump(2)<<"\n";
    return report["status"]=="pass"?0:1;
}
