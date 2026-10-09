#include "runtime.hpp"
#include "router_ids.hpp"
#include "synthetic_minimax_m2.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
using namespace minimax_m2;
using json=nlohmann::ordered_json;
using Floats=std::vector<float>;
static bool equal(const Floats &a,const Floats &b) {
    return a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*sizeof(float)) &&
        std::all_of(a.begin(),a.end(),[](float x){return std::isfinite(x);});
}
static Floats run(llama_context *ctx,int batch,int salt) {
    clear(ctx);Floats out;std::vector<llama_token> tokens(25);
    for(size_t i=0;i<tokens.size();++i)tokens[i]=(int(i)*7+11+salt)%64;
    for(int pos=0;pos<17;) {
        const int count=std::min(batch,17-pos);decode(ctx,tokens,pos,count,pos,true);
        for(int i=0;i<count;++i) {const auto *p=llama_get_logits_ith(ctx,i);out.insert(out.end(),p,p+64);}
        pos+=count;
    }
    for(int pos=17;pos<25;++pos) {
        decode(ctx,tokens,pos,1,pos);const auto *p=llama_get_logits_ith(ctx,-1);out.insert(out.end(),p,p+64);
    }
    return out;
}
int main(int argc,char **argv) {
    try {
        require(argc==2,"usage: router-ids-check NEW_DIRECTORY");const std::filesystem::path dir=argv[1];
        require(!std::filesystem::exists(dir),"output directory exists");std::filesystem::create_directories(dir);
        json tests=json::array(),runs=json::array();
        auto check=[&](const std::string &name,bool ok){tests.push_back({{"name",name},{"pass",ok}});};
        ggml_context_ptr cpu(ggml_init({ggml_tensor_overhead()*8,nullptr,true}));
        auto *ids=ggml_new_tensor_2d(cpu.get(),GGML_TYPE_I32,8,16);
        auto *weight=ggml_new_tensor_3d(cpu.get(),GGML_TYPE_Q4_K,256,16,256);
        ggml_tensor node{};node.op=GGML_OP_MUL_MAT_ID;node.src[0]=weight;node.src[2]=ids;
        int device_identity=0;ids->data=&device_identity;ggml_tensor source=*ids;
        std::vector<int32_t> data(128),copy(128);for(int i=0;i<128;++i)data[i]=i;
        RouterIds slot;auto publish=[&](){return slot.publish(&node,&source,data.data(),512);};
        auto take=[&](){return slot.take(&node,copy.data(),512);};
        check("unpublished_fallback",!take());check("published",publish());
        data[0]=255;check("owned_snapshot",take() && copy[0]==0);data[0]=0;
        check("single_use",!take());
        publish();auto other=node;check("same_gpu_address_other_node_fallback",!slot.take(&other,copy.data(),512));
        check("mismatch_consumes",!take());
        publish();ids->data=data.data();check("moved_gpu_storage_fallback",!take());ids->data=&device_identity;
        publish();ids->nb[1]+=4;check("changed_stride_fallback",!take());*ids=source;
        publish();ids->ne[1]=8;check("changed_shape_fallback",!take());*ids=source;
        publish();ids->view_offs=4;check("changed_view_fallback",!take());*ids=source;
        publish();weight->ne[2]=255;check("changed_expert_count_fallback",!take());weight->ne[2]=256;
        publish();check("short_destination_fallback",!slot.take(&node,copy.data(),32));
        publish();slot.clear();check("split_boundary_fallback",!take());
        try {RouterIdsScope scope(slot);publish();throw std::runtime_error("simulated split failure");}catch(const std::exception &){}
        check("exception_scope_clears",!take());
        {RouterIdsScope scope(slot);publish();}check("normal_scope_clears",!take());
        data[3]=-1;check("invalid_expert_rejected",!publish());data[3]=3;
        source.nb[1]+=4;check("copied_ids_layout_mismatch",!publish());source=*ids;
        check("oversize_rejected",!slot.publish(&node,&source,data.data(),65540));
        check("new_graph_same_addresses_fresh_bytes",publish() && take() && copy==data);

        environment();ggml_backend_load_all();
        const auto path=(dir/"mixed.gguf").string();write_synthetic_minimax_m2(path,true);
        for(int events:{0,2})for(int batch:{1,8,16}) {
            strata_mm27_mode(2);strata_mm27_pipeline(2,4,true,true,events);
            strata_mm27_cache_configure(32ull<<20,true);
            auto model=load(path,false,true);auto ctx=context(model.get(),512,batch);
            for(int salt:{0,3,0}) {
                const auto name=std::to_string(events)+"/"+std::to_string(batch)+"/"+std::to_string(salt)+"/"+std::to_string(runs.size());
                strata_mm27_router_host_ids(false);strata_mm27_reset();const auto baseline=run(ctx.get(),batch,salt);
                check(name+"disabled_no_hits",strata_mm27_snapshot().router_ids_hits==0);
                strata_mm27_router_host_ids(true);strata_mm27_reset();const auto candidate=run(ctx.get(),batch,salt);
                const auto s=strata_mm27_snapshot();
                check(name+"exact_all_logits",equal(baseline,candidate));
                check(name+"reuse_exercised",s.router_ids_hits>0 && s.router_ids_bytes>0 && s.router_ids_published>=s.router_ids_hits);
                check(name+"gpu_only_drained",s.gpu_nodes>0 && !s.rejected_cpu_nodes && !s.rejected_full_copies && !s.pipeline_pending_copy);
                runs.push_back({{"name",name},{"floats",candidate.size()},{"hits",s.router_ids_hits},{"misses",s.router_ids_misses},{"bytes",s.router_ids_bytes}});
            }
            ctx.reset();model.reset();strata_mm27_release();check("release/"+std::to_string(events)+"/"+std::to_string(batch),strata_mm27_snapshot().router_ids_hits==0);
        }
        size_t failures=0;for(const auto &t:tests)if(!t.at("pass").get<bool>())++failures;
        json report={{"tests",tests},{"runs",runs},{"cases",tests.size()},{"failures",failures},{"pass",!failures}};
        std::ofstream(dir/"report.json")<<report.dump(2)<<'\n';std::cout<<report.dump(2)<<'\n';return failures?1:0;
    }catch(const std::exception &e) {std::cerr<<e.what()<<'\n';strata_mm27_release();return 2;}
}
