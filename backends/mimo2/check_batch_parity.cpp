// Diagnostic teacher-forced replay, not a throughput benchmark. Capture only
// named 2-D activations; scheduler readbacks can inhibit graph fusion.
#include "draft_probe.hpp"
#include "nlohmann/json.hpp"
#include <map>
#include <iostream>
using namespace mimo2;
using json=nlohmann::ordered_json;
using Floats=std::vector<float>;
struct Activation {int64_t width,columns;Floats values;};
struct Capture {
    bool enabled=false;
    std::map<std::string,Activation> tensors;
    std::vector<std::string> order;
    static bool callback(ggml_tensor *t,bool ask,void *data) {
        auto &c=*static_cast<Capture *>(data);
        if(ask) {
            if(!c.enabled || t->type!=GGML_TYPE_F32 || t->ne[2]!=1 || t->ne[3]!=1 || t->ne[1]>8)return false;
            const std::string name=t->name;
            for(const char *p:{"attn_norm-","wqkv-","kqv_out-","attn_out-","attn_out_scaled-","ffn_inp-","ffn_norm-",
                              "ffn_up-","ffn_gate-","ffn_down-","ffn_out-","ffn_moe_logits-","ffn_moe_out-","l_out-","result_norm","result_output"})
                if(name.rfind(p,0)==0)return true;
            return false;
        }
        require(ggml_nbytes(t)<16u*1024*1024,"unexpected capture size");
        std::vector<uint8_t> bytes(ggml_nbytes(t));ggml_backend_tensor_get(t,bytes.data(),0,bytes.size());
        Activation a{t->ne[0],t->ne[1],Floats(size_t(t->ne[0]*t->ne[1]))};
        for(int64_t col=0;col<t->ne[1];++col)for(int64_t row=0;row<t->ne[0];++row)
            std::memcpy(&a.values[size_t(col*t->ne[0]+row)],bytes.data()+col*t->nb[1]+row*t->nb[0],4);
        if(!c.tensors.count(t->name))c.order.push_back(t->name);
        c.tensors[t->name]=std::move(a);return true;
    }
    void reset() {tensors.clear();order.clear();}
    json dump(const std::filesystem::path &p) const {
        std::ofstream f(p,std::ios::binary);require(bool(f),"capture open failed");json index=json::array();size_t offset=0;
        for(const auto &name:order) {
            const auto &a=tensors.at(name);f.write(reinterpret_cast<const char *>(a.values.data()),a.values.size()*4);
            index.push_back({{"name",name},{"width",a.width},{"columns",a.columns},{"offset",offset}});offset+=a.values.size()*4;
        }
        require(bool(f),"capture write failed");return index;
    }
};
static void mode(int m) {_putenv_s("STRATA_MIMO_TOKENWISE_MATMUL",std::to_string(m).c_str());}
static json compare(const Capture &batch,const Capture &single,int column) {
    json out=json::array();
    for(const auto &name:single.order) {
        const auto &a=single.tensors.at(name);auto it=batch.tensors.find(name);
        if(it==batch.tensors.end()) {out.push_back({{"name",name},{"missing",true}});continue;}
        const auto &b=it->second;
        require(a.columns==1 && b.columns==2 && a.width==b.width,"capture geometry mismatch: "+name);
        double maxabs=0,error=0,energy=0;bool finite=true;const float *v=b.values.data()+column*b.width;
        for(int64_t i=0;i<a.width;++i) {const double d=double(v[i])-a.values[i];maxabs=std::max(maxabs,std::abs(d));error+=d*d;energy+=double(a.values[i])*a.values[i];finite&=std::isfinite(v[i])&&std::isfinite(a.values[i]);}
        out.push_back({{"name",name},{"width",a.width},{"bit_exact",std::memcmp(v,a.values.data(),a.width*4)==0},
            {"finite",finite},{"max_abs",maxabs},{"nmse",error/std::max(energy,1e-30)}});
    }
    return out;
}
int main(int argc,char **argv) {
    try {
        require(argc==4,"usage: batch-parity-check MODEL REQUEST.json FRESH_DIR");
        const std::filesystem::path out=argv[3];require(!std::filesystem::exists(out),"fresh output directory required");std::filesystem::create_directories(out);
        std::ifstream input(argv[2]);require(bool(input),"request open failed");json requests;input>>requests;
        const auto prompt=requests.at(0).at("tokens").get<std::vector<llama_token>>();
        const auto oracle=requests.at(0).at("oracle_ids").get<std::vector<llama_token>>();
        require(oracle.size()>=18 && !prompt.empty() && prompt.size()+oracle.size()<480,"invalid replay request");
        environment();mode(0);_putenv_s("STRATA_MIMO_TARGET_HEAD_COLUMNS","1");ggml_backend_load_all();strata_mimo_mode(2);
        json report={{"status","pass"},{"scope","Teacher-forced same-history activation diagnostic; callback may change fusion; no speed claim"},{"cases",json::array()}};
        {
            auto model=load(argv[1]);Capture capture;auto ctx=probe_context(model.get(),false,nullptr,Capture::callback,&capture);
            strata_mimo_reader(true);strata_mimo_phase(true);std::vector<llama_token> warm(8,11);decode(ctx.get(),warm,0,8,0,true);
            clear(ctx.get());strata_mimo_cache(size_t(8)<<30);strata_mimo_cache_prefill(false);strata_mimo_pipeline_config(1,8);
            auto prefix=[&](int previous) {
                capture.enabled=false;mode(0);clear(ctx.get());strata_mimo_phase(true);
                for(size_t i=0;i<prompt.size();i+=8)decode(ctx.get(),prompt,i,int(std::min(size_t(8),prompt.size()-i)),int(i));
                strata_mimo_phase(false);
                for(int i=0;i<previous;++i)decode(ctx.get(),oracle,i,1,int(prompt.size())+i,true);
            };
            for(int previous:{0,16}) {
                prefix(previous);Capture refs[2];
                for(int t=0;t<2;++t) {
                    capture.reset();capture.enabled=true;decode(ctx.get(),oracle,previous+t,1,int(prompt.size())+previous+t,true);refs[t]=capture;
                    const auto name="p"+std::to_string(previous)+"-single"+std::to_string(t);
                    const auto index=capture.dump(out/(name+".f32"));std::ofstream(out/(name+".json"))<<index.dump(2)<<'\n';
                }
                for(int m:{5,7}) {
                    prefix(previous);capture.reset();capture.enabled=true;mode(m);
                    decode(ctx.get(),oracle,previous,2,int(prompt.size())+previous,true);mode(0);
                    const auto name="p"+std::to_string(previous)+"-mode"+std::to_string(m);
                    const auto index=capture.dump(out/(name+".f32"));std::ofstream(out/(name+".json"))<<index.dump(2)<<'\n';
                    report["cases"].push_back({{"previous",previous},{"mode",m},{"first",compare(capture,refs[0],0)},{"second",compare(capture,refs[1],1)},
                        {"ids",{draft_greedy(ctx.get(),0),draft_greedy(ctx.get(),1)}}});
                    std::ofstream(out/"report.json")<<report.dump(2)<<'\n';
                    std::cerr<<"CAPTURE p"<<previous<<" mode"<<m<<" complete\n";
                }
            }
            auto s=strata_mimo_snapshot();require(!s.rejected_cpu_nodes && !s.pipeline_reader_owned && !s.pipeline_queued,"GPU/drain audit failed");
            report["gpu_nodes"]=s.gpu_nodes;report["rejected_cpu_nodes"]=s.rejected_cpu_nodes;strata_mimo_release();
        }
        llama_backend_free();std::ofstream(out/"report.json")<<report.dump(2)<<'\n';std::cout<<"DONE\n";return 0;
    } catch(const std::exception &e) {strata_mimo_release();std::cerr<<e.what()<<'\n';return 1;}
}
