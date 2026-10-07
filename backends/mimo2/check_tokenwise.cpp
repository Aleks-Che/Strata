// Real MiMo matrix geometries, GPU single-token reference, synthetic weights.
// No CPU matrix oracle: the CPU only generates/quantizes weights and compares.
#include "runtime.hpp"
#include "ggml-cpp.h"
#include "nlohmann/json.hpp"
#include <cstring>
#include <iostream>
using namespace mimo2;
using json=nlohmann::ordered_json;
using Floats=std::vector<float>;
static void mode(int n) {_putenv_s("STRATA_MIMO_TOKENWISE_MATMUL",std::to_string(n).c_str());}
struct Weights {
    int k,m,experts;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor *base;
    std::vector<uint8_t> packed;
    Weights(ggml_backend_t backend,ggml_type type,int in,int out):k(in),m(out),experts(type==GGML_TYPE_BF16?1:16),
        ctx(ggml_init({ggml_tensor_overhead()*4,nullptr,true})) {
        require(bool(ctx),"weight context failed");base=ggml_new_tensor_3d(ctx.get(),type,k,m+2,experts);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(),backend));require(bool(buffer),"weight GPU allocation failed");
        ggml_backend_buffer_set_usage(buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        Floats source(ggml_nelements(base)),importance(k,1.f);uint32_t s=0x27185u;
        for(auto &x:source) {s^=s<<13;s^=s>>17;s^=s<<5;x=(float(s&65535)/32767.5f-1)*.03f;}
        packed.resize(ggml_nbytes(base));
        require(ggml_quantize_chunk(type,source.data(),packed.data(),0,(m+2)*experts,k,importance.data())==packed.size(),"quantization size");
        ggml_backend_tensor_set(base,packed.data(),0,packed.size());
    }
    void unchanged() {std::vector<uint8_t> copy(packed.size());ggml_backend_tensor_get(base,copy.data(),0,copy.size());require(copy==packed,"weights changed");}
};
static Floats run(ggml_backend_t backend,const Weights &w,const Floats &input,const std::vector<int32_t> &ids,
                  int tokens,int lanes,bool padded,int strategy) {
    mode(strategy);const bool routed=w.experts>1;
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead()*32+ggml_graph_overhead(),nullptr,true}));require(bool(ctx),"graph context failed");
    auto *a=ggml_view_3d(ctx.get(),w.base,w.k,w.m,w.experts,w.base->nb[1],w.base->nb[2],w.base->nb[1]);
    const int input_rows=lanes+(padded?2:0),id_rows=8+(padded?3:0);const size_t offset=padded?w.k*4:0;
    auto *storage=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,w.k,input_rows,tokens);
    auto *ib=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,id_rows,tokens);
    auto *b=routed?ggml_view_3d(ctx.get(),storage,w.k,lanes,tokens,storage->nb[1],storage->nb[2],offset):
                    ggml_view_2d(ctx.get(),storage,w.k,tokens,storage->nb[2],offset);
    auto *selected=ggml_view_2d(ctx.get(),ib,8,tokens,ib->nb[1],padded?4:0);
    auto *y=routed?ggml_mul_mat_id(ctx.get(),a,b,selected):ggml_mul_mat(ctx.get(),a,b);
    auto *graph=ggml_new_graph(ctx.get());ggml_build_forward_expand(graph,y);
    for(int i=0;i<ggml_graph_n_nodes(graph);++i)require(ggml_backend_supports_op(backend,ggml_graph_node(graph,i)),"unsupported GPU node");
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend));require(bool(buffer),"graph allocation failed");
    Floats physical(ggml_nelements(storage),123.f);std::vector<int32_t> selected_physical(ggml_nelements(ib),15);
    for(int t=0;t<tokens;++t) {
        std::copy_n(input.data()+t*w.k*lanes,w.k*lanes,physical.data()+t*w.k*input_rows+offset/4);
        std::copy_n(ids.data()+t*8,8,selected_physical.data()+t*id_rows+(padded?1:0));
    }
    ggml_backend_tensor_set(storage,physical.data(),0,ggml_nbytes(storage));ggml_backend_tensor_set(ib,selected_physical.data(),0,ggml_nbytes(ib));
    require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"tokenwise GPU compute failed");
    Floats output(ggml_nelements(y)),after(physical.size());std::vector<int32_t> after_ids(selected_physical.size());
    ggml_backend_tensor_get(y,output.data(),0,ggml_nbytes(y));ggml_backend_tensor_get(storage,after.data(),0,ggml_nbytes(storage));ggml_backend_tensor_get(ib,after_ids.data(),0,ggml_nbytes(ib));
    require(std::memcmp(after.data(),physical.data(),physical.size()*4)==0 && after_ids==selected_physical,"input or canary changed");
    return output;
}
static json compare(const Floats &a,const Floats &b) {
    require(a.size()==b.size(),"shape mismatch");double maxabs=0,error=0,energy=0;bool finite=true;
    for(size_t i=0;i<a.size();++i) {const double d=double(a[i])-b[i];maxabs=std::max(maxabs,std::abs(d));error+=d*d;energy+=double(b[i])*b[i];finite&=std::isfinite(a[i])&&std::isfinite(b[i]);}
    return {{"elements",a.size()},{"finite",finite},{"max_abs",maxabs},{"nmse",error/std::max(energy,1e-30)},{"bit_exact",std::memcmp(a.data(),b.data(),a.size()*4)==0}};
}
int main() {
    try {
        environment();ggml_backend_load_all();strata_mimo_memory(size_t(512)<<20,size_t(1)<<30);
        ggml_backend_ptr backend(ggml_backend_dev_init(ggml_backend_dev_by_name("CUDA0"),nullptr));require(bool(backend),"CUDA backend required");
        json report={{"status","pass"},{"scope","Synthetic BF16/Q2_K/Q3_K/MXFP4; same-input GPU single-token oracle; actual MiMo matmul shapes"},{"cases",json::array()}};
        for(ggml_type type:{GGML_TYPE_BF16,GGML_TYPE_Q2_K,GGML_TYPE_Q3_K,GGML_TYPE_MXFP4})for(bool down:{false,true}) {
            Weights w(backend.get(),type,down?2048:4096,down?4096:2048);const int lanes=type==GGML_TYPE_BF16?1:down?8:1;
            for(int tokens:{1,2,8})for(bool padded:{false,true}) {
                Floats input(size_t(w.k)*lanes*tokens);uint32_t s=0x88181u;
                for(float &x:input) {s^=s<<13;s^=s>>17;s^=s<<5;x=(float(s&65535)/32767.5f-1)*.5f;}
                std::vector<int32_t> ids(tokens*8);const int choices[8]={0,15,1,14,2,13,3,12};
                for(int t=0;t<tokens;++t)for(int i=0;i<8;++i)ids[t*8+i]=(choices[i]+t*3)%16;
                Floats reference;
                for(int t=0;t<tokens;++t) {
                    auto x=run(backend.get(),w,{input.begin()+t*w.k*lanes,input.begin()+(t+1)*w.k*lanes},
                        {ids.begin()+t*8,ids.begin()+(t+1)*8},1,lanes,padded,0);
                    reference.insert(reference.end(),x.begin(),x.end());
                }
                const auto old=run(backend.get(),w,input,ids,tokens,lanes,padded,0);
                const auto current=run(backend.get(),w,input,ids,tokens,lanes,padded,3);const auto parity=compare(current,reference);
                const bool pass=parity.at("finite").get<bool>() && parity.at("bit_exact").get<bool>();
                if(!pass)report["status"]="mismatch";
                report["cases"].push_back({{"type",ggml_type_name(type)},{"k",w.k},{"m",w.m},{"lanes",lanes},{"tokens",tokens},{"padded",padded},
                    {"candidate",parity},{"old",compare(old,reference)},{"inputs_and_canaries_unchanged",true}});
            }
            w.unchanged();strata_mimo_memory();
        }
        std::cout<<report.dump()<<'\n';return report["status"]=="pass"?0:1;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
