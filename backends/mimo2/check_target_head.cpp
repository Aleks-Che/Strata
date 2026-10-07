// Actual target BF16 weights, synthetic identical F32 activations. All matrix
// operations use CUDA. The reference projects independent single columns.
#include "runtime.hpp"
#include "target_head.hpp"
#include "ggml-cpp.h"
#include "nlohmann/json.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
using namespace mimo2;
using json=nlohmann::ordered_json;
using Floats=std::vector<float>;
struct Result {Floats values;json metrics;};

static Result project(ggml_tensor *weights,const Floats &input,int tokens,bool padded,bool columns) {
    ggml_backend_ptr backend(ggml_backend_dev_init(ggml_backend_dev_by_name("CUDA0"),nullptr));
    require(bool(backend),"CUDA initialization failed");
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead()*64+ggml_graph_overhead(),nullptr,true}));
    require(bool(ctx),"head graph context failed");
    const int k=int(weights->ne[0]),stride=k+(padded?32:0),offset=padded?16:0;
    auto *storage=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,stride,tokens);
    auto *x=ggml_view_2d(ctx.get(),storage,k,tokens,storage->nb[1],offset*sizeof(float));
    auto mul=[&](ggml_tensor *column) {return ggml_mul_mat(ctx.get(),weights,column);};
    auto *y=columns?mimo2_project_head_columns(ctx.get(),x,mul):mul(x);
    auto *graph=ggml_new_graph(ctx.get());ggml_build_forward_expand(graph,y);
    for(int i=0;i<ggml_graph_n_nodes(graph);++i)
        require(ggml_backend_supports_op(backend.get(),ggml_graph_node(graph,i)),"non-CUDA head operation");
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
    require(bool(buffer),"head graph allocation failed");
    Floats source(size_t(stride)*tokens,123.f);
    for(int t=0;t<tokens;++t)std::copy_n(input.data()+t*k,k,source.data()+t*stride+offset);
    ggml_backend_tensor_set(storage,source.data(),0,ggml_nbytes(storage));
    strata_mimo_memory();const auto before=strata_mimo_snapshot();
    auto execute=[&] {require(ggml_backend_graph_compute(backend.get(),graph)==GGML_STATUS_SUCCESS,"head compute failed");};
    execute();strata_mimo_memory();const auto after=strata_mimo_snapshot();
    const auto start=std::chrono::steady_clock::now();
    for(int i=0;i<3;++i)execute();
    const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/3;
    Floats output(ggml_nelements(y)),unchanged(source.size());
    ggml_backend_tensor_get(y,output.data(),0,ggml_nbytes(y));
    ggml_backend_tensor_get(storage,unchanged.data(),0,ggml_nbytes(storage));
    require(std::memcmp(source.data(),unchanged.data(),source.size()*4)==0,"head input mutated");
    return {std::move(output),{{"mean_graph_ms",elapsed},{"gpu_free_before_compute",before.gpu_free},
        {"gpu_free_after_compute",after.gpu_free},{"global_gpu_used_delta",int64_t(before.gpu_free)-int64_t(after.gpu_free)},
        {"gpu_nodes",ggml_graph_n_nodes(graph)},{"input_unchanged",true}}};
}
static json compare(const Floats &actual,const Floats &expected) {
    require(actual.size()==expected.size(),"head comparison shape");
    double maxabs=0,error=0,energy=0;bool finite=true;
    for(size_t i=0;i<actual.size();++i) {
        finite&=std::isfinite(actual[i])&&std::isfinite(expected[i]);
        const double d=double(actual[i])-expected[i];maxabs=std::max(maxabs,std::abs(d));
        error+=d*d;energy+=double(expected[i])*expected[i];
    }
    return {{"elements",actual.size()},{"finite",finite},{"max_abs",maxabs},{"nmse",error/std::max(energy,1e-30)},
        {"bit_exact",std::memcmp(actual.data(),expected.data(),actual.size()*4)==0}};
}
int main(int argc,char **argv) {
    try {
        require(argc==2,"usage: strata-mimo2-head-check MODEL");environment();ggml_backend_load_all();strata_mimo_mode(2);
        json report={{"status","pass"},{"scope","Actual BF16 target head; synthetic inputs; independent GPU single-column oracle; global allocation deltas"},{"cases",json::array()}};
        {
            auto model=load(argv[1]);auto *w=model->output;
            require(w && w->type==GGML_TYPE_BF16 && w->ne[0]==4096 && w->ne[1]==152576,"unexpected target head");
            for(int tokens:{1,2,8})for(bool padded:{false,true}) {
                Floats input(size_t(tokens)*w->ne[0]);uint32_t random=0x71234u;
                for(float &x:input) {random^=random<<13;random^=random>>17;random^=random<<5;x=(float(random&65535)/32767.5f-1)*.5f;}
                Floats reference;
                for(int t=0;t<tokens;++t) {
                    Floats column(input.begin()+t*w->ne[0],input.begin()+(t+1)*w->ne[0]);
                    auto one=project(w,column,1,false,false);
                    reference.insert(reference.end(),one.values.begin(),one.values.end());
                }
                // Candidate first: the cuBLAS pool in a separate backend must not
                // contaminate its measured allocation delta.
                auto candidate=project(w,input,tokens,padded,true),batched=project(w,input,tokens,padded,false);
                const auto parity=compare(candidate.values,reference);
                require(parity.at("finite").get<bool>() && parity.at("bit_exact").get<bool>(),"column head differs from independent GPU oracle");
                report["cases"].push_back({{"tokens",tokens},{"padded",padded},{"columns",candidate.metrics},
                    {"batched",batched.metrics},{"columns_vs_single",parity},{"batched_vs_single",compare(batched.values,reference)}});
            }
            strata_mimo_release();
        }
        llama_backend_free();std::cout<<report.dump()<<'\n';return 0;
    } catch(const std::exception &e) {strata_mimo_release();std::cerr<<e.what()<<'\n';return 1;}
}
