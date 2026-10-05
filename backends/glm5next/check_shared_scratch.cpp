#include "shared_scratch.h"
#include "ggml-cpp.h"
#include "nlohmann/json.hpp"
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
using json=nlohmann::ordered_json;
static void require(bool v,const char * what) {if(!v)throw std::runtime_error(what);}
struct Graph {
    ggml_context_ptr ctx;
    ggml_tensor *input,*output;
    ggml_cgraph *graph;
    Graph(int n):ctx(ggml_init({8*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true})) {
        input=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,n);ggml_set_input(input);
        output=ggml_scale(ctx.get(),input,2.0f);ggml_set_output(output);
        graph=ggml_new_graph(ctx.get());ggml_build_forward_expand(graph,output);
    }
    void run(ggml_gallocr_t alloc,ggml_backend_t gpu,int seed) {
        require(ggml_gallocr_alloc_graph(alloc,graph),"graph allocation failed");
        std::vector<float> in(ggml_nelements(input)),out(in.size());
        for(size_t i=0;i<in.size();++i)in[i]=float(int(i%127)-seed)*.125f;
        ggml_backend_tensor_set(input,in.data(),0,in.size()*sizeof(float));
        require(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"compute failed");
        ggml_backend_tensor_get(output,out.data(),0,out.size()*sizeof(float));
        for(size_t i=0;i<in.size();++i)require(out[i]==2*in[i],"cross-context scratch corruption");
    }
};
using Alloc=std::unique_ptr<ggml_gallocr,decltype(&ggml_gallocr_free)>;
int main() {
    json report={{"status","error"},{"checks",json::array()}};
    try {
        ggml_backend_load_all();auto *dev=ggml_backend_dev_by_name("CUDA0");require(dev,"CUDA0 required");
        ggml_backend_ptr gpu(ggml_backend_dev_init(dev,nullptr));require(bool(gpu),"CUDA init failed");
        auto buft=ggml_backend_get_default_buffer_type(gpu.get());
        for(bool reverse:{false,true}) for(bool grow:{false,true}) {
            Graph a(4096),b(8192),larger(65536);
            Alloc aa(ggml_gallocr_new(buft),ggml_gallocr_free),bb(ggml_gallocr_new(buft),ggml_gallocr_free);
            require(ggml_gallocr_reserve(aa.get(),a.graph)&&ggml_gallocr_reserve(bb.get(),b.graph),"reserve failed");
            const auto saved=strata_glm_gallocr_share_scratch(reverse?bb.get():aa.get(),reverse?aa.get():bb.get());
            require(saved>0,"scratch sharing not activated");
            require(strata_glm_gallocr_scratch_is_shared(aa.get(),bb.get()),"missing shared ownership");
            require(strata_glm_gallocr_share_scratch(aa.get(),bb.get())==0,"nested sharing accepted");
            for(int i=0;i<8;++i) {a.run(aa.get(),gpu.get(),i);b.run(bb.get(),gpu.get(),i+20);}
            require(ggml_backend_buffer_get_base(a.input->buffer)==ggml_backend_buffer_get_base(b.input->buffer),"buffers do not alias");
            if(grow) {
                // Either side may grow. The other graph must retain its original
                // allocation and remain executable, with no dangling ownership.
                auto *growing=reverse?bb.get():aa.get();
                require(ggml_gallocr_reserve(growing,larger.graph),"growth failed");
                larger.run(growing,gpu.get(),7);
                require(!strata_glm_gallocr_scratch_is_shared(aa.get(),bb.get()),"growth failed to detach");
                (reverse?a:b).run(reverse?aa.get():bb.get(),gpu.get(),13);
            }
            // Both destruction orders: borrower first and original owner first.
            if(reverse) {bb.reset();a.run(aa.get(),gpu.get(),29);}
            else {aa.reset();b.run(bb.get(),gpu.get(),31);}
            report["checks"].push_back({{"reverse",reverse},{"grow",grow},{"saved_bytes",saved},{"status","pass"}});
        }
        Graph cpu_a(64),cpu_b(64);
        auto *cpu_dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);require(cpu_dev,"CPU backend required");
        auto cpu_buft=ggml_backend_dev_buffer_type(cpu_dev);
        Alloc ca(ggml_gallocr_new(cpu_buft),ggml_gallocr_free),cb(ggml_gallocr_new(cpu_buft),ggml_gallocr_free);
        require(ggml_gallocr_reserve(ca.get(),cpu_a.graph)&&ggml_gallocr_reserve(cb.get(),cpu_b.graph),"CPU reserve failed");
        require(strata_glm_gallocr_share_scratch(ca.get(),cb.get())==0,"CPU buffer sharing allowed");
        report["checks"].push_back({{"name","CPU sharing rejected"},{"status","pass"}});
        report["status"]="pass";
    }catch(const std::exception &e) {report["error"]=e.what();}
    std::cout<<report.dump(2)<<'\n';return report["status"]=="pass"?0:1;
}
