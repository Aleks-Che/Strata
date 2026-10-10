#include "runtime.hpp"
#include "sort_table.hpp"
#include "synthetic_minimax_m2.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <thread>
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
    for(int pos=0;pos<25;) {
        const int count=pos<17?std::min(batch,17-pos):1;
        decode(ctx,tokens,pos,count,pos,true);
        for(int i=0;i<count;++i) {const auto *p=llama_get_logits_ith(ctx,i);out.insert(out.end(),p,p+64);}
        pos+=count;
    }
    return out;
}
int main(int argc,char **argv) {
    try {
        require(argc==2,"usage: sort-table-check NEW_DIRECTORY");const std::filesystem::path dir=argv[1];
        require(!std::filesystem::exists(dir),"output directory exists");std::filesystem::create_directories(dir);
        json tests=json::array(),runs=json::array();
        auto check=[&](const std::string &name,bool ok){tests.push_back({{"name",name},{"pass",ok}});};
        auto cuda=[](cudaError_t rc){require(rc==cudaSuccess,cudaGetErrorString(rc));};
        environment();ggml_backend_load_all();
        constexpr size_t count=48,bytes=SortTableRing::slot_bytes;
        void *device=nullptr;cuda(cudaMalloc(&device,count*bytes));
        cudaStream_t streams[2];for(auto &s:streams)cuda(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
        std::vector<unsigned char> input(bytes),output(count*bytes),expected(count*bytes);
        SortTableRing ring;
        check("empty_and_oversize_fallback",!ring.copy(device,input.data(),0,streams[0]) &&
            !ring.copy(device,input.data(),bytes+1,streams[0]) && !ring.pinned_bytes());
        check("null_fallback",!ring.copy(nullptr,input.data(),bytes,streams[0]) && !ring.copy(device,nullptr,bytes,streams[0]));
        // Deliberately keep the first stream busy while wrapping the host ring.
        // A real event wait must protect slot zero from the fifth upload.
        cuda(cudaLaunchHostFunc(streams[0],[](void *){std::this_thread::sleep_for(std::chrono::milliseconds(200));},nullptr));
        for(size_t i=0;i<count;++i) {
            std::fill(input.begin(),input.end(),static_cast<unsigned char>(i+1));
            std::memcpy(expected.data()+i*bytes,input.data(),bytes);
            require(ring.copy(static_cast<char *>(device)+i*bytes,input.data(),bytes,streams[i%2]),"ring copy refused");
            std::fill(input.begin(),input.end(),255); // caller storage dies/changes immediately
        }
        ring.drain();cuda(cudaMemcpy(output.data(),device,output.size(),cudaMemcpyDeviceToHost));
        check("two_stream_wrap_owned_bytes",output==expected);
        check("busy_slot_wait_exercised",ring.counters().reuse_waits>0);
        check("bounded_pinned_and_pending",ring.pinned_bytes()==16384 && !ring.pending() && ring.counters().peak_pending==4);
        check("copy_accounting",ring.counters().copies==count && ring.counters().bytes==count*bytes);
        ring.reset_counters();check("reset_drains_counts",!ring.pending() && !ring.counters().copies && ring.pinned_bytes()==16384);
        ring.close();check("close_releases",!ring.pinned_bytes() && !ring.pending());ring.close();
        try {
            SortTableRing abandoned;
            std::fill(input.begin(),input.end(),117);
            require(abandoned.copy(device,input.data(),bytes,streams[1]),"abandoned copy refused");
            std::fill(input.begin(),input.end(),33);
            throw std::runtime_error("injected caller failure after enqueue");
        }catch(const std::runtime_error &){}
        cuda(cudaMemcpy(output.data(),device,bytes,cudaMemcpyDeviceToHost));
        check("exception_destruction_preserves_bytes",std::all_of(output.begin(),output.begin()+bytes,[](unsigned char x){return x==117;}));
        std::fill(input.begin(),input.end(),71);require(ring.copy(device,input.data(),bytes,streams[0]),"reopen refused");
        ring.close();cuda(cudaMemcpy(output.data(),device,bytes,cudaMemcpyDeviceToHost));
        check("close_pending_and_reopen",std::all_of(output.begin(),output.begin()+bytes,[](unsigned char x){return x==71;}) && !ring.pinned_bytes());
        for(auto s:streams)cuda(cudaStreamDestroy(s));cuda(cudaFree(device));

        const auto path=(dir/"mixed.gguf").string();write_synthetic_minimax_m2(path,true);
        for(int events:{0,2})for(int batch:{1,8,16}) {
            strata_mm27_mode(2);strata_mm27_pipeline(2,4,true,true,events);
            strata_mm27_cache_configure(32ull<<20,true);strata_mm27_router_host_ids(true);
            auto model=load(path,false,true);auto ctx=context(model.get(),512,batch);
            for(int salt:{0,3,0}) {
                const auto name=std::to_string(events)+"/"+std::to_string(batch)+"/"+std::to_string(runs.size());
                strata_mm27_sort_table_async(false);strata_mm27_reset();const auto baseline=run(ctx.get(),batch,salt);
                check(name+"disabled_no_staging",!strata_mm27_snapshot().sort_table_copies && !strata_mm27_snapshot().sort_table_pinned_bytes);
                strata_mm27_sort_table_async(true);strata_mm27_reset();const auto candidate=run(ctx.get(),batch,salt);
                const auto s=strata_mm27_snapshot();
                check(name+"exact_all_logits",equal(baseline,candidate));
                check(name+"copies_exercised",s.sort_table_copies>0 && s.sort_table_bytes==25*9*8*2*sizeof(int32_t) && !s.sort_table_fallbacks);
                check(name+"gpu_only_drained",s.gpu_nodes>0 && !s.rejected_cpu_nodes && !s.rejected_full_copies && !s.pipeline_pending_copy && !s.sort_table_pending);
                check(name+"bounded_staging",s.sort_table_pinned_bytes==16384 && s.sort_table_peak_pending<=4);
                runs.push_back({{"name",name},{"floats",candidate.size()},{"copies",s.sort_table_copies},
                    {"bytes",s.sort_table_bytes},{"reuse_waits",s.sort_table_reuse_waits},{"drain_waits",s.sort_table_drain_waits}});
            }
            ctx.reset();model.reset();strata_mm27_release();
            check("release/"+std::to_string(events)+"/"+std::to_string(batch),!strata_mm27_snapshot().sort_table_pinned_bytes && !strata_mm27_snapshot().sort_table_copies);
        }
        size_t failures=0;for(const auto &t:tests)if(!t.at("pass").get<bool>())++failures;
        json report={{"tests",tests},{"runs",runs},{"cases",tests.size()},{"failures",failures},{"pass",!failures}};
        std::ofstream(dir/"report.json")<<report.dump(2)<<'\n';std::cout<<report.dump(2)<<'\n';return failures?1:0;
    }catch(const std::exception &e) {std::cerr<<e.what()<<'\n';strata_mm27_release();return 2;}
}
