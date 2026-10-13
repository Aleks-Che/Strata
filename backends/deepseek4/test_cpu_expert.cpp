#include "expert_transfer.h"
#include "cpu_lane.hpp"
#include "../common/expert_file.hpp"
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include "ggml-cpp.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
static void check(bool ok,const char *s){if(!ok)throw std::runtime_error(s);}
static void env(const char *k,const char *v){
#ifdef _WIN32
_putenv_s(k,v);
#else
setenv(k,v,1);
#endif
}
constexpr int E=8,D=256,H=512,K=2;
struct Weights {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor *w[3];
    std::vector<uint8_t> data[3];
    std::vector<float> dequant[3];
    bool registered=false;
    void register_files() {
#ifdef _WIN32
        std::filesystem::create_directories("cpu-shared-test");
        for(int j=0;j<3;++j) {
            auto path=std::filesystem::path("cpu-shared-test")/("weight-"+std::to_string(j)+".bin");
            {std::ofstream f(path,std::ios::binary);f.write((const char *)data[j].data(),data[j].size());}
            int fd=_wopen(path.c_str(),_O_RDONLY|_O_BINARY);check(fd>=0,"open shared weight source");
            strata_expert_file::add(w[j]->data,data[j].size(),fd);_close(fd);
        }
        registered=true;
#endif
    }
    ~Weights(){if(registered)for(auto *v:w)strata_expert_file::remove(v->data);}

    Weights(ggml_backend_t cpu,ggml_type type):ctx(ggml_init({3*ggml_tensor_overhead(),nullptr,true})) {
        for(int j=0;j<3;++j)w[j]=ggml_new_tensor_3d(ctx.get(),j==1?GGML_TYPE_Q8_0:type,j==2?H:D,j==2?D:H,E);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(),cpu));check(bool(buffer),"weights allocation");
        ggml_backend_buffer_set_usage(buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for(int j=0;j<3;++j) {
            std::vector<float> f(ggml_nelements(w[j])),importance(w[j]->ne[0],1);
            for(size_t i=0;i<f.size();++i) {
                uint32_t v=uint32_t(i)+uint32_t(j*991+1)*0x9e3779b9u;
                v^=v>>16;v*=0x7feb352du;v^=v>>15;v*=0x846ca68bu;v^=v>>16;
                f[i]=0.03f*(float(v>>8)/8388608.0f-1.0f);
            }
            data[j].resize(ggml_nbytes(w[j]));
            check(ggml_quantize_chunk(w[j]->type,f.data(),data[j].data(),0,w[j]->ne[1]*E,w[j]->ne[0],importance.data())==data[j].size(),"quantize");
            ggml_backend_tensor_set(w[j],data[j].data(),0,data[j].size());
            dequant[j].resize(f.size());auto *traits=ggml_get_type_traits(w[j]->type);
            traits->to_float(data[j].data(),dequant[j].data(),f.size());
        }
    }
};
struct Graph {
    ggml_context_ptr ctx;
    ggml_backend_sched_t sched=nullptr;
    ggml_tensor *x,*ids,*up,*gate,*act,*down,*weights,*out;
    ggml_cgraph *graph;
    int tokens;
    Graph(Weights &w,ggml_backend_t cpu,ggml_backend_t gpu,int n,bool parallel,float clamp):ctx(ggml_init({48*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true})),tokens(n) {
        x=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,D,1,n);ggml_set_input(x);
        ids=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,K,n);
        ids->nb[1]=E*4;ids->nb[2]=ids->nb[3]=E*n*4;ggml_set_input(ids);
        weights=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,1,K,n);ggml_set_input(weights);
        up=ggml_mul_mat_id(ctx.get(),w.w[0],x,ids);
        gate=ggml_mul_mat_id(ctx.get(),w.w[1],x,ids);
        act=clamp>0?ggml_swiglu_clamp(ctx.get(),gate,up,clamp):ggml_swiglu_split(ctx.get(),gate,up);
        down=ggml_mul_mat_id(ctx.get(),w.w[2],act,ids);ggml_set_output(down);
        auto *weighted=ggml_mul(ctx.get(),down,weights);
        auto *first=ggml_view_2d(ctx.get(),weighted,D,n,weighted->nb[2],0);
        auto *second=ggml_view_2d(ctx.get(),weighted,D,n,weighted->nb[2],weighted->nb[1]);
        out=ggml_add(ctx.get(),first,second);ggml_set_output(out);
        graph=ggml_new_graph(ctx.get());ggml_build_forward_expand(graph,out);
        ggml_backend_t backends[]={gpu,cpu};sched=ggml_backend_sched_new(backends,nullptr,2,GGML_DEFAULT_GRAPH_SIZE,parallel,true);
        for(auto *v:{x,ids,weights})ggml_backend_sched_set_tensor_backend(sched,v,gpu);
        for(int i=0;i<ggml_graph_n_nodes(graph);++i)ggml_backend_sched_set_tensor_backend(sched,ggml_graph_node(graph,i),gpu);
        check(ggml_backend_sched_alloc_graph(sched,graph),"graph allocation");
    }
    ~Graph(){ggml_backend_sched_free(sched);}
    std::vector<float> run(const std::vector<float> &input,const std::vector<int32_t> &route) {
        std::vector<float> weight(tokens*K);for(size_t i=0;i<weight.size();++i)weight[i]=i%K?0.37f:0.63f;
        ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
        ggml_backend_tensor_set(ids,route.data(),0,ggml_nbytes(ids));
        ggml_backend_tensor_set(weights,weight.data(),0,weight.size()*4);
        check(ggml_backend_sched_graph_compute(sched,graph)==GGML_STATUS_SUCCESS,"compute");
        std::vector<float> result(tokens*K*D),sum(tokens*D);
        ggml_backend_tensor_get(down,result.data(),0,result.size()*4);ggml_backend_tensor_get(out,sum.data(),0,sum.size()*4);
        for(int t=0;t<tokens;++t)for(int i=0;i<D;++i) {
            float ref=result[(t*K)*D+i]*0.63f+result[(t*K+1)*D+i]*0.37f;
            check(std::isfinite(sum[t*D+i]) && std::abs(sum[t*D+i]-ref)<1e-7f,"downstream or reduction stale");
        }
        return result;
    }
};
static double scalar_error(const Weights &w,const float *x,int id,float clamp,const float *actual) {
    std::vector<double> h(H);for(int r=0;r<H;++r) {
        double up=0,gate=0;for(int i=0;i<D;++i) {
            up+=double(w.dequant[0][(id*H+r)*D+i])*x[i];gate+=double(w.dequant[1][(id*H+r)*D+i])*x[i];
        }
        if(clamp>0){gate=std::min(gate,double(clamp));up=std::clamp(up,-double(clamp),double(clamp));}
        h[r]=gate/(1+std::exp(-gate))*up;
    }
    double error=0,norm=0;for(int r=0;r<D;++r) {
        double ref=0;for(int i=0;i<H;++i)ref+=double(w.dequant[2][(id*D+r)*H+i])*h[i];
        error+=(actual[r]-ref)*(actual[r]-ref);norm+=ref*ref;
    }
    return std::sqrt(error/std::max(norm,1e-20));
}
static void fetch_cache_test(ggml_backend_t cpu,ggml_backend_dev_t dev,StrataExpertStats stats) {
#ifdef _WIN32
    env("STRATA_DS4_CPU_SHARED_RAM","1");env("STRATA_DS4_RAM_CACHE_MIB","16");
    env("STRATA_DS4_RAM_RESIDENCY","1");env("STRATA_DS4_CPU_RAM_MIB","0");
    // Exercise the ordinary all-miss eligibility rule, including a RAM-ready
    // expert becoming GPU-resident. LRU ties still choose the second top-k row.
    env("STRATA_DS4_CPU_TEST_FIXED","0");env("STRATA_DS4_CPU_MAX_EXPERTS","1");
    auto control=(StrataExpertControl)ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),"strata_expert_control");
    check(bool(control),"missing cache controller");
    const bool demote=std::getenv("STRATA_DS4_RAM_DEMOTE_MIB") && std::atoi(std::getenv("STRATA_DS4_RAM_DEMOTE_MIB"));
    const bool keep_ram=std::getenv("STRATA_DS4_GPU_FETCH_KEEP_RAM") && std::atoi(std::getenv("STRATA_DS4_GPU_FETCH_KEEP_RAM"));
    for(bool managed:{false,true})for(auto type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_MXFP4,GGML_TYPE_Q8_0}) {
        Weights target(cpu,type),victim(cpu,type);target.register_files();victim.register_files();
        env("STRATA_EXPERT_CACHE_MIB","0");env("STRATA_DS4_CPU_THREADS","0");
        ggml_backend_ptr warmer(ggml_backend_dev_init(dev,nullptr)),reference(ggml_backend_dev_init(dev,nullptr));
        Graph warm(target,cpu,warmer.get(),3,false,0.1f),ref(target,cpu,reference.get(),3,false,0.1f);
        std::vector<float> x(3*D);for(size_t i=0;i<x.size();++i)x[i]=0.8f*std::cos(float(i)*0.031f);
        std::vector<int32_t> ids(3*E),old_ids(3*E);
        for(int t=0;t<3;++t){ids[t*E]=1;ids[t*E+1]=2;old_ids[t*E]=3;old_ids[t*E+1]=4;}
        for(int i=0;i<16;++i)warm.run(x,ids);
        env("STRATA_EXPERT_CACHE_MIB","8");
        ggml_backend_ptr hybrid(ggml_backend_dev_init(dev,nullptr));
        Graph actual(target,cpu,hybrid.get(),3,false,0.1f),old(victim,cpu,hybrid.get(),3,false,0.1f);
        if(managed){StrataVramPolicy policy;StrataVramStatus status;check(control(0,&policy,&status),"enable managed loans");}
        env("STRATA_DS4_CPU_THREADS","4");const auto old_result=old.run(x,old_ids);
        StrataExpertCounters before;stats(hybrid.get(),&before);
        check(before.misses>=6,"victim cache not populated");
        strata_ds4::CpuLanePool pool(4);
        for(int iteration=0;iteration<12;++iteration) {
            if(iteration==6) {
                StrataExpertCounters partial;stats(hybrid.get(),&partial);
                if(keep_ram && !demote)check(!partial.gpu_fetch_cache_admitted && partial.gpu_fetch_cache_ram_bypasses>0,
                    "ready RAM triple was split by a GPU admission");
                // Keep the CPU expert hot, but demand a GPU expert whose RAM
                // triple is absent: protection must not disable all admissions.
                for(int t=0;t<3;++t)ids[t*E]=5;
            }
            for(size_t i=0;i<x.size();++i)x[i]=0.8f*std::cos(float(i+iteration*23)*0.031f);
            env("STRATA_EXPERT_CACHE_MIB","0");env("STRATA_DS4_CPU_THREADS","0");const auto expected=ref.run(x,ids);
            env("STRATA_EXPERT_CACHE_MIB","8");env("STRATA_DS4_CPU_THREADS","4");const auto result=actual.run(x,ids);
            strata_ds4::CpuExpertWork work;work.columns=D;work.hidden=H;work.tokens=3;work.topk=K;work.clamp=0.1f;
            work.up_type=target.w[0]->type;work.gate_type=target.w[1]->type;work.down_type=target.w[2]->type;work.input=x.data();
            std::vector<float> native(3*D);work.output=native.data();
            for(int t=0;t<3;++t)work.experts.push_back({target.data[1].data()+2*target.w[1]->nb[2],target.data[0].data()+2*target.w[0]->nb[2],target.data[2].data()+2*target.w[2]->nb[2],t,t*K+1});
            work.compute(pool);
            for(int t=0;t<3;++t) {
                check(!std::memcmp(result.data()+t*K*D,expected.data()+t*K*D,D*4),"loan GPU row corrupted");
                check(!std::memcmp(result.data()+(t*K+1)*D,native.data()+t*D,D*4),"loan changed CPU row");
            }
        }
        StrataExpertCounters after;stats(hybrid.get(),&after);
        check(after.gpu_fetch_branches==12,"loan test fell back to host routing");
        if(const auto *v=std::getenv("STRATA_DS4_GPU_FETCH_EARLY_RETIRE"))if(std::atoi(v))
            check(after.gpu_fetch_early_retire==after.gpu_fetch_branches,"branch copies were not retired before the final consumer fence");
        if(const auto *v=std::getenv("STRATA_DS4_GPU_FETCH_RELEASE_RAM"))if(std::atoi(v)) {
            check(after.cpu_snapshot_released>0,"unused CPU snapshots were retained");
            check(after.gpu_fetch_ram_released_unused>0 && after.gpu_fetch_ram_released_used>0,
                "transfer snapshots were not released after route/copy acknowledgement");
        }
        if(demote)check(!after.gpu_fetch_cache_offered && !after.gpu_fetch_cache_admitted,"loans bypassed demotion policy");
        else {
            check(after.gpu_fetch_cache_admitted>0 && after.gpu_fetch_cache_bytes>0,"no GPU cache admissions");
            check(after.gpu_fetch_cache_restored>0,"unused loans never restored");
            check(after.gpu_fetch_cache_offered==after.gpu_fetch_cache_restored+after.gpu_fetch_cache_admitted,"loan ownership leaked");
            check(after.direct_hits>before.direct_hits,"admitted matrix not reused");
        }
        env("STRATA_DS4_CPU_THREADS","0");for(size_t i=0;i<x.size();++i)x[i]=0.8f*std::cos(float(i)*0.031f);
        check(old.run(x,old_ids)==old_result,"restored victim cache contains new weights");
        if(managed || type==GGML_TYPE_Q8_0) {
            StrataVramPolicy release;release.mode=1;StrataVramStatus status;
            check(control(0,&release,&status) && !status.matrices && !status.cache_bytes,"cache loans prevent explicit release");
            StrataVramPolicy restore;check(control(0,&restore,&status),"restore cache policy");
        }
    }
    std::puts("GPU fetch cache: arena/managed, three quantizations, 72 mixed replays, unused restoration, demotion guard and explicit release");
#else
    throw std::runtime_error("GPU fetch cache fixture requires native Windows file registration");
#endif
}
int main(){try {
    for(int threads:{1,4,12,16}) {
        strata_ds4::CpuLanePool pool(threads);std::array<std::atomic<int>,32> hits{};
        for(int generation=0;generation<128;++generation) {
            for(auto &v:hits)v=0;int count=generation%33;
            auto job=[&](int i){hits[i].fetch_add(1);};
            if(generation%2){pool.submit(count,job);pool.wait();}else pool.run(count,job);
            for(int i=0;i<32;++i)check(hits[i]==int(i<count),"worker pool lost or duplicated a job");
        }
    }
    env("STRATA_EXPERT_PIPELINE","1");env("STRATA_EXPERT_STAGE_MIB","1");env("STRATA_EXPERT_CACHE_MIB","0");
    env("STRATA_EXPERT_CACHE_POLICY","lru");env("GGML_OP_OFFLOAD_MIN_BATCH","1");
    env("STRATA_DS4_DIRECT_CACHE","1");env("STRATA_DS4_ORDERED_EXPERTS","1");
    env("STRATA_DS4_CPU_MATRIX_LEGACY","0");env("STRATA_DS4_CPU_POLICY","force");env("STRATA_DS4_CPU_RAM_MIB","16");
    ggml_backend_load_all();ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr));
    auto *dev=ggml_backend_dev_by_name("CUDA0");if(!dev || !cpu)return 77;
    auto stats=(StrataExpertStats)ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),"strata_expert_stats");
    auto phase=(StrataCpuExpertPhase)ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),"strata_cpu_expert_phase");
    check(bool(phase),"missing CPU phase control");
    if(std::getenv("STRATA_DS4_TEST_FETCH_CACHE")){fetch_cache_test(cpu.get(),dev,stats);return 0;}
    int cases=0;uint64_t operations=0;double cpu_error=0,gpu_error=0,fallback_error=0;
    strata_ds4::CpuLanePool ref_pool(4);
    for(const char *policy:{"force","auto","no_ram"})for(auto type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_MXFP4,GGML_TYPE_Q8_0}) {
        const bool automatic=std::strcmp(policy,"auto")==0,no_ram=std::strcmp(policy,"no_ram")==0;
        env("STRATA_DS4_CPU_POLICY",automatic?"auto":"force");env("STRATA_DS4_CPU_RAM_MIB",no_ram?"0":"16");
        Weights w(cpu.get(),type);
        for(int n:{1,3,8,9})for(int cap:{1,6})for(float clamp:{0.0f,0.1f})for(bool parallel:{false,true}) {
            env("STRATA_DS4_CPU_MAX_EXPERTS",std::to_string(cap).c_str());
            ggml_backend_ptr ga(ggml_backend_dev_init(dev,nullptr)),gb(ggml_backend_dev_init(dev,nullptr));
            Graph a(w,cpu.get(),ga.get(),n,parallel,clamp),b(w,cpu.get(),gb.get(),n,parallel,clamp);
            for(int iter=0;iter<10;++iter) {
                std::vector<float> x(n*D);for(size_t i=0;i<x.size();++i)x[i]=0.8f*std::cos(float(i+iter*23)*0.031f);
                std::vector<int32_t> ids(n*E);
                for(int t=0;t<n;++t)for(int j=0;j<K;++j) {
                    int id=(t*3+j+iter)%E;ids[t*E+j]=id;
                }
                env("STRATA_DS4_CPU_THREADS","0");auto expected=a.run(x,ids);
                env("STRATA_DS4_CPU_THREADS","4");phase(iter!=3 && iter!=6);auto actual=b.run(x,ids);phase(true);
                const bool eligible=n<=8 && !parallel && iter!=3 && iter!=6;
                strata_ds4::CpuExpertWork ref;ref.columns=D;ref.hidden=H;ref.tokens=n;ref.topk=K;ref.clamp=clamp;
                ref.up_type=w.w[0]->type;ref.gate_type=w.w[1]->type;ref.down_type=w.w[2]->type;ref.input=x.data();
                std::vector<float> native(n*K*D);ref.output=native.data();
                std::vector<int> mapping(n*K,-1);
                for(int t=0;t<n;++t)for(int j=0;j<K;++j) {
                    int id=ids[t*E+j];if(!eligible || no_ram || (!automatic && j<K-cap))continue;
                    mapping[t*K+j]=int(ref.experts.size());
                    ref.experts.push_back({w.data[1].data()+id*w.w[1]->nb[2],w.data[0].data()+id*w.w[0]->nb[2],
                        w.data[2].data()+id*w.w[2]->nb[2],t,t*K+j});
                }
                ref.compute(ref_pool);
                for(int slot=0;slot<n*K;++slot) {
                    if(automatic && !std::memcmp(expected.data()+slot*D,actual.data()+slot*D,D*4))mapping[slot]=-1;
                    const float *want=mapping[slot]<0?expected.data()+slot*D:native.data()+mapping[slot]*D;
                    if(std::memcmp(want,actual.data()+slot*D,D*4)) {
                        std::fprintf(stderr,"type=%s n=%d cap=%d clamp=%g parallel=%d iter=%d slot=%d cpu=%d diff=%g\n",
                            ggml_type_name(type),n,cap,clamp,int(parallel),iter,slot,mapping[slot],actual[slot*D]-want[0]);
                        check(false,"whole expert placement or replay mismatch");
                    }
                    if(iter==0) {
                        double err=scalar_error(w,x.data()+(slot/K)*D,ids[(slot/K)*E+slot%K],clamp,actual.data()+slot*D);
                        // Batch 9 uses the unchanged GPU MMQ path (including its
                        // activation precision). Its contract here is bitwise
                        // equality to GPU-only, already checked above.
                        if(n>8){fallback_error=std::max(fallback_error,err);continue;}
                        if(mapping[slot]>=0)cpu_error=std::max(cpu_error,err);else gpu_error=std::max(gpu_error,err);
                        if(err>=0.05)std::fprintf(stderr,"reference error type=%s n=%d cap=%d clamp=%g slot=%d cpu=%d rel=%g\n",
                            ggml_type_name(type),n,cap,clamp,slot,mapping[slot],err);
                        check(err<0.05,"whole expert differs excessively from independent double reference");
                    }
                }
                ++cases;
            }
            StrataExpertCounters s;stats(gb.get(),&s);operations+=s.cpu_lane_ops;
            check(!s.pipeline_unused && !s.pipeline_fallbacks,"CPU weights leaked into pipeline");
            if(n<=8 && !parallel && !no_ram)check(automatic?s.cpu_lane_ops<=8:s.cpu_lane_ops==8,"whole branch operation count");
            else check(s.cpu_lane_ops==0,"unsupported branch used CPU");
        }
    }
    // A future draft may reuse a preceding expert on the other device. The
    // first CPU row must retain its placement/value under append and replay.
    env("STRATA_DS4_CPU_POLICY","force");env("STRATA_DS4_CPU_RAM_MIB","16");
    env("STRATA_DS4_CPU_TEST_FIXED","1");env("STRATA_DS4_CPU_MAX_EXPERTS","1");
    int prefix_cases=0;
    for(auto type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_MXFP4,GGML_TYPE_Q8_0}) {
        Weights w(cpu.get(),type);
        for(float clamp:{0.0f,0.1f}) {
            ggml_backend_ptr single_gpu(ggml_backend_dev_init(dev,nullptr));
            Graph single(w,cpu.get(),single_gpu.get(),1,false,clamp);
            std::vector<float> x(8*D);for(size_t i=0;i<x.size();++i)x[i]=0.5f*std::sin(float(i)*0.033f);
            std::vector<int32_t> first(E);first[0]=1;first[1]=2;
            env("STRATA_DS4_CPU_THREADS","4");
            auto prefix=single.run(std::vector<float>(x.begin(),x.begin()+D),first);
            for(int n:{3,8}) {
                ggml_backend_ptr batch_gpu(ggml_backend_dev_init(dev,nullptr)),control_gpu(ggml_backend_dev_init(dev,nullptr));
                Graph batch(w,cpu.get(),batch_gpu.get(),n,false,clamp),control(w,cpu.get(),control_gpu.get(),n,false,clamp);
                for(int change=0;change<6;++change) {
                    std::vector<int32_t> route(n*E);route[0]=1;route[1]=2;
                    for(int t=1;t<n;++t){route[t*E]=(t+change)%E;route[t*E+1]=(t+change+1)%E;}
                    // Expert 2 is CPU for token 0 and GPU for token 1; expert
                    // 1 may be GPU then CPU. Transfers may not be skipped.
                    if(change%2==0){route[E]=2;route[E+1]=1;}
                    auto input=std::vector<float>(x.begin(),x.begin()+n*D);
                    env("STRATA_DS4_CPU_THREADS","0");auto gpu=control.run(input,route);
                    env("STRATA_DS4_CPU_THREADS","4");auto actual=batch.run(input,route);
                    check(!std::memcmp(prefix.data()+D,actual.data()+D,D*4),"future route changed first CPU row");
                    for(int t=0;t<n;++t)check(!std::memcmp(gpu.data()+t*K*D,actual.data()+t*K*D,D*4),"mixed expert GPU row missing or stale");
                    ++prefix_cases;
                }
            }
        }
    }
    env("STRATA_DS4_CPU_TEST_FIXED","0");
    std::printf("Causal per-token CPU placement: %d append/replay/mixed-device cases\n",prefix_cases);
#ifdef _WIN32
    env("STRATA_DS4_CPU_SHARED_RAM","1");env("STRATA_DS4_RAM_CACHE_MIB","4");
    env("STRATA_DS4_CPU_RAM_MIB","0"); // no private CPU weight allocations
    env("STRATA_DS4_CPU_MAX_EXPERTS","1");env("STRATA_DS4_CPU_POLICY","force");
    int shared_cases=0;
    for(auto type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_MXFP4,GGML_TYPE_Q8_0}) {
        Weights w(cpu.get(),type);w.register_files();
        ggml_backend_ptr hybrid(ggml_backend_dev_init(dev,nullptr)),gpu(ggml_backend_dev_init(dev,nullptr));
        Graph a(w,cpu.get(),hybrid.get(),3,false,0.1f),b(w,cpu.get(),gpu.get(),3,false,0.1f);
        std::vector<int32_t> ids(3*E);ids[0]=1;ids[1]=2;ids[E]=2;ids[E+1]=1;ids[2*E]=3;ids[2*E+1]=2;
        std::vector<float> x(3*D);for(size_t i=0;i<x.size();++i)x[i]=0.8f*std::cos(float(i)*0.031f);
        env("STRATA_DS4_CPU_THREADS","4");
        a.run(x,ids);StrataExpertCounters cold;stats(hybrid.get(),&cold);
        check(cold.cpu_lane_ops==0,"unpublished shared RAM used by CPU");
        phase(false);for(int i=0;i<24;++i)a.run(x,ids);phase(true);
        const char *fetch_flag=std::getenv("STRATA_DS4_GPU_FETCH_QUEUE");
        const bool fetch_enabled=fetch_flag && std::atoi(fetch_flag) && !std::getenv("STRATA_DS4_ROUTE_TRACE");
        // One GPU row has no RAM entry: an eligible CPU branch must also safely
        // fetch a different expert from its retained native file source.
        if(fetch_enabled)ids[0]=5;
        uint64_t warm_file_bytes=0;
        for(int change=0;change<10;++change) {
            for(size_t i=0;i<x.size();++i)x[i]=0.8f*std::cos(float(i+change*23)*0.031f);
            // The GPU-only reference must not warm the tested shared cache.
            env("STRATA_DS4_CPU_SHARED_RAM","0");env("STRATA_DS4_RAM_CACHE_MIB","0");
            env("STRATA_DS4_CPU_THREADS","0");auto expected=b.run(x,ids);
            env("STRATA_DS4_CPU_SHARED_RAM","1");env("STRATA_DS4_RAM_CACHE_MIB","4");
            env("STRATA_DS4_CPU_THREADS","4");auto actual=a.run(x,ids);
            strata_ds4::CpuExpertWork ref;ref.columns=D;ref.hidden=H;ref.tokens=3;ref.topk=K;ref.clamp=0.1f;
            ref.up_type=w.w[0]->type;ref.gate_type=w.w[1]->type;ref.down_type=w.w[2]->type;ref.input=x.data();
            std::vector<float> native(3*D);ref.output=native.data();
            for(int t=0;t<3;++t) {
                int id=ids[t*E+1];ref.experts.push_back({w.data[1].data()+id*w.w[1]->nb[2],w.data[0].data()+id*w.w[0]->nb[2],
                    w.data[2].data()+id*w.w[2]->nb[2],t,t*K+1});
            }
            ref.compute(ref_pool);
            for(int t=0;t<3;++t) {
                check(!std::memcmp(native.data()+t*D,actual.data()+(t*K+1)*D,D*4),"shared CPU matrix output stale");
                check(!std::memcmp(expected.data()+t*K*D,actual.data()+t*K*D,D*4),"shared mixed GPU row stale");
            }
            ++shared_cases;
            if(change==5) {StrataExpertCounters partial;stats(hybrid.get(),&partial);warm_file_bytes=partial.file_read_bytes;}
        }
        StrataExpertCounters warm;stats(hybrid.get(),&warm);
        check(warm.cpu_lane_ops==10 && warm.cpu_lane_experts==30,"shared ready matrices not used by CPU");
        if(const char *v=std::getenv("STRATA_DS4_CPU_GPU_QUEUE"))if(std::atoi(v))
            check(warm.cpu_queue_jobs==10 && warm.cpu_queue_rows==30,"queue test silently used host callback fallback");
        if(fetch_enabled) {
            check(warm.gpu_fetch_branches==10 && warm.gpu_fetch_matrices>0,"GPU fetch silently used host route fallback");
            check(warm.file_read_bytes>cold.file_read_bytes,"missing native cold-matrix fetch");
            if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_RAM_MIB"))if(std::atoi(v)) {
                check(warm.gpu_fetch_ram_reservations>=3 && warm.gpu_fetch_ram_completed>=3,"native fetch never admitted cold matrices");
                check(warm.gpu_fetch_ram_bytes>0,"native fetch never reused pinned RAM");
                check(warm.file_read_bytes==warm_file_bytes,"native fetch reread already admitted matrices");
            }
        } else check(!warm.gpu_fetch_branches,"unexpected GPU fetch bypassed the shadow route recorder");
        check(!warm.pipeline_unused && !warm.pipeline_fallbacks,"shared RAM transfer mismatch");
        for(int transition=0;transition<6;++transition) {
            env("STRATA_DS4_CPU_THREADS","0");const auto expected=b.run(x,ids);
            env("STRATA_DS4_CPU_THREADS","4");phase(false);const auto prefill=a.run(x,ids);phase(true);
            check(prefill==expected,"captured queue graph changed short GPU-only prefill");
            const auto hybrid_result=a.run(x,ids);
            for(int t=0;t<3;++t)check(!std::memcmp(hybrid_result.data()+t*K*D,expected.data()+t*K*D,D*4),
                "graph transport transition corrupted a GPU row");
        }
    }
    env("STRATA_DS4_CPU_SHARED_RAM","0");env("STRATA_DS4_RAM_CACHE_MIB","0");
    std::printf("Shared pinned CPU/pipeline matrices: %d mixed-placement replay cases\n",shared_cases);
#endif
    std::printf("CPU whole expert: %d cases, %llu operations, independent double rel-L2 CPU=%.9g GPU-MMVQ=%.9g unchanged-GPU-MMQ=%.9g\n",
        cases,(unsigned long long)operations,cpu_error,gpu_error,fallback_error);
    return 0;
}catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
