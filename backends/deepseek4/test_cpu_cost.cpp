#include "cpu_cost_profile.hpp"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
using namespace strata_ds4;
using json=nlohmann::json;
static void check(bool ok,const char *s){if(!ok)throw std::runtime_error(s);}
static void env(const char *name,const char *value) {
#ifdef _WIN32
    _putenv_s(name,value);
#else
    setenv(name,value,1);
#endif
}
int main()try {
#ifdef _WIN32
    _putenv_s("STRATA_DS4_CPU_MMVQ_FAST","1");
#else
    setenv("STRATA_DS4_CPU_MMVQ_FAST","1",1);
#endif
    CpuCostModel m;m.enabled=true;m.copy_us_per_byte=1;m.budget_us_per_token=1000;
    int counts[4]={},ids[2]={0,1},occ[2]={1,1};uint64_t missing[2]={100,100};bool gpu[4]={};
    for(int t=0;t<4;++t){m.compute[t][1]=30;m.compute[t][2]=55;m.compute[t][9]=35;
        for(int r=1;r<=8;++r)m.boundary[t][r]=5;}
    auto choose=[&](int n,int cap,uint64_t bytes,int rows){return cpu_cost_choose(m,n,counts,gpu,4,ids,occ,missing,2,cap,bytes,rows);};
    check(choose(1,1,200,0).mask==1,"deterministic equal-cost tie");
    check(choose(1,2,200,0).mask==3,"parallel cost must be max, not a sum");
    m.compute[0][1]=250;m.compute[0][2]=350;check(!choose(1,2,200,0).mask,"zero-CPU alternative lost");
    m.compute[0][1]=30;m.compute[0][2]=55;m.budget_us_per_token=20;check(!choose(1,2,200,0).mask,"CPU budget ignored");m.budget_us_per_token=1000;
    counts[0]=1; // Prefer repeating expert A even though B appears first.
    ids[0]=1;ids[1]=0;missing[0]=missing[1]=50;check(choose(2,1,100,1).mask==2,"repeat reuse not priced");
    gpu[0]=true;auto mixed=choose(2,2,200,1);check(!(mixed.mask&2),"already-required H2D counted as saved");gpu[0]=false;
    m.compute[1][9]=0;check(!(choose(2,1,200,1).mask&2),"unmeasured grouping used");
    int huge[]={4,4,1};check(cpu_cost_key(huge,3)<0,"unbounded row plan");
    check(!choose(5,1,200,1).mask,"unsupported window");
    {
        CpuCostModel q;q.enabled=true;q.copy_us_per_byte=1;q.loaded_copy_us_per_byte=5;
        q.compute[0][1]=30;q.compute[0][2]=55;q.loaded_compute[0][1]=125;q.loaded_compute[0][2]=150;
        q.boundary[0][1]=q.boundary[0][2]=5;
        int zero[4]={},which[]={0,1},once[]={1,1};bool required[4]={};uint64_t bytes[]={100,100};
        auto plan=[&](int cap,int topk=8){return cpu_cost_choose(q,1,zero,required,4,which,once,bytes,2,cap,200,0,topk);};
        check(plan(1).mask==1,"idle control choice");q.contention=true;
        check(!plan(1).mask,"slower competing DMA did not reject CPU work");
        check(cpu_cost_transfer(q,50,0)==50 && cpu_cost_transfer(q,50,1)==250,"DMA contention applied without CPU work");
        q.contention=false;q.gpu_enabled=true;q.gpu_compute[0][0]=300;q.gpu_compute[0][1]=220;q.gpu_compute[0][2]=100;
        check(plan(2).mask==3,"GPU kernel savings not used");
        check(plan(2).total_us==105,"overlapped GPU compute and transfer double counted");
        check(plan(1).total_us==325,"sequential GPU cost not included");q.gpu_overlap=1;
        check(plan(1).total_us==225,"GPU overlap double counted");q.gpu_overlap=0;
        check(!plan(2,4).mask,"unmeasured GPU top-k accepted");
        q.gpu_compute[0][1]=0;check(!plan(1).mask,"unmeasured GPU mask used");
        q.gpu_enabled=false;q.contention=true;q.budget_us_per_token=100;
        check(!plan(1).mask,"loaded CPU budget ignored");
        check(cpu_cost_compute(q,1,once,2,false)==55,"idle path incorrectly used loaded compute");
    }
    // Invalid external measurements must fail closed, not fall back to old auto.
    if(cpu_cost_mode()=="unsupported")return 77;
    CpuExpertWork w{};w.gate_type=w.up_type=w.down_type=GGML_TYPE_Q8_0;w.columns=4096;w.hidden=2048;w.clamp=10;
    auto root=std::filesystem::temp_directory_path()/("strata-cost-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(root);std::string reason;
    json j={{"version",1},{"build",cpu_cost_build_id()},{"machine",cpu_cost_machine()},{"gpu","test-gpu"},
        {"mode",cpu_cost_mode()},{"threads",12},{"profiles",json::array()}};
    json p={{"gate",8},{"up",8},{"down",8},{"columns",4096},{"hidden",2048},{"clamp",10},
        {"resident_weight_bytes",100ULL<<20},{"copy_us_per_byte",0.001},{"cases",json::array()}, {"boundary",json::array()}};
    for(int t=1;t<=4;++t){p["cases"].push_back({{"tokens",t},{"counts",{1}},{"compute_us",123}});
        for(int r=1;r<=8;++r)p["boundary"].push_back({{"tokens",t},{"rows",r},{"us",7}});}
    j["profiles"].push_back(p);
    auto save=[&](const char *name,const json &value){const auto path=root/name;std::ofstream(path)<<value.dump();return path.string();};
    auto good=save("good.json",j);check(cpu_cost_load(good,w,12,"test-gpu",reason).enabled,"valid profile rejected");
    check(!cpu_cost_load(good,w,16,"test-gpu",reason).enabled,"wrong threads accepted");
    auto tuning=j;tuning["mode"]=cpu_cost_mode()+"/different-pool";
    check(!cpu_cost_load(save("pool-mismatch.json",tuning),w,12,"test-gpu",reason).enabled,"wrong pool tuning accepted");
    check(!cpu_cost_load(good,w,12,"another-gpu",reason).enabled,"wrong device accepted");
    w.clamp=0;check(!cpu_cost_load(good,w,12,"test-gpu",reason).enabled,"wrong clamp accepted");w.clamp=10;
    auto bad=j;bad["build"]="old";check(!cpu_cost_load(save("old.json",bad),w,12,"test-gpu",reason).enabled,"stale build accepted");
    bad=j;bad["profiles"][0]["cases"][0]["compute_us"]=-1;
    check(!cpu_cost_load(save("negative.json",bad),w,12,"test-gpu",reason).enabled,"negative cost accepted");
    auto duplicate=j;duplicate["profiles"][0]["boundary"].push_back(duplicate["profiles"][0]["boundary"][0]);
    check(!cpu_cost_load(save("duplicate.json",duplicate),w,12,"test-gpu",reason).enabled,"duplicate boundary accepted");
    auto v2=j;v2["version"]=2;auto &p2=v2["profiles"][0];p2["loaded_copy_us_per_byte"]=0.002;
    for(auto &row:p2["cases"])row["loaded_compute_us"]=234;
    p2["gpu"]={{"topk",8},{"cases",json::array()}, {"overlap",{{"tokens",3},{"cpu_rows",3},{"serial_us",300},{"parallel_us",250}}}};
    for(int t=1;t<=4;++t)for(int r=0;r<=4;++r)for(const char *layout:{"repeated","distinct"})for(const char *mask:{"packed","spread"})
        p2["gpu"]["cases"].push_back({{"tokens",t},{"cpu_rows",r},{"layout",layout},{"mask",mask},{"us",100-r}});
    auto extended=cpu_cost_load(save("v2.json",v2),w,12,"test-gpu",reason);
    check(extended.enabled && extended.contention && extended.gpu_enabled && extended.gpu_compute[2][1]==99,"version 2 profile rejected");
    auto v3=v2;v3["version"]=3;v3["profiles"][0]["burst_gap_requested_us"]=20000;
    for(auto &row:v3["profiles"][0]["cases"]) {
        row["burst_compute_us"]=180;row["burst_loaded_compute_us"]=300;
        row["burst_samples_us"]=row["burst_loaded_samples_us"]=std::vector<float>(8,200);
        row["burst_gaps_us"]=row["burst_loaded_gaps_us"]=std::vector<float>(8,20000);
    }
    auto burst_path=save("v3.json",v3);env("STRATA_DS4_CPU_COST_BURST","0");
    auto burst_model=cpu_cost_load(burst_path,w,12,"test-gpu",reason);
    check(burst_model.enabled && !burst_model.burst && burst_model.compute[0][1]==123,"continuous ablation changed");
    env("STRATA_DS4_CPU_COST_BURST","1");burst_model=cpu_cost_load(burst_path,w,12,"test-gpu",reason);
    check(burst_model.enabled && burst_model.burst && burst_model.compute[0][1]==180 && burst_model.loaded_compute[0][1]==300,"burst cost not applied");
    int one[]={1};check(cpu_cost_compute(burst_model,1,one,1,true)==300,"burst DMA contention lost");
    auto missing_burst=v3;missing_burst["profiles"][0]["cases"][0]["burst_gaps_us"]=json::array();
    check(!cpu_cost_load(save("burst-incomplete.json",missing_burst),w,12,"test-gpu",reason).enabled,"incomplete burst profile accepted");
    env("STRATA_DS4_CPU_COST_BURST","0");
    auto invalid=v2;invalid["profiles"][0]["gpu"]["cases"].push_back(invalid["profiles"][0]["gpu"]["cases"][0]);
    check(!cpu_cost_load(save("gpu-duplicate.json",invalid),w,12,"test-gpu",reason).enabled,"duplicate GPU measurement accepted");
    invalid=v2;invalid["profiles"][0]["cases"][0]["loaded_compute_us"]=-1;
    check(!cpu_cost_load(save("loaded-negative.json",invalid),w,12,"test-gpu",reason).enabled,"negative loaded compute accepted");
    // An edit to the file does not adapt the active experiment mid-request.
    std::ofstream(good)<<bad.dump();check(cpu_cost_load(good,w,12,"test-gpu",reason).enabled,"loaded profile was not frozen");
    for(const char *name:{"good.json","old.json","negative.json","duplicate.json","v2.json","gpu-duplicate.json","loaded-negative.json","pool-mismatch.json","v3.json","burst-incomplete.json"})std::filesystem::remove(root/name);
    std::filesystem::remove(root);
    std::cout<<"CPU cost: budget, reuse, unique H2D, zero work, exact shapes, fingerprints, frozen profile passed\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
