#include "cpu_cost_refresh.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <filesystem>
#include <fstream>
#include <chrono>
using namespace strata_ds4;
static void check(bool ok,const char *s){if(!ok)throw std::runtime_error(s);}
static void env(const char *name,const std::string &value) {
#ifdef _WIN32
    _putenv_s(name,value.c_str());
#else
    setenv(name,value.c_str(),1);
#endif
}
static CpuCostModel base() {
    CpuCostModel m;m.enabled=true;m.contention=true;m.compute[0][1]=100;m.loaded_compute[0][1]=120;
    m.compute[3][9]=400;m.loaded_compute[3][9]=450;
    m.copy_us_per_byte=.05f;m.loaded_copy_us_per_byte=.07f;m.gpu_compute[0][1]=80;m.boundary[0][1]=30;
    return m;
}
static void samples(CpuCostRefresh &p,double us,int n=16,bool copying=false) {
    const auto g=p.snapshot().generation;for(int i=0;i<n;++i)p.observe(g,1,1,copying,us);
}
int main()try {
    {
        auto model=base();model.budget_us_per_token=110;
        CpuCostRefresh proof(model);
        check(proof.budget_possible(1),"initial proof rejected feasible CPU");
        for(int request=0;request<2;++request) {
            proof.begin(true);samples(proof,160);
            check(proof.budget_possible(1),"proof changed inside request");proof.finish(true);
        }
        check(!proof.budget_possible(1),"proof did not update after successor");
        for(int request=0;request<2;++request) {
            proof.begin(true);samples(proof,1);
            proof.finish(true,[](const auto &){throw std::runtime_error("injected write failure");});
        }
        check(!proof.budget_possible(1),"failed publication changed cached proof");
        for(int request=0;request<2;++request) {
            proof.begin(true);samples(proof,1);proof.finish(true);
        }
        check(proof.budget_possible(1),"cached proof prevented CPU re-entry");
        for(int t=1;t<=4;++t)check(proof.budget_possible(t)==cpu_cost_budget_possible(proof.snapshot(),t),"refresh prefix proof mismatch");
    }
    CpuCostRefresh p(base());
    p.begin(true);samples(p,150);
    check(p.snapshot().compute[0][1]==100,"table mutated inside request");
    check(!p.finish(true).changed,"one request triggered rebuild");
    p.begin(true);samples(p,150);
    bool saved=false;
    const auto r=p.finish(true,[&](const auto &m){check(m.generation==1 && m.compute[0][1]==125,"invalid successor");saved=true;});
    check(saved && r.changed==1 && p.snapshot().generation==1,"successor not published");
    check(p.snapshot().copy_us_per_byte==.05f && p.snapshot().gpu_compute[0][1]==80 && p.snapshot().boundary[0][1]==30,
        "CPU timing changed unmeasured GPU/PCIe/queue coefficients");
    p.begin(true);samples(p,300);p.finish(true);
    p.begin(true);samples(p,300);auto failed=p.finish(true,[](const auto &){throw std::runtime_error("disk full");});
    check(!failed.changed && !failed.error.empty() && p.snapshot().generation==1,"write failure published partial profile");
    p.begin(true);samples(p,300);check(!p.finish(true).changed,"failed write retained drift streak");
    p.begin(true);samples(p,300);check(p.finish(true).changed==1,"recovery after failed write");
    for(int i=0;i<30;++i){p.begin(true);samples(p,10000);p.finish(true);}
    check(p.snapshot().compute[0][1]==200,"baseline upper guard lost");
    for(int i=0;i<30;++i){p.begin(true);samples(p,1);p.finish(true);}
    check(p.snapshot().compute[0][1]==50,"baseline lower guard lost");
    CpuCostRefresh abort(base());
    abort.begin(true);samples(abort,160);abort.finish(true);
    abort.begin(true);samples(abort,160);check(!abort.finish(false).changed,"cancel updated profile");
    abort.begin(true);samples(abort,160);check(!abort.finish(true).changed,"cancel retained streak");
    abort.begin(false);samples(abort,160);abort.finish(true);
    abort.begin(true);samples(abort,160);check(!abort.finish(true).changed,"seeded request retained streak");
    abort.begin(true);abort.finish(true);
    abort.begin(true);samples(abort,160);check(!abort.finish(true).changed,"zero CPU request retained streak");
    abort.begin(true);samples(abort,160,7);check(!abort.finish(true).changed,"undersampled request updated profile");
    CpuCostRefresh invalid(base());
    for(int j=0;j<3;++j){invalid.begin(true);
        for(int i=0;i<16;++i) {
            invalid.observe(99,1,1,false,200);invalid.observe(0,5,1,false,200);invalid.observe(0,1,400,false,200);
            invalid.observe(0,1,1,false,std::numeric_limits<double>::infinity());invalid.observe(0,1,1,false,-1);
        }
        check(invalid.finish(true).samples==0,"invalid samples admitted");
    }
    CpuCostRefresh outlier(base());
    for(int j=0;j<3;++j){outlier.begin(true);samples(outlier,100,62);samples(outlier,100000,1);check(!outlier.finish(true).changed,"outlier triggered update");}
    CpuCostRefresh loaded(base());
    for(int j=0;j<2;++j){loaded.begin(true);samples(loaded,180,16,true);loaded.finish(true);}
    check(loaded.snapshot().compute[0][1]==100 && loaded.snapshot().loaded_compute[0][1]==150,"loaded/idle samples mixed");
    for(int j=0;j<20;++j){loaded.begin(true);samples(loaded,1,16,true);loaded.finish(true);}
    check(loaded.snapshot().compute[0][1]==100 && loaded.snapshot().loaded_compute[0][1]==100,"loaded samples lowered idle floor");
    CpuCostRefresh shared(base());
    for(int j=0;j<2;++j){shared.begin(true);
        std::thread a([&]{samples(shared,160,1024);}),b([&]{samples(shared,160,1024);});a.join();b.join();
        auto result=shared.finish(true);check(result.samples==2048,"shared target/draft observations lost");
    }
    check(shared.snapshot().compute[0][1]==125,"shared successor failed");
    env("STRATA_DS4_CPU_COST_AUTO","1");
    const auto output=std::filesystem::current_path()/("cost-refresh-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    env("STRATA_DS4_CPU_COST_AUTO_DIR",output.string());
    CpuExpertWork w;w.gate_type=w.up_type=w.down_type=GGML_TYPE_Q8_0;w.columns=256;w.hidden=128;w.topk=6;
    cpu_cost_request_boundary(0);
    auto registered=cpu_cost_refresh_for("test-source",w,12,"test-gpu",base());
    check(registered==cpu_cost_refresh_for("test-source",w,12,"test-gpu",base()),"target/draft identity did not share profile");
    check(registered!=cpu_cost_refresh_for("test-source",w,16,"test-gpu",base()),"different worker pools share profile");
    samples(*registered,160);cpu_cost_request_boundary(2);
    cpu_cost_request_boundary(0);samples(*registered,160);cpu_cost_request_boundary(2);
    check(registered->snapshot().generation==1,"registry did not publish");
    int files=0;for(const auto &f:std::filesystem::directory_iterator(output)) {
        check(f.path().extension()==".json","temporary snapshot was left behind");
        std::ifstream in(f.path());std::string text((std::istreambuf_iterator<char>(in)),{});
        check(text.find("strata-cpu-runtime-cost")!=std::string::npos && text.find("test-source")!=std::string::npos,"snapshot identity missing");++files;
    }
    check(files==1,"unexpected snapshot count");
    const auto blocked=output/"not-a-directory";std::ofstream(blocked)<<"block";
    env("STRATA_DS4_CPU_COST_AUTO_DIR",blocked.string());
    for(int i=0;i<2;++i){cpu_cost_request_boundary(0);samples(*registered,300);cpu_cost_request_boundary(2);}
    check(registered->snapshot().generation==1,"filesystem failure replaced live registry profile");
    cpu_cost_request_boundary(1);samples(*registered,300);cpu_cost_request_boundary(2);
    cpu_cost_request_boundary(0);samples(*registered,300);cpu_cost_request_boundary(3);
    check(registered->snapshot().generation==1,"excluded/aborted request changed registry");
    std::cout<<"CPU cost refresh: request freeze, drift, cancellation, exclusions, bounded samples, concurrent services, guards and atomic write rollback passed\n";
    return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
