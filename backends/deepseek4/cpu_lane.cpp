#include "cpu_lane.hpp"
#include "ggml-cpu.h"
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>

namespace strata_ds4 {
struct CpuLanePool::Impl {
    struct Batch {
        std::function<void(int)> job;
        std::atomic<int> next{0}, remaining;
        int count;
        std::mutex mu;
        std::condition_variable done;
        std::exception_ptr error;
        Batch(int n,std::function<void(int)> f):job(std::move(f)),remaining(n),count(n) {}
        void work() {
            for(int i;(i=next.fetch_add(1,std::memory_order_relaxed))<count;) {
                try {job(i);} catch(...) {std::lock_guard<std::mutex> lock(mu);if(!error)error=std::current_exception();}
                if(remaining.fetch_sub(1,std::memory_order_acq_rel)==1) {
                    std::lock_guard<std::mutex> lock(mu);done.notify_all();
                }
            }
        }
        void wait() {
            std::unique_lock<std::mutex> lock(mu);
            done.wait(lock,[&]{return remaining.load(std::memory_order_acquire)==0;});
            if(error)std::rethrow_exception(error);
        }
    };
    std::mutex mu;
    std::condition_variable start;
    std::vector<std::thread> workers;
    std::shared_ptr<Batch> batch;
    int worker_limit=0;
    uint64_t generation=0;
    bool stop=false;
    explicit Impl(int n) {
        if(n<1 || n>16)throw std::invalid_argument("CPU lane pool needs 1..16 workers");
        ggml_cpu_init();
        try {
            for(int i=0;i<n;++i) workers.emplace_back([this,i] {
                uint64_t seen=0;
                std::unique_lock<std::mutex> lock(mu);
                for(;;) {
                    start.wait(lock,[&]{return stop || (generation!=seen && i<worker_limit);});
                    if(stop)return;
                    seen=generation;auto current=batch;lock.unlock();
                    current->work();lock.lock();
                }
            });
        } catch(...) {shutdown();throw;}
    }
    void shutdown() {
        {std::lock_guard<std::mutex> lock(mu);stop=true;start.notify_all();}
        for(auto &t:workers)if(t.joinable())t.join();
    }
    ~Impl(){shutdown();}
    std::shared_ptr<Batch> start_batch(int n,std::function<void(int)> job,bool caller) {
        if(batch)batch->wait();
        auto current=std::make_shared<Batch>(n,std::move(job));
        {std::lock_guard<std::mutex> lock(mu);batch=current;
         worker_limit=std::min(n,int(workers.size())-(caller?1:0));++generation;}
        start.notify_all();return current;
    }
};
CpuLanePool::CpuLanePool(int n):p(std::make_unique<Impl>(n)){}
CpuLanePool::~CpuLanePool(){try{wait();}catch(...) {}}
void CpuLanePool::wait() {
    if(p->batch)p->batch->wait();
}
void CpuLanePool::submit(int n,std::function<void(int)> job) {
    p->start_batch(n,std::move(job),false);
}
void CpuLanePool::run(int n,const std::function<void(int)> &job) {
    auto batch=p->start_batch(n,job,true);batch->work();batch->wait();
}
int CpuLanePool::size()const{return int(p->workers.size());}
bool cpu_lane_format(ggml_type type,int columns) {
    if(type!=GGML_TYPE_IQ3_XXS && type!=GGML_TYPE_MXFP4 && type!=GGML_TYPE_Q8_0)return false;
    const auto *t=ggml_get_type_traits_cpu(type);
    return t->vec_dot && t->nrows==1 && columns>0 && columns%ggml_blck_size(t->vec_dot_type)==0 &&
        ggml_get_type_traits_cpu(t->vec_dot_type)->from_float;
}
size_t cpu_lane_activation_bytes(ggml_type type,int columns) {
    return ggml_row_size(ggml_get_type_traits_cpu(type)->vec_dot_type,columns);
}
void cpu_lane_quantize(ggml_type type,int columns,const float *x,void *q) {
    const auto qt=ggml_get_type_traits_cpu(type)->vec_dot_type;
    ggml_get_type_traits_cpu(qt)->from_float(x,q,columns);
}
void cpu_lane_rows(ggml_type type,int columns,const void *weight,const void *q,float *out,int first,int last) {
    const auto *t=ggml_get_type_traits_cpu(type);
    const size_t stride=ggml_row_size(type,columns);
    for(int row=first;row<last;++row)t->vec_dot(columns,out+row,0,(const char *)weight+row*stride,0,q,0,1);
}
void CpuExpertWork::compute(CpuLanePool &pool) {
    if(experts.empty())return;
    const size_t gbytes=cpu_lane_activation_bytes(gate_type,columns);
    const size_t ubytes=cpu_lane_activation_bytes(up_type,columns);
    const size_t hbytes=cpu_lane_activation_bytes(down_type,hidden);
    const bool shared_q=ggml_get_type_traits_cpu(gate_type)->vec_dot_type==ggml_get_type_traits_cpu(up_type)->vec_dot_type;
    gate_q.resize(tokens*gbytes);if(!shared_q)up_q.resize(tokens*ubytes);
    hidden_q.resize(experts.size()*hbytes);hidden_f.resize(experts.size()*hidden);
    for(int t=0;t<tokens;++t) {
        cpu_lane_quantize(gate_type,columns,input+size_t(t)*columns,gate_q.data()+t*gbytes);
        if(!shared_q)cpu_lane_quantize(up_type,columns,input+size_t(t)*columns,up_q.data()+t*ubytes);
    }
    const auto *gt=ggml_get_type_traits_cpu(gate_type),*ut=ggml_get_type_traits_cpu(up_type);
    const auto *dt=ggml_get_type_traits_cpu(down_type);
    const size_t gs=ggml_row_size(gate_type,columns),us=ggml_row_size(up_type,columns),ds=ggml_row_size(down_type,hidden);
    // Group tokens of one expert, keeping each weight row hot across the window.
    std::vector<std::vector<size_t>> groups;
    for(size_t i=0;i<experts.size();++i) {
        auto it=std::find_if(groups.begin(),groups.end(),[&](const auto &g){return experts[g[0]].gate==experts[i].gate;});
        if(it==groups.end())groups.push_back({i});else it->push_back(i);
    }
    const int partitions=std::max(1,2*pool.size()/int(groups.size()));
    pool.run(int(groups.size())*partitions,[&](int job) {
        const auto &group=groups[job/partitions];const int part=job%partitions;
        for(int r=hidden*part/partitions;r<hidden*(part+1)/partitions;++r)for(size_t i:group) {
            const auto &e=experts[i];float g,u;
            gt->vec_dot(columns,&g,0,(const char *)e.gate+r*gs,0,gate_q.data()+e.token*gbytes,0,1);
            ut->vec_dot(columns,&u,0,(const char *)e.up+r*us,0,(shared_q?gate_q.data():up_q.data())+e.token*ubytes,0,1);
            if(clamp>0){g=std::min(g,clamp);u=std::max(-clamp,std::min(u,clamp));}
            hidden_f[i*hidden+r]=(g/(1.0f+std::exp(-g)))*u;
        }
    });
    for(size_t i=0;i<experts.size();++i)cpu_lane_quantize(down_type,hidden,hidden_f.data()+i*hidden,hidden_q.data()+i*hbytes);
    pool.run(int(groups.size())*partitions,[&](int job) {
        const auto &group=groups[job/partitions];const int part=job%partitions;
        for(int r=columns*part/partitions;r<columns*(part+1)/partitions;++r)for(size_t i:group)
            dt->vec_dot(hidden,output+i*columns+r,0,(const char *)experts[i].down+r*ds,0,hidden_q.data()+i*hbytes,0,1);
    });
}
}
