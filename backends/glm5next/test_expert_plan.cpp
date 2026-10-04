#include "expert_plan.hpp"
#include <algorithm>
#include <cstdio>
#include <functional>

using namespace strata_glm;
static void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
static const char *quants[]={"IQ2_S","IQ3_S","IQ4_XS","Q2_K","Q3_K","IQ3_XXS","Q6_K","Q4_K"};
static constexpr uint64_t sizes[]={82,110,136,84,110,98,210,144};
struct Fixture {
    ExpertLayerLayout layer{3,45,46,3,{}};
    std::vector<ExpertShardLayout> shards{{"metadata.gguf",128,99},{"weights.gguf",(uint64_t(1)<<35)+96,0}};
    explicit Fixture(size_t group=0) {
        auto end=shards[1].data_start;
        for(size_t p=0;p<3;++p) {
            const auto bytes=512*sizes[(group+p)%8]*288;
            layer.tensors[p]={Projection(p),quants[(group+p)%8],p==2?512u:256u,p==2?256u:512u,288,1,end,bytes};
            end+=bytes+32;
        }
        shards[1].file_bytes=end-32; // Final expert ends exactly at EOF.
    }
    std::vector<ExpertKey> plan(const std::vector<int> &ids={287,0,287,1,2,3,4,5,6,0},size_t limit=4096) const {
        return plan_experts("model-full-identity",71,layer,shards,ids,limit);
    }
};
static void test_valid() {
    for(size_t group=0;group<8;++group)for(int layer:{3,45}) {
        Fixture f(group);f.layer.layer=layer;
        const auto original=f.layer.tensors;
        std::swap(f.layer.tensors[0],f.layer.tensors[2]); // Input order is not projection order.
        auto plan=f.plan();require(plan.size()==24,"duplicate routes lost a triple or added copies");
        const int experts[]={287,0,1,2,3,4,5,6};
        for(size_t i=0;i<plan.size();++i) {
            const size_t p=i%3;const auto &t=original[p];
            const uint64_t bytes=512*sizes[(group+p)%8];
            ExpertKey expected{"model-full-identity",71,layer==45?Branch::mtp:Branch::main,layer,experts[i/3],
                               Projection(p),quants[(group+p)%8],p==2?512u:256u,p==2?256u:512u,
                               "weights.gguf",t.offset+uint64_t(experts[i/3])*bytes,bytes};
            require(plan[i]==expected,"native plan differs from independent key/range geometry");
        }
        require(plan[2].offset+plan[2].bytes==f.shards[1].file_bytes,"last expert does not end at EOF");
        require(f.plan({}).empty(),"empty route plan is not empty");
    }
    Fixture f;
    const auto a=f.plan();
    auto changed=plan_experts("other-model",71,f.layer,f.shards,{287});
    require(!(a[0]==changed[0]),"model identity lost");
    changed=plan_experts("model-full-identity",72,f.layer,f.shards,{287});
    require(!(a[0]==changed[0]),"load generation lost");
    // A triple may span separate shards, including the same offset in each.
    for(size_t p=0;p<3;++p) {
        auto &t=f.layer.tensors[p];t.shard=f.shards.size();t.offset=64;
        f.shards.push_back({"split-"+std::to_string(p)+".gguf",64,64+t.bytes});
    }
    const auto split=f.plan({0});
    require(split.size()==3 && split[0].shard!=split[1].shard && split[1].shard!=split[2].shard &&
            split[0].offset==64 && split[2].offset==64,"split tensor shards mixed");
    std::puts("PASS: 384 native matrix keys, 8 quant types, main/MTP, dedup order, transposed down, split shards and >4 GiB/EOF ranges");
}
static void test_invalid() {
    for(const auto &mutate:std::vector<std::function<void(Fixture&)>>{
        [](auto &f){f.layer.layer=-1;},[](auto &f){f.layer.layer=46;},
        [](auto &f){f.layer.layer=0;},[](auto &f){f.layer.leading_dense_blocks=-1;},
        [](auto &f){f.layer.leading_dense_blocks=46;},
        [](auto &f){f.layer.main_blocks=0;},[](auto &f){f.layer.block_count=44;},
        [](auto &f){f.layer.tensors[0].projection=Projection::up;},
        [](auto &f){f.layer.tensors[0].projection=Projection(9);},
        [](auto &f){f.layer.tensors[0].quant="BF16";},
        [](auto &f){f.layer.tensors[0].columns=255;},
        [](auto &f){f.layer.tensors[0].rows=0;},
        [](auto &f){f.layer.tensors[0].experts=0;},
        [](auto &f){f.layer.tensors[0].experts=uint64_t(INT_MAX)+1;},
        [](auto &f){f.layer.tensors[0].columns=UINT64_MAX-255;f.layer.tensors[0].rows=UINT64_MAX;},
        [](auto &f){--f.layer.tensors[0].bytes;},
        [](auto &f){f.layer.tensors[0].shard=2;},
        [](auto &f){f.layer.tensors[0].shard=0;},
        [](auto &f){f.layer.tensors[0].offset=0;},
        [](auto &f){f.layer.tensors[0].offset=UINT64_MAX;},
        [](auto &f){--f.shards[1].file_bytes;},
        [](auto &f){f.shards[1].name.clear();},
        [](auto &f){f.shards[1].name=f.shards[0].name;},
        [](auto &f){f.layer.tensors[1].offset=f.layer.tensors[0].offset;},
        [](auto &f){std::swap(f.layer.tensors[2].columns,f.layer.tensors[2].rows);},
        [](auto &f){f.layer.tensors[1].experts=144;f.layer.tensors[1].bytes/=2;}}) {
        Fixture f;mutate(f);bool rejected=false;
        try {f.plan();}catch(const std::invalid_argument &) {rejected=true;}
        require(rejected,"malformed layout accepted");
    }
    Fixture f;
    for(const auto &ids:std::vector<std::vector<int>>{{0,-1},{0,288},{0,INT_MAX}}) {
        bool rejected=false;
        try {f.plan(ids);}catch(const std::invalid_argument &) {rejected=true;}
        require(rejected,"bad router ID accepted after valid prefix");
    }
    for(size_t limit:{size_t(0),size_t(1)}) {
        bool rejected=false;
        try {f.plan({0,0},limit);}catch(const std::invalid_argument &) {rejected=true;}
        require(rejected,"route input limit ignored");
    }
    bool rejected=false;
    try {plan_experts("",1,f.layer,f.shards,{0});}catch(const std::invalid_argument &) {rejected=true;}
    require(rejected,"empty identity accepted");
    require(f.plan({0}).size()==3,"failed plan polluted later call");
    std::puts("PASS: invalid routes, limits, duplicate projections/shards, shapes, overlap, quant, overflow and truncated payload rejected");
}
int main() {
    try {test_valid();test_invalid();return 0;}
    catch(const std::exception &e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
