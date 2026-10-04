#include "expert_warm_profile.hpp"
#include <iostream>
using namespace strata_glm;
using json=nlohmann::json;
static void check(bool value,const char *why) {if(!value)throw std::runtime_error(why);}
int main(int argc,char **argv) {
    try {
        check(argc==2,"test profile path required");auto path=std::filesystem::u8path(argv[1]);
        std::filesystem::remove(path);
        const json identity={{"model","model.gguf"},{"file_bytes",1234},{"file_time",5678},
            {"tensors",{{"gate","iq2_s",64,32,4,1024},{"down","iq3_s",32,64,8,2048}}}};
        const std::vector<size_t> counts{4,8};
        auto absent=read_warm_profile(path,identity,counts);
        check(absent.status=="missing" && absent.writable,"missing profile fallback failed");
        write_warm_profile(path,identity,{{0,3,2},{1,7,19},{0,0,19}});
        auto loaded=read_warm_profile(path,identity,counts);
        check(loaded.status=="loaded" && loaded.writable && loaded.entries.size()==3,"profile roundtrip failed");
        check(loaded.entries[0].tensor==0 && loaded.entries[0].expert==0 && loaded.entries[1].expert==7 && loaded.entries[2].score==2,"profile order differs");
        // Replace an existing profile, as after a second complete request.
        write_warm_profile(path,identity,{{1,2,5}});
        check(read_warm_profile(path,identity,counts).entries[0].expert==2,"atomic replacement failed");
        for(const auto *field:{"model","file_bytes","file_time","tensors"}) {
            auto other=identity;other[field]=nullptr;
            auto mismatch=read_warm_profile(path,other,counts);
            check(mismatch.status=="mismatch" && !mismatch.writable && mismatch.entries.empty(),"wrong model/layout profile accepted");
        }
        auto write=[&](const json &rows) {std::ofstream out(path);out<<json({{"version",1},{"identity",identity},{"entries",rows}});};
        for(const auto &rows:std::vector<json>{
            json::array({{-1,0,1}}),json::array({{2,0,1}}),json::array({{0,-1,1}}),json::array({{0,4,1}}),
            json::array({{1,8,1}}),json::array({{0,0,0}}),json::array({{0,0,256}}),
            json::array({{0,0,1},{0,0,2}}),json::array({{0,0,1.5}}),json::array({{0,0}}),json::object()}) {
            write(rows);auto invalid=read_warm_profile(path,identity,counts);
            check(invalid.status=="invalid" && !invalid.writable && invalid.entries.empty(),"invalid/partial metadata accepted");
        }
        {std::ofstream out(path);out<<"{\"version\":1,";}
        check(read_warm_profile(path,identity,counts).status=="invalid","truncated profile accepted");
        {std::ofstream out(path,std::ios::binary);out.seekp(8*1024*1024);out.put('x');}
        check(read_warm_profile(path,identity,counts).status=="invalid","oversized profile accepted");
        std::filesystem::remove(path);
        std::cout<<"PASS warm metadata: roundtrip, atomic replace, model/layout binding, bounds/duplicates, truncation and size limit\n";
        return 0;
    }catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
}
