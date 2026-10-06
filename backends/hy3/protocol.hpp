#pragma once
#include <cmath>
#include <cstdint>
#include <map>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hy3 {
inline int integer(const std::string & text) {
    size_t end=0; int n=std::stoi(text,&end);
    if (end!=text.size()) throw std::runtime_error("invalid integer");
    return n;
}
struct Request {
    int count=0;
    std::vector<int32_t> tokens;
    std::map<std::string,double> sampling;
    double value(const char * key,double fallback) const {
        auto it=sampling.find(key); return it==sampling.end() ? fallback : it->second;
    }
};
inline Request request(const std::string & line,int context,int vocab) {
    std::istringstream in(line); std::string verb,word; Request r;
    if (!(in>>verb>>word) || verb!="GEN") throw std::runtime_error("expected GEN <count> [sampling] <ids>");
    r.count=integer(word);
    if (r.count<1 || r.count>context) throw std::runtime_error("invalid generation length");
    bool have_tokens=false,have_session=false;
    const std::set<std::string> allowed={"temperature","top_k","top_p","min_p","seed","penalty_last_n","penalty_repeat","penalty_freq","penalty_present"};
    while (in>>word) {
        const auto eq=word.find('=');
        if (eq!=std::string::npos && !have_tokens) {
            const auto key=word.substr(0,eq),value=word.substr(eq+1);
            if (key=="session") {
                if (have_session || value.size()!=64 || value.find_first_not_of("0123456789abcdef")!=std::string::npos)
                    throw std::runtime_error("invalid session hash");
                have_session=true; // Accepted for isolation; every request starts empty.
            } else {
                if (!allowed.count(key) || r.sampling.count(key)) throw std::runtime_error("unsupported/duplicate sampling key: "+key);
                size_t end=0; double n=std::stod(value,&end);
                if (end!=value.size() || !std::isfinite(n)) throw std::runtime_error("invalid sampling value");
                r.sampling[key]=n;
            }
        } else {
            if (have_tokens || word.empty() || word.back()==',') throw std::runtime_error("invalid/extra token list");
            have_tokens=true; std::istringstream list(word); std::string id;
            while (std::getline(list,id,',')) {
                const int n=integer(id);
                if (n<0 || n>=vocab) throw std::runtime_error("token outside vocabulary");
                r.tokens.push_back(n);
                if (r.tokens.size()>size_t(context)) throw std::runtime_error("prompt exceeds context");
            }
        }
    }
    if (r.tokens.empty() || r.tokens.size()+r.count>size_t(context)) throw std::runtime_error("prompt plus generation exceeds context");
    const double temp=r.value("temperature",0),top_p=r.value("top_p",1),min_p=r.value("min_p",0);
    if (temp<0 || temp>100 || top_p<=0 || top_p>1 || min_p<0 || min_p>1 || r.value("penalty_repeat",1)<=0)
        throw std::runtime_error("invalid sampling range");
    for (const auto & key : {"top_k","penalty_last_n","seed"}) {
        const double n=r.value(key,0),limit=std::string(key)=="seed" ? UINT32_MAX : std::string(key)=="top_k" ? vocab : 1048576;
        if (n<0 || n>limit || n!=std::floor(n)) throw std::runtime_error("invalid integer sampling value");
    }
    for (const auto & p : r.sampling) if (std::abs(p.second)>double(std::numeric_limits<float>::max()) && p.first!="seed")
        throw std::runtime_error("sampling value exceeds float range");
    return r;
}
} // namespace hy3
