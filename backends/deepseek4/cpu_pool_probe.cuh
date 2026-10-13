#pragma once

// Offline paired sweep. Delays are outside the expert interval and the same
// inputs, banks and row shapes are used for every scheduling option.
static json pool_probe(CpuExpertWork w,std::array<Bank,8> &banks,const std::vector<float> &input,int threads) {
    const std::vector<CpuPoolOptions> options={{0,2},{0,1},{0,4},{50,2},{200,2},{200,4}};
    json cases=json::array();std::vector<std::vector<float>> expected(3);
    std::vector<float> output(3*w.columns);w.input=input.data();w.output=output.data();
    auto setup=[&](int shape,int bank) {
        w.tokens=shape?3:1;w.experts.clear();
        for(int t=0;t<w.tokens;++t) {
            auto &b=banks[(bank+(shape==2?t:0))%8];
            w.experts.push_back({b.weight(0),b.weight(1),b.weight(2),t,t*w.topk});
        }
    };
    // Exact baseline for all banks, including distinct and repeated experts.
    {CpuLanePool reference(threads,{0,2});for(int shape=0;shape<3;++shape)for(int bank=0;bank<8;++bank) {
        setup(shape,bank);w.compute(reference);auto &v=expected[shape];
        v.insert(v.end(),output.begin(),output.begin()+w.tokens*w.columns);
    }}
    for(int gap:{0,20000,80000})for(int shape=0;shape<3;++shape) {
        std::vector<json> rows(options.size());
        for(size_t i=0;i<options.size();++i)rows[i]={{"spin_us",options[i].spin_us},{"partitions",options[i].partitions},
            {"gap_requested_us",gap},{"tokens",shape?3:1},{"counts",shape==0?json{1}:shape==1?json{3}:json{1,1,1}},
            {"samples",json::array()}};
        // Forward/reverse rounds reduce monotonic warmup and clock bias.
        for(int round=0;round<4;++round)for(size_t step=0;step<options.size();++step) {
            const size_t index=round%2?options.size()-1-step:step;
            CpuLanePool pool(threads,options[index]);
            setup(shape,0);w.compute(pool); // creation/first dispatch excluded
            for(int b=0;b<8;++b) {
                const int bank=(b+1)%8;setup(shape,bank);
                const auto pause=Clock::now();if(gap)std::this_thread::sleep_for(std::chrono::microseconds(gap));
                const double actual_gap=us(pause);CpuExpertTiming timing;const auto start=Clock::now();
                w.compute(pool,&timing);const auto duration=us(start);
                if(std::memcmp(output.data(),expected[shape].data()+bank*w.tokens*w.columns,w.tokens*w.columns*4))
                    throw std::runtime_error("pool tuning changed expert output");
                if(round)rows[index]["samples"].push_back({{"us",duration},{"gap_us",actual_gap},
                    {"prepare_us",timing.prepare_us},{"gate_up_us",timing.gate_up_us},{"hidden_quant_us",timing.hidden_quant_us},{"down_us",timing.down_us},
                    {"gu_first_us",timing.gate_up_pool.first_worker_us},{"gu_last_us",timing.gate_up_pool.last_worker_us},
                    {"down_first_us",timing.down_pool.first_worker_us},{"down_last_us",timing.down_pool.last_worker_us}});
            }
        }
        for(auto &row:rows){std::vector<double> samples;for(const auto &s:row["samples"])samples.push_back(s["us"]);
            row["median_us"]=median(samples);cases.push_back(row);}
    }
    return {{"gate",int(w.gate_type)},{"up",int(w.up_type)},{"down",int(w.down_type)},
        {"columns",w.columns},{"hidden",w.hidden},{"clamp",w.clamp},{"cases",cases},{"exact",true}};
}
