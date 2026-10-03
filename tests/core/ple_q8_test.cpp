// Q8_0 PLE: mmap/direct, page boundaries, tail, dedup, cache and batched gathers against ggml.
#include "gguf_fixture.hpp"
#include "strata/kernels/ngram.hpp"
#include "ggml.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

namespace k = strata::kernels;
namespace fs = std::filesystem;

static bool verify(const fs::path& path) {
    std::vector<uint32_t> rows;
    std::vector<float> want;
    uint64_t count = 0;
    {   // Release the reference mapping before testing unbuffered reads.
        strata::GgufFile file(path.string());
        const auto* t = file.find("per_layer_token_embd.weight");
        if (!t || t->type != GGML_TYPE_Q8_0 || t->shape.size() != 2 || t->shape[0] != k::PLE_HEAD_DIM)
            return false;
        count = t->shape[1];
        if (count < 256) return false;
        rows = {0, 1, 1, 2, (uint32_t) (count - 1), (uint32_t) count, UINT32_MAX};
        for (uint32_t r = 0; r < 256; ++r) {
            const auto at = file.data_start() + t->offset + uint64_t(r) * k::PLE_ROW_BYTES_Q8_0;
            if (at / 4096 != (at + k::PLE_ROW_BYTES_Q8_0 - 1) / 4096) rows.push_back(r);
        }
        std::mt19937 rng(8);
        while (rows.size() < 16 * 16) rows.push_back((uint32_t) (rng() % count));
        want.resize(rows.size() * k::PLE_HEAD_DIM, 0.0f);
        for (size_t i = 0; i < rows.size(); ++i)
            if (rows[i] < count)
                ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(
                    file.tensor_data(*t) + uint64_t(rows[i]) * k::PLE_ROW_BYTES_Q8_0,
                    want.data() + i * k::PLE_HEAD_DIM, k::PLE_HEAD_DIM);
    }
    k::PleTable table;
    for (auto mode : {k::PleIo::Mmap, k::PleIo::Direct})
    for (bool worker : {false, true})
    for (uint64_t cache : {0ull, 128ull}) {
        k::PleIoOptions opt;
        opt.mode = mode;
        opt.io_thread = worker;
        opt.cache_rows = cache;
        opt.max_inflight = 2; // batch exceeds the window; rows can span two pages
        std::string err;
        if (!table.open(path.string(), err, opt)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return false; }
        if (std::strcmp(table.format(), "Q8_0") || table.rows() != count) return false;
        std::vector<float> got(want.size(), -123.0f);
        for (size_t i = 0; i < rows.size(); ++i) table.read_row(rows[i], got.data() + i * k::PLE_HEAD_DIM);
        if (std::memcmp(got.data(), want.data(), want.size() * sizeof(float))) {
            std::fprintf(stderr, "read_row mismatch (direct=%d worker=%d cache=%llu)\n", mode == k::PleIo::Direct,
                         worker, (unsigned long long) cache); return false;
        }
        for (int repeat = 0; repeat < 2; ++repeat) {
            if (!table.issue(rows.data()) || !table.collect(got.data(), err)) return false;
            if (std::memcmp(got.data(), want.data(), k::NG_N_EMBD * sizeof(float))) return false;
            table.gather(rows.data(), got.data());
            if (std::memcmp(got.data(), want.data(), k::NG_N_EMBD * sizeof(float))) return false;
            if (!table.gather_batch(rows.data(), rows.size() / k::PLE_N_HEADS, got.data(), err)) {
                std::fprintf(stderr, "gather_batch: %s\n", err.c_str()); return false;
            }
            if (std::memcmp(got.data(), want.data(), want.size() * sizeof(float))) return false;
        }
        table.close();
        if (table.is_open()) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc == 2) {
        const bool ok = verify(argv[1]);
        std::printf("Q8_0 PLE real rows vs ggml: %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    if (argc != 1) return 2;
    const auto path = fs::temp_directory_path() / ("strata-ple-q8-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    constexpr uint64_t n = 513; // unaligned file tail
    const auto layout = fixture::write(path, {}, {{"per_layer_token_embd.weight", {160, n}, 8}});
    {
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        file.seekp((std::streamoff) layout.data_start);
        for (uint64_t r = 0; r < n; ++r)
        for (int b = 0; b < 5; ++b) {
            const uint16_t scales[] = {0x3c00, 0x3800, 0xc000, 0x0000, 0x0400};
            file.write((const char*) &scales[(r + b) % 5], 2);
            for (int j = 0; j < 32; ++j) {
                const int8_t q = (int8_t) ((r * 17 + b * 31 + j * 7) % 256 - 128);
                file.write((const char*) &q, 1);
            }
        }
        if (!file) return 2;
    }
    bool ok = verify(path);
    fs::resize_file(path, fs::file_size(path) - 1);
    k::PleTable truncated;
    std::string err;
    ok = !truncated.open(path.string(), err) && ok;
    fs::remove(path);
    std::printf("Q8_0 PLE synthetic rows vs ggml, truncated file refusal: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
