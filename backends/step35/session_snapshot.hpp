#pragma once

// Step-local, in-process checkpoint for a single append-only text sequence.
// Native sequence serialization compacts SWA cells; this keeps their physical
// indices (including currently masked history) for bit-exact continuation.
// This is KV only: callers must checkpoint tokens, sampler and output state.
#include "llama-context.h"
#include "llama-model.h"
#include "llama-kv-cache-iswa.h"
#include "llama-io.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace step35 {
class SessionSnapshot {
    struct Writer : llama_io_write_i {
        std::vector<uint8_t> bytes;
        void write(const void * src, size_t count) override {
            const auto * p = static_cast<const uint8_t *>(src);
            bytes.insert(bytes.end(), p, p + count);
        }
        void write_tensor(ggml_tensor * t, size_t offset, size_t count) override {
            const size_t start = bytes.size(); bytes.resize(start + count);
            ggml_backend_tensor_get(t, bytes.data() + start, offset, count);
        }
        size_t n_bytes() override { return bytes.size(); }
    };
    struct Reader : llama_io_read_i {
        const std::vector<uint8_t> & bytes;
        size_t offset = 0;
        explicit Reader(const std::vector<uint8_t> & b) : bytes(b) {}
        void bounds(size_t count) const {
            if (count > bytes.size() - offset) throw std::runtime_error("truncated Step checkpoint");
        }
        void read(void * dst, size_t count) override {
            bounds(count); std::memcpy(dst, bytes.data() + offset, count); offset += count;
        }
        void read_tensor(ggml_tensor * t, size_t begin, size_t count) override {
            bounds(count); ggml_backend_tensor_set(t, bytes.data() + offset, begin, count); offset += count;
        }
        size_t n_bytes() override { return offset; }
    };
    struct Cache {
        std::vector<uint8_t> bytes;
        llama_kv_cache::slot_info_vec_t slots;
        llama_pos last = -1;
        uint32_t last_slot = 0;
    };
    llama_context * owner_;
    Cache base_, swa_;

    static llama_kv_cache_iswa * memory(llama_context * ctx) {
        auto * mem = dynamic_cast<llama_kv_cache_iswa *>(ctx->get_memory());
        if (!mem || ctx->get_model().arch != LLM_ARCH_STEP35 || ctx->n_seq_max() != 1 ||
            !ctx->get_cparams().causal_attn || ctx->get_model().hparams.swa_type != LLAMA_SWA_TYPE_STANDARD)
            throw std::runtime_error("Step checkpoint requires one Step text sequence");
        return mem;
    }
    static Cache capture_cache(llama_kv_cache & kv) {
        if (kv.get_n_stream() != 1 || kv.get_has_shift() || kv.has_cell_ext())
            throw std::runtime_error("unsupported Step checkpoint cache layout");
        Cache out;
        const auto & cells = kv.get_cells(0);
        llama_kv_cache::slot_info slots{0, 0, {0}, {{}}};
        std::vector<llama_pos> positions;
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells.is_empty(i)) continue;
            if (cells.seq_count(i) != 1 || !cells.seq_has(i, 0))
                throw std::runtime_error("Step checkpoint found another sequence");
            slots.idxs[0].push_back(i); positions.push_back(cells.pos_get(i));
            if (cells.pos_get(i) > out.last) { out.last = cells.pos_get(i); out.last_slot = i; }
        }
        std::sort(positions.begin(), positions.end());
        for (size_t i = 1; i < positions.size(); ++i)
            if (positions[i] != positions[i-1] + 1)
                throw std::runtime_error("Step checkpoint requires contiguous retained positions");
        // Empty streams must use an empty slot_info, not one empty index vector.
        out.slots.push_back(positions.empty() ? llama_kv_cache::slot_info{} : slots);
        Writer io; kv.state_write(io, -1); // -1 retains SWA-masked cells too
        out.bytes = std::move(io.bytes);
        return out;
    }
    static void append_cursor(llama_kv_cache & kv, llama_pos last_pos, uint32_t last_slot) {
        if (last_pos < 0) return;
        // state_read_sinfo sets the search cursor after the highest physical
        // index. For an append-only ring it must follow the last token instead.
        // Remove/reapply just its metadata through the native allocation API;
        // seq_rm never changes K/V bytes, and applying an empty slot purges none.
        if (!kv.seq_rm(0, last_pos, last_pos + 1)) throw std::runtime_error("Step cursor reset failed");
        llama_batch_allocr allocator(1);
        auto batch = allocator.ubatch_reserve(1, 1);
        llama_seq_id seq = 0;
        batch.token = nullptr;
        batch.pos[0] = last_pos; batch.n_seq_id[0] = 1;
        batch.seq_id[0] = &seq; batch.seq_id_unq[0] = 0;
        llama_kv_cache::slot_info last{0, 0, {0}, {{last_slot}}};
        kv.apply_ubatch(last, batch);
    }
    static void restore_cache(llama_kv_cache & kv, const Cache & saved) {
        Reader io(saved.bytes);
        kv.state_read_sinfo(io, 0, 0, nullptr, &saved.slots);
        if (io.n_bytes() != saved.bytes.size()) throw std::runtime_error("Step checkpoint trailing bytes");
        append_cursor(kv, saved.last, saved.last_slot);
    }
    SessionSnapshot(llama_context * ctx, Cache base, Cache swa)
        : owner_(ctx), base_(std::move(base)), swa_(std::move(swa)) {}
public:
    static SessionSnapshot capture(llama_context * ctx) {
        auto * mem = memory(ctx);
        llama_synchronize(ctx);
        if (mem->get_base()->seq_pos_min(0) > 0 || mem->get_base()->seq_pos_max(0) != mem->get_swa()->seq_pos_max(0))
            throw std::runtime_error("Step checkpoint requires an unshifted full prefix and matching SWA tail");
        return SessionSnapshot(ctx, capture_cache(*mem->get_base()), capture_cache(*mem->get_swa()));
    }
    size_t size() const { return base_.bytes.size() + swa_.bytes.size(); }
    void restore(llama_context * ctx) const {
        // Snapshot lifetime must be shorter than its owner context's lifetime.
        if (ctx != owner_) throw std::runtime_error("Step checkpoint belongs to another context");
        auto * mem = memory(ctx);
        llama_synchronize(ctx);
        mem->clear(true);
        try {
            restore_cache(*mem->get_base(), base_);
            restore_cache(*mem->get_swa(), swa_);
        } catch (...) { mem->clear(true); throw; }
    }
};
} // namespace step35
