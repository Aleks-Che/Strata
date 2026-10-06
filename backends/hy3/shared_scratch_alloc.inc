// Included after ggml_gallocr is defined. One reference per distinct allocator
// buffer, not per backend index (the allocator already deduplicates indices).
bool strata_hy3_gallocr_scratch_is_shared(ggml_gallocr_t a, ggml_gallocr_t b) {
    if (!a || !b || a == b) return false;
    for (int i = 0; i < a->n_buffers; ++i) {
        if (strcmp(ggml_backend_buft_name(a->bufts[i]), "CUDA0") != 0 || !a->buffers[i]) continue;
        for (int j = 0; j < b->n_buffers; ++j) {
            if (a->bufts[i] == b->bufts[j] && a->buffers[i] == b->buffers[j]) return true;
        }
    }
    return false;
}

size_t strata_hy3_gallocr_share_scratch(ggml_gallocr_t a, ggml_gallocr_t b) {
    if (!a || !b || a == b) return 0;
    int ai = -1, bi = -1;
    for (int i = 0; i < a->n_buffers; ++i) {
        if (strcmp(ggml_backend_buft_name(a->bufts[i]), "CUDA0") == 0) {
            if (ai >= 0 && a->buffers[ai] != a->buffers[i]) return 0;
            ai = i;
        }
    }
    for (int i = 0; i < b->n_buffers; ++i) {
        if (strcmp(ggml_backend_buft_name(b->bufts[i]), "CUDA0") == 0) {
            if (bi >= 0 && b->buffers[bi] != b->buffers[i]) return 0;
            bi = i;
        }
    }
    if (ai < 0 || bi < 0 || a->bufts[ai] != b->bufts[bi]) return 0;
    struct vbuffer * ab = a->buffers[ai], * bb = b->buffers[bi];
    if (!ab || !bb || ab == bb || ab->strata_refs != 1 || bb->strata_refs != 1) return 0;
    bool a_fits = true, b_fits = true;
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS; ++i) {
        size_t as = ggml_vbuffer_chunk_size(ab, i), bs = ggml_vbuffer_chunk_size(bb, i);
        a_fits = a_fits && as >= bs;
        b_fits = b_fits && bs >= as;
    }
    // No reallocation or partial update if neither chunk layout contains the other.
    if (!a_fits && !b_fits) return 0;
    struct ggml_gallocr * borrower = a_fits ? b : a;
    struct vbuffer * keep = a_fits ? ab : bb, * release = a_fits ? bb : ab;
    const size_t saved = ggml_vbuffer_size(release);
    if (!saved) return 0;
    ++keep->strata_refs;
    for (int i = 0; i < borrower->n_buffers; ++i) {
        if (borrower->buffers[i] == release) borrower->buffers[i] = keep;
    }
    ggml_vbuffer_free(release);
    return saved;
}
