// Sharing is a setup-only operation. Live graph tensor addresses may already
// point into private allocations, so reject either allocated scheduler.
extern "C" bool strata_hy3_sched_scratch_is_shared(ggml_backend_sched_t a, ggml_backend_sched_t b) {
    return a && b && strata_hy3_gallocr_scratch_is_shared(a->galloc, b->galloc);
}

extern "C" size_t strata_hy3_sched_share_scratch(ggml_backend_sched_t a, ggml_backend_sched_t b) {
    if (!a || !b || a == b || a->is_alloc || b->is_alloc || a->n_copies != 1 || b->n_copies != 1) return 0;
    ggml_backend_sched_synchronize(a);
    ggml_backend_sched_synchronize(b);
    return strata_hy3_gallocr_share_scratch(a->galloc, b->galloc);
}
