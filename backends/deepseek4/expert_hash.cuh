// A bounded duplicate of integer GET_ROWS, used only to schedule I/O early.
// The model still executes its original routing nodes on the GPU.
struct StrataHashLookup {
    const int32_t *tables[3];
    int32_t tokens[8];
    int widths[3];
    int count, n_tokens;
};
static __global__ void strata_hash_lookup(StrataHashLookup args,int32_t *out) {
    const int i=threadIdx.x, layer=i/64, index=i%64;
    if(layer>=args.count)return;
    const int width=args.widths[layer];
    if(index<width*args.n_tokens)
        out[i]=args.tables[layer][size_t(args.tokens[index/width])*width+index%width];
}
struct StrataHashScratch {
    int32_t *device=nullptr,*host=nullptr;
    bool allocate() {
        constexpr size_t bytes=3*64*sizeof(int32_t);
        if(!device && cudaMalloc((void **)&device,bytes)!=cudaSuccess) {
            cudaGetLastError();return false;
        }
        if(!host && cudaHostAlloc((void **)&host,bytes,cudaHostAllocDefault)!=cudaSuccess) {
            cudaGetLastError();return false;
        }
        return true;
    }
    ~StrataHashScratch() {
        if(device)CUDA_CHECK(cudaFree(device));
        if(host)CUDA_CHECK(cudaFreeHost(host));
    }
};
