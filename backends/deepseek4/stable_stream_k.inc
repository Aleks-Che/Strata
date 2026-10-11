        // This helper is instantiated only for the private D=512 MMA path.
        // Its dispatcher requires sparse, single-sequence, 64-head attention
        // with ncols1=1, ncols2=8 on Ada or newer NVIDIA GPUs.
        const int stock_blocks=blocks_num.x;
        const char *long_mode=std::getenv("STRATA_DS4_FA_COMPACT");
        const int stable_blocks=strata_ds4::stable_stream_k_blocks(
            int(Q->ne[1]),ntiles_z_gqa,ntiles_KV,max_blocks,long_mode && std::strcmp(long_mode,"4")==0?4096:4);
        if(!stable_blocks) {
            // A fractional one-query grid cannot be replicated by merely
            // scaling gridDim.x: tile order interleaves queries and heads.
            if(Q->ne[1]>1) {
                for(int64_t i=0;i<Q->ne[1];++i) {
                    ggml_tensor one_q=*Q,one_mask=*mask,one_dst=*dst;
                    one_q.ne[1]=1;one_q.data=(char *)Q->data+i*Q->nb[1];
                    one_mask.ne[1]=1;one_mask.data=(char *)mask->data+i*mask->nb[1];
                    one_dst.ne[2]=1;one_dst.data=(char *)dst->data+i*dst->nb[2];
                    one_dst.src[0]=&one_q;one_dst.src[3]=&one_mask;
                    strata_ds4_launch_fattn<DV,ncols1,ncols2>(ctx,&one_dst,fattn_kernel,nwarps,
                        nbytes_shared,nbatch_fa,need_f16_K,need_f16_V,stream_k,use_sparse,warp_size);
                }
                return;
            }
        } else {
            blocks_num.x=stable_blocks;
        }
        if(const char *trace=std::getenv("STRATA_DS4_FA_STREAM_K_TRACE");trace && std::strcmp(trace,"1")==0) {
            static std::mutex trace_mutex;
            static std::set<std::tuple<int,int,int,int,int,int>> seen;
            std::lock_guard<std::mutex> lock(trace_mutex);
            if(seen.size()<128 && seen.emplace(int(Q->ne[1]),int(n_kv),ntiles_KV,max_blocks,stock_blocks,int(blocks_num.x)).second)
                std::fprintf(stderr,"STRATA_FA_STREAM_K queries=%d kv=%lld kv_tiles=%d tiles_per_query=%d max_blocks=%d stock_blocks=%d stable_blocks=%u\n",
                    int(Q->ne[1]),(long long)n_kv,ntiles_KV,ntiles_z_gqa,max_blocks,stock_blocks,blocks_num.x);
        }
