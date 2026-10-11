# Private Q8_0 host launch helpers: trace stock grid selection and provide a
# fixed J / fixed K partition research path. Device math stays unchanged.
set(ds4_mmq_dir "${STRATA_LLAMA_DIR}/ggml/src/ggml-cuda")
file(SHA256 "${ds4_mmq_dir}/mmq.cuh" ds4_mmq_header_hash)
file(SHA256 "${ds4_mmq_dir}/mmq.cu" ds4_mmq_source_hash)
if(NOT ds4_mmq_header_hash STREQUAL "50db23f5055af6e28c142fcb4c7fb16c51f7137184ebf8c0ca113ed88906851b" OR
   NOT ds4_mmq_source_hash STREQUAL "a35bb96dfc917c9026023482e24dcf47539fc08ccee1bb4236f40932e9fe540e")
  message(FATAL_ERROR "Review DeepSeek fixed projection MMQ for changed source")
endif()
set(ds4_fixed_dir "${CMAKE_BINARY_DIR}/ds4-projection-mmq")
file(MAKE_DIRECTORY "${ds4_fixed_dir}")
file(READ "${ds4_mmq_dir}/mmq.cuh" ds4_mmq_header)
string(FIND "${ds4_mmq_header}" "template <ggml_type type, int J, bool fallback>\nstatic void launch_mul_mat_q" ds4_launch_start)
string(FIND "${ds4_mmq_header}" "#define DECL_MMQ_CASE" ds4_launch_end)
if(ds4_launch_start LESS 0 OR ds4_launch_end LESS ds4_launch_start)
  message(FATAL_ERROR "Missing MMQ launch helper boundaries")
endif()
math(EXPR ds4_launch_length "${ds4_launch_end}-${ds4_launch_start}")
string(SUBSTRING "${ds4_mmq_header}" ${ds4_launch_start} ${ds4_launch_length} ds4_mmq_header)
foreach(host launch_mul_mat_q mul_mat_q_switch_J mul_mat_q_case)
  string(REPLACE "${host}" "strata_projection_${host}" ds4_mmq_header "${ds4_mmq_header}")
endforeach()
string(REPLACE "template <ggml_type type, int J, bool fallback>\nstatic void strata_projection_launch_mul_mat_q"
  "template <ggml_type type, int J, bool fallback, bool fixed = false, int fixed_splits = 0>\nstatic void strata_projection_launch_mul_mat_q" ds4_mmq_header "${ds4_mmq_header}")
set(ds4_grid "const dim3 block_nums_stream_k(GGML_CUDA_CC_IS_NVIDIA(cc) && tiles_efficiency_percent >= 90 ? ntiles_dst : nsm, 1, 1);")
string(FIND "${ds4_mmq_header}" "${ds4_grid}" ds4_grid_at)
if(ds4_grid_at LESS 0)
  message(FATAL_ERROR "Missing MMQ stream-K grid")
endif()
string(REPLACE "${ds4_grid}" "const dim3 block_nums_stream_k(fixed ? ntiles_dst*(fixed_splits ? fixed_splits : strata_projection_splits()) : (GGML_CUDA_CC_IS_NVIDIA(cc) && tiles_efficiency_percent >= 90 ? ntiles_dst : nsm), 1, 1);\n    strata_projection_grid_trace(args, J, config.I, block_nums_stream_k.x, ntiles_dst, fixed);" ds4_mmq_header "${ds4_mmq_header}")
string(PREPEND ds4_mmq_header "#pragma once\n#include \"mmq.cuh\"\n#include \"projection_grid_trace.hpp\"\n")
string(APPEND ds4_mmq_header [=[
static void strata_projection_launch_fixed(ggml_backend_cuda_context &ctx,const mmq_args &args,cudaStream_t stream) {
    // Q_b has enough output rows for one whole-K block per tile. Avoid a
    // multi-gigabyte fixup buffer for the largest prefill batch.
    if(args.ncols_x==1024 && args.nrows_x==32768) {
        strata_projection_launch_mul_mat_q<GGML_TYPE_Q8_0,64,false,true,1>(ctx,args,stream);return;
    }
    switch(strata_projection_tile()) {
        case 32: strata_projection_launch_mul_mat_q<GGML_TYPE_Q8_0,32,false,true>(ctx,args,stream);break;
        case 64: strata_projection_launch_mul_mat_q<GGML_TYPE_Q8_0,64,false,true>(ctx,args,stream);break;
        default: strata_projection_launch_mul_mat_q<GGML_TYPE_Q8_0,8,false,true>(ctx,args,stream);break;
    }
}
]=])
file(WRITE "${ds4_fixed_dir}/strata-projection-mmq.cuh" "${ds4_mmq_header}")

# Copy only the upstream dense entry path. The routed expert path remains in
# the ordinary backend. Reuse upstream quantization, padding and pool handling.
file(READ "${ds4_mmq_dir}/mmq.cu" ds4_mmq_source)
string(FIND "${ds4_mmq_source}" "void ggml_cuda_mul_mat_q(" ds4_dense_start)
string(FIND "${ds4_mmq_source}" "    GGML_ASSERT(ne13 == 1);" ds4_dense_end)
if(ds4_dense_start LESS 0 OR ds4_dense_end LESS ds4_dense_start)
  message(FATAL_ERROR "Missing dense MMQ entry boundaries")
endif()
math(EXPR ds4_dense_length "${ds4_dense_end}-${ds4_dense_start}")
string(SUBSTRING "${ds4_mmq_source}" ${ds4_dense_start} ${ds4_dense_length} ds4_dense)
string(REPLACE "void ggml_cuda_mul_mat_q(" "void strata_projection_fixed_mmq(" ds4_dense "${ds4_dense}")
string(REPLACE "ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);"
  "strata_projection_launch_fixed(ctx, args, stream);" ds4_dense "${ds4_dense}")
file(WRITE "${ds4_fixed_dir}/strata-projection-mmq.cu" "#include \"strata-projection-mmq.cuh\"\n#include \"quantize.cuh\"\n${ds4_dense}\n    GGML_ABORT(\"fixed projection MMQ requires dense inputs\");\n}\n")

# Replace only the Q8_0 instantiation with a forwarding definition through our
# otherwise identical stock launch helpers. This makes the trace report the
# actual J/grid, including for the unmodified reference path.
file(WRITE "${ds4_fixed_dir}/strata-mmq-q8_0.cu" [=[
#define mul_mat_q_case strata_projection_unused_case
#include "strata-projection-mmq.cuh"
#undef mul_mat_q_case
template <ggml_type type>
void mul_mat_q_case(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    strata_projection_mul_mat_q_case<type>(ctx, args, stream);
}
template void mul_mat_q_case<GGML_TYPE_Q8_0>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
]=])
get_target_property(ds4_mmq_sources ggml-cuda SOURCES)
set(ds4_q8_sources "${ds4_mmq_sources}")
list(FILTER ds4_q8_sources INCLUDE REGEX "(^|/)mmq-instance-q8_0\\.cu$")
list(LENGTH ds4_q8_sources ds4_q8_count)
if(NOT ds4_q8_count EQUAL 1)
  message(FATAL_ERROR "Expected one Q8_0 MMQ instantiation")
endif()
list(REMOVE_ITEM ds4_mmq_sources ${ds4_q8_sources})
set_property(TARGET ggml-cuda PROPERTY SOURCES "${ds4_mmq_sources};${ds4_fixed_dir}/strata-mmq-q8_0.cu;${ds4_fixed_dir}/strata-projection-mmq.cu")
set_source_files_properties("${ds4_fixed_dir}/strata-mmq-q8_0.cu" "${ds4_fixed_dir}/strata-projection-mmq.cu" TARGET_DIRECTORY ggml-cuda
  PROPERTIES INCLUDE_DIRECTORIES "${ds4_mmq_dir};${CMAKE_CURRENT_LIST_DIR}")
