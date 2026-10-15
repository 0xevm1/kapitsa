#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Shared helpers for nace-infer kernels. Activations are fp32 on the decode path;
// weights are MLX affine-quantized (uint32-packed, bf16 scales/biases per group).

#define NI_INLINE inline __attribute__((always_inline))

NI_INLINE float silu(float x) { return x / (1.0f + metal::fast::exp(-x)); }
NI_INLINE float sigmoid(float x) { return 1.0f / (1.0f + metal::fast::exp(-x)); }
NI_INLINE float softplus(float x) { return x > 20.0f ? x : metal::precise::log(1.0f + metal::precise::exp(x)); }

// Sum across a threadgroup of up to 1024 threads (32 simdgroups). `scratch` holds >= 32 floats.
NI_INLINE float tg_sum(float v, threadgroup float* scratch, uint tid, uint ntg) {
  uint lane = tid & 31, sg = tid >> 5, nsg = (ntg + 31) >> 5;
  v = simd_sum(v);
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float r = lane < nsg ? scratch[lane] : 0.0f;
  r = simd_sum(r);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}

NI_INLINE float tg_max(float v, threadgroup float* scratch, uint tid, uint ntg) {
  uint lane = tid & 31, sg = tid >> 5, nsg = (ntg + 31) >> 5;
  v = simd_max(v);
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float r = lane < nsg ? scratch[lane] : -INFINITY;
  r = simd_max(r);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}
