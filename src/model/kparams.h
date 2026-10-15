// Host mirrors of the kernel parameter structs (must match src/kernels/*.metal field for field).
#pragma once
#include <cstdint>

namespace ni::kp {

struct Qmv {
  uint32_t N, K, x_stride, y_stride;
  float eps;
};

struct QmvMulti {
  uint32_t K, nseg;
  float eps;
  uint32_t N[4];
  uint32_t tg_begin[5];
  uint32_t bits[4];
};

struct Embed {
  uint32_t D, gs, tok_off;
};

struct GdnPrep {
  uint32_t T, Hk, Hv, C;
};

struct GdnStep {
  uint32_t T, Hk, Hv;
  float eps;
};

struct AttnPrep {
  uint32_t T, Hq, Hkv, D, rot, pos, max_seq;
  float eps, theta;
};

struct AttnDec {
  uint32_t Hq, Hkv, L, chunk, max_seq;
  float scale;
  uint32_t nchunks;
};

struct Topk {
  uint32_t E, k, norm;
};

struct Moe {
  uint32_t D, I, SI, k;
  float eps;
};

struct TopkStage {
  uint32_t V, K, chunk;
};

struct Sample {
  uint32_t n, K;
  float temp, top_p;
  uint32_t seed, step, out_index;
};

struct Qmm {
  uint32_t M, N, K, x_ld, y_ld, add, gs, w_expert_stride, s_expert_stride;
};

struct AttnPref {
  uint32_t T, Hq, Hkv, pos0, max_seq;
  float scale;
};

struct Bucket {
  uint32_t P, k, E, BM;
};

struct Rows {
  uint32_t D, norm;
  float eps;
};

struct Combine {
  uint32_t D, k;
};

} // namespace ni::kp
