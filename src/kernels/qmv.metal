#include "common.h"

// Quantized matrix x vector(s) for MLX affine weights.
//   W: [N, K] packed little-endian in uint32 words (value j of a word = (w >> BITS*j) & mask)
//   scales/biases: bf16 [N, K/GS];  w_real = scale * q + bias
// y[t, n] = sum_k W[n, k] * x[t, k]   for t < NT (NT > 1 = several tokens share one weight read,
// used to verify speculative drafts).
//
// Each simdgroup owns ROWS consecutive rows; each lane reads VPL values per row per step.
// Shift-free dequant: the value at slot j of a word is (w & (mask << BITS*j)) == q * 2^(BITS*j);
// x is pre-scaled by 2^-(BITS*j), so each value costs one AND + convert + FMA.
// NORM=true fuses an RMSNorm of x (weight `nw`, bf16) into the load: x' = x * rsqrt(mean(x^2)+eps) * nw.

struct QmvParams {
  uint N;
  uint K;
  uint x_stride;  // floats between consecutive input vectors (t)
  uint y_stride;  // floats between consecutive output vectors (t)
  float eps;      // NORM only
};

template <int BITS, int VPL, int ROWS, int GS, int NT, bool ADD, bool NORM>
NI_INLINE void qmv_impl(const device uint32_t* w, const device bfloat* scales, const device bfloat* biases,
                        const device float* x, device float* y, constant QmvParams& p,
                        const device float* resid, const device bfloat* nw, uint row0, uint lane) {
  constexpr int PER_WORD = 32 / BITS;
  static_assert(VPL % PER_WORD == 0, "VPL must cover whole words");
  static_assert(GS % VPL == 0, "a lane chunk must stay inside one group");
  const uint K = p.K;
  const uint words_per_row = K / PER_WORD;
  const uint groups_per_row = K / GS;

  NormCtx<NT> nc;
  if (NORM) nc = rms_ctx<NT>(x, p.x_stride, K, p.eps, lane);

  float acc[ROWS][NT];
  for (int r = 0; r < ROWS; ++r)
    for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;

  // Rows past N are clamped (computed redundantly, never stored) so lanes never diverge.
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) rows[r] = min(row0 + r, p.N - 1);

  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    XChunk<BITS, VPL, NT> c;
    load_x<BITS, VPL, NT, NORM>(c, x, p.x_stride, k0, nc, nw);
    const uint g = k0 / GS;
    const uint wo = k0 / PER_WORD;
    for (int r = 0; r < ROWS; ++r) {
      const float s = float(scales[rows[r] * groups_per_row + g]);
      const float b = float(biases[rows[r] * groups_per_row + g]);
      dot_row<BITS, VPL, NT>(c, w + rows[r] * words_per_row + wo, s, b, acc[r]);
    }
  }

  for (int t = 0; t < NT; ++t) {
    for (int r = 0; r < ROWS; ++r) {
      float v = simd_sum(acc[r][t]);
      uint row = row0 + r;
      if (lane == 0 && row < p.N) {
        if (ADD) v += resid[t * p.y_stride + row];
        y[t * p.y_stride + row] = v;
      }
    }
  }
}

#define NI_QMV(NAME, BITS, VPL, ROWS, GS, NT, ADD, NORM)                                                 \
  [[kernel]] void NAME(const device uint32_t* w [[buffer(0)]], const device bfloat* scales [[buffer(1)]], \
                       const device bfloat* biases [[buffer(2)]], const device float* x [[buffer(3)]],     \
                       device float* y [[buffer(4)]], constant QmvParams& p [[buffer(5)]],                 \
                       const device float* resid [[buffer(6)]], const device bfloat* nw [[buffer(7)]],     \
                       uint3 tgp [[threadgroup_position_in_grid]],                                         \
                       uint sg [[simdgroup_index_in_threadgroup]],                                         \
                       uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {  \
    uint row0 = (tgp.x * nsg + sg) * ROWS;                                                                \
    qmv_impl<BITS, VPL, ROWS, GS, NT, ADD, NORM>(w, scales, biases, x, y, p, resid, nw, row0, lane);     \
  }

// name: qmv_b{bits}_g{group}_v{values per lane}_r{rows per simdgroup}_t{vectors}[_add][_norm]
NI_QMV(qmv_b4_g64_v16_r4_t1, 4, 16, 4, 64, 1, false, false)
NI_QMV(qmv_b4_g64_v32_r4_t1, 4, 32, 4, 64, 1, false, false)
NI_QMV(qmv_b4_g64_v16_r8_t1, 4, 16, 8, 64, 1, false, false)
NI_QMV(qmv_b4_g64_v32_r2_t1, 4, 32, 2, 64, 1, false, false)
NI_QMV(qmv_b4_g64_v16_r2_t1, 4, 16, 2, 64, 1, false, false)
NI_QMV(qmv_b4_g64_v16_r4_t1_add, 4, 16, 4, 64, 1, true, false)
NI_QMV(qmv_b4_g64_v16_r4_t1_norm, 4, 16, 4, 64, 1, false, true)
NI_QMV(qmv_b8_g64_v16_r4_t1, 8, 16, 4, 64, 1, false, false)
NI_QMV(qmv_b8_g64_v16_r4_t1_norm, 8, 16, 4, 64, 1, false, true)
NI_QMV(qmv_b4_g64_v16_r4_t4, 4, 16, 4, 64, 4, false, false)

// ---- Multi-segment: several matrices that share the same input (e.g. q/k/v or qkv/z/a/b).
// Threadgroups are partitioned across segments: segment i owns tg.x in [tg_begin[i], tg_begin[i+1]).
struct QmvMultiParams {
  uint K;
  uint nseg;
  float eps;
  uint N[4];
  uint tg_begin[5];
  uint bits[4]; // per segment: 4 or 8
};

template <int VPL, int ROWS, int GS, bool NORM>
NI_INLINE void qmv_seg(const device uint32_t* w, const device bfloat* s, const device bfloat* b,
                       const device float* x, device float* y, uint N, uint bits, uint K, float eps,
                       const device bfloat* nw, uint row0, uint lane) {
  NormCtx<1> nc;
  if (NORM) nc = rms_ctx<1>(x, K, K, eps, lane);
  float acc[ROWS][1];
  for (int r = 0; r < ROWS; ++r) acc[r][0] = 0.0f;
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) rows[r] = min(row0 + r, N - 1);
  const uint groups_per_row = K / GS;
  if (bits == 4) {
    const uint wpr = K / 8;
    for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
      XChunk<4, VPL, 1> c;
      load_x<4, VPL, 1, NORM>(c, x, K, k0, nc, nw);
      for (int r = 0; r < ROWS; ++r)
        dot_row<4, VPL, 1>(c, w + rows[r] * wpr + k0 / 8, float(s[rows[r] * groups_per_row + k0 / GS]),
                           float(b[rows[r] * groups_per_row + k0 / GS]), acc[r]);
    }
  } else {
    const uint wpr = K / 4;
    for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
      XChunk<8, VPL, 1> c;
      load_x<8, VPL, 1, NORM>(c, x, K, k0, nc, nw);
      for (int r = 0; r < ROWS; ++r)
        dot_row<8, VPL, 1>(c, w + rows[r] * wpr + k0 / 4, float(s[rows[r] * groups_per_row + k0 / GS]),
                           float(b[rows[r] * groups_per_row + k0 / GS]), acc[r]);
    }
  }
  for (int r = 0; r < ROWS; ++r) {
    float v = simd_sum(acc[r][0]);
    if (lane == 0 && row0 + r < N) y[row0 + r] = v;
  }
}

#define NI_QMV_MULTI(NAME, VPL, ROWS, GS, NORM)                                                          \
  [[kernel]] void NAME(                                                                                  \
      const device uint32_t* w0 [[buffer(0)]], const device bfloat* s0 [[buffer(1)]],                    \
      const device bfloat* b0 [[buffer(2)]], device float* y0 [[buffer(3)]],                             \
      const device uint32_t* w1 [[buffer(4)]], const device bfloat* s1 [[buffer(5)]],                    \
      const device bfloat* b1 [[buffer(6)]], device float* y1 [[buffer(7)]],                             \
      const device uint32_t* w2 [[buffer(8)]], const device bfloat* s2 [[buffer(9)]],                    \
      const device bfloat* b2 [[buffer(10)]], device float* y2 [[buffer(11)]],                           \
      const device uint32_t* w3 [[buffer(12)]], const device bfloat* s3 [[buffer(13)]],                  \
      const device bfloat* b3 [[buffer(14)]], device float* y3 [[buffer(15)]],                           \
      const device float* x [[buffer(16)]], constant QmvMultiParams& p [[buffer(17)]],                   \
      const device bfloat* nw [[buffer(18)]], uint3 tgp [[threadgroup_position_in_grid]],                \
      uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],               \
      uint lane [[thread_index_in_simdgroup]]) {                                                         \
    uint seg = 0;                                                                                        \
    for (uint i = 1; i < p.nseg; ++i)                                                                    \
      if (tgp.x >= p.tg_begin[i]) seg = i;                                                               \
    uint row0 = ((tgp.x - p.tg_begin[seg]) * nsg + sg) * ROWS;                                           \
    switch (seg) {                                                                                       \
      case 0: qmv_seg<VPL, ROWS, GS, NORM>(w0, s0, b0, x, y0, p.N[0], p.bits[0], p.K, p.eps, nw, row0, lane); break; \
      case 1: qmv_seg<VPL, ROWS, GS, NORM>(w1, s1, b1, x, y1, p.N[1], p.bits[1], p.K, p.eps, nw, row0, lane); break; \
      case 2: qmv_seg<VPL, ROWS, GS, NORM>(w2, s2, b2, x, y2, p.N[2], p.bits[2], p.K, p.eps, nw, row0, lane); break; \
      default: qmv_seg<VPL, ROWS, GS, NORM>(w3, s3, b3, x, y3, p.N[3], p.bits[3], p.K, p.eps, nw, row0, lane); break; \
    }                                                                                                    \
  }

NI_QMV_MULTI(qmv_multi_g64_v16_r4_norm, 16, 4, 64, true)
NI_QMV_MULTI(qmv_multi_g64_v16_r4, 16, 4, 64, false)
