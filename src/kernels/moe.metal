#include "common.h"

// ---- Mixture of experts (Qwen3.5/3.6 MoE): softmax router, top-k, SwiGLU experts + gated shared expert ----

struct TopkParams {
  uint E;     // experts (multiple of 32, <= 512)
  uint k;     // active experts
  uint norm;  // renormalize top-k probabilities
};

// One simdgroup. logits: [E] router logits, shared: [1] shared-expert gate logit.
// Writes idx[k] and wts[k+1] (wts[k] = sigmoid(shared gate)).
[[kernel]] void moe_topk(const device float* logits [[buffer(0)]], const device float* shared [[buffer(1)]],
                         device uint* idx [[buffer(2)]], device float* wts [[buffer(3)]],
                         constant TopkParams& p [[buffer(4)]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint MAXPER = 16;
  const uint per = p.E / 32;
  float v[MAXPER];
  float mx = -INFINITY;
  for (uint i = 0; i < per; ++i) {
    v[i] = logits[lane * per + i];
    mx = max(mx, v[i]);
  }
  mx = simd_max(mx);
  float sum = 0.0f;
  for (uint i = 0; i < per; ++i) {
    v[i] = metal::precise::exp(v[i] - mx);
    sum += v[i];
  }
  sum = simd_sum(sum);
  for (uint i = 0; i < per; ++i) v[i] /= sum;

  uint taken = 0;
  float sel_sum = 0.0f;
  float my_w[16];
  uint my_i[16];
  for (uint r = 0; r < p.k; ++r) {
    float best = -1.0f;
    uint bi = 0xffffffff;
    for (uint i = 0; i < per; ++i)
      if (!(taken & (1u << i)) && v[i] > best) {
        best = v[i];
        bi = lane * per + i;
      }
    float gbest = simd_max(best);
    uint cand = best == gbest ? bi : 0xffffffff;
    uint win = simd_min(cand);
    if (win / per == lane && win != 0xffffffff) taken |= 1u << (win % per);
    my_w[r] = gbest;
    my_i[r] = win;
    sel_sum += gbest;
  }
  if (lane == 0) {
    for (uint r = 0; r < p.k; ++r) {
      idx[r] = my_i[r];
      wts[r] = p.norm ? my_w[r] / sel_sum : my_w[r];
    }
    wts[p.k] = sigmoid(shared[0]);
  }
}


struct MoeParams {
  uint D;    // hidden size (input of gate/up, output of down)
  uint I;    // expert intermediate size
  uint SI;   // shared expert intermediate size
  uint k;    // active experts
  float eps; // fused input RMSNorm
};

// grid: x = row tiles over max(I, SI) (ROWS*simdgroups rows each), y = slot (0..k-1 routed, k shared).
// h[slot, row] = silu(gate_row . xn) * (up_row . xn), xn = rmsnorm(x) * nw.
[[kernel]] void moe_gate_up(const device uint32_t* gw [[buffer(0)]], const device bfloat* gsc [[buffer(1)]],
                            const device bfloat* gbi [[buffer(2)]], const device uint32_t* uw [[buffer(3)]],
                            const device bfloat* usc [[buffer(4)]], const device bfloat* ubi [[buffer(5)]],
                            const device uint32_t* sgw [[buffer(6)]], const device bfloat* sgs [[buffer(7)]],
                            const device bfloat* sgb [[buffer(8)]], const device uint32_t* suw [[buffer(9)]],
                            const device bfloat* sus [[buffer(10)]], const device bfloat* sub [[buffer(11)]],
                            const device float* x [[buffer(12)]], const device bfloat* nw [[buffer(13)]],
                            const device uint* idx [[buffer(14)]], device float* h [[buffer(15)]],
                            constant MoeParams& p [[buffer(16)]], uint2 tgp [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  constexpr int ROWS = 4, VPL = 16, GS = 64;
  const uint slot = tgp.y;
  const bool shared = slot == p.k;
  const uint N = shared ? p.SI : p.I;
  const uint row0 = (tgp.x * nsg + sg) * ROWS;
  if (row0 >= N) return;
  const uint K = p.D, wpr = K / 8, gpr = K / GS;
  const device uint32_t *G = gw, *U = uw;
  const device bfloat *Gs = gsc, *Gb = gbi, *Us = usc, *Ub = ubi;
  if (shared) {
    G = sgw; Gs = sgs; Gb = sgb; U = suw; Us = sus; Ub = sub;
  } else {
    uint e = idx[slot];
    G += e * N * wpr; U += e * N * wpr;
    Gs += e * N * gpr; Gb += e * N * gpr; Us += e * N * gpr; Ub += e * N * gpr;
  }
  NormCtx<1> nc = rms_ctx<1>(x, K, K, p.eps, lane);
  float ag[ROWS][1], au[ROWS][1];
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    ag[r][0] = au[r][0] = 0.0f;
    rows[r] = min(row0 + r, N - 1);
  }
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    XChunk<4, VPL, 1> c;
    load_x<4, VPL, 1, true>(c, x, K, k0, nc, nw);
    const uint g = k0 / GS;
    for (int r = 0; r < ROWS; ++r) {
      uint ro = rows[r];
      dot_row<4, VPL, 1>(c, G + ro * wpr + k0 / 8, float(Gs[ro * gpr + g]), float(Gb[ro * gpr + g]), ag[r]);
      dot_row<4, VPL, 1>(c, U + ro * wpr + k0 / 8, float(Us[ro * gpr + g]), float(Ub[ro * gpr + g]), au[r]);
    }
  }
  for (int r = 0; r < ROWS; ++r) {
    float gv = simd_sum(ag[r][0]), uv = simd_sum(au[r][0]);
    if (lane == 0 && row0 + r < N) h[slot * max(p.I, p.SI) + row0 + r] = silu(gv) * uv;
  }
}

// grid: x = row tiles over D. x[row] += sum_slot wts[slot] * (down_slot[row] . h[slot]).
// Expert intermediate sizes must be a multiple of 512 (one 16-values-per-lane step).
[[kernel]] void moe_down(const device uint32_t* dw [[buffer(0)]], const device bfloat* dsc [[buffer(1)]],
                         const device bfloat* dbi [[buffer(2)]], const device uint32_t* sdw [[buffer(3)]],
                         const device bfloat* sds [[buffer(4)]], const device bfloat* sdb [[buffer(5)]],
                         const device float* h [[buffer(6)]], const device uint* idx [[buffer(7)]],
                         const device float* wts [[buffer(8)]], device float* x [[buffer(9)]],
                         constant MoeParams& p [[buffer(10)]], uint tgp [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  constexpr int ROWS = 4, VPL = 16, GS = 64;
  const uint row0 = (tgp * nsg + sg) * ROWS;
  if (row0 >= p.D) return;
  const uint hstride = max(p.I, p.SI);
  NormCtx<1> nc;
  float acc[ROWS];
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    acc[r] = 0.0f;
    rows[r] = min(row0 + r, p.D - 1);
  }
  for (uint slot = 0; slot <= p.k; ++slot) {
    const bool shared = slot == p.k;
    const uint K = shared ? p.SI : p.I;
    const uint wpr = K / 8, gpr = K / GS;
    const device uint32_t* W = shared ? sdw : dw + idx[slot] * p.D * wpr;
    const device bfloat* S = shared ? sds : dsc + idx[slot] * p.D * gpr;
    const device bfloat* B = shared ? sdb : dbi + idx[slot] * p.D * gpr;
    const float ws = wts[slot];
    for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
      XChunk<4, VPL, 1> c;
      load_x<4, VPL, 1, false>(c, h + slot * hstride, K, k0, nc, nullptr);
      const uint g = k0 / GS;
      for (int r = 0; r < ROWS; ++r) {
        float a[1] = {0.0f};
        dot_row<4, VPL, 1>(c, W + rows[r] * wpr + k0 / 8, float(S[rows[r] * gpr + g]), float(B[rows[r] * gpr + g]), a);
        acc[r] += ws * a[0];
      }
    }
  }
  for (int r = 0; r < ROWS; ++r) {
    float v = simd_sum(acc[r]);
    if (lane == 0 && row0 + r < p.D) x[row0 + r] += v;
  }
}

// ---- Prefill MoE helpers ----

// Router top-k for T tokens: one simdgroup per token (grid x = T, 32 threads).
[[kernel]] void moe_topk_rows(const device float* logits [[buffer(0)]], const device float* shared [[buffer(1)]],
                              device uint* idx [[buffer(2)]], device float* wts [[buffer(3)]],
                              constant TopkParams& p [[buffer(4)]], uint t [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const uint per = p.E / 32;
  float v[16];
  float mx = -INFINITY;
  for (uint i = 0; i < per; ++i) {
    v[i] = logits[t * p.E + lane * per + i];
    mx = max(mx, v[i]);
  }
  mx = simd_max(mx);
  float sum = 0.0f;
  for (uint i = 0; i < per; ++i) {
    v[i] = metal::precise::exp(v[i] - mx);
    sum += v[i];
  }
  sum = simd_sum(sum);
  for (uint i = 0; i < per; ++i) v[i] /= sum;
  uint taken = 0;
  float sel_sum = 0.0f;
  float my_w[16];
  uint my_i[16];
  for (uint r = 0; r < p.k; ++r) {
    float best = -1.0f;
    uint bi = 0xffffffff;
    for (uint i = 0; i < per; ++i)
      if (!(taken & (1u << i)) && v[i] > best) { best = v[i]; bi = lane * per + i; }
    float gbest = simd_max(best);
    uint win = simd_min(best == gbest ? bi : 0xffffffff);
    if (win != 0xffffffff && win / per == lane) taken |= 1u << (win % per);
    my_w[r] = gbest;
    my_i[r] = win;
    sel_sum += gbest;
  }
  if (lane == 0) {
    for (uint r = 0; r < p.k; ++r) {
      idx[t * p.k + r] = my_i[r];
      wts[t * (p.k + 1) + r] = p.norm ? my_w[r] / sel_sum : my_w[r];
    }
    wts[t * (p.k + 1) + p.k] = sigmoid(shared[t]);
  }
}

// Groups the T*k (token, slot) pairs by expert. One threadgroup of 1024 threads.
// Outputs: sorted_pair[P] (pair ids grouped by expert), sorted_tok[P] (= pair / k),
// tiles[3*i] = (expert, begin, end) row ranges of at most BM sorted rows, ntiles[0].
struct BucketParams {
  uint P;   // pairs = T * k
  uint k;
  uint E;
  uint BM;
};

[[kernel]] void moe_bucket(const device uint* idx [[buffer(0)]], device uint* sorted_pair [[buffer(1)]],
                           device uint* sorted_tok [[buffer(2)]], device uint* tiles [[buffer(3)]],
                           device uint* ntiles [[buffer(4)]], constant BucketParams& p [[buffer(5)]],
                           uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup atomic_uint cnt[512];
  threadgroup uint off[513];
  for (uint e = tid; e < p.E; e += ntg) atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < p.P; i += ntg) atomic_fetch_add_explicit(&cnt[idx[i]], 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint acc = 0, nt = 0;
    for (uint e = 0; e < p.E; ++e) {
      uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
      off[e] = acc;
      for (uint r = 0; r < c; r += p.BM) {
        tiles[nt * 3 + 0] = e;
        tiles[nt * 3 + 1] = acc + r;
        tiles[nt * 3 + 2] = acc + min(r + p.BM, c);
        ++nt;
      }
      acc += c;
    }
    ntiles[0] = nt;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < p.E; e += ntg) atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < p.P; i += ntg) {
    uint e = idx[i];
    uint pos = off[e] + atomic_fetch_add_explicit(&cnt[e], 1u, memory_order_relaxed);
    sorted_pair[pos] = i;
    sorted_tok[pos] = i / p.k;
  }
}

// h[row] = silu(g[row]) * u[row] as half (rows of width I). grid: rows x I/4 threads.
[[kernel]] void swiglu_half(const device float* g [[buffer(0)]], const device float* u [[buffer(1)]],
                            device half* h [[buffer(2)]], constant uint& n [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
  if (i >= n) return;
  h[i] = half(silu(g[i]) * u[i]);
}

// x[t, :] += sum_s wts[t, s] * y[t*k + s, :] + wts[t, k] * ys[t, :]. grid: (D/4, T)
struct CombineParams {
  uint D;
  uint k;
};
[[kernel]] void moe_combine(const device float* y [[buffer(0)]], const device float* ys [[buffer(1)]],
                            const device float* wts [[buffer(2)]], device float* x [[buffer(3)]],
                            constant CombineParams& p [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
  uint d4 = gid.x, t = gid.y;
  if (d4 >= p.D / 4) return;
  float4 acc = 0.0f;
  for (uint s = 0; s < p.k; ++s) acc += wts[t * (p.k + 1) + s] * ((const device float4*)(y + (t * p.k + s) * p.D))[d4];
  acc += wts[t * (p.k + 1) + p.k] * ((const device float4*)(ys + t * p.D))[d4];
  ((device float4*)(x + t * p.D))[d4] += acc;
}
