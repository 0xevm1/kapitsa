#include "common.h"

// ---- Token embedding (MLX affine 4-bit, row per token) ----
struct EmbedParams {
  uint D;        // hidden size
  uint gs;       // group size
  uint tok_off;  // index into `tokens` of the first token
};

// grid: x = D/8 words, y = T tokens
[[kernel]] void embed_q4(const device uint32_t* w [[buffer(0)]], const device bfloat* s [[buffer(1)]],
                         const device bfloat* b [[buffer(2)]], const device int* tokens [[buffer(3)]],
                         device float* x [[buffer(4)]], constant EmbedParams& p [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
  uint i = gid.x, t = gid.y;
  if (i >= p.D / 8) return;
  int tok = tokens[p.tok_off + t];
  uint32_t word = w[uint(tok) * (p.D / 8) + i];
  uint gpr = p.D / p.gs;
  for (uint j = 0; j < 8; ++j) {
    uint k = i * 8 + j;
    float sc = float(s[uint(tok) * gpr + k / p.gs]), bi = float(b[uint(tok) * gpr + k / p.gs]);
    x[t * p.D + k] = sc * float((word >> (4 * j)) & 0xF) + bi;
  }
}

// ---- Gated DeltaNet: conv1d + SiLU + q/k L2 norm + decay/beta ----
// Channels of the qkv projection: [q: Hk*Dk | k: Hk*Dk | v: Hv*Dv]; one threadgroup of 128 threads
// per 128-channel head chunk. conv_state holds the previous 3 raw inputs ([3, C]).
struct GdnPrepParams {
  uint T;
  uint Hk;
  uint Hv;
  uint C;  // conv channels = 2*Hk*Dk + Hv*Dv
};

[[kernel]] void gdn_prep(const device float* qkv [[buffer(0)]], const device float* a [[buffer(1)]],
                         const device float* bb [[buffer(2)]], device float* conv_state [[buffer(3)]],
                         const device bfloat* conv_w [[buffer(4)]], const device bfloat* A_log [[buffer(5)]],
                         const device bfloat* dt_bias [[buffer(6)]], device float* q_out [[buffer(7)]],
                         device float* k_out [[buffer(8)]], device float* v_out [[buffer(9)]],
                         device float* g_out [[buffer(10)]], device float* beta_out [[buffer(11)]],
                         constant GdnPrepParams& p [[buffer(12)]], uint tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[32];
  constexpr uint D = 128;
  const uint c = tg * D + tid;
  const uint C = p.C;
  const uint qk = p.Hk * D;
  const bool is_q = c < qk, is_k = c >= qk && c < 2 * qk;
  float w0 = conv_w[c * 4 + 0], w1 = conv_w[c * 4 + 1], w2 = conv_w[c * 4 + 2], w3 = conv_w[c * 4 + 3];

  for (uint t = 0; t < p.T; ++t) {
    // concat = [state0, state1, state2, qkv[0], qkv[1], ...]; output t uses concat[t..t+3]
    float in0 = t + 0 < 3 ? conv_state[(t + 0) * C + c] : qkv[(t + 0 - 3) * C + c];
    float in1 = t + 1 < 3 ? conv_state[(t + 1) * C + c] : qkv[(t + 1 - 3) * C + c];
    float in2 = t + 2 < 3 ? conv_state[(t + 2) * C + c] : qkv[(t + 2 - 3) * C + c];
    float in3 = qkv[t * C + c];
    float o = silu(w0 * in0 + w1 * in1 + w2 * in2 + w3 * in3);
    if (is_q || is_k) {
      float ss = tg_sum(o * o, scratch, tid, D);
      o *= metal::precise::rsqrt(ss + 1e-6f);
      if (is_q) {
        o *= metal::precise::rsqrt(float(D));
        q_out[t * qk + c] = o;
      } else {
        k_out[t * qk + (c - qk)] = o;
      }
    } else {
      v_out[t * (C - 2 * qk) + (c - 2 * qk)] = o;
    }
  }
  // Roll the conv window: new rows = concat[T..T+2]. Ascending r only reads rows > r.
  for (uint r = 0; r < 3; ++r) {
    uint pos = p.T + r;
    float v = pos < 3 ? conv_state[pos * C + c] : qkv[(pos - 3) * C + c];
    conv_state[r * C + c] = v;
  }
  if (tg == 0 && tid < p.Hv) {
    float A = metal::precise::exp(float(A_log[tid]));
    float dtb = float(dt_bias[tid]);
    for (uint t = 0; t < p.T; ++t) {
      float av = a[t * p.Hv + tid];
      g_out[t * p.Hv + tid] = metal::precise::exp(-A * softplus(av + dtb));
      beta_out[t * p.Hv + tid] = sigmoid(bb[t * p.Hv + tid]);
    }
  }
}

// ---- Gated DeltaNet recurrence + gated RMSNorm, one threadgroup (1024 threads) per value head ----
// state: [Hv, Dv=128, Dk=128] fp32. Simdgroup sg owns rows dv = sg*4..sg*4+3; lane owns dk = lane*4..+3.
//   S = g*S;  kv = S k;  delta = (v - kv) * beta;  S += delta k^T;  o = S q
//   out = rmsnorm(o) * w * silu(z)
struct GdnStepParams {
  uint T;
  uint Hk;
  uint Hv;
  float eps;
};

[[kernel]] void gdn_step(const device float* q [[buffer(0)]], const device float* k [[buffer(1)]],
                         const device float* v [[buffer(2)]], const device float* g [[buffer(3)]],
                         const device float* beta [[buffer(4)]], device float* state [[buffer(5)]],
                         const device float* z [[buffer(6)]], const device bfloat* norm_w [[buffer(7)]],
                         device float* out [[buffer(8)]], constant GdnStepParams& p [[buffer(9)]],
                         uint h [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[32];
  constexpr uint D = 128;
  const uint hk = h / (p.Hv / p.Hk);
  device float* S = state + h * D * D;
  float4 st[4];
  for (int r = 0; r < 4; ++r) st[r] = *(device float4*)(S + (sg * 4 + r) * D + lane * 4);
  float nw[4];
  for (int r = 0; r < 4; ++r) nw[r] = float(norm_w[sg * 4 + r]);

  for (uint t = 0; t < p.T; ++t) {
    float4 q4 = *(const device float4*)(q + (t * p.Hk + hk) * D + lane * 4);
    float4 k4 = *(const device float4*)(k + (t * p.Hk + hk) * D + lane * 4);
    float gg = g[t * p.Hv + h], be = beta[t * p.Hv + h];
    float o[4];
    float ss = 0.0f;
    for (int r = 0; r < 4; ++r) {
      st[r] *= gg;
      float kv = simd_sum(dot(st[r], k4));
      float delta = (v[(t * p.Hv + h) * D + sg * 4 + r] - kv) * be;
      st[r] += k4 * delta;
      o[r] = simd_sum(dot(st[r], q4));
      ss += o[r] * o[r];
    }
    if (lane == 0) scratch[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = simd_sum(scratch[lane]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = metal::precise::rsqrt(tot / float(D) + p.eps);
    if (lane < 4) {
      uint dv = sg * 4 + lane;
      float ov = lane == 0 ? o[0] : lane == 1 ? o[1] : lane == 2 ? o[2] : o[3];
      float nv = lane == 0 ? nw[0] : lane == 1 ? nw[1] : lane == 2 ? nw[2] : nw[3];
      uint idx = (t * p.Hv + h) * D + dv;
      out[idx] = ov * inv * nv * silu(z[idx]);
    }
  }
  for (int r = 0; r < 4; ++r) *(device float4*)(S + (sg * 4 + r) * D + lane * 4) = st[r];
}

// ---- Prefill helpers ----
// Y[t, :] = half(rmsnorm(X[t, :]) * w) when w is given (norm=1), else half(X[t, :]).
// grid: T threadgroups x 256 threads.
struct RowsParams {
  uint D;
  uint norm;
  float eps;
};
[[kernel]] void rows_to_half(const device float* X [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                             device half* Y [[buffer(2)]], constant RowsParams& p [[buffer(3)]],
                             uint t [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                             uint ntg [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const device float* x = X + t * p.D;
  float inv = 1.0f;
  if (p.norm) {
    float ss = 0.0f;
    for (uint i = tid; i < p.D; i += ntg) ss += x[i] * x[i];
    ss = tg_sum(ss, scratch, tid, ntg);
    inv = metal::precise::rsqrt(ss / float(p.D) + p.eps);
  }
  for (uint i = tid; i < p.D; i += ntg) Y[t * p.D + i] = half(p.norm ? x[i] * inv * float(w[i]) : x[i]);
}
