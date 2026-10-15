#include "common.h"

// ---- Gated full attention (Qwen3.5/3.6/3.8): q/k RMSNorm, partial RoPE, KV cache write ----
// q_proj output per head is [query (D) | gate (D)]. KV cache: half, [Hkv, max_seq, D].
struct AttnPrepParams {
  uint T;
  uint Hq;
  uint Hkv;
  uint D;        // head dim (256)
  uint rot;      // rotary dims (64)
  uint pos;      // position of the first token
  uint max_seq;
  float eps;
  float theta;
};

// grid: x = Hq + 2*Hkv (q heads, k heads, v heads), y = T; threadgroup = D threads
[[kernel]] void attn_prep(const device float* qg [[buffer(0)]], const device float* kr [[buffer(1)]],
                          const device float* vr [[buffer(2)]], const device bfloat* qn_w [[buffer(3)]],
                          const device bfloat* kn_w [[buffer(4)]], device float* q_out [[buffer(5)]],
                          device half* kc [[buffer(6)]], device half* vc [[buffer(7)]],
                          constant AttnPrepParams& p [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint2 tid2 [[thread_position_in_threadgroup]]) {
  const uint tid = tid2.x;
  threadgroup float scratch[32];
  threadgroup float row[512];
  const uint h = tg.x, t = tg.y, D = p.D;
  const uint pos = p.pos + t;
  if (h >= p.Hq + p.Hkv) {  // value head: plain copy into the cache
    uint hv = h - p.Hq - p.Hkv;
    vc[(hv * p.max_seq + pos) * D + tid] = half(vr[(t * p.Hkv + hv) * D + tid]);
    return;
  }
  const bool is_q = h < p.Hq;
  float x = is_q ? qg[(t * p.Hq + h) * 2 * D + tid] : kr[(t * p.Hkv + (h - p.Hq)) * D + tid];
  float ss = tg_sum(x * x, scratch, tid, D);
  x = x * metal::precise::rsqrt(ss / float(D) + p.eps) * float(is_q ? qn_w[tid] : kn_w[tid]);
  row[tid] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint half_rot = p.rot / 2;
  if (tid < p.rot) {
    uint i = tid % half_rot;
    float inv_freq = metal::precise::pow(p.theta, -2.0f * float(i) / float(p.rot));
    float ang = float(pos) * inv_freq;
    float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
    if (tid < half_rot) x = row[tid] * cs - row[tid + half_rot] * sn;
    else x = row[tid] * cs + row[tid - half_rot] * sn;
  }
  if (is_q) q_out[(t * p.Hq + h) * D + tid] = x;
  else kc[((h - p.Hq) * p.max_seq + pos) * D + tid] = half(x);
}

// ---- Decode attention, split-K over the cache ("flash decoding") ----
// One threadgroup per (chunk, kv head); its G = Hq/Hkv simdgroups are the query heads sharing that
// kv head, so K/V tiles are read from memory once and reused from threadgroup memory.
struct AttnDecParams {
  uint Hq;
  uint Hkv;
  uint L;       // number of valid cache positions (including the current token)
  uint chunk;   // positions per threadgroup
  uint max_seq;
  float scale;
  uint nchunks;
};

constant constexpr uint kTile = 16;
constant constexpr uint kHD = 256;

[[kernel]] void attn_decode_partial(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                    const device half* vc [[buffer(2)]], device float* part_o [[buffer(3)]],
                                    device float* part_ml [[buffer(4)]], constant AttnDecParams& p [[buffer(5)]],
                                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                    uint2 ntg2 [[threads_per_threadgroup]]) {
  const uint ntg = ntg2.x;
  threadgroup half ktile[kTile * kHD];
  threadgroup half vtile[kTile * kHD];
  const uint chunk = tg.x, kvh = tg.y;
  const uint G = p.Hq / p.Hkv;
  const uint hq = kvh * G + sg;
  const uint start = chunk * p.chunk;
  const uint end = min(start + p.chunk, p.L);

  float qv[8], acc[8];
  for (int i = 0; i < 8; ++i) {
    qv[i] = q[hq * kHD + lane * 8 + i] * p.scale;
    acc[i] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  const device half* kbase = kc + kvh * p.max_seq * kHD;
  const device half* vbase = vc + kvh * p.max_seq * kHD;

  for (uint t0 = start; t0 < end; t0 += kTile) {
    uint n = min(kTile, end - t0);
    // cooperative load: n*256 halfs each for K and V, as half4
    for (uint i = tid; i < n * kHD / 4; i += ntg) {
      ((threadgroup half4*)ktile)[i] = ((const device half4*)(kbase + t0 * kHD))[i];
      ((threadgroup half4*)vtile)[i] = ((const device half4*)(vbase + t0 * kHD))[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < G) {
      for (uint j = 0; j < n; ++j) {
        const threadgroup half* kr = ktile + j * kHD + lane * 8;
        float s = 0.0f;
        for (int i = 0; i < 8; ++i) s += qv[i] * float(kr[i]);
        s = simd_sum(s);
        float mn = max(m, s);
        float corr = metal::fast::exp(m - mn);
        float e = metal::fast::exp(s - mn);
        l = l * corr + e;
        const threadgroup half* vr = vtile + j * kHD + lane * 8;
        for (int i = 0; i < 8; ++i) acc[i] = acc[i] * corr + e * float(vr[i]);
        m = mn;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (sg < G) {
    uint base = (hq * p.nchunks + chunk);
    for (int i = 0; i < 8; ++i) part_o[base * kHD + lane * 8 + i] = acc[i];
    if (lane == 0) {
      part_ml[base * 2 + 0] = m;
      part_ml[base * 2 + 1] = l;
    }
  }
}

// grid: Hq threadgroups x 256 threads. out[h*D + d] = softmax-combined o * sigmoid(gate)
[[kernel]] void attn_decode_reduce(const device float* part_o [[buffer(0)]], const device float* part_ml [[buffer(1)]],
                                   const device float* qg [[buffer(2)]], device float* out [[buffer(3)]],
                                   constant AttnDecParams& p [[buffer(4)]], uint h [[threadgroup_position_in_grid]],
                                   uint d [[thread_position_in_threadgroup]]) {
  float M = -INFINITY;
  for (uint c = 0; c < p.nchunks; ++c) M = max(M, part_ml[(h * p.nchunks + c) * 2]);
  float num = 0.0f, den = 0.0f;
  for (uint c = 0; c < p.nchunks; ++c) {
    float mc = part_ml[(h * p.nchunks + c) * 2];
    if (mc == -INFINITY) continue;
    float w = metal::fast::exp(mc - M);
    num += w * part_o[(h * p.nchunks + c) * kHD + d];
    den += w * part_ml[(h * p.nchunks + c) * 2 + 1];
  }
  float gate = qg[h * 2 * kHD + kHD + d];
  out[h * kHD + d] = (num / den) * sigmoid(gate);
}
