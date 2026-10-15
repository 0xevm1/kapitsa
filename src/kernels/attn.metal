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

// ---- Prefill attention: causal flash attention with simdgroup matrices ----
// One threadgroup per (block of 4 queries, kv head); its 4 simdgroups each own one query and the
// G=8 query heads that share the kv head (8 rows of an 8x8 fragment), so K/V tiles are loaded once
// for 32 rows and every row in a simdgroup has the same causal limit.
// q: [T, Hq, D] fp32 (normed + roped); kc/vc: [Hkv, max_seq, D] half; qg: [T, Hq, 2D] (gate half).
// out: [T, Hq*D] = softmax(q k^T * scale) v * sigmoid(gate).
struct AttnPrefParams {
  uint T;
  uint Hq;
  uint Hkv;
  uint pos0;     // absolute position of query 0
  uint max_seq;
  float scale;
};

constant constexpr uint kQB = 4;    // queries per threadgroup
constant constexpr uint kKB = 16;   // keys per tile
constant constexpr uint kQLD = kHD + 8;

[[kernel]] void attn_prefill(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                             const device half* vc [[buffer(2)]], const device float* qg [[buffer(3)]],
                             device float* out [[buffer(4)]], constant AttnPrefParams& p [[buffer(5)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half qs[kQB * 8 * kQLD];   // 32 rows x 264
  threadgroup half kv[kKB * kQLD];       // 16 keys x 264 (K, then V)
  threadgroup float sst[kQB][8 * kKB];   // per-simdgroup S staging (and output staging)
  threadgroup half pst[kQB][8 * kKB];    // per-simdgroup P (probabilities) as half
  threadgroup float diag[kQB][64];
  const uint G = p.Hq / p.Hkv;  // must be 8
  const uint kvh = tg.y;
  const uint t0 = tg.x * kQB;
  const uint qi = t0 + sg;                    // this simdgroup's query index
  const bool qvalid = qi < p.T;
  const uint qpos = p.pos0 + min(qi, p.T - 1);
  const uint last_q = min(t0 + kQB, p.T) - 1;
  const uint kend = p.pos0 + last_q + 1;      // keys needed by the block

  // load Q rows: row r = s*8 + g  -> q[t0+s, kvh*G+g, :]
  for (uint i = tid; i < kQB * 8 * kHD / 4; i += 128) {
    uint r = i / (kHD / 4), c4 = i % (kHD / 4);
    uint s = r / 8, g = r % 8;
    uint t = min(t0 + s, p.T - 1);
    float4 v = *(const device float4*)(q + (t * p.Hq + kvh * G + g) * kHD + c4 * 4);
    *(threadgroup half4*)(qs + r * kQLD + c4 * 4) = half4(v * p.scale);
  }
  for (uint i = lane; i < 64; i += 32) diag[sg][i] = 0.0f;

  simdgroup_float8x8 o[kHD / 8];
  for (uint j = 0; j < kHD / 8; ++j) o[j] = simdgroup_float8x8(0);
  // row stats: lane handles row (lane / 4), cols (lane % 4)*4 .. +3 of the 8x16 S tile
  const uint my_row = lane / 4, my_c0 = (lane % 4) * 4;
  float m_row = -INFINITY, l_row = 0.0f;

  const device half* kb = kc + kvh * p.max_seq * kHD;
  const device half* vb = vc + kvh * p.max_seq * kHD;
  for (uint k0 = 0; k0 < kend; k0 += kKB) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < kKB * kHD / 4; i += 128) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      half4 v = k0 + r < kend ? *(const device half4*)(kb + (k0 + r) * kHD + c4 * 4) : half4(0);
      *(threadgroup half4*)(kv + r * kQLD + c4 * 4) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // S = Q K^T  (8 x 16)
    simdgroup_float8x8 s0 = simdgroup_float8x8(0), s1 = simdgroup_float8x8(0);
    for (uint d = 0; d < kHD; d += 8) {
      simdgroup_half8x8 a, b0, b1;
      simdgroup_load(a, qs + sg * 8 * kQLD + d, kQLD);
      simdgroup_load(b0, kv + 0 * kQLD + d, kQLD, ulong2(0, 0), true);
      simdgroup_load(b1, kv + 8 * kQLD + d, kQLD, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(s0, a, b0, s0);
      simdgroup_multiply_accumulate(s1, a, b1, s1);
    }
    simdgroup_store(s0, sst[sg], kKB);
    simdgroup_store(s1, sst[sg] + 8, kKB);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // online softmax on this lane's 4 values of row my_row
    float sv[4];
    float mx = -INFINITY;
    for (uint c = 0; c < 4; ++c) {
      uint key = k0 + my_c0 + c;
      sv[c] = (key <= qpos && key < kend) ? sst[sg][my_row * kKB + my_c0 + c] : -INFINITY;
      mx = max(mx, sv[c]);
    }
    mx = max(mx, simd_shuffle_xor(mx, 1));
    mx = max(mx, simd_shuffle_xor(mx, 2));
    float m_new = max(m_row, mx);
    float corr = m_new == -INFINITY ? 1.0f : metal::fast::exp(m_row - m_new);
    float ps = 0.0f;
    for (uint c = 0; c < 4; ++c) {
      float e = sv[c] == -INFINITY ? 0.0f : metal::fast::exp(sv[c] - m_new);
      ps += e;
      pst[sg][my_row * kKB + my_c0 + c] = half(e);
    }
    ps += simd_shuffle_xor(ps, 1);
    ps += simd_shuffle_xor(ps, 2);
    l_row = l_row * corr + ps;
    m_row = m_new;
    if (lane % 4 == 0) diag[sg][my_row * 9] = corr;
    // load V into the same tile buffer
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < kKB * kHD / 4; i += 128) {
      uint r = i / (kHD / 4), c4 = i % (kHD / 4);
      half4 v = k0 + r < kend ? *(const device half4*)(vb + (k0 + r) * kHD + c4 * 4) : half4(0);
      *(threadgroup half4*)(kv + r * kQLD + c4 * 4) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // O = diag(corr) O + P V
    simdgroup_float8x8 dg;
    simdgroup_half8x8 p0, p1;
    simdgroup_load(dg, diag[sg], 8);
    simdgroup_load(p0, pst[sg], kKB);
    simdgroup_load(p1, pst[sg] + 8, kKB);
    for (uint j = 0; j < kHD / 8; ++j) {
      simdgroup_half8x8 v0, v1;
      simdgroup_load(v0, kv + 0 * kQLD + j * 8, kQLD);
      simdgroup_load(v1, kv + 8 * kQLD + j * 8, kQLD);
      simdgroup_multiply(o[j], dg, o[j]);
      simdgroup_multiply_accumulate(o[j], p0, v0, o[j]);
      simdgroup_multiply_accumulate(o[j], p1, v1, o[j]);
    }
  }
  // finalize: o / l, times sigmoid(gate)
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // row sums to diag (1/l)
  if (lane % 4 == 0) diag[sg][my_row * 9] = l_row > 0.0f ? 1.0f / l_row : 0.0f;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 dg;
  simdgroup_load(dg, diag[sg], 8);
  for (uint j = 0; j < kHD / 8; ++j) {
    simdgroup_multiply(o[j], dg, o[j]);
    simdgroup_store(o[j], sst[sg], 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (qvalid) {
      for (uint e = lane; e < 64; e += 32) {
        uint g = e / 8, d = j * 8 + e % 8;
        uint h = kvh * G + g;
        float gate = qg[(qi * p.Hq + h) * 2 * kHD + kHD + d];
        out[qi * p.Hq * kHD + h * kHD + d] = sst[sg][e] * sigmoid(gate);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}
