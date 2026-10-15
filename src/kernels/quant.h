// Shared helpers for MLX affine-quantized dot products (included after common.h).

template <int NT>
struct NormCtx {
  float inv[NT];
};

// Per-simdgroup RMS of each input vector (K floats; every lane reads K/32 of them).
template <int NT>
NI_INLINE NormCtx<NT> rms_ctx(const device float* x, uint x_stride, uint K, float eps, uint lane) {
  NormCtx<NT> c;
  for (int t = 0; t < NT; ++t) {
    const device float4* xp = (const device float4*)(x + t * x_stride);
    float ss = 0.0f;
    for (uint i = lane; i < K / 4; i += 32) {
      float4 v = xp[i];
      ss += dot(v, v);
    }
    ss = simd_sum(ss);
    c.inv[t] = metal::precise::rsqrt(ss / float(K) + eps);
  }
  return c;
}

template <int BITS, int VPL, int NT>
struct XChunk {
  float xs[NT][VPL];
  float xsum[NT];
};

template <int BITS, int VPL, int NT, bool NORM>
NI_INLINE void load_x(thread XChunk<BITS, VPL, NT>& c, const device float* x, uint x_stride, uint k0,
                      thread const NormCtx<NT>& nc, const device bfloat* nw) {
  constexpr int PER_WORD = 32 / BITS;
  for (int t = 0; t < NT; ++t) {
    const device float4* xp = (const device float4*)(x + t * x_stride + k0);
    c.xsum[t] = 0.0f;
    for (int i = 0; i < VPL / 4; ++i) {
      float4 v = xp[i];
      if (NORM) {
        float4 w = float4(nw[k0 + i * 4 + 0], nw[k0 + i * 4 + 1], nw[k0 + i * 4 + 2], nw[k0 + i * 4 + 3]);
        v = v * nc.inv[t] * w;
      }
      c.xsum[t] += v.x + v.y + v.z + v.w;
      for (int e = 0; e < 4; ++e) {
        int j = (i * 4 + e) % PER_WORD;
        c.xs[t][i * 4 + e] = v[e] * float(1.0f / float(1u << (BITS * j)));
      }
    }
  }
}

// acc[t] += dot(row chunk, x chunk) for one row; wp points at the lane's first word.
template <int BITS, int VPL, int NT>
NI_INLINE void dot_row(thread const XChunk<BITS, VPL, NT>& c, const device uint32_t* wp, float s, float b,
                       thread float* acc) {
  constexpr int PER_WORD = 32 / BITS;
  constexpr int WORDS = VPL / PER_WORD;
  constexpr uint MASK = (1u << BITS) - 1u;
  uint32_t wv[WORDS];
  if (WORDS == 4) {
    uint4 v = *(const device uint4*)wp;
    wv[0] = v.x; wv[1] = v.y; wv[2] = v.z; wv[3] = v.w;
  } else if (WORDS == 2) {
    uint2 v = *(const device uint2*)wp;
    wv[0] = v.x; wv[1] = v.y;
  } else {
    for (int i = 0; i < WORDS; ++i) wv[i] = wp[i];
  }
  for (int t = 0; t < NT; ++t) {
    float d = 0.0f;
    for (int i = 0; i < WORDS; ++i)
      for (int j = 0; j < PER_WORD; ++j) d += float(wv[i] & (MASK << (BITS * j))) * c.xs[t][i * PER_WORD + j];
    acc[t] += s * d + b * c.xsum[t];
  }
}

