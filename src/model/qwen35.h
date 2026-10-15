// Qwen3.5 / 3.6 / 3.8 hybrid model (Gated DeltaNet + gated full attention, 3:1), dense or MoE,
// MLX affine-quantized safetensors.
#pragma once
#include "loader/safetensors.h"
#include "metal/gpu.h"

#include <string>
#include <vector>

namespace ni {

struct QLinear {
  const Tensor* w = nullptr;
  const Tensor* s = nullptr;
  const Tensor* b = nullptr;
  uint32_t N = 0, K = 0;
  int bits = 4, gs = 64;
  explicit operator bool() const { return w != nullptr; }
};

struct Qwen35Config {
  int hidden = 0, layers = 0, vocab = 0;
  std::vector<bool> is_linear;
  // Gated DeltaNet
  int lin_hk = 0, lin_hv = 0, lin_dk = 0, lin_dv = 0, conv_k = 4;
  // full attention
  int hq = 0, hkv = 0, head_dim = 0, rot_dims = 0;
  float theta = 1e7f, eps = 1e-6f;
  // MLP
  bool moe = false;
  int experts = 0, top_k = 0, moe_I = 0, shared_I = 0, dense_I = 0;
  bool norm_topk = true;
  std::vector<int> eos;
  int group_size = 64;
};

struct Layer {
  bool linear = false;
  const Tensor* in_norm = nullptr;
  const Tensor* post_norm = nullptr;
  // GDN
  QLinear qkv, z, a, b, out;
  const Tensor *conv_w = nullptr, *A_log = nullptr, *dt_bias = nullptr, *gnorm = nullptr;
  // attention
  QLinear q, k, v, o;
  const Tensor *q_norm = nullptr, *k_norm = nullptr;
  // MoE
  QLinear router, shared_gate;
  QLinear eg, eu, ed;          // stacked experts [E, ...]
  QLinear sg, su, sd;          // shared expert
  // dense MLP
  QLinear mg, mu, md;
};

struct SamplingParams {
  float temperature = 0.0f;
  float top_p = 1.0f;
  int top_k = 20;
  uint32_t seed = 0;
};

class Qwen35 {
 public:
  explicit Qwen35(const std::string& model_dir, int max_seq = 32768, int prefill_chunk = 512);
  const Qwen35Config& config() const { return cfg_; }

  // Clears recurrent state (KV entries are overwritten by position, so they need no reset).
  void reset();
  // Encodes one decode step for the token at tokens[pos]. With `sample`, computes logits and
  // writes the sampled id to tokens[pos+1]; without, only advances the state (prompt tokens).
  void encode_step(Encoder& e, int pos, const SamplingParams& sp, uint32_t step, bool sample = true);
  // Encodes T prompt tokens tokens[pos0 .. pos0+T) as one batch (T <= prefill_chunk).
  // With `sample`, also samples tokens[pos0+T] from the last position.
  void encode_prefill(Encoder& e, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample);
  int prefill_chunk() const { return tmax_; }
  int* tokens() { return tok_.as<int>(); }
  float* logits() { return logits_.as<float>(); }
  // Debug: copy of the residual stream after the last step.
  const float* hidden() { return x_.as<float>(); }
  Buffer debug_buf(const std::string& name);

 private:
  void load_config(const std::string& dir);
  QLinear qlin(const std::string& name, bool required = true);
  void alloc_state();

  void enc_qmv(Encoder& e, const QLinear& L, const Buffer& x, const Buffer& y, const Buffer* resid,
               const Tensor* norm);
  struct Seg {
    const QLinear* L;
    Buffer y;
  };
  void enc_qmv_multi(Encoder& e, std::vector<Seg> segs, const Buffer& x, const Tensor* norm);
  void enc_gdn(Encoder& e, int li, const Layer& L);
  void enc_attn(Encoder& e, int li, const Layer& L, int pos);
  void enc_moe(Encoder& e, const Layer& L);
  // prefill
  void enc_rows_half(Encoder& e, const Buffer& x, const Tensor* norm, const Buffer& y, int T, int D);
  void enc_qmm(Encoder& e, const QLinear& L, const Buffer& xh, const Buffer& y, int M, bool add);
  void enc_qmm_moe(Encoder& e, const QLinear& L, const Buffer& xh, const Buffer& y, const Buffer& xrows,
                   const Buffer& yrows, int P, int bm);
  void enc_sample(Encoder& e, const Buffer& xrow, int out_index, const SamplingParams& sp, uint32_t step);
  void enc_moe_prefill(Encoder& e, const Layer& L, int T);

  std::string dir_;
  std::string prefix_;
  Qwen35Config cfg_;
  WeightStore ws_;
  std::vector<Layer> layers_;
  const Tensor* final_norm_ = nullptr;
  QLinear embed_, lm_head_;
  int max_seq_;
  int tmax_;

  // state
  Buffer tok_, x_, logits_;
  Buffer qkv_, zb_, ab_, bb_, gq_, gk_, gv_, gg_, gbeta_, gout_;
  Buffer qg_, kr_, vr_, qo_, attn_out_, part_o_, part_ml_;
  Buffer rlog_, slog_, eidx_, ewts_, eh_;
  Buffer cand_v_, cand_i_, sprob_;
  // prefill-only
  Buffer xh_, hh_, ridx_, rwts_, spair_, stok_, tiles_, ntiles_, hg_, hu_, hhalf_, yd_, sgb_, sub_, ys_;
  std::vector<Buffer> conv_state_, rec_state_, kc_, vc_;
};

} // namespace ni
