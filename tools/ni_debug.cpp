// Prints residual-stream stats after N layers for the first prompt token (debugging aid).
#include "metal/gpu.h"
#include "model/qwen35.h"
#include <cmath>
#include <cstdio>
using namespace ni;
int main(int argc, char** argv) {
  Qwen35 m(argv[1]);
  auto seq = m.new_seq(256);
  int tok = argc > 2 ? atoi(argv[2]) : 760;
  seq->tokens()[0] = tok;
  SamplingParams sp;
  auto e = Gpu::get().begin();
  m.encode_step(*e, *seq, 0, sp, 0, getenv("NI_DEBUG_LAYERS") == nullptr);
  e->commit_and_wait();
  const float* x = m.hidden();
  int D = m.config().hidden, nan = 0;
  double ss = 0;
  for (int i = 0; i < D; ++i) { if (!std::isfinite(x[i])) ++nan; else ss += x[i] * x[i]; }
  printf("nan=%d rms=%.5f x[0..4]= %.5f %.5f %.5f %.5f %.5f\n", nan, std::sqrt(ss / D), x[0], x[1], x[2], x[3], x[4]);
  auto stat = [&](const char* n, int cnt) {
    Buffer b = m.debug_buf(n);
    const float* p = b.as<float>();
    int bad = 0; double s2 = 0, mx = 0;
    for (int i = 0; i < cnt; ++i) { if (!std::isfinite(p[i])) ++bad; else { s2 += p[i]*p[i]; mx = std::max(mx, (double)std::fabs(p[i])); } }
    printf("  %-6s nan=%d rms=%.4g max=%.4g first=%.4g %.4g %.4g\n", n, bad, std::sqrt(s2/cnt), mx, p[0], p[1], p[2]);
  };
  stat("qkv", 8192); stat("zb", 4096); stat("gq", 2048); stat("gv", 4096); stat("gg", 32); stat("gbeta", 32); stat("gout", 4096);
  stat("rlog", 256); stat("ewts", 9); stat("eh", 9*512);
  { const float* h = m.debug_buf("eh").as<float>(); for (int s = 0; s < 9; ++s) { int bad=0; for (int i=0;i<512;++i) if(!std::isfinite(h[s*512+i])) ++bad; printf("  slot %d nan=%d\n", s, bad);} }
  { const unsigned* ix = m.debug_buf("eidx").as<unsigned>(); printf("  eidx:"); for (int i = 0; i < 8; ++i) printf(" %u", ix[i]); printf("\n"); }
  if (!getenv("NI_DEBUG_LAYERS")) {
    const float* l = m.logits();
    int best = 0; for (int i = 1; i < m.config().vocab; ++i) if (l[i] > l[best]) best = i;
    printf("argmax=%d logit=%.4f next_tok=%d\n", best, l[best], seq->tokens()[1]);
  }
}
