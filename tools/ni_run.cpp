// Runs a prompt (token ids) through the engine and generates tokens.
//   ni_run <model_dir> <ids: comma-separated> [n_gen=32] [temp=0] [--dump-logits file]
// Prints generated ids, prompt/decode speeds; optionally dumps the logits of the first
// generated position (raw float32) for comparison against a reference implementation.
#include "metal/gpu.h"
#include "model/qwen35.h"
#include "util/log.h"

#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: ni_run <model_dir> <ids> [n_gen] [temp] [--dump-logits f]\n");
    return 1;
  }
  std::string dir = argv[1];
  std::vector<int> ids;
  {
    std::stringstream ss(argv[2]);
    std::string t;
    while (std::getline(ss, t, ',')) ids.push_back(std::stoi(t));
  }
  int n_gen = argc > 3 ? atoi(argv[3]) : 32;
  float temp = argc > 4 ? float(atof(argv[4])) : 0.0f;
  const char* dump = nullptr;
  for (int i = 1; i < argc - 1; ++i)
    if (!strcmp(argv[i], "--dump-logits")) dump = argv[i + 1];

  Gpu& g = Gpu::get();
  Qwen35 m(dir, 4096);
  SamplingParams sp;
  sp.temperature = temp;
  sp.top_p = 0.95f;
  sp.top_k = 20;
  sp.seed = 42;

  int n = int(ids.size());
  for (int i = 0; i < n; ++i) m.tokens()[i] = ids[i];

  // Prompt: one command buffer per 32 tokens (T=1 path for now; batched prefill comes later).
  double t0 = now_ms();
  for (int p = 0; p < n; p += 32) {
    auto e = g.begin();
    for (int q = p; q < std::min(n, p + 32); ++q) m.encode_step(*e, q, sp, 0, q == n - 1);
    e->commit_and_wait();
  }
  double t_prompt = now_ms() - t0;
  if (dump) {
    FILE* f = fopen(dump, "wb");
    fwrite(m.logits(), 4, m.config().vocab, f);
    fclose(f);
  }
  std::vector<int> out;
  out.push_back(m.tokens()[n]);

  // Decode: keep up to 2 command buffers in flight; the GPU writes each sampled id into the
  // token buffer, which the next step's embedding reads, so the CPU never waits per token.
  double t1 = now_ms();
  std::deque<std::unique_ptr<Encoder>> inflight;
  int pos = n;
  int done = 1;
  bool stop = false;
  for (int s = 1; s < n_gen && !stop; ++s) {
    auto e = g.begin();
    m.encode_step(*e, pos, sp, uint32_t(s), true);
    e->commit();
    inflight.push_back(std::move(e));
    ++pos;
    while (inflight.size() > 2 || (s == n_gen - 1 && !inflight.empty())) {
      inflight.front()->wait();
      inflight.pop_front();
      int tok = m.tokens()[n + done];
      out.push_back(tok);
      ++done;
      for (int eos : m.config().eos)
        if (tok == eos) stop = true;
    }
  }
  while (!inflight.empty()) {
    inflight.front()->wait();
    inflight.pop_front();
    out.push_back(m.tokens()[n + done]);
    ++done;
  }
  double t_dec = now_ms() - t1;
  printf("ids:");
  for (int t : out) printf(" %d", t);
  printf("\n");
  fprintf(stderr, "prompt %d tok in %.0f ms (%.1f tok/s); decode %zu tok in %.0f ms (%.1f tok/s)\n", n, t_prompt,
          n / (t_prompt / 1000), out.size() - 1, t_dec, (out.size() - 1) / (t_dec / 1000));
  return 0;
}
