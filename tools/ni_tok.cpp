// Tokenizer check: reads texts from stdin separated by "\n\x1e\n" and prints one id list per text.
#include "tokenizer/tokenizer.h"
#include <iostream>
#include <iterator>
#include <string>
int main(int argc, char** argv) {
  ni::Tokenizer t;
  t.load(argv[1]);
  std::string all((std::istreambuf_iterator<char>(std::cin)), {});
  const std::string sep = "\n\x1e\n";
  size_t p = 0;
  while (p <= all.size()) {
    size_t q = all.find(sep, p);
    if (q == std::string::npos) q = all.size();
    auto ids = t.encode(all.substr(p, q - p));
    for (size_t i = 0; i < ids.size(); ++i) printf(i ? ",%d" : "%d", ids[i]);
    printf("\n");
    if (t.decode(ids) != all.substr(p, q - p)) fprintf(stderr, "roundtrip mismatch\n");
    p = q + sep.size();
  }
}
