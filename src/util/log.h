#pragma once
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace ni {

inline double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

inline int& log_level() {
  static int lvl = getenv("NI_LOG") ? atoi(getenv("NI_LOG")) : 1;
  return lvl;
}

[[noreturn]] inline void fatal(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "[nace-infer] fatal: ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  abort();
}

inline void logi(const char* fmt, ...) {
  if (log_level() < 1) return;
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "[nace-infer] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

inline void logd(const char* fmt, ...) {
  if (log_level() < 2) return;
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "[nace-infer] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

} // namespace ni
