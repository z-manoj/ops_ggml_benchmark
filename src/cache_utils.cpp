#include "cache_utils.h"

#if COLD_CACHE
#include <string>
#include <fstream>
#include <filesystem>
#include <vector>
#include <emmintrin.h>

#define CACHE_LINE_SIZE 64

static size_t read_cache_size(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return 0;
  }
  std::string size_str;
  file >> size_str;
  file.close();

  size_t multiplier = 1;
  if (!size_str.empty()) {
    if (size_str.back() == 'K') {
      multiplier = 1024;
    }
    else if (size_str.back() == 'M') {
      multiplier = 1024 * 1024;
    }
    if (size_str.back() == 'K' || size_str.back() == 'M') {
      size_str.pop_back();
    }
    return std::stoul(size_str) * multiplier;
  }
  return 0;
}

size_t get_cache_size() {
  size_t cache_size = 0;

  std::filesystem::path cache_path = "/sys/devices/system/cpu/cpu0/cache";

  if (std::filesystem::exists(cache_path)) {
    for (const auto &index : std::filesystem::directory_iterator(cache_path)) {
      if (index.path().filename().string().find("index") == 0) {
        std::string size_path = index.path().string() + "/size";

        size_t size_in_bytes = read_cache_size(size_path);
        cache_size += size_in_bytes;
      }
    }
  }
  return cache_size;
}

void flush_cache(size_t cache_size) {
  if (cache_size == 0) return;
  // Pre-calculate to avoid runtime variability
  size_t buffer_size = cache_size * 2;

  #pragma omp parallel
  {
    static thread_local std::vector<char> tls_buffer(buffer_size);
    if (tls_buffer.size() != buffer_size) {
        tls_buffer.resize(buffer_size);
    }

    char *buffer = tls_buffer.data();

    // Pollute cache lines - simple sequential write
    for (size_t i = 0; i < buffer_size; i += CACHE_LINE_SIZE) {
      buffer[i] = (char)(i & 0xFF);
    }

    // Prevent optimization
    asm volatile("" : : "r"(buffer), "r"(buffer_size) : "memory");

    // Flush cache
    for (size_t i = 0; i < buffer_size; i += CACHE_LINE_SIZE) {
      _mm_clflush(&buffer[i]);
    }
    _mm_mfence();
  }
}
#endif
