#pragma once

#include <cstddef>

#ifndef COLD_CACHE
#define COLD_CACHE 0
#endif

#if COLD_CACHE
size_t get_cache_size();
void flush_cache(size_t cache_size);
#endif
