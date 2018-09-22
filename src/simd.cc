#include "simd.h"

#if defined(__AVX__)
#include <immintrin.h>

namespace {

// number of single precision floats per 256-bit AVX vector
constexpr size_t FLOAT_STRIDE_256 = 256 / (8 * sizeof(float));

}

namespace simd {

float dotProduct(const float* a, const float* b, size_t len) {
  // initialize vector of sums with zeros
  auto sum256 = _mm256_setzero_ps();

  // process all vectors with at least 256 bits of data
  for (auto k = 0; k < len / FLOAT_STRIDE_256; k++) {
    auto index = k * FLOAT_STRIDE_256;

    // load source operands
    auto a256 = _mm256_loadu_ps(&a[index]);
    auto b256 = _mm256_loadu_ps(&b[index]);

    // compute sum256 += a256 * b256
    sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(a256, b256));
  }

  // add adjacent pairs: [(0:1), (2:3), (0:1), (2:3), (4:5), (6:7), (4:5), (6:7)]
  sum256 = _mm256_hadd_ps(sum256, sum256);

  // add adjacent pairs: [(0:3), (0:3), (0:3), (0:3), (4:7), (4:7), (4:7), (4:7)]
  sum256 = _mm256_hadd_ps(sum256, sum256);

  // store result into array
  float sumArray[FLOAT_STRIDE_256];
  _mm256_storeu_ps(sumArray, sum256);

  auto sum = sumArray[0] + sumArray[4];

  // process remaining elements
  auto remainder = len % FLOAT_STRIDE_256;
  for (auto i = len - remainder; i < len; i++)
    sum += a[i] * b[i];

  // avoid the penalty of switching from AVX to SSE
  _mm256_zeroupper();

  return sum;
}

}

#else

namespace simd {

float dotProduct(const float* a, const float* b, size_t len) {
  float sum = 0.f;

  for (int i = 0; i < len; i++)
    sum += a[i] * b[i];

  return sum;
}

}

#endif
