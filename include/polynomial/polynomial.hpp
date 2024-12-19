#ifndef POLYNOMIAL_POLYNOMIAL_HPP
#define POLYNOMIAL_POLYNOMIAL_HPP

#include <cstdint>
#include <cuda.h>

__global__ void polynomial_expansion(float* __restrict__ input, float* __restrict__ coeffs, std::int32_t const degree, std::int32_t const size);

#endif // POLYNOMIAL_POLYNOMIAL_HPP
