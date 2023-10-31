#ifndef POLYNOMIAL_POLYNOMIAL_HPP
#define POLYNOMIAL_POLYNOMIAL_HPP

#include <cuda.h>

constexpr static unsigned long const elements_per_thread = 4;

__global__ void polynomial_expansion(float* __restrict__ input, float const* __restrict__ coeffs, std::size_t const degree, std::size_t const size);

#endif // POLYNOMIAL_POLYNOMIAL_HPP
