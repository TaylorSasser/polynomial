#ifndef POLYNOMIAL_POLYNOMIAL_HPP
#define POLYNOMIAL_POLYNOMIAL_HPP

#include <cuda.h>

__global__ void polynomial_expansion(float* __restrict__ input, float const* __restrict__ coeffs, unsigned int const degree, unsigned int const size);

#endif // POLYNOMIAL_POLYNOMIAL_HPP
