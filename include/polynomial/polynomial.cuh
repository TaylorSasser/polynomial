#ifndef POLYNOMIAL_POLYNOMIAL_CUH
#define POLYNOMIAL_POLYNOMIAL_CUH

#include <cuda.h>


__global__ void polynomial_expansion(float* __restrict__ input, float const* __restrict__ coeffs, std::size_t degree, std::size_t size);

#endif //POLYNOMIAL_POLYNOMIAL_CUH
