#include <polynomial/polynomial.hpp>

__global__ void polynomial_expansion(float* __restrict__ input, float const* __restrict__ coeffs, std::size_t const degree, std::size_t size)
{
    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size)
        return;

    float x = input[index];
    float res = 0;
    float power = 1;


    for (size_t i = 0; i <= degree; i++)
    {
        res += x * coeffs[i];
        power *= x;
    }

    input[index] = res;
}


