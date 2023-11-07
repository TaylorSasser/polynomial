#include <polynomial/polynomial.hpp>

constexpr static unsigned int shared_size = 1024;

__global__ void polynomial_expansion(float* __restrict__ input, float const* __restrict__ coeffs, unsigned int const degree, unsigned int const size)
{
    for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < size;
         index += blockDim.x * gridDim.x)
    {
        __shared__ float shared_coeffs[shared_size];
        float        x     = input[index];
        float        res   = 0;
        float        power = 1;

        unsigned int batch = 0;
        for (; batch + (shared_size - 1) <= degree; batch += shared_size)
        {
            if (batch + threadIdx.x < shared_size)
            {
                shared_coeffs[threadIdx.x] = coeffs[batch + threadIdx.x];
            }

            __syncthreads();

            for (unsigned int i = 0; i < shared_size; i++)
            {
                res += x * shared_coeffs[batch + i];
                power *= x;
            }

            __syncthreads();
        }


        if (batch + threadIdx.x <= degree)
        {
            shared_coeffs[threadIdx.x] = coeffs[batch + threadIdx.x];
        }

        __syncthreads();

        for (; batch <= degree; batch++)
        {
            res += x * shared_coeffs[batch];
            power *= x;
        }

        __syncthreads();

        input[index] = res;
    }
}


