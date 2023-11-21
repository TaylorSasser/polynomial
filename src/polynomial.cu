#include <polynomial/polynomial.hpp>
#include <cstdint>

constexpr static std::uint32_t block_size = 1024;

__global__ void polynomial_expansion(float* __restrict__ input, float const* __restrict__ coeffs, std::int32_t const degree, std::int32_t const size)
{
    __shared__ float shared_coeffs[block_size];

    std::int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    std::int32_t thread_id = threadIdx.x;
    std::int32_t total_batches = (degree + block_size - 1) / block_size;

    if (index < size)
    {
        float        x     = input[index];
        float        res   = 0;
        float        power = 1;

        for (uint32_t batch = 0; batch < total_batches; ++batch)
        {
            std::uint32_t coeff_idx = batch * block_size + thread_id;
            if (coeff_idx < degree)
            {
                shared_coeffs[thread_id] = coeffs[coeff_idx];
            } else
            {
                shared_coeffs[thread_id] = 0;
            }

            __syncthreads();

            for (std::int32_t i = 0; i < block_size; i++)
            {
                res += x * shared_coeffs[i];
                power *= x;
            }

            __syncthreads();
        }

        input[index] = res;
    }
}


