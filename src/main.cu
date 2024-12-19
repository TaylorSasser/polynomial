#include <random>
#include <charconv>
#include <cstdio>
#include <string>
#include <algorithm>
#include <iostream>
#include <cuda.h>
#include <experimental/iterator>

#define PROGRAM_EXIT(...) do{ std::fprintf(stderr,__VA_ARGS__); exit(1); } while(0)


constexpr static std::uint32_t thread_results = 16;
constexpr static std::uint32_t block_cols = 32;
constexpr static std::uint32_t global_memory_load_size = 4;


__device__ void dump_shared_memory(float* shared_coeffs, int total_elements)
{
    printf("Shared Memory Dump (Total Elements: %d):\n", total_elements);

    for (int i = 0; i < total_elements; i++)
    {
        printf("%05.0f ", shared_coeffs[i]);
        if ((i + 1) % block_cols == 0)
        {
            printf("\n");
        }
    }
    printf("\n");
}

__global__ void polynomial_expansion(float* __restrict__ input,
                                     float* __restrict__ coeffs,
                                     std::int32_t const degree,
                                     std::int32_t const size)
{
    __shared__ float shared_coeffs[8192];

    float input_buffer[thread_results] = {0.0};
    float output_buffer[thread_results] = {0.0};

    for (int i = 0; i < (thread_results / global_memory_load_size); i++)
    {
        int index = (blockIdx.x * blockDim.x + threadIdx.x) * global_memory_load_size + (i * gridDim.x * blockDim.x * global_memory_load_size);

        if (index + 4 <= size)
        {
            float4 tmp = reinterpret_cast<const float4 *>(&input[index])[0];
            input_buffer[i * 4 + 0] = tmp.x;
            input_buffer[i * 4 + 1] = tmp.y;
            input_buffer[i * 4 + 2] = tmp.z;
            input_buffer[i * 4 + 3] = tmp.w;
        }
        else
        {
            for (int j = 0; j < 4; ++j)
            {
                int input_offset = index + j;
                float value = (input_offset < size) ? input[input_offset] : 0.0f;
                input_buffer[i * 4 + j] = value;
            }
        }
    }

    __syncthreads();

    uint const row = threadIdx.x / (block_cols / global_memory_load_size);
    uint const col = threadIdx.x % (block_cols / global_memory_load_size);


    // It's possible our degree exceeds 8192, in which case we need to save our results and load
    // The new coeffs in
    for (uint degree_block = 0; degree_block < degree; degree_block += 8192)
    {
        for (uint inner_idx = 0; inner_idx < global_memory_load_size; ++inner_idx)
        {
            int coeff_index = degree_block + (row * 32 + (col + (512 * inner_idx)) * 4);

            if (coeff_index + degree_block + 4 <= degree)
            {
                float4 tmp = reinterpret_cast<const float4 *>(&coeffs[coeff_index])[0];
                shared_coeffs[((col * 64) + (4 * inner_idx + 00)) * 16 + row] = tmp.x;
                shared_coeffs[((col * 64) + (4 * inner_idx + 16)) * 16 + row] = tmp.y;
                shared_coeffs[((col * 64) + (4 * inner_idx + 32)) * 16 + row] = tmp.z;
                shared_coeffs[((col * 64) + (4 * inner_idx + 48)) * 16 + row] = tmp.w;
            }
            else
            {
                for (int j = 0; j < 4; ++j)
                {
                    int coeff_offset = coeff_index + j;
                    float value = (coeff_offset < degree) ? coeffs[coeff_offset] : 0.0f;
                    shared_coeffs[((col * 64) + (4 * inner_idx + (16 * j))) * 16 + row] = value;
                }
            }
        }
        
        __syncthreads();

        for (uint result = 0; result < thread_results; result++)
        {
            float x = input_buffer[result];
            float s = degree_block == 0 ? 0.0 : output_buffer[result];
            for(int k = 255; k >= 0; k--)
            {
                s = s * x + shared_coeffs[(threadIdx.x % block_cols) * 256 + k];
            }
            output_buffer[result] = s;
        };

        __syncthreads();
    }

    for (int i = 0; i < (thread_results / global_memory_load_size); i++)
    {
        int index = (blockIdx.x * blockDim.x + threadIdx.x) * global_memory_load_size + (i * gridDim.x * blockDim.x * global_memory_load_size);
        for (int j = 0; j < 4; j++)
        {
            if (threadIdx.x == 0 && index < size)
            {
                printf("index: %d\n", index);
                input[index + j] = output_buffer[i * 4 + j];
            }
        }
    }
}


template<class T>
auto read_cli_argument(std::string str, char const* error_message)
{
    if constexpr (std::is_arithmetic_v<T>)
    {
        T result{};
        auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), result);
        if (ec != std::errc())
            PROGRAM_EXIT(error_message, str.data());
        return result;
    }
    static_assert("Invalid CLI argument type");
}

__host__ __device__ inline std::int32_t div_up(std::int32_t x, std::int32_t constant)
{
    return (x + constant - 1) / constant;
}


void launch_polynomial_evaluation(float* input, std::int32_t len, float* coeffs, std::int32_t degree)
{
    std::int32_t device_count = 0;

    if (auto err = cudaGetDeviceCount(&device_count); err != cudaSuccess)
        PROGRAM_EXIT("Error: %s %s\n", cudaGetErrorName(err), cudaGetErrorString(err));

    std::int32_t num_blocks = 0;
    for (std::int32_t i = 0; i < device_count; i++)
    {
        cudaSetDevice(i);
        int device_sm_count;
        if (cudaDeviceGetAttribute(&device_sm_count, cudaDevAttrMultiProcessorCount, i) != cudaSuccess)
            PROGRAM_EXIT("Error retrieving SM count for device %d\n", i);

        num_blocks += device_sm_count * 3;
    }

    std::int32_t blocks = div_up(len / 16, num_blocks);

    dim3 grid_dim(blocks);
    dim3 block_dim(32 * 16);

    float* dev_coeffs_ = nullptr;
    float* dev_values_ = nullptr;

    if (cudaMalloc(&dev_coeffs_, sizeof(float) * degree) != cudaSuccess ||
        cudaMalloc(&dev_values_, sizeof(float) * len) != cudaSuccess)
    {
        PROGRAM_EXIT("Error allocating device memory\n");
    }

    if (cudaMemcpy(dev_values_, input, sizeof(float) * len, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_coeffs_, coeffs, sizeof(float) * degree, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        PROGRAM_EXIT("Error copying data to device\n");
    }

    polynomial_expansion<<<grid_dim, block_dim>>>(dev_values_, dev_coeffs_, degree, len);

    if (cudaPeekAtLastError() != cudaSuccess)
    {
        PROGRAM_EXIT("Kernel launch failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    }

    if (cudaMemcpy(input, dev_values_, len * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        std::cout << cudaGetErrorString(cudaGetLastError()) << '\n';
        PROGRAM_EXIT("Error copying data back to host\n");
    }

    cudaFree(dev_coeffs_);
    cudaFree(dev_values_);
}



int main(int argc, char* argv[])
{    if (argc != 3)
         PROGRAM_EXIT("Error: Bad command line parameters\nUsage: ./polynomial <num> <deg>\nEx ./polynomial 3500000000 30000");
     auto const len = (read_cli_argument<std::size_t>(argv[1], "Unable to convert \"%s\" to std::size_t\n"));
     auto const deg = read_cli_argument<std::size_t>(argv[2], "Unable to convert \"%s\" to std::size_t\n") + 1;

     float* host_values = nullptr;
     float* host_coeffs = nullptr;

     if (auto err = cudaMallocHost(&host_values, sizeof(float) * len); err != cudaSuccess)
         PROGRAM_EXIT("Error: %s %s\n", cudaGetErrorName(err), cudaGetErrorString(err));

     if (auto err = cudaMallocHost(&host_coeffs, sizeof(float) * deg); err != cudaSuccess)
         PROGRAM_EXIT("Error: %s %s\n", cudaGetErrorName(err), cudaGetErrorString(err));


     std::fill(host_values + 0, host_values + len, 1);
     std::fill(host_coeffs + 0, host_coeffs + deg, 1);

     launch_polynomial_evaluation(host_values, len, host_coeffs, deg);

     std::copy(host_values, host_values + 512,
               std::experimental::make_ostream_joiner(std::cout, ", "));
     return 0;

}
