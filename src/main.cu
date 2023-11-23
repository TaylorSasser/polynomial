#include <polynomial/polynomial.hpp>
#include <random>
#include <charconv>
#include <cstdio>
#include <string>
#include <algorithm>

constexpr static std::size_t const streams_per_device{4};

#define PROGRAM_EXIT(...) do{ std::fprintf(stderr,__VA_ARGS__); exit(1); } while(0)


struct cuda_device_args
{
public:
    float* dev_coeffs_;
    float* dev_values_;
    cudaStream_t coeff_stream_;
    cudaStream_t value_stream_[streams_per_device];
};

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

std::int32_t div_up(std::int32_t x, std::int32_t constant)
{
    return (x + constant - 1) / constant;
}


void launch_multiple_devices(float* input, std::int32_t len, float* coeffs, std::int32_t degree)
{
    std::int32_t device_count = 0;

    if (auto err = cudaGetDeviceCount(&device_count); err != cudaSuccess)
        PROGRAM_EXIT("Error: %s %s\n", cudaGetErrorName(err), cudaGetErrorString(err));

    std::int32_t blocks = streams_per_device * device_count;
    std::int32_t block_size = div_up(len, blocks);

    cuda_device_args* args = new cuda_device_args[device_count];

    for (std::int32_t i = 0; i < device_count; i++)
    {
        cudaSetDevice(i);
        cuda_device_args current = args[i];

        cudaStream_t coeff_alloc_stream;
        cudaStream_t value_alloc_stream;

        cudaStreamCreate(&coeff_alloc_stream);
        cudaStreamCreate(&value_alloc_stream);

        cudaMallocAsync(&current.dev_coeffs_, sizeof(float) * degree, coeff_alloc_stream);
        cudaMallocAsync(&current.dev_values_, sizeof(float) * div_up(len, device_count), value_alloc_stream);

        cudaStreamSynchronize(coeff_alloc_stream);
        cudaStreamSynchronize(value_alloc_stream);

        cudaStreamCreate(&current.coeff_stream_);
        cudaMemcpyAsync(current.dev_coeffs_, coeffs, sizeof(float) * degree, cudaMemcpyHostToDevice, current.coeff_stream_);

        for (std::int32_t j = 0; j < blocks; j++)
        {
            cudaStreamCreate(current.value_stream_ + j);
            std::int32_t block_beg = (i * streams_per_device + j) * block_size;
            std::int32_t block_len = std::min(block_beg + block_size, len) - block_beg;

            cudaMemcpyAsync(
                current.dev_values_ + (block_size * j),
                input + block_beg,
                sizeof(float) * block_len,
                cudaMemcpyHostToDevice,
                current.value_stream_[j]
            );
        }

        for (std::int32_t j = 0; j < blocks; j++)
        {
            dim3 grid_dim(div_up(len, 1024));
            dim3 block_dim(1024);
            std::int32_t block_beg = (i * streams_per_device + j) * block_size;
            std::int32_t block_len = std::min(block_beg + block_size, len) - block_beg;

            polynomial_expansion<<<grid_dim, block_dim, 1024 * sizeof(float), current.value_stream_[j]>>>(current.dev_values_ + block_beg, current.dev_coeffs_, degree, block_len);
        }


        for (std::int32_t j = 0; j < blocks; j++)
        {
            std::int32_t block_beg = (i * streams_per_device + j) * block_size;
            std::int32_t block_len = std::min(block_beg + block_size, len) - block_beg;

            cudaMemcpyAsync(input + block_beg, current.dev_values_ + (block_size * j), block_len * sizeof(float), cudaMemcpyDeviceToHost, current.value_stream_[j]);
        }

        for (std::int32_t j = 0; j < blocks; j++)
        {
            cudaStreamSynchronize(current.value_stream_[j]);
        }
    }
}


int main(int argc, char* argv[])
{
    if (argc != 3)
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


    launch_multiple_devices(host_values, len, host_coeffs, deg);

    return 0;
}
