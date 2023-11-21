#include <polynomial/polynomial.hpp>
#include <iostream>
#include <random>
#include <charconv>
#include <cstdio>
#include <string>
#include <algorithm>
#include <chrono>
#include <iomanip>



constexpr static std::size_t const streams{8};

#define PROGRAM_EXIT(...) do{ std::fprintf(stderr,__VA_ARGS__); exit(1); } while(0)

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



int main(int argc, char* argv[])
{
    if (argc != 3)
        PROGRAM_EXIT("Error: Bad command line parameters\nUsage: ./polynomial <num> <deg>\nEx ./polynomial 3500000000 30000");
    auto const len = (read_cli_argument<std::size_t>(argv[1], "Unable to convert \"%s\" to std::size_t\n"));
    auto const deg = read_cli_argument<std::size_t>(argv[2], "Unable to convert \"%s\" to std::size_t\n") + 1;

    std::size_t stream_data_len = len / streams;

    float* host_values = nullptr;
    float* host_coeffs = nullptr;

    if (auto err = cudaMallocHost(&host_values, sizeof(float) * len); err != cudaSuccess)
        PROGRAM_EXIT("Error: %s %s\n", cudaGetErrorName(err), cudaGetErrorString(err));

    if (auto err = cudaMallocHost(&host_coeffs, sizeof(float) * deg); err != cudaSuccess)
        PROGRAM_EXIT("Error: %s %s\n", cudaGetErrorName(err), cudaGetErrorString(err));

    std::fill(host_values + 0, host_values + len, 1);
    std::fill(host_coeffs + 0, host_coeffs + deg, 1);

    int device_count = 0;
    int streaming_multiprocessor_count = 0;

    if (auto err = cudaGetDeviceCount(&device_count); err != cudaSuccess)
        PROGRAM_EXIT("Error: %s %s\n", cudaGetErrorName(err), cudaGetErrorString(err));

    for (int i = 0; i < device_count; i++)
    {
        cudaDeviceProp properties;
        if (auto err = cudaGetDeviceProperties_v2(&properties, i); err != cudaSuccess)
            PROGRAM_EXIT("Error: %s %s\n", cudaGetErrorName(err), cudaGetErrorString(err));

        if (properties.multiProcessorCount > streaming_multiprocessor_count)
        {
            streaming_multiprocessor_count = properties.multiProcessorCount;
            if (auto err = cudaSetDevice(i); err != cudaSuccess)
                PROGRAM_EXIT("Error: %s %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        }
    }

    float* dev_values = nullptr;
    float* dev_coeffs = nullptr;

    cudaStream_t coeffs_stream;
    cudaStream_t values_stream[streams];
    cudaEvent_t beg_poly[streams];
    cudaEvent_t end_poly[streams];

    if (auto err = cudaMalloc(&dev_values, sizeof(float) * len))
        PROGRAM_EXIT("Error: %s %s\n ", cudaGetErrorName(err), cudaGetErrorString(err));
    if (auto err = cudaMalloc(&dev_coeffs, sizeof(float) * deg))
        PROGRAM_EXIT("Error: %s %s\n ", cudaGetErrorName(err), cudaGetErrorString(err));
    if (auto err = cudaStreamCreateWithFlags(&coeffs_stream, cudaStreamNonBlocking); err != cudaSuccess)
        PROGRAM_EXIT("Error: %s %s\n ", cudaGetErrorName(err), cudaGetErrorString(err));

    for (std::size_t i = 0; i < streams; i++)
    {
        if (auto err = cudaStreamCreateWithFlags(values_stream + i, cudaStreamNonBlocking); err != cudaSuccess)
            PROGRAM_EXIT("Error: %s %s\n ", cudaGetErrorName(err), cudaGetErrorString(err));
        cudaEventCreate(beg_poly + i);
        cudaEventCreate(end_poly + i);
    }

    if (auto err = cudaMemcpyAsync(dev_coeffs, host_coeffs, deg * sizeof(float), cudaMemcpyHostToDevice, coeffs_stream); err != cudaSuccess)
        PROGRAM_EXIT("Coeffs Stream: %s %s\n ", cudaGetErrorName(err), cudaGetErrorString(err));

    for (std::size_t i = 0; i < streams; i++)
    {
        std::size_t offset = i * stream_data_len;
        if (auto err = cudaMemcpyAsync(dev_values + offset, host_values + offset, stream_data_len * sizeof(float), cudaMemcpyHostToDevice, values_stream[i]); err != cudaSuccess)
            PROGRAM_EXIT("Values Stream: %s %s\n ", cudaGetErrorName(err), cudaGetErrorString(err));
    }


    cudaStreamSynchronize(coeffs_stream);
    dim3 grid_dim(div_up(len, 1024));
    dim3 block_dim(1024);

    auto beg = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < streams; i++)
    {
        std::size_t offset = i * stream_data_len;
        cudaEventRecord(beg_poly[i], values_stream[i]);
        polynomial_expansion<<<grid_dim, block_dim, 1024 * sizeof(float), values_stream[i]>>>(dev_values + offset, dev_coeffs, deg, stream_data_len);
        cudaEventRecord(end_poly[i], values_stream[i]);
        std::cout << cudaGetErrorString(cudaGetLastError()) << '\n';
    }

    for (std::size_t i = 0; i < streams; i++)
    {
        std::size_t offset = i * stream_data_len;
        if (auto err = cudaMemcpyAsync(host_values + offset, dev_values + offset, stream_data_len * sizeof(float), cudaMemcpyDeviceToHost, values_stream[i]); err != cudaSuccess)
            PROGRAM_EXIT("Values Stream: %s %s\n ", cudaGetErrorName(err), cudaGetErrorString(err));
    }

    for (std::size_t i = 0; i < streams; i++)
    {
        cudaStreamSynchronize(values_stream[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();


    std::cout << std::setprecision(16);
    float kernel_time = 0;
    cudaEventElapsedTime(&kernel_time, beg_poly[0], end_poly[streams - 1]);
    kernel_time /= 1e3;

    double program_time = (end - beg).count() / 1e9;
    double giga_flops = static_cast<double>(3 * (deg + 1) * len) / kernel_time / 1e9;
    double bandwidth = static_cast<float>((2 * len) + deg) * sizeof(float) / (program_time - kernel_time) / 1e9;



    std::cout << "Program Seconds: " << program_time << '\n';
    std::cout << "Kernel Seconds: " << kernel_time << '\n';
    std::cout << "Memory GBs: " << bandwidth << '\n';
    std::cout << "GFlops / Second: " << giga_flops << '\n';

    return 0;
}
