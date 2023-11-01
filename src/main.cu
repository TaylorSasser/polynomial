#include <polynomial/polynomial.hpp>
#include <iostream>
#include <random>
#include <iostream>
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


int main(int argc, char* argv[])
{
    if (argc != 3)
        PROGRAM_EXIT("Error: Bad command line parameters\nUsage: ./polynomial <num> <deg>\nEx ./polynomial 10000 50");
    auto const len = (read_cli_argument<std::size_t>(argv[1], "Unable to convert \"%s\" to std::size_t\n") / streams) * streams;
    auto const deg = read_cli_argument<std::size_t>(argv[2], "Unable to convert \"%s\" to std::size_t\n") + 1;

    std::size_t stream_data_len = len / streams;
    std::size_t block_dimension = 512;
    std::size_t grid_dimension = (len + block_dimension - 1) / block_dimension;

    float* host_values = nullptr;
    float* host_coeffs = nullptr;

    if (auto err = cudaMallocHost(&host_values, sizeof(float) * len); err != cudaSuccess)
        PROGRAM_EXIT("Error: %s %s\n", cudaGetErrorName(err), cudaGetErrorString(err));

    if (auto err = cudaMallocHost(&host_coeffs, sizeof(float) * deg); err != cudaSuccess)
        PROGRAM_EXIT("Error: %s %s\n", cudaGetErrorName(err), cudaGetErrorString(err));

    std::fill(host_values + 0, host_values + len, 1);
    std::fill(host_coeffs + 0, host_coeffs + deg, 1);

    float* dev_values = nullptr;
    float* dev_coeffs = nullptr;

    cudaStream_t coeffs_stream;
    cudaStream_t values_stream[streams];
    cudaEvent_t beg_poly[streams];
    cudaEvent_t end_poly[streams];

    std::cout << "Streams: " << streams << '\n';
    std::cout << "Dimension: " << block_dimension << '\n';
    std::cout << "Grid Size: " << grid_dimension << '\n';


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

    auto beg = std::chrono::high_resolution_clock::now();

    if (auto err = cudaMemcpyAsync(dev_coeffs, host_coeffs, deg * sizeof(float), cudaMemcpyHostToDevice, coeffs_stream); err != cudaSuccess)
        PROGRAM_EXIT("Coeffs Stream: %s %s\n ", cudaGetErrorName(err), cudaGetErrorString(err));

    for (std::size_t i = 0; i < streams; i++)
    {
        std::size_t offset = i * stream_data_len;
        if (auto err = cudaMemcpyAsync(dev_values + offset, host_values + offset, stream_data_len * sizeof(float), cudaMemcpyHostToDevice, values_stream[i]); err != cudaSuccess)
            PROGRAM_EXIT("Values Stream: %s %s\n ", cudaGetErrorName(err), cudaGetErrorString(err));

    }

    //We need to wait before the coeff's are done first.
    cudaStreamSynchronize(coeffs_stream);

    for (std::size_t i = 0; i < streams; i++)
    {
        std::size_t offset = i * stream_data_len;
        cudaEventRecord(beg_poly[i], values_stream[i]);
        polynomial_expansion<<<grid_dimension / streams, block_dimension, 0, values_stream[i]>>>(dev_values + offset, dev_coeffs, deg, stream_data_len);
        cudaEventRecord(end_poly[i], values_stream[i]);
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

    for (std::size_t i = 0; i < len; i++)
    {
        if (fabs(host_values[i] - static_cast<float>(deg)) > 0.01)
            std::cout << "host_values[" << i << "] should be " << deg << " not " << host_values[i] << '\n';
    }

    std::cout << std::setprecision(16);
    float total_time = 0;
    cudaEventElapsedTime(&total_time, beg_poly[streams - 1], end_poly[streams - 1]);
    total_time /= 1e3;

    double giga_flops = static_cast<double>(2 * (deg + 1) * len) / total_time / 1e9;


    std::cout << "Application Seconds: " << std::chrono::duration_cast<std::chrono::seconds>(end - beg).count() << '\n';
    std::cout << "Kernel Seconds: " << total_time << '\n';
    std::cout << "GFlops / Second: " << giga_flops << '\n';

    return 0;
}
