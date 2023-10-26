#include <polynomial/polynomial.hpp>
#include <iostream>
#include <random>
#include <iostream>
#include <charconv>
#include <cstdio>
#include <algorithm>
#include <chrono>


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
    /*
    if (argc != 3)
        PROGRAM_EXIT("Error: Bad command line parameters\nUsage: ./polynomial <num> <deg>\nEx ./polynomial 10000 50");
    auto const len  = read_cli_argument<std::size_t>(argv[1], "Unable to convert \"%s\" to std::size_t\n");
    auto const deg = read_cli_argument<std::size_t>(argv[2], "Unable to convert \"%s\" to std::size_t\n");
    */
    std::size_t len = 2147483647;
    std::size_t deg = 10000;

    float* host_values = new float[len];
    float* host_coeffs = new float[deg];

    std::cout << "Generating values and coeffs (may take a moment)...\n";


    std::random_device dev{};
    std::mt19937 rng(dev());

    std::uniform_real_distribution<float> coeff_dist(-1, 1);

    std::iota(host_values + 0, host_values + len, 1);
    std::generate(host_coeffs + 0, host_coeffs + deg, [&]() {
          return coeff_dist(rng);
    });


    float* dev_values = nullptr;
    float* dev_coeffs = nullptr;

    if (auto err = cudaMalloc(&dev_values, sizeof(float) * len))
        PROGRAM_EXIT("Error: %s %s\n ", cudaGetErrorName(err), cudaGetErrorString(err));

    if (auto err = cudaMalloc(&dev_coeffs, sizeof(float) * deg))
        PROGRAM_EXIT("Error: %s %s\n ", cudaGetErrorName(err), cudaGetErrorString(err));

    std::size_t block_dimension = 512;
    std::size_t grid_dimension = (len + block_dimension - 1) / block_dimension;
    std::cout << "Dimension: " << block_dimension << '\n';
    std::cout << "Grid Size: " << grid_dimension << '\n';

    auto beg = std::chrono::system_clock::now();

    if (auto err = cudaMemcpy(dev_coeffs, host_coeffs, sizeof(float) * deg, cudaMemcpyHostToDevice))
        PROGRAM_EXIT("Error: %s %s\n ", cudaGetErrorName(err), cudaGetErrorString(err));

    if (auto err = cudaMemcpy(dev_values, host_values, sizeof(float) * len, cudaMemcpyHostToDevice))
        PROGRAM_EXIT("Error: %s %s\n ", cudaGetErrorName(err), cudaGetErrorString(err));

    polynomial_expansion<<<grid_dimension,block_dimension>>>(dev_values, dev_coeffs, deg, len);

    if (auto err = cudaMemcpy(host_values, dev_values, sizeof(float) * len, cudaMemcpyDeviceToHost))
        PROGRAM_EXIT("Error: %s %s\n ", cudaGetErrorName(err), cudaGetErrorString(err));


    auto end = std::chrono::system_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count();

    double total_time = static_cast<double>(time) / 1e9;

    //Each cycle does one fma instruction, and an extra mul for the x^n.
    double flops = static_cast<double>(3 * deg * len) / total_time / 1e9;

    std::cout << "Seconds: " << total_time << '\n';
    std::cout << "GFlops / Second: " << flops << '\n';

    return 0;
}
