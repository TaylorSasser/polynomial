#include <iostream>
#include <random>

#include <polynomial/polynomial.cuh>



int main(int argc, char* argv[])
{

    std::size_t len = std::numeric_limits<int>::max() - 10;
    std::size_t deg = 10;

    std::vector<float> host_values;
    std::vector<float> host_coeffs;

    host_coeffs.reserve(len);
    host_values.reserve(deg);

    std::random_device dev{};
    std::mt19937 rng(dev());

    std::uniform_real_distribution<float> coeff_dist(-10, 10);
    std::uniform_real_distribution<float> value_dist(-1024, 1024);

    for (int i = 0; i < len; i++)
        host_values.emplace_back(value_dist(rng));
    for (int i = 0; i < deg; i++)
        host_coeffs.emplace_back(coeff_dist(rng));

    float* dev_values = nullptr;
    float* dev_coeffs = nullptr;

    cudaMalloc(&dev_values, sizeof(float) * len);
    cudaMalloc(&dev_coeffs, sizeof(float) * deg);

    cudaMemcpy(dev_coeffs, host_coeffs.data(), sizeof(float) * deg, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_values, host_values.data(), sizeof(float) * len, cudaMemcpyHostToDevice);


    std::size_t block_dimension = 512;
    std::size_t grid_dimension = (len + block_dimension - 1) / block_dimension;
    std::cout << "Dimension: " << block_dimension << '\n';
    std::cout << "Grid Size: " << grid_dimension << '\n';

    polynomial_expansion<<<grid_dimension,block_dimension>>>(dev_values, dev_coeffs, deg, len);


    cudaMemcpy(host_values.data(), dev_values, sizeof(float) * len, cudaMemcpyDeviceToHost);




    std::cout << "Hello, World!" << std::endl;
    return 0;
}
