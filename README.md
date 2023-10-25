Dependencies
`Catch2 3.0 & Fmt`


Run in project root directory

Create the debug build

```cmake -S . -B build/debug -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE=<path-to-vcpkg.cmake> -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc```

And the release build

```cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=<path-to-vcpkg.cmake> -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc```


To Build

```
cmake --build build/release --target ggm_example -- -j 18
cmake --build build/debug --target ggm_example -- -j 18
```

To run unit tests
```
cmake --build build/debug --target ggm_tree_tests -j 18
./build/debug/tests/ggm_tree_tests --skip-benchmarks -r xml -d yes --order lex
```

To run benchmarks and unit tests
```
cmake --build build/release --target ggm_tree_tests -j 18
./build/release/tests/ggm_tree_tests -r xml -d yes --order lex
```
