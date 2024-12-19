cmake_minimum_required(VERSION 3.10)
project(polynomial VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


add_executable(polynomial
    src/main.cu
)

add_executable(
    polynomial::polynomial ALIAS polynomial
)

target_include_directories(
    polynomial
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(polynomial
    PRIVATE
)


target_compile_features(polynomial
    PRIVATE
        cxx_alignas
        cxx_alignof
        cxx_attributes
        cxx_auto_type
        cxx_constexpr
        cxx_defaulted_functions
        cxx_deleted_functions
        cxx_final
        cxx_lambdas
        cxx_noexcept
        cxx_override
        cxx_range_for
        cxx_rvalue_references
        cxx_static_assert
        cxx_strong_enums
        cxx_trailing_return_types
        cxx_unicode_literals
        cxx_user_literals
        cxx_variable_templates
        cxx_variadic_macros
)

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/polynomial_utils)

install(
    TARGETS polynomial
    EXPORT polynomial_targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/polynomial_config_version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES  DESTINATION ${INSTALL_CONFIGDIR})

export(
        EXPORT polynomial_targets
        FILE ${CMAKE_CURRENT_BINARY_DIR}/polynomial_targets.cmake
        NAMESPACE polynomial::
)

export(PACKAGE polynomial)