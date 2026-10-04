cmake_minimum_required(VERSION 3.15)
project(PyFastQuant VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 REQUIRED)
find_package(OpenMP)

# -----------------------------------------------------------------------
# Compiler-specific flags. MSVC and GCC/Clang use *different* flag
# spellings to enable the same AVX2 hardware feature, which is why a
# single -mavx2 flag (GCC/Clang-only) breaks on MSVC and vice versa --
# this is the bug the original single-branch CMakeLists.txt had.
# -----------------------------------------------------------------------
if(MSVC)
    add_compile_options(/O2 /EHsc /arch:AVX2 /bigobj /MP)
    if(OpenMP_CXX_FOUND)
        add_compile_options(/openmp)
    endif()
else()
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f" AVX512_SUPPORTED)
    check_cxx_compiler_flag("-mavx2" AVX2_SUPPORTED)

    add_compile_options(-O3 -Wall -Wextra)

    if(AVX512_SUPPORTED)
        add_compile_options(-mavx512f -mavx512cd -mavx512dq -mavx512bw -mfma)
        message(STATUS "PyFastQuant: building with AVX-512")
    elseif(AVX2_SUPPORTED)
        add_compile_options(-mavx2 -mfma -mbmi2)
        message(STATUS "PyFastQuant: building with AVX2 + FMA")
    else()
        message(STATUS "PyFastQuant: no AVX2/AVX-512 detected, using scalar fallback path")
    endif()
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${pybind11_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${NumPy_INCLUDE_DIRS}
)

set(ENGINE_SOURCES
    src/engine/monte_carlo.cpp
    src/engine/black_scholes.cpp
    src/engine/linear_algebra.cpp
)

add_library(pyfastquant_engine STATIC ${ENGINE_SOURCES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(pyfastquant_engine PUBLIC OpenMP::OpenMP_CXX)
endif()

pybind11_add_module(_core src/bindings/pybind_wrapper.cpp)
target_link_libraries(_core PRIVATE pyfastquant_engine)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# So `import pyfastquant._core` finds the compiled module next to the
# rest of the Python package during `pip install -e .`
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
)
