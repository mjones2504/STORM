cmake_minimum_required(VERSION 3.18)
project(STORM)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find PyTorch
find_package(Torch REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find NVIDIA profiling tools (optional)
find_package(NVTX QUIET)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})

# CUDA settings
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++17)

# Add executable
add_executable(storm main.cpp)

# Link libraries
target_link_libraries(storm ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})

# Add NVTX if available
if(NVTX_FOUND)
    target_link_libraries(storm ${NVTX_LIBRARIES})
    target_compile_definitions(storm PRIVATE NVTX_ENABLED)
endif()

# Set properties for PyTorch
set_property(TARGET storm PROPERTY CXX_STANDARD 17)

# Add PyTorch extension for STORM with ANCF
pybind11_add_module(storm_cuda storm_bindings.cpp storm_ancf_bindings.cpp)

# Add CUDA source files
set_property(SOURCE storm_ancf_kernels.cu PROPERTY LANGUAGE CUDA)

# Link ANCF CUDA kernels
target_link_libraries(storm_cuda PRIVATE ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})

# Set CUDA properties for ANCF kernels
set_target_properties(storm_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Compile CUDA kernels
target_compile_definitions(storm_cuda PRIVATE TORCH_EXTENSION_NAME=storm_cuda)
target_compile_definitions(storm_cuda PRIVATE ANCF_ENABLED)

# Add CUB library for ANCF kernels (if available)
find_path(CUB_INCLUDE_DIR cub/cub.cuh
    HINTS /usr/local/cuda/include
          /usr/cuda/include
          ${CUDA_TOOLKIT_ROOT_DIR}/include
)

if(CUB_INCLUDE_DIR)
    target_include_directories(storm_cuda PRIVATE ${CUB_INCLUDE_DIR})
    target_compile_definitions(storm_cuda PRIVATE CUB_AVAILABLE)
endif()

# Set properties for ANCF extension
set_property(TARGET storm_cuda PROPERTY CXX_STANDARD 17)