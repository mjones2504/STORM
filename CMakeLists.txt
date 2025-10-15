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
