/**
 * STORM Python Bindings - Simplified Version
 * 
 * This file provides Python bindings for the STORM system.
 * Simplified to avoid complex PyTorch module binding issues.
 */

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Include our header-only STORM implementations
#include "storm_core.h"

namespace py = pybind11;

/**
 * Python module definition for STORM
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "STORM - Synchronous Transfer Orchestration for RAM Memory";
    
    // Expose the storm namespace
    py::module storm_module = m.def_submodule("storm", "STORM core functionality");
    
    // STORM System
    py::class_<storm::StormSystem>(storm_module, "StormSystem")
        .def(py::init<>())
        .def("initialize", &storm::StormSystem::initialize)
        .def("is_initialized", &storm::StormSystem::isInitialized);
    
    // CUDA Stream
    py::class_<storm::CUDAStream>(storm_module, "CUDAStream")
        .def(py::init<>())
        .def("is_valid", &storm::CUDAStream::isValid)
        .def("synchronize", &storm::CUDAStream::synchronize);
    
    // CUDA Event
    py::class_<storm::CUDAEvent>(storm_module, "CUDAEvent")
        .def(py::init<>())
        .def("is_valid", &storm::CUDAEvent::isValid)
        .def("record", &storm::CUDAEvent::record)
        .def("wait", &storm::CUDAEvent::wait);
    
    // Pinned Memory Buffer
    py::class_<storm::PinnedMemoryBuffer<float>>(storm_module, "PinnedMemoryBuffer")
        .def(py::init<size_t>())
        .def("is_valid", &storm::PinnedMemoryBuffer<float>::isValid)
        .def("size", &storm::PinnedMemoryBuffer<float>::size);
    
    // Simple test function
    storm_module.def("test_function", []() {
        return "STORM extension loaded successfully!";
    });
    
    // Utility functions
    storm_module.def("initialize_cuda", []() {
        cudaError_t error = cudaSetDevice(0);
        if (error != cudaSuccess) {
            throw std::runtime_error("Failed to initialize CUDA");
        }
        return true;
    });
    
    storm_module.def("get_cuda_device_count", []() {
        int count;
        cudaGetDeviceCount(&count);
        return count;
    });
    
    storm_module.def("get_cuda_device_name", [](int device_id) {
        cudaDeviceProp prop;
        cudaError_t error = cudaGetDeviceProperties(&prop, device_id);
        if (error != cudaSuccess) {
            throw std::runtime_error("Failed to get CUDA device properties");
        }
        return std::string(prop.name);
    });
    
    // Version information
    storm_module.attr("__version__") = "1.0.0";
    storm_module.attr("__author__") = "STORM Development Team";
}