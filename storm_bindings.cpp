/**
 * STORM Python Bindings - Header-Only Implementation
 * 
 * This file provides Python bindings for the STORM system.
 * All implementations are in the header files for simplicity.
 */

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Include our header-only STORM implementations
#include "storm_core.h"
#include "storm_autograd.h"
#include "storm_orchestration.h"
#include "storm_profiling.h"

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
    
    // STORM Model
    py::class_<storm::StormModel>(storm_module, "StormModel")
        .def(py::init<int, int, int>())
        .def("forward", &storm::StormModel::forward);
    
    // STORM Trainer
    py::class_<storm::StormTrainer>(storm_module, "StormTrainer")
        .def(py::init<int, int, int, double>())
        .def("train_step", &storm::StormTrainer::trainStep);
    
    // STORM Orchestrator
    py::class_<storm::StormOrchestrator>(storm_module, "StormOrchestrator")
        .def(py::init<>())
        .def("initialize", &storm::StormOrchestrator::initialize)
        .def("orchestrated_forward", &storm::StormOrchestrator::orchestratedForward)
        .def("orchestrated_backward", &storm::StormOrchestrator::orchestratedBackward);
    
    // STORM Profiler
    py::class_<storm::StormProfiler>(storm_module, "StormProfiler")
        .def(py::init<>())
        .def("start_profiling", &storm::StormProfiler::startProfiling)
        .def("stop_profiling", &storm::StormProfiler::stopProfiling);
    
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
    
    // Version information
    storm_module.attr("__version__") = "1.0.0";
    storm_module.attr("__author__") = "STORM Development Team";
}
