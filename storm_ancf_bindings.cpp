#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include <vector>
#include <string>
#include "storm_ancf_encoder.h"
#include "storm_ancf_integration.h"

/**
 * ANCF Python Bindings
 * 
 * This file exposes the ANCF encoding system to Python/PyTorch:
 * - encode_activation(tensor, layer_id): Encode tensor with adaptive dictionary
 * - decode_activation(encoded_data, layer_id): Lossless reconstruction
 * - get_compression_stats(): Report compression ratios and throughput
 * - set_dictionary_size_policy(policy): Configure adaptive sizing strategy
 */

namespace py = pybind11;

/**
 * Python wrapper for ANCFEncoder
 */
class ANCFEncoderWrapper {
private:
    std::unique_ptr<storm::ANCFEncoder> encoder_;
    
public:
    ANCFEncoderWrapper(
        int policy = 1,  // Default to ADAPTIVE
        bool enable_caching = true,
        bool enable_profiling = true,
        float outlier_tolerance = 1e-6f
    ) {
        storm::DictionarySizePolicy enum_policy;
        switch (policy) {
            case 0: enum_policy = storm::DictionarySizePolicy::CONSERVATIVE; break;
            case 1: enum_policy = storm::DictionarySizePolicy::ADAPTIVE; break;
            case 2: enum_policy = storm::DictionarySizePolicy::AGGRESSIVE; break;
            default: enum_policy = storm::DictionarySizePolicy::ADAPTIVE; break;
        }
        encoder_ = std::make_unique<storm::ANCFEncoder>(enum_policy, enable_caching, enable_profiling, outlier_tolerance);
    }
    
    /**
     * Encode activation tensor
     */
    py::dict encode_activation(py::object activation_tensor, int layer_id = 0) {
        // Convert Python tensor to torch::Tensor
        torch::Tensor activation;
        if (py::isinstance<torch::Tensor>(activation_tensor)) {
            activation = activation_tensor.cast<torch::Tensor>();
        } else {
            throw std::runtime_error("Input must be a PyTorch tensor");
        }
        
        // Ensure tensor is on GPU and contiguous
        if (!activation.is_cuda()) {
            activation = activation.cuda();
        }
        if (!activation.is_contiguous()) {
            activation = activation.contiguous();
        }
        
        // Encode using ANCF
        auto encoded_data = encoder_->encodeActivation(activation, layer_id);
        
        // Convert to Python dictionary
        py::dict result;
        result["indices"] = py::cast(encoded_data.indices);
        result["dictionary"] = py::cast(encoded_data.dictionary);
        result["outliers"] = py::cast(encoded_data.outliers);
        result["outlier_positions"] = py::cast(encoded_data.outlier_positions);
        result["dictionary_size"] = encoded_data.dictionary_size;
        result["escape_code"] = encoded_data.escape_code;
        result["original_shape"] = py::cast(encoded_data.original_shape);
        result["compression_ratio"] = encoded_data.compression_ratio;
        result["encode_time_us"] = encoded_data.encode_time.count();
        
        return result;
    }
    
    /**
     * Decode activation tensor
     */
    py::object decode_activation(py::dict encoded_dict, py::object device = py::none()) {
        // Extract encoded data from dictionary
        storm::ANCFEncodedData encoded_data;
        
        encoded_data.indices = encoded_dict["indices"].cast<std::vector<uint8_t>>();
        encoded_data.dictionary = encoded_dict["dictionary"].cast<std::vector<float>>();
        encoded_data.outliers = encoded_dict["outliers"].cast<std::vector<float>>();
        encoded_data.outlier_positions = encoded_dict["outlier_positions"].cast<std::vector<size_t>>();
        encoded_data.dictionary_size = encoded_dict["dictionary_size"].cast<int>();
        encoded_data.escape_code = encoded_dict["escape_code"].cast<int>();
        encoded_data.original_shape = encoded_dict["original_shape"].cast<std::vector<int64_t>>();
        encoded_data.compression_ratio = encoded_dict["compression_ratio"].cast<float>();
        
        // Determine target device
        torch::Device target_device = torch::kCUDA;
        if (!device.is_none()) {
            if (py::isinstance<torch::Device>(device)) {
                target_device = device.cast<torch::Device>();
            } else if (py::isinstance<py::str>(device)) {
                std::string device_str = device.cast<std::string>();
                if (device_str == "cuda" || device_str == "cuda:0") {
                    target_device = torch::kCUDA;
                } else if (device_str == "cpu") {
                    target_device = torch::kCPU;
                } else {
                    throw std::runtime_error("Unsupported device: " + device_str);
                }
            }
        }
        
        // Decode using ANCF
        auto decoded_tensor = encoder_->decodeActivation(encoded_data, target_device);
        
        return py::cast(decoded_tensor);
    }
    
    /**
     * Encode for CPU storage
     */
    py::bytes encode_for_cpu_storage(py::object activation_tensor, int layer_id = 0) {
        torch::Tensor activation = activation_tensor.cast<torch::Tensor>();
        if (!activation.is_cuda()) {
            activation = activation.cuda();
        }
        if (!activation.is_contiguous()) {
            activation = activation.contiguous();
        }
        
        auto compressed_data = encoder_->encodeForCPUStorage(activation, layer_id);
        return py::bytes(reinterpret_cast<const char*>(compressed_data.data()), compressed_data.size());
    }
    
    /**
     * Encode for PCIe transfer
     */
    py::bytes encode_for_pcie_transfer(py::object activation_tensor, int layer_id = 0) {
        torch::Tensor activation = activation_tensor.cast<torch::Tensor>();
        if (!activation.is_cuda()) {
            activation = activation.cuda();
        }
        if (!activation.is_contiguous()) {
            activation = activation.contiguous();
        }
        
        auto compressed_data = encoder_->encodeForPCIeTransfer(activation, layer_id);
        return py::bytes(reinterpret_cast<const char*>(compressed_data.data()), compressed_data.size());
    }
    
    /**
     * Get compression statistics
     */
    std::string get_compression_stats() {
        return encoder_->getCompressionStats();
    }
    
    /**
     * Get average compression ratio
     */
    float get_average_compression_ratio() {
        return encoder_->getAverageCompressionRatio();
    }
    
    /**
     * Reset statistics
     */
    void reset_stats() {
        encoder_->resetStats();
    }
    
    /**
     * Set dictionary policy
     */
    void set_dictionary_policy(int policy) {
        storm::DictionarySizePolicy enum_policy;
        switch (policy) {
            case 0: enum_policy = storm::DictionarySizePolicy::CONSERVATIVE; break;
            case 1: enum_policy = storm::DictionarySizePolicy::ADAPTIVE; break;
            case 2: enum_policy = storm::DictionarySizePolicy::AGGRESSIVE; break;
            default: throw std::runtime_error("Invalid policy value");
        }
        encoder_->setDictionaryPolicy(enum_policy);
    }
    
    /**
     * Set caching enabled
     */
    void set_caching_enabled(bool enabled) {
        encoder_->setCachingEnabled(enabled);
    }
    
    /**
     * Set outlier tolerance
     */
    void set_outlier_tolerance(float tolerance) {
        encoder_->setOutlierTolerance(tolerance);
    }
    
    /**
     * Verify lossless reconstruction
     */
    bool verify_lossless(py::object original_tensor, py::object decoded_tensor) {
        torch::Tensor original = original_tensor.cast<torch::Tensor>();
        torch::Tensor decoded = decoded_tensor.cast<torch::Tensor>();
        
        return storm::ANCFEncoder::verifyLossless(original, decoded);
    }
};

/**
 * Python wrapper for ANCFCPUStorageManager
 */
class ANCFCPUStorageWrapper {
private:
    std::unique_ptr<storm::ANCFCPUStorageManager> storage_manager_;
    
public:
    ANCFCPUStorageWrapper(
        size_t max_storage = 8ULL * 1024 * 1024 * 1024, // 8GB default
        int policy = 1 // ADAPTIVE
    ) {
        storm::DictionarySizePolicy enum_policy;
        switch (policy) {
            case 0: enum_policy = storm::DictionarySizePolicy::CONSERVATIVE; break;
            case 1: enum_policy = storm::DictionarySizePolicy::ADAPTIVE; break;
            case 2: enum_policy = storm::DictionarySizePolicy::AGGRESSIVE; break;
            default: enum_policy = storm::DictionarySizePolicy::ADAPTIVE; break;
        }
        storage_manager_ = std::make_unique<storm::ANCFCPUStorageManager>(max_storage, enum_policy);
    }
    
    /**
     * Store activation
     */
    bool store_activation(py::object activation_tensor, int layer_id) {
        torch::Tensor activation = activation_tensor.cast<torch::Tensor>();
        if (!activation.is_cuda()) {
            activation = activation.cuda();
        }
        if (!activation.is_contiguous()) {
            activation = activation.contiguous();
        }
        
        return storage_manager_->storeActivation(activation, layer_id);
    }
    
    /**
     * Retrieve activation
     */
    py::object retrieve_activation(int layer_id, py::object device = py::none()) {
        torch::Device target_device = torch::kCUDA;
        if (!device.is_none()) {
            if (py::isinstance<torch::Device>(device)) {
                target_device = device.cast<torch::Device>();
            } else if (py::isinstance<py::str>(device)) {
                std::string device_str = device.cast<std::string>();
                if (device_str == "cuda" || device_str == "cuda:0") {
                    target_device = torch::kCUDA;
                } else if (device_str == "cpu") {
                    target_device = torch::kCPU;
                }
            }
        }
        
        auto tensor = storage_manager_->retrieveActivation(layer_id, target_device);
        return py::cast(tensor);
    }
    
    /**
     * Check if activation exists
     */
    bool has_activation(int layer_id) {
        return storage_manager_->hasActivation(layer_id);
    }
    
    /**
     * Remove activation
     */
    void remove_activation(int layer_id) {
        storage_manager_->removeActivation(layer_id);
    }
    
    /**
     * Clear storage
     */
    void clear_storage() {
        storage_manager_->clearStorage();
    }
    
    /**
     * Get storage statistics
     */
    std::string get_storage_stats() {
        return storage_manager_->getStorageStats();
    }
    
    /**
     * Get compression ratio
     */
    float get_compression_ratio() {
        return storage_manager_->getCompressionRatio();
    }
    
    /**
     * Check if storage is full
     */
    bool is_storage_full() {
        return storage_manager_->isStorageFull();
    }
    
    /**
     * Get available storage
     */
    size_t get_available_storage() {
        return storage_manager_->getAvailableStorage();
    }
};

/**
 * Python wrapper for ANCFStormIntegration
 */
class ANCFStormIntegrationWrapper {
private:
    std::unique_ptr<storm::ANCFStormIntegration> integration_;
    
public:
    ANCFStormIntegrationWrapper() {
        integration_ = std::make_unique<storm::ANCFStormIntegration>();
    }
    
    /**
     * Start integration
     */
    void start_integration() {
        integration_->startIntegration();
    }
    
    /**
     * Stop integration
     */
    void stop_integration() {
        integration_->stopIntegration();
    }
    
    /**
     * Check if integration is active
     */
    bool is_active() {
        return integration_->isActive();
    }
    
    /**
     * ANCF forward pass
     */
    py::object ancf_forward_pass(py::object input_tensor, int layer_id) {
        torch::Tensor input = input_tensor.cast<torch::Tensor>();
        auto output = integration_->ancfForwardPass(input, layer_id);
        return py::cast(output);
    }
    
    /**
     * ANCF backward pass
     */
    py::object ancf_backward_pass(py::object grad_output, int layer_id) {
        torch::Tensor grad = grad_output.cast<torch::Tensor>();
        auto grad_input = integration_->ancfBackwardPass(grad, layer_id);
        return py::cast(grad_input);
    }
    
    /**
     * Store activation with ANCF
     */
    void store_activation_with_ancf(py::object activation_tensor, int layer_id) {
        torch::Tensor activation = activation_tensor.cast<torch::Tensor>();
        integration_->storeActivationWithANCF(activation, layer_id);
    }
    
    /**
     * Retrieve activation with ANCF
     */
    py::object retrieve_activation_with_ancf(int layer_id, py::object device = py::none()) {
        torch::Device target_device = torch::kCUDA;
        if (!device.is_none()) {
            if (py::isinstance<torch::Device>(device)) {
                target_device = device.cast<torch::Device>();
            } else if (py::isinstance<py::str>(device)) {
                std::string device_str = device.cast<std::string>();
                if (device_str == "cuda" || device_str == "cuda:0") {
                    target_device = torch::kCUDA;
                } else if (device_str == "cpu") {
                    target_device = torch::kCPU;
                }
            }
        }
        
        auto tensor = integration_->retrieveActivationWithANCF(layer_id, target_device);
        return py::cast(tensor);
    }
    
    /**
     * Get performance report
     */
    std::string get_performance_report() {
        return integration_->getANCFPerformanceReport();
    }
    
    /**
     * Check if targets are met
     */
    bool meets_targets() {
        return integration_->meetsANCFTargets();
    }
    
    /**
     * Configure ANCF
     */
    void configure_ancf(int policy = 1, size_t cpu_storage_limit = 8ULL * 1024 * 1024 * 1024) {
        storm::DictionarySizePolicy enum_policy;
        switch (policy) {
            case 0: enum_policy = storm::DictionarySizePolicy::CONSERVATIVE; break;
            case 1: enum_policy = storm::DictionarySizePolicy::ADAPTIVE; break;
            case 2: enum_policy = storm::DictionarySizePolicy::AGGRESSIVE; break;
            default: enum_policy = storm::DictionarySizePolicy::ADAPTIVE; break;
        }
        integration_->configureANCF(enum_policy, cpu_storage_limit);
    }
    
    /**
     * Get configuration
     */
    std::string get_configuration() {
        return integration_->getANCFConfiguration();
    }
};

/**
 * Python module definition
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ANCF (Asynchronous Near-Compute Memory Fabric) - Lossless encoding system for STORM";
    
    // Dictionary size policy enum
    py::enum_<storm::DictionarySizePolicy>(m, "DictionarySizePolicy")
        .value("CONSERVATIVE", storm::DictionarySizePolicy::CONSERVATIVE)
        .value("ADAPTIVE", storm::DictionarySizePolicy::ADAPTIVE)
        .value("AGGRESSIVE", storm::DictionarySizePolicy::AGGRESSIVE);
    
    // ANCFEncoder wrapper
    py::class_<ANCFEncoderWrapper>(m, "ANCFEncoder")
        .def(py::init<int, bool, bool, float>(), 
             py::arg("policy") = 1, 
             py::arg("enable_caching") = true, 
             py::arg("enable_profiling") = true, 
             py::arg("outlier_tolerance") = 1e-6f)
        .def("encode_activation", &ANCFEncoderWrapper::encode_activation,
             py::arg("activation_tensor"), py::arg("layer_id") = 0)
        .def("decode_activation", &ANCFEncoderWrapper::decode_activation,
             py::arg("encoded_dict"), py::arg("device") = py::none())
        .def("encode_for_cpu_storage", &ANCFEncoderWrapper::encode_for_cpu_storage,
             py::arg("activation_tensor"), py::arg("layer_id") = 0)
        .def("encode_for_pcie_transfer", &ANCFEncoderWrapper::encode_for_pcie_transfer,
             py::arg("activation_tensor"), py::arg("layer_id") = 0)
        .def("get_compression_stats", &ANCFEncoderWrapper::get_compression_stats)
        .def("get_average_compression_ratio", &ANCFEncoderWrapper::get_average_compression_ratio)
        .def("reset_stats", &ANCFEncoderWrapper::reset_stats)
        .def("set_dictionary_policy", &ANCFEncoderWrapper::set_dictionary_policy,
             py::arg("policy"))
        .def("set_caching_enabled", &ANCFEncoderWrapper::set_caching_enabled,
             py::arg("enabled"))
        .def("set_outlier_tolerance", &ANCFEncoderWrapper::set_outlier_tolerance,
             py::arg("tolerance"))
        .def("verify_lossless", &ANCFEncoderWrapper::verify_lossless,
             py::arg("original_tensor"), py::arg("decoded_tensor"));
    
    // ANCFCPUStorageManager wrapper
    py::class_<ANCFCPUStorageWrapper>(m, "ANCFCPUStorage")
        .def(py::init<size_t, int>(), 
             py::arg("max_storage") = 8ULL * 1024 * 1024 * 1024, 
             py::arg("policy") = 1)
        .def("store_activation", &ANCFCPUStorageWrapper::store_activation,
             py::arg("activation_tensor"), py::arg("layer_id"))
        .def("retrieve_activation", &ANCFCPUStorageWrapper::retrieve_activation,
             py::arg("layer_id"), py::arg("device") = py::none())
        .def("has_activation", &ANCFCPUStorageWrapper::has_activation,
             py::arg("layer_id"))
        .def("remove_activation", &ANCFCPUStorageWrapper::remove_activation,
             py::arg("layer_id"))
        .def("clear_storage", &ANCFCPUStorageWrapper::clear_storage)
        .def("get_storage_stats", &ANCFCPUStorageWrapper::get_storage_stats)
        .def("get_compression_ratio", &ANCFCPUStorageWrapper::get_compression_ratio)
        .def("is_storage_full", &ANCFCPUStorageWrapper::is_storage_full)
        .def("get_available_storage", &ANCFCPUStorageWrapper::get_available_storage);
    
    // ANCFStormIntegration wrapper
    py::class_<ANCFStormIntegrationWrapper>(m, "ANCFStormIntegration")
        .def(py::init<>())
        .def("start_integration", &ANCFStormIntegrationWrapper::start_integration)
        .def("stop_integration", &ANCFStormIntegrationWrapper::stop_integration)
        .def("is_active", &ANCFStormIntegrationWrapper::is_active)
        .def("ancf_forward_pass", &ANCFStormIntegrationWrapper::ancf_forward_pass,
             py::arg("input_tensor"), py::arg("layer_id"))
        .def("ancf_backward_pass", &ANCFStormIntegrationWrapper::ancf_backward_pass,
             py::arg("grad_output"), py::arg("layer_id"))
        .def("store_activation_with_ancf", &ANCFStormIntegrationWrapper::store_activation_with_ancf,
             py::arg("activation_tensor"), py::arg("layer_id"))
        .def("retrieve_activation_with_ancf", &ANCFStormIntegrationWrapper::retrieve_activation_with_ancf,
             py::arg("layer_id"), py::arg("device") = py::none())
        .def("get_performance_report", &ANCFStormIntegrationWrapper::get_performance_report)
        .def("meets_targets", &ANCFStormIntegrationWrapper::meets_targets)
        .def("configure_ancf", &ANCFStormIntegrationWrapper::configure_ancf,
             py::arg("policy") = 1, 
             py::arg("cpu_storage_limit") = 8ULL * 1024 * 1024 * 1024)
        .def("get_configuration", &ANCFStormIntegrationWrapper::get_configuration);
    
    // Utility functions
    m.def("get_ancf_version", []() {
        return std::string("ANCF v1.0.0 - Lossless Encoding for STORM");
    });
    
    m.def("check_cuda_compatibility", []() {
        if (!torch::cuda::is_available()) {
            return std::string("CUDA not available - ANCF requires GPU");
        }
        
        auto device_props = torch::cuda::get_device_properties(torch::cuda::current_device());
        return std::string("CUDA Device: ") + device_props.name + 
               std::string(", Compute Capability: ") + std::to_string(device_props.major) + 
               std::string(".") + std::to_string(device_props.minor);
    });
}
