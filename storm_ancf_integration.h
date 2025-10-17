#pragma once

#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <thread>
#include <future>
#include "storm_ancf_encoder.h"
#include "storm_core.h"
#include "storm_orchestration.h"

/**
 * ANCF Integration Layer
 * 
 * This header integrates ANCF encoding into the existing STORM system:
 * 1. Pre-CPU Storage Encoding: Compress activations before storing in CPU RAM (4x memory savings)
 * 2. Pre-PCIe Transfer Encoding: Compress tensors before PCIe transfer (4x bandwidth gain)
 * 3. Async Transfer Coordination: Overlap encoding/transfer with GPU compute
 * 4. Event-based Synchronization: Ensure decoded data ready before backward pass
 */

namespace storm {

/**
 * ANCF Transfer Coordinator
 * 
 * Manages asynchronous encoding/decoding operations with proper synchronization
 */
class ANCFTransferCoordinator {
private:
    std::unique_ptr<CUDAStream> encoding_stream_;
    std::unique_ptr<CUDAStream> transfer_stream_;
    std::unique_ptr<CUDAStream> decoding_stream_;
    
    std::vector<std::unique_ptr<CUDAEvent>> encoding_events_;
    std::vector<std::unique_ptr<CUDAEvent>> transfer_events_;
    std::vector<std::unique_ptr<CUDAEvent>> decoding_events_;
    
    std::unique_ptr<ANCFEncoder> encoder_;
    std::unique_ptr<ANCFPerformanceMonitor> performance_monitor_;
    
    std::mutex coordinator_mutex_;
    bool coordination_active_;
    
    // Async operation tracking
    std::unordered_map<int, std::future<ANCFEncodedData>> pending_encodings_;
    std::unordered_map<int, std::future<torch::Tensor>> pending_decodings_;
    
public:
    ANCFTransferCoordinator() : coordination_active_(false) {
        initialize();
    }
    
    ~ANCFTransferCoordinator() = default;
    
    // Delete copy constructor and assignment
    ANCFTransferCoordinator(const ANCFTransferCoordinator&) = delete;
    ANCFTransferCoordinator& operator=(const ANCFTransferCoordinator&) = delete;
    
    /**
     * Initialize the transfer coordinator
     */
    bool initialize();
    
    /**
     * Start coordination system
     */
    void startCoordination();
    
    /**
     * Stop coordination system
     */
    void stopCoordination();
    
    /**
     * Check if coordination is active
     */
    bool isActive() const { return coordination_active_; }
    
    /**
     * Async encode activation for CPU storage
     * 
     * @param activation Input activation tensor
     * @param layer_id Layer identifier
     * @return Future containing encoded data
     */
    std::future<ANCFEncodedData> asyncEncodeForCPUStorage(
        const torch::Tensor& activation,
        int layer_id
    );
    
    /**
     * Async encode activation for PCIe transfer
     * 
     * @param activation Input activation tensor
     * @param layer_id Layer identifier
     * @return Future containing encoded data
     */
    std::future<ANCFEncodedData> asyncEncodeForPCIeTransfer(
        const torch::Tensor& activation,
        int layer_id
    );
    
    /**
     * Async decode activation from encoded data
     * 
     * @param encoded_data Encoded data structure
     * @param target_device Target device for decoded tensor
     * @return Future containing decoded tensor
     */
    std::future<torch::Tensor> asyncDecodeActivation(
        const ANCFEncodedData& encoded_data,
        torch::Device target_device = torch::kCUDA
    );
    
    /**
     * Synchronize encoding operations
     */
    void synchronizeEncoding();
    
    /**
     * Synchronize transfer operations
     */
    void synchronizeTransfer();
    
    /**
     * Synchronize decoding operations
     */
    void synchronizeDecoding();
    
    /**
     * Wait for specific encoding to complete
     */
    void waitForEncoding(int layer_id);
    
    /**
     * Wait for specific decoding to complete
     */
    void waitForDecoding(int layer_id);
    
    /**
     * Get performance monitor
     */
    ANCFPerformanceMonitor& getPerformanceMonitor() { return *performance_monitor_; }
    
    /**
     * Get encoder
     */
    ANCFEncoder& getEncoder() { return *encoder_; }
};

/**
 * ANCF CPU Storage Manager
 * 
 * Manages compressed activation storage in CPU RAM with ANCF encoding
 */
class ANCFCPUStorageManager {
private:
    std::unique_ptr<ANCFEncoder> encoder_;
    std::unique_ptr<ANCFTransferCoordinator> coordinator_;
    
    // Compressed storage
    std::unordered_map<int, ANCFEncodedData> compressed_activations_;
    std::unordered_map<int, std::chrono::high_resolution_clock::time_point> storage_times_;
    
    std::mutex storage_mutex_;
    size_t max_storage_size_;
    size_t current_storage_size_;
    
    // Storage statistics
    size_t total_original_bytes_;
    size_t total_compressed_bytes_;
    int storage_operations_;
    
public:
    explicit ANCFCPUStorageManager(
        size_t max_storage = 8ULL * 1024 * 1024 * 1024, // 8GB default
        DictionarySizePolicy policy = DictionarySizePolicy::ADAPTIVE
    ) : encoder_(std::make_unique<ANCFEncoder>(policy)),
        coordinator_(std::make_unique<ANCFTransferCoordinator>()),
        max_storage_size_(max_storage),
        current_storage_size_(0),
        total_original_bytes_(0),
        total_compressed_bytes_(0),
        storage_operations_(0) {}
    
    /**
     * Store activation in compressed CPU storage
     * 
     * @param activation Input activation tensor
     * @param layer_id Layer identifier
     * @return True if storage successful
     */
    bool storeActivation(const torch::Tensor& activation, int layer_id);
    
    /**
     * Retrieve activation from compressed CPU storage
     * 
     * @param layer_id Layer identifier
     * @param target_device Target device for reconstructed tensor
     * @return Reconstructed tensor
     */
    torch::Tensor retrieveActivation(int layer_id, torch::Device target_device = torch::kCUDA);
    
    /**
     * Check if activation exists in storage
     * 
     * @param layer_id Layer identifier
     * @return True if activation exists
     */
    bool hasActivation(int layer_id) const;
    
    /**
     * Remove activation from storage
     * 
     * @param layer_id Layer identifier
     */
    void removeActivation(int layer_id);
    
    /**
     * Clear all stored activations
     */
    void clearStorage();
    
    /**
     * Get storage statistics
     * 
     * @return Storage statistics string
     */
    std::string getStorageStats() const;
    
    /**
     * Get compression ratio achieved
     * 
     * @return Average compression ratio
     */
    float getCompressionRatio() const;
    
    /**
     * Check if storage is full
     * 
     * @return True if storage is full
     */
    bool isStorageFull() const;
    
    /**
     * Get available storage space
     * 
     * @return Available storage space in bytes
     */
    size_t getAvailableStorage() const;
    
private:
    /**
     * Evict oldest stored activations to make space
     */
    void evictOldestActivations();
    
    /**
     * Calculate compressed data size
     */
    size_t calculateCompressedSize(const ANCFEncodedData& encoded_data) const;
};

/**
 * ANCF PCIe Transfer Manager
 * 
 * Manages compressed tensor transfers across PCIe with ANCF encoding
 */
class ANCFPCIeTransferManager {
private:
    std::unique_ptr<ANCFEncoder> encoder_;
    std::unique_ptr<ANCFTransferCoordinator> coordinator_;
    
    // Transfer streams and events
    std::unique_ptr<CUDAStream> h2d_stream_;
    std::unique_ptr<CUDAStream> d2h_stream_;
    std::vector<std::unique_ptr<CUDAEvent>> transfer_events_;
    
    // Transfer statistics
    size_t total_transferred_bytes_;
    size_t total_compressed_bytes_;
    double total_transfer_time_;
    int transfer_operations_;
    
    std::mutex transfer_mutex_;
    
public:
    ANCFPCIeTransferManager() 
        : encoder_(std::make_unique<ANCFEncoder>(DictionarySizePolicy::ADAPTIVE)),
          coordinator_(std::make_unique<ANCFTransferCoordinator>()),
          total_transferred_bytes_(0),
          total_compressed_bytes_(0),
          total_transfer_time_(0.0),
          transfer_operations_(0) {
        initialize();
    }
    
    /**
     * Initialize transfer manager
     */
    bool initialize();
    
    /**
     * Transfer tensor from GPU to CPU with ANCF compression
     * 
     * @param gpu_tensor Source tensor on GPU
     * @param layer_id Layer identifier
     * @return Compressed data on CPU
     */
    std::vector<uint8_t> transferGPUToCPU(
        const torch::Tensor& gpu_tensor,
        int layer_id
    );
    
    /**
     * Transfer tensor from CPU to GPU with ANCF decompression
     * 
     * @param compressed_data Compressed data on CPU
     * @param layer_id Layer identifier
     * @param target_shape Target tensor shape
     * @param target_device Target device
     * @return Reconstructed tensor on GPU
     */
    torch::Tensor transferCPUToGPU(
        const std::vector<uint8_t>& compressed_data,
        int layer_id,
        const std::vector<int64_t>& target_shape,
        torch::Device target_device = torch::kCUDA
    );
    
    /**
     * Async transfer GPU to CPU with compression
     * 
     * @param gpu_tensor Source tensor on GPU
     * @param layer_id Layer identifier
     * @return Future containing compressed data
     */
    std::future<std::vector<uint8_t>> asyncTransferGPUToCPU(
        const torch::Tensor& gpu_tensor,
        int layer_id
    );
    
    /**
     * Async transfer CPU to GPU with decompression
     * 
     * @param compressed_data Compressed data on CPU
     * @param layer_id Layer identifier
     * @param target_shape Target tensor shape
     * @param target_device Target device
     * @return Future containing reconstructed tensor
     */
    std::future<torch::Tensor> asyncTransferCPUToGPU(
        const std::vector<uint8_t>& compressed_data,
        int layer_id,
        const std::vector<int64_t>& target_shape,
        torch::Device target_device = torch::kCUDA
    );
    
    /**
     * Synchronize all transfer operations
     */
    void synchronizeTransfers();
    
    /**
     * Get transfer statistics
     * 
     * @return Transfer statistics string
     */
    std::string getTransferStats() const;
    
    /**
     * Get effective bandwidth achieved
     * 
     * @return Effective bandwidth in GB/s
     */
    double getEffectiveBandwidth() const;
    
    /**
     * Get compression ratio achieved
     * 
     * @return Average compression ratio
     */
    float getCompressionRatio() const;
    
    /**
     * Reset transfer statistics
     */
    void resetStats();
    
private:
    /**
     * Record transfer operation
     */
    void recordTransfer(size_t original_size, size_t compressed_size, double transfer_time);
};

/**
 * ANCF Integration with STORM Orchestrator
 * 
 * Integrates ANCF encoding into the existing STORM orchestration system
 */
class ANCFStormIntegration {
private:
    std::unique_ptr<ANCFCPUStorageManager> cpu_storage_manager_;
    std::unique_ptr<ANCFPCIeTransferManager> pcie_transfer_manager_;
    std::unique_ptr<ANCFTransferCoordinator> transfer_coordinator_;
    
    // Reference to existing STORM components
    std::shared_ptr<StormOrchestrator> storm_orchestrator_;
    std::shared_ptr<ActivationCache> activation_cache_;
    std::shared_ptr<LayerEventManager> event_manager_;
    
    bool integration_active_;
    std::mutex integration_mutex_;
    
public:
    explicit ANCFStormIntegration(
        std::shared_ptr<StormOrchestrator> storm_orchestrator = nullptr
    ) : storm_orchestrator_(storm_orchestrator),
        integration_active_(false) {
        initialize();
    }
    
    /**
     * Initialize ANCF integration with STORM
     */
    bool initialize();
    
    /**
     * Start ANCF integration
     */
    void startIntegration();
    
    /**
     * Stop ANCF integration
     */
    void stopIntegration();
    
    /**
     * Check if integration is active
     */
    bool isActive() const { return integration_active_; }
    
    /**
     * Enhanced forward pass with ANCF encoding
     * 
     * @param input Input tensor
     * @param layer_id Layer identifier
     * @return Output tensor with ANCF-encoded activations stored
     */
    torch::Tensor ancfForwardPass(torch::Tensor input, int layer_id);
    
    /**
     * Enhanced backward pass with ANCF decoding
     * 
     * @param grad_output Gradient tensor
     * @param layer_id Layer identifier
     * @return Gradient input tensor
     */
    torch::Tensor ancfBackwardPass(torch::Tensor grad_output, int layer_id);
    
    /**
     * Store activation with ANCF compression
     * 
     * @param activation Activation tensor
     * @param layer_id Layer identifier
     */
    void storeActivationWithANCF(const torch::Tensor& activation, int layer_id);
    
    /**
     * Retrieve activation with ANCF decompression
     * 
     * @param layer_id Layer identifier
     * @param target_device Target device
     * @return Reconstructed activation tensor
     */
    torch::Tensor retrieveActivationWithANCF(int layer_id, torch::Device target_device = torch::kCUDA);
    
    /**
     * Get comprehensive ANCF performance report
     * 
     * @return Performance report string
     */
    std::string getANCFPerformanceReport() const;
    
    /**
     * Check if ANCF meets performance targets
     * 
     * @return True if ANCF meets all performance targets
     */
    bool meetsANCFTargets() const;
    
    /**
     * Set ANCF configuration
     * 
     * @param policy Dictionary size policy
     * @param cpu_storage_limit CPU storage limit in bytes
     */
    void configureANCF(
        DictionarySizePolicy policy = DictionarySizePolicy::ADAPTIVE,
        size_t cpu_storage_limit = 8ULL * 1024 * 1024 * 1024
    );
    
    /**
     * Get ANCF configuration
     * 
     * @return Configuration string
     */
    std::string getANCFConfiguration() const;
};

} // namespace storm
