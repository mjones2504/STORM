#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <cuda.h>

/**
 * STORM Core Classes
 * 
 * This header file contains the core classes for our STORM system.
 * Let me teach you advanced C++ concepts as we build this!
 */

namespace storm {

/**
 * CUDA Stream Manager
 * 
 * This class demonstrates several important C++ concepts:
 * 1. RAII (Resource Acquisition Is Initialization)
 * 2. Move semantics
 * 3. Exception safety
 * 4. Smart pointers
 */
class CUDAStream {
private:
    cudaStream_t stream_;
    bool is_valid_;
    
public:
    // Constructor - RAII pattern
    explicit CUDAStream() : is_valid_(false) {
        cudaError_t error = cudaStreamCreate(&stream_);
        if (error == cudaSuccess) {
            is_valid_ = true;
        } else {
            std::cerr << "Failed to create CUDA stream: " 
                      << cudaGetErrorString(error) << std::endl;
        }
    }
    
    // Destructor - automatic cleanup
    ~CUDAStream() {
        if (is_valid_) {
            cudaStreamDestroy(stream_);
        }
    }
    
    // Delete copy constructor and assignment (streams can't be copied)
    CUDAStream(const CUDAStream&) = delete;
    CUDAStream& operator=(const CUDAStream&) = delete;
    
    // Move constructor - transfers ownership
    CUDAStream(CUDAStream&& other) noexcept 
        : stream_(other.stream_), is_valid_(other.is_valid_) {
        other.is_valid_ = false;  // Prevent double-destruction
    }
    
    // Move assignment
    CUDAStream& operator=(CUDAStream&& other) noexcept {
        if (this != &other) {
            if (is_valid_) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            is_valid_ = other.is_valid_;
            other.is_valid_ = false;
        }
        return *this;
    }
    
    // Getter methods
    cudaStream_t get() const { return stream_; }
    bool isValid() const { return is_valid_; }
    
    // Synchronization method
    void synchronize() const {
        if (is_valid_) {
            cudaStreamSynchronize(stream_);
        }
    }
};

/**
 * CUDA Event Manager
 * 
 * Demonstrates:
 * 1. Template specialization
 * 2. Const correctness
 * 3. Exception safety
 */
class CUDAEvent {
private:
    cudaEvent_t event_;
    bool is_valid_;
    
public:
    explicit CUDAEvent() : is_valid_(false) {
        cudaError_t error = cudaEventCreate(&event_);
        if (error == cudaSuccess) {
            is_valid_ = true;
        } else {
            std::cerr << "Failed to create CUDA event: " 
                      << cudaGetErrorString(error) << std::endl;
        }
    }
    
    ~CUDAEvent() {
        if (is_valid_) {
            cudaEventDestroy(event_);
        }
    }
    
    // Delete copy constructor and assignment
    CUDAEvent(const CUDAEvent&) = delete;
    CUDAEvent& operator=(const CUDAEvent&) = delete;
    
    // Move semantics
    CUDAEvent(CUDAEvent&& other) noexcept 
        : event_(other.event_), is_valid_(other.is_valid_) {
        other.is_valid_ = false;
    }
    
    CUDAEvent& operator=(CUDAEvent&& other) noexcept {
        if (this != &other) {
            if (is_valid_) {
                cudaEventDestroy(event_);
            }
            event_ = other.event_;
            is_valid_ = other.is_valid_;
            other.is_valid_ = false;
        }
        return *this;
    }
    
    // Event operations
    void record(const CUDAStream& stream) const {
        if (is_valid_ && stream.isValid()) {
            cudaEventRecord(event_, stream.get());
        }
    }
    
    void wait(const CUDAStream& stream) const {
        if (is_valid_ && stream.isValid()) {
            cudaStreamWaitEvent(stream.get(), event_, 0);
        }
    }
    
    bool isValid() const { return is_valid_; }
};

/**
 * Pinned Memory Buffer
 * 
 * This class manages pinned (page-locked) memory for fast CPU-GPU transfers.
 * Demonstrates:
 * 1. Template classes
 * 2. Memory management
 * 3. Exception safety
 */
template<typename T>
class PinnedMemoryBuffer {
private:
    T* data_;
    size_t size_;
    bool is_valid_;
    
public:
    explicit PinnedMemoryBuffer(size_t size) 
        : data_(nullptr), size_(size), is_valid_(false) {
        cudaError_t error = cudaHostAlloc(&data_, size * sizeof(T), 
                                        cudaHostAllocDefault);
        if (error == cudaSuccess) {
            is_valid_ = true;
        } else {
            std::cerr << "Failed to allocate pinned memory: " 
                      << cudaGetErrorString(error) << std::endl;
        }
    }
    
    ~PinnedMemoryBuffer() {
        if (is_valid_ && data_) {
            cudaFreeHost(data_);
        }
    }
    
    // Delete copy constructor and assignment
    PinnedMemoryBuffer(const PinnedMemoryBuffer&) = delete;
    PinnedMemoryBuffer& operator=(const PinnedMemoryBuffer&) = delete;
    
    // Move semantics
    PinnedMemoryBuffer(PinnedMemoryBuffer&& other) noexcept 
        : data_(other.data_), size_(other.size_), is_valid_(other.is_valid_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.is_valid_ = false;
    }
    
    PinnedMemoryBuffer& operator=(PinnedMemoryBuffer&& other) noexcept {
        if (this != &other) {
            if (is_valid_ && data_) {
                cudaFreeHost(data_);
            }
            data_ = other.data_;
            size_ = other.size_;
            is_valid_ = other.is_valid_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.is_valid_ = false;
        }
        return *this;
    }
    
    // Access methods
    T& operator[](size_t index) {
        return data_[index];
    }
    
    const T& operator[](size_t index) const {
        return data_[index];
    }
    
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    bool isValid() const { return is_valid_; }
};

/**
 * STORM System Core
 * 
 * The main orchestrator class that manages all STORM components.
 * Demonstrates:
 * 1. Composition over inheritance
 * 2. Factory pattern
 * 3. Resource management
 */
class StormSystem {
private:
    std::unique_ptr<CUDAStream> compute_stream_;
    std::unique_ptr<CUDAStream> transfer_h2d_stream_;  // Host to Device
    std::unique_ptr<CUDAStream> transfer_d2h_stream_;  // Device to Host
    
    std::vector<std::unique_ptr<CUDAEvent>> compute_events_;
    std::vector<std::unique_ptr<CUDAEvent>> transfer_events_;
    
    bool is_initialized_;
    
public:
    explicit StormSystem() : is_initialized_(false) {
        initialize();
    }
    
    ~StormSystem() = default;
    
    // Delete copy constructor and assignment
    StormSystem(const StormSystem&) = delete;
    StormSystem& operator=(const StormSystem&) = delete;
    
    // Move semantics
    StormSystem(StormSystem&&) = default;
    StormSystem& operator=(StormSystem&&) = default;
    
    bool initialize() {
        try {
            // Create streams
            compute_stream_ = std::make_unique<CUDAStream>();
            transfer_h2d_stream_ = std::make_unique<CUDAStream>();
            transfer_d2h_stream_ = std::make_unique<CUDAStream>();
            
            if (!compute_stream_->isValid() || 
                !transfer_h2d_stream_->isValid() || 
                !transfer_d2h_stream_->isValid()) {
                return false;
            }
            
            is_initialized_ = true;
            std::cout << "STORM system initialized successfully!" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize STORM system: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool isInitialized() const { return is_initialized_; }
    
    CUDAStream& getComputeStream() { return *compute_stream_; }
    CUDAStream& getTransferH2DStream() { return *transfer_h2d_stream_; }
    CUDAStream& getTransferD2HStream() { return *transfer_d2h_stream_; }
};

} // namespace storm
