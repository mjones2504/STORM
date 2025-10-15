# C++ Learning Guide for STORM Development

## C++ Fundamentals You Need to Know

### 1. Basic Syntax and Structure
```cpp
#include <iostream>  // Include standard library headers
#include <vector>    // For dynamic arrays
#include <memory>    // For smart pointers

// Main function - entry point of our program
int main() {
    // Code goes here
    return 0;  // Return 0 means success
}
```

### 2. Variables and Data Types
```cpp
// Basic types
int age = 25;                    // Integer
float temperature = 98.6f;       // Floating point
double precision = 3.14159;     // Double precision
bool isActive = true;           // Boolean
char letter = 'A';              // Single character

// C++11+ auto keyword (type inference)
auto count = 42;                // Compiler figures out this is int
auto name = "STORM";            // This is const char*
```

### 3. Memory Management (CRITICAL for STORM!)
```cpp
// Raw pointers (dangerous - avoid when possible)
int* ptr = new int(42);         // Allocate memory
delete ptr;                     // Must free memory manually!

// Smart pointers (SAFE - use these!)
std::unique_ptr<int> smartPtr = std::make_unique<int>(42);
// Automatically freed when out of scope!

// Shared pointers (for shared ownership)
std::shared_ptr<int> sharedPtr = std::make_shared<int>(42);
```

### 4. Classes and Objects
```cpp
class StormSystem {
private:
    int deviceId;
    bool isInitialized;
    
public:
    // Constructor
    StormSystem(int id) : deviceId(id), isInitialized(false) {
        std::cout << "STORM system created with device ID: " << id << std::endl;
    }
    
    // Destructor
    ~StormSystem() {
        std::cout << "STORM system destroyed" << std::endl;
    }
    
    // Methods
    bool initialize() {
        isInitialized = true;
        return true;
    }
    
    int getDeviceId() const { return deviceId; }
};
```

### 5. CUDA-Specific C++ Concepts
```cpp
// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Usage:
CUDA_CHECK(cudaSetDevice(0));
```

### 6. Templates (Advanced but useful)
```cpp
template<typename T>
class MemoryBuffer {
private:
    T* data;
    size_t size;
    
public:
    MemoryBuffer(size_t s) : size(s) {
        data = new T[size];
    }
    
    ~MemoryBuffer() {
        delete[] data;
    }
    
    T& operator[](size_t index) {
        return data[index];
    }
};

// Usage:
MemoryBuffer<float> buffer(1000);  // Buffer of 1000 floats
```

## Key Concepts for STORM Development

### 1. RAII (Resource Acquisition Is Initialization)
- Objects manage their own resources
- Destructors automatically clean up
- Prevents memory leaks

### 2. Exception Safety
- Use try-catch blocks for error handling
- Smart pointers for automatic cleanup
- RAII ensures cleanup even if exceptions occur

### 3. Move Semantics (C++11+)
```cpp
class StormBuffer {
private:
    float* data;
    size_t size;
    
public:
    // Move constructor
    StormBuffer(StormBuffer&& other) noexcept 
        : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }
    
    // Move assignment
    StormBuffer& operator=(StormBuffer&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
};
```

## STORM-Specific Patterns

### 1. Stream Management
```cpp
class CUDAStream {
private:
    cudaStream_t stream;
    
public:
    CUDAStream() {
        cudaStreamCreate(&stream);
    }
    
    ~CUDAStream() {
        cudaStreamDestroy(stream);
    }
    
    cudaStream_t get() const { return stream; }
};
```

### 2. Event Synchronization
```cpp
class CUDAEvent {
private:
    cudaEvent_t event;
    
public:
    CUDAEvent() {
        cudaEventCreate(&event);
    }
    
    ~CUDAEvent() {
        cudaEventDestroy(event);
    }
    
    void record(cudaStream_t stream) {
        cudaEventRecord(event, stream);
    }
    
    void wait(cudaStream_t stream) {
        cudaStreamWaitEvent(stream, event, 0);
    }
};
```

This guide will grow as we build STORM together!
