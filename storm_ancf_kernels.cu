#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <vector>
#include <algorithm>
#include <cmath>
#include "storm_ancf_encoder.h"

/**
 * ANCF CUDA Kernels
 * 
 * GPU-accelerated kernels for ANCF encoding/decoding operations:
 * 1. Fast K-means clustering for dictionary generation
 * 2. Parallel encoding kernel (FP16 → 8-bit indices)
 * 3. Parallel decoding kernel (8-bit indices → FP16)
 * 4. Sparsity analysis kernel for optimal dictionary sizing
 * 5. Outlier detection and escape code handling
 */

namespace storm {

// CUDA kernel configurations
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int MAX_DICTIONARY_SIZE = 512;
constexpr int MAX_ITERATIONS = 10;
constexpr float CONVERGENCE_THRESHOLD = 1e-6f;

/**
 * Utility function to get optimal CUDA grid size
 */
__forceinline__ dim3 getGridSize(int num_elements, int block_size = BLOCK_SIZE) {
    return dim3((num_elements + block_size - 1) / block_size);
}

/**
 * CUDA kernel for analyzing activation sparsity
 * 
 * Determines the optimal dictionary size based on value distribution
 */
__global__ void analyzeSparsityKernel(
    const float* __restrict__ activation,
    int num_elements,
    float* __restrict__ sparsity_metrics,
    float* __restrict__ value_range
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Thread-local accumulators
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    int local_zeros = 0;
    int local_unique_approx = 0;
    
    // Process elements with stride for better memory access
    for (int i = tid; i < num_elements; i += stride) {
        float val = activation[i];
        
        // Update range
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
        
        // Count zeros (approximate sparsity)
        if (fabsf(val) < 1e-8f) {
            local_zeros++;
        }
        
        // Approximate unique values (simplified hash-based counting)
        if (fabsf(val - floorf(val * 1000.0f) / 1000.0f) < 1e-6f) {
            local_unique_approx++;
        }
    }
    
    // Reduce within block
    __shared__ float shared_min[BLOCK_SIZE];
    __shared__ float shared_max[BLOCK_SIZE];
    __shared__ int shared_zeros[BLOCK_SIZE];
    __shared__ int shared_unique[BLOCK_SIZE];
    
    shared_min[threadIdx.x] = local_min;
    shared_max[threadIdx.x] = local_max;
    shared_zeros[threadIdx.x] = local_zeros;
    shared_unique[threadIdx.x] = local_unique_approx;
    
    __syncthreads();
    
    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_min[threadIdx.x] = fminf(shared_min[threadIdx.x], shared_min[threadIdx.x + s]);
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + s]);
            shared_zeros[threadIdx.x] += shared_zeros[threadIdx.x + s];
            shared_unique[threadIdx.x] += shared_unique[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Write results (only thread 0 of each block)
    if (threadIdx.x == 0) {
        atomicMin(reinterpret_cast<int*>(&value_range[0]), __float_as_int(shared_min[0]));
        atomicMax(reinterpret_cast<int*>(&value_range[1]), __float_as_int(shared_max[0]));
        atomicAdd(&sparsity_metrics[0], shared_zeros[0]);
        atomicAdd(&sparsity_metrics[1], shared_unique[0]);
    }
}

/**
 * CUDA kernel for K-means clustering initialization
 * 
 * Initializes centroids using k-means++ algorithm
 */
__global__ void kmeansInitKernel(
    const float* __restrict__ data,
    float* __restrict__ centroids,
    int num_elements,
    int num_clusters,
    curandState* __restrict__ states
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    curandState local_state = states[tid];
    
    // First centroid: randomly select
    if (tid == 0) {
        float random_val = curand_uniform(&local_state);
        int random_idx = static_cast<int>(random_val * num_elements) % num_elements;
        centroids[0] = data[random_idx];
    }
    
    __syncthreads();
    
    // K-means++ initialization
    for (int k = 1; k < num_clusters; k++) {
        float max_distance = 0.0f;
        int selected_idx = 0;
        
        // Find point with maximum distance to nearest centroid
        for (int i = tid; i < num_elements; i += stride) {
            float min_distance = FLT_MAX;
            
            // Find minimum distance to existing centroids
            for (int j = 0; j < k; j++) {
                float distance = fabsf(data[i] - centroids[j]);
                min_distance = fminf(min_distance, distance);
            }
            
            // Update maximum distance
            if (min_distance > max_distance) {
                max_distance = min_distance;
                selected_idx = i;
            }
        }
        
        // Reduce to find global maximum
        __shared__ float shared_max_dist[BLOCK_SIZE];
        __shared__ int shared_max_idx[BLOCK_SIZE];
        
        shared_max_dist[threadIdx.x] = max_distance;
        shared_max_idx[threadIdx.x] = selected_idx;
        
        __syncthreads();
        
        // Block-level reduction
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                if (shared_max_dist[threadIdx.x + s] > shared_max_dist[threadIdx.x]) {
                    shared_max_dist[threadIdx.x] = shared_max_dist[threadIdx.x + s];
                    shared_max_idx[threadIdx.x] = shared_max_idx[threadIdx.x + s];
                }
            }
            __syncthreads();
        }
        
        // Global reduction across blocks (simplified)
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            centroids[k] = data[shared_max_idx[0]];
        }
        
        __syncthreads();
    }
    
    states[tid] = local_state;
}

/**
 * CUDA kernel for K-means clustering assignment step
 * 
 * Assigns each data point to the nearest centroid
 */
__global__ void kmeansAssignKernel(
    const float* __restrict__ data,
    const float* __restrict__ centroids,
    int* __restrict__ assignments,
    int num_elements,
    int num_clusters
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < num_elements; i += stride) {
        float min_distance = FLT_MAX;
        int best_cluster = 0;
        
        // Find nearest centroid
        for (int j = 0; j < num_clusters; j++) {
            float distance = fabsf(data[i] - centroids[j]);
            if (distance < min_distance) {
                min_distance = distance;
                best_cluster = j;
            }
        }
        
        assignments[i] = best_cluster;
    }
}

/**
 * CUDA kernel for K-means clustering update step
 * 
 * Updates centroids based on current assignments
 */
__global__ void kmeansUpdateKernel(
    const float* __restrict__ data,
    const int* __restrict__ assignments,
    float* __restrict__ new_centroids,
    int* __restrict__ cluster_counts,
    int num_elements,
    int num_clusters
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Initialize cluster accumulators
    for (int i = tid; i < num_clusters; i += stride) {
        new_centroids[i] = 0.0f;
        cluster_counts[i] = 0;
    }
    
    __syncthreads();
    
    // Accumulate values for each cluster
    for (int i = tid; i < num_elements; i += stride) {
        int cluster = assignments[i];
        float val = data[i];
        
        atomicAdd(&new_centroids[cluster], val);
        atomicAdd(&cluster_counts[cluster], 1);
    }
    
    __syncthreads();
    
    // Compute new centroids
    for (int i = tid; i < num_clusters; i += stride) {
        if (cluster_counts[i] > 0) {
            new_centroids[i] /= cluster_counts[i];
        }
    }
}

/**
 * CUDA kernel for ANCF encoding
 * 
 * Converts FP16 values to 8-bit indices using dictionary lookup
 */
__global__ void ancfEncodeKernel(
    const float* __restrict__ activation,
    const float* __restrict__ dictionary,
    uint8_t* __restrict__ indices,
    float* __restrict__ outliers,
    int* __restrict__ outlier_positions,
    int* __restrict__ outlier_count,
    int num_elements,
    int dictionary_size,
    int escape_code,
    float tolerance
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < num_elements; i += stride) {
        float val = activation[i];
        bool found = false;
        int best_index = 0;
        float min_distance = FLT_MAX;
        
        // Find best match in dictionary
        for (int j = 0; j < dictionary_size; j++) {
            if (j == escape_code) continue; // Skip escape code
            
            float distance = fabsf(val - dictionary[j]);
            if (distance < tolerance) {
                indices[i] = static_cast<uint8_t>(j);
                found = true;
                break;
            }
            
            if (distance < min_distance) {
                min_distance = distance;
                best_index = j;
            }
        }
        
        if (!found) {
            // Use escape code and store as outlier
            indices[i] = static_cast<uint8_t>(escape_code);
            int outlier_idx = atomicAdd(outlier_count, 1);
            outliers[outlier_idx] = val;
            outlier_positions[outlier_idx] = i;
        } else {
            indices[i] = static_cast<uint8_t>(best_index);
        }
    }
}

/**
 * CUDA kernel for ANCF decoding
 * 
 * Reconstructs FP16 values from 8-bit indices and dictionary
 */
__global__ void ancfDecodeKernel(
    const uint8_t* __restrict__ indices,
    const float* __restrict__ dictionary,
    const float* __restrict__ outliers,
    const int* __restrict__ outlier_positions,
    float* __restrict__ output,
    int num_elements,
    int dictionary_size,
    int escape_code,
    int num_outliers
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < num_elements; i += stride) {
        uint8_t index = indices[i];
        
        if (index == escape_code) {
            // Find corresponding outlier value
            float outlier_val = 0.0f;
            for (int j = 0; j < num_outliers; j++) {
                if (outlier_positions[j] == i) {
                    outlier_val = outliers[j];
                    break;
                }
            }
            output[i] = outlier_val;
        } else {
            // Use dictionary value
            output[i] = dictionary[index];
        }
    }
}

/**
 * CUDA kernel for outlier detection
 * 
 * Identifies values that don't have close matches in the dictionary
 */
__global__ void detectOutliersKernel(
    const float* __restrict__ activation,
    const float* __restrict__ dictionary,
    bool* __restrict__ is_outlier,
    int num_elements,
    int dictionary_size,
    float tolerance
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < num_elements; i += stride) {
        float val = activation[i];
        bool found_close_match = false;
        
        // Check if value has a close match in dictionary
        for (int j = 0; j < dictionary_size; j++) {
            if (fabsf(val - dictionary[j]) < tolerance) {
                found_close_match = true;
                break;
            }
        }
        
        is_outlier[i] = !found_close_match;
    }
}

/**
 * Host functions to launch CUDA kernels
 */

/**
 * Analyze activation sparsity on GPU
 */
std::vector<float> analyzeSparsityGPU(const torch::Tensor& activation) {
    TORCH_CHECK(activation.is_cuda(), "Activation must be on GPU");
    TORCH_CHECK(activation.dtype() == torch::kFloat32 || activation.dtype() == torch::kFloat16,
                "Activation must be FP32 or FP16");
    
    int num_elements = activation.numel();
    auto device = activation.device();
    
    // Allocate GPU memory for results
    auto sparsity_metrics = torch::zeros({2}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto value_range = torch::zeros({2}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    // Initialize value range
    value_range[0] = FLT_MAX;
    value_range[1] = -FLT_MAX;
    
    // Launch kernel
    dim3 grid_size = getGridSize(num_elements);
    dim3 block_size(BLOCK_SIZE);
    
    analyzeSparsityKernel<<<grid_size, block_size>>>(
        activation.data_ptr<float>(),
        num_elements,
        sparsity_metrics.data_ptr<float>(),
        value_range.data_ptr<float>()
    );
    
    cudaDeviceSynchronize();
    
    // Copy results to host
    std::vector<float> result(4);
    result[0] = sparsity_metrics[0].item<float>() / num_elements; // Zero ratio
    result[1] = sparsity_metrics[1].item<float>(); // Unique value approximation
    result[2] = value_range[0].item<float>(); // Min value
    result[3] = value_range[1].item<float>(); // Max value
    
    return result;
}

/**
 * Perform K-means clustering on GPU
 */
torch::Tensor kmeansClusteringGPU(const torch::Tensor& activation, int num_clusters) {
    TORCH_CHECK(activation.is_cuda(), "Activation must be on GPU");
    TORCH_CHECK(num_clusters > 0 && num_clusters <= MAX_DICTIONARY_SIZE, "Invalid cluster count");
    
    int num_elements = activation.numel();
    auto device = activation.device();
    
    // Allocate GPU memory
    auto centroids = torch::zeros({num_clusters}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto assignments = torch::zeros({num_elements}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto new_centroids = torch::zeros({num_clusters}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto cluster_counts = torch::zeros({num_clusters}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    // Initialize random states for K-means++
    auto states = torch::zeros({BLOCK_SIZE * 4}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    // Launch initialization kernel
    dim3 grid_size = getGridSize(BLOCK_SIZE * 4);
    dim3 block_size(BLOCK_SIZE);
    
    kmeansInitKernel<<<grid_size, block_size>>>(
        activation.data_ptr<float>(),
        centroids.data_ptr<float>(),
        num_elements,
        num_clusters,
        reinterpret_cast<curandState*>(states.data_ptr<int>())
    );
    
    // K-means iterations
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        // Assignment step
        kmeansAssignKernel<<<grid_size, block_size>>>(
            activation.data_ptr<float>(),
            centroids.data_ptr<float>(),
            assignments.data_ptr<int>(),
            num_elements,
            num_clusters
        );
        
        // Update step
        kmeansUpdateKernel<<<grid_size, block_size>>>(
            activation.data_ptr<float>(),
            assignments.data_ptr<int>(),
            new_centroids.data_ptr<float>(),
            cluster_counts.data_ptr<int>(),
            num_elements,
            num_clusters
        );
        
        cudaDeviceSynchronize();
        
        // Check convergence (simplified)
        auto diff = torch::abs(centroids - new_centroids).max().item<float>();
        if (diff < CONVERGENCE_THRESHOLD) {
            break;
        }
        
        // Update centroids
        centroids = new_centroids.clone();
    }
    
    return centroids;
}

/**
 * Encode activation using ANCF on GPU
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> encodeActivationGPU(
    const torch::Tensor& activation,
    const torch::Tensor& dictionary,
    int escape_code,
    float tolerance = 1e-6f
) {
    TORCH_CHECK(activation.is_cuda() && dictionary.is_cuda(), "Tensors must be on GPU");
    
    int num_elements = activation.numel();
    int dictionary_size = dictionary.numel();
    auto device = activation.device();
    
    // Allocate GPU memory for results
    auto indices = torch::zeros({num_elements}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    auto outliers = torch::zeros({num_elements}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto outlier_positions = torch::zeros({num_elements}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto outlier_count = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    // Launch encoding kernel
    dim3 grid_size = getGridSize(num_elements);
    dim3 block_size(BLOCK_SIZE);
    
    ancfEncodeKernel<<<grid_size, block_size>>>(
        activation.data_ptr<float>(),
        dictionary.data_ptr<float>(),
        indices.data_ptr<uint8_t>(),
        outliers.data_ptr<float>(),
        outlier_positions.data_ptr<int>(),
        outlier_count.data_ptr<int>(),
        num_elements,
        dictionary_size,
        escape_code,
        tolerance
    );
    
    cudaDeviceSynchronize();
    
    int actual_outlier_count = outlier_count.item<int>();
    
    // Trim outliers to actual count
    if (actual_outlier_count > 0) {
        outliers = outliers.slice(0, 0, actual_outlier_count);
        outlier_positions = outlier_positions.slice(0, 0, actual_outlier_count);
    } else {
        outliers = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        outlier_positions = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    }
    
    return std::make_tuple(indices, outliers, outlier_positions, outlier_count);
}

/**
 * Decode activation using ANCF on GPU
 */
torch::Tensor decodeActivationGPU(
    const torch::Tensor& indices,
    const torch::Tensor& dictionary,
    const torch::Tensor& outliers,
    const torch::Tensor& outlier_positions,
    int escape_code,
    const std::vector<int64_t>& original_shape
) {
    TORCH_CHECK(indices.is_cuda() && dictionary.is_cuda(), "Tensors must be on GPU");
    
    int num_elements = indices.numel();
    int dictionary_size = dictionary.numel();
    int num_outliers = outliers.numel();
    auto device = indices.device();
    
    // Allocate GPU memory for output
    auto output = torch::zeros(original_shape, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    // Launch decoding kernel
    dim3 grid_size = getGridSize(num_elements);
    dim3 block_size(BLOCK_SIZE);
    
    ancfDecodeKernel<<<grid_size, block_size>>>(
        indices.data_ptr<uint8_t>(),
        dictionary.data_ptr<float>(),
        outliers.data_ptr<float>(),
        outlier_positions.data_ptr<int>(),
        output.data_ptr<float>(),
        num_elements,
        dictionary_size,
        escape_code,
        num_outliers
    );
    
    cudaDeviceSynchronize();
    
    return output;
}

} // namespace storm
