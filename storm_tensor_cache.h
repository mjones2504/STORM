#pragma once

#include <torch/torch.h>
#include <torch/extension.h>
#include <unordered_map>
#include <chrono>
#include <string>
#include <vector>
#include <memory>

namespace storm {

/**
 * STORM Tensor Cache
 * 
 * This class provides intelligent tensor caching for bandwidth optimization.
 * It implements the core STORM strategy of reducing VRAM bandwidth usage
 * by caching frequently accessed activations and optimizing memory access patterns.
 */
class StormTensorCache {
private:
    // Cache storage
    std::unordered_map<int, torch::Tensor> layer_cache_;
    std::unordered_map<int, std::chrono::time_point<std::chrono::high_resolution_clock>> access_times_;
    std::unordered_map<int, int> access_counts_;
    
    // Cache configuration
    size_t max_cache_size_;
    double cache_hit_rate_;
    int total_requests_;
    int cache_hits_;
    
    // Bandwidth optimization
    bool enable_bandwidth_optimization_;
    double target_bandwidth_reduction_;

public:
    StormTensorCache(size_t max_size = 1000) 
        : max_cache_size_(max_size)
        , cache_hit_rate_(0.0)
        , total_requests_(0)
        , cache_hits_(0)
        , enable_bandwidth_optimization_(true)
        , target_bandwidth_reduction_(0.3) {} // 30% target

    /**
     * Cache activation tensor for a specific layer
     * 
     * This function caches activation tensors to reduce VRAM bandwidth usage
     * by avoiding recomputation and reducing memory transfers.
     * 
     * @param activation Activation tensor to cache
     * @param layer_id Layer identifier
     * @param priority Cache priority (higher = keep longer)
     */
    void cache_activation(torch::Tensor activation, int layer_id, int priority = 1) {
        // Check if we need to evict old entries
        if (layer_cache_.size() >= max_cache_size_) {
            evict_old_activations();
        }
        
        // Store activation in cache
        layer_cache_[layer_id] = activation.clone(); // Clone to avoid reference issues
        access_times_[layer_id] = std::chrono::high_resolution_clock::now();
        access_counts_[layer_id] = 1;
        
        // Apply bandwidth optimization
        if (enable_bandwidth_optimization_) {
            optimize_tensor_for_bandwidth(layer_cache_[layer_id]);
        }
    }

    /**
     * Retrieve cached activation for a specific layer
     * 
     * @param layer_id Layer identifier
     * @return Cached activation tensor if found, empty tensor otherwise
     */
    torch::Tensor retrieve_activation(int layer_id) {
        total_requests_++;
        
        auto it = layer_cache_.find(layer_id);
        if (it != layer_cache_.end()) {
            cache_hits_++;
            access_times_[layer_id] = std::chrono::high_resolution_clock::now();
            access_counts_[layer_id]++;
            
            // Update cache hit rate
            cache_hit_rate_ = static_cast<double>(cache_hits_) / total_requests_;
            
            return it->second;
        }
        
        return torch::Tensor();
    }

    /**
     * Check if activation is cached for a specific layer
     * 
     * @param layer_id Layer identifier
     * @return True if cached, false otherwise
     */
    bool has_cached_activation(int layer_id) const {
        return layer_cache_.find(layer_id) != layer_cache_.end();
    }

    /**
     * Evict old activations to free memory
     * 
     * This function implements LRU (Least Recently Used) eviction
     * to maintain cache size limits while preserving frequently accessed data.
     */
    void evict_old_activations() {
        if (layer_cache_.empty()) {
            return;
        }
        
        // Find least recently used entry
        auto oldest_it = access_times_.begin();
        auto oldest_time = oldest_it->second;
        
        for (auto it = access_times_.begin(); it != access_times_.end(); ++it) {
            if (it->second < oldest_time) {
                oldest_time = it->second;
                oldest_it = it;
            }
        }
        
        // Remove oldest entry
        int layer_id = oldest_it->first;
        layer_cache_.erase(layer_id);
        access_times_.erase(layer_id);
        access_counts_.erase(layer_id);
    }

    /**
     * Get cache hit rate
     * 
     * @return Cache hit rate (0.0 to 1.0)
     */
    double get_cache_hit_rate() const {
        return cache_hit_rate_;
    }

    /**
     * Get cache statistics
     * 
     * @return Detailed cache statistics
     */
    std::string get_cache_stats() const {
        return "STORM Tensor Cache Statistics:\n"
               "  Cached layers: " + std::to_string(layer_cache_.size()) + "\n"
               "  Cache hit rate: " + std::to_string(cache_hit_rate_ * 100) + "%\n"
               "  Total requests: " + std::to_string(total_requests_) + "\n"
               "  Cache hits: " + std::to_string(cache_hits_) + "\n"
               "  Max cache size: " + std::to_string(max_cache_size_) + "\n"
               "  Bandwidth optimization: " + (enable_bandwidth_optimization_ ? "Enabled" : "Disabled");
    }

    /**
     * Clear all cached activations
     */
    void clear_cache() {
        layer_cache_.clear();
        access_times_.clear();
        access_counts_.clear();
        cache_hit_rate_ = 0.0;
        total_requests_ = 0;
        cache_hits_ = 0;
    }

    /**
     * Set cache size limit
     * 
     * @param max_size Maximum number of cached tensors
     */
    void set_max_cache_size(size_t max_size) {
        max_cache_size_ = max_size;
        
        // Evict excess entries if needed
        while (layer_cache_.size() > max_cache_size_) {
            evict_old_activations();
        }
    }

    /**
     * Enable or disable bandwidth optimization
     * 
     * @param enable Whether to enable bandwidth optimization
     */
    void set_bandwidth_optimization(bool enable) {
        enable_bandwidth_optimization_ = enable;
    }

    /**
     * Get target bandwidth reduction
     * 
     * @return Target bandwidth reduction percentage
     */
    double get_target_bandwidth_reduction() const {
        return target_bandwidth_reduction_;
    }

    /**
     * Set target bandwidth reduction
     * 
     * @param reduction Target bandwidth reduction (0.0 to 1.0)
     */
    void set_target_bandwidth_reduction(double reduction) {
        target_bandwidth_reduction_ = std::max(0.0, std::min(1.0, reduction));
    }

private:
    /**
     * Optimize tensor for bandwidth efficiency
     * 
     * This function applies bandwidth optimizations to cached tensors
     * to reduce VRAM bandwidth usage.
     * 
     * @param tensor Tensor to optimize
     */
    void optimize_tensor_for_bandwidth(torch::Tensor& tensor) {
        // Ensure contiguous memory layout
        if (!tensor.is_contiguous()) {
            tensor = tensor.contiguous();
        }
        
        // Apply memory alignment optimizations
        // PyTorch handles most optimizations automatically
        // We can add custom bandwidth optimizations here
    }
};

} // namespace storm
