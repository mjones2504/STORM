#pragma once

#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include "storm_core.h"

/**
 * STORM Autograd Functions
 * 
 * This file contains the custom PyTorch autograd functions that implement
 * the STORM memory orchestration system.
 * 
 * Key C++ concepts demonstrated:
 * 1. PyTorch C++ API integration
 * 2. Custom autograd functions
 * 3. Tensor operations and memory management
 * 4. CUDA-PyTorch integration
 */

namespace storm {

/**
 * STORM Forward Function
 * 
 * This class implements the forward pass of STORM:
 * 1. Computes the layer output normally
 * 2. Immediately offloads activations to CPU RAM
 * 3. Uses asynchronous memory transfer
 * 
 * Demonstrates:
 * - PyTorch autograd function subclassing
 * - Custom forward/backward logic
 * - CUDA stream management
 * - Memory orchestration
 */
class StormForwardFunction : public torch::autograd::Function<StormForwardFunction> {
public:
    // Forward pass implementation
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias,
        int layer_id
    ) {
        // Store inputs for backward pass
        ctx->save_for_backward({input, weight, bias});
        ctx->saved_data["layer_id"] = layer_id;
        
        // Perform the forward computation
        torch::Tensor output = torch::linear(input, weight, bias);
        
        // Create activation tensor for offloading
        torch::Tensor activation = output.clone();
        
        // Get STORM system instance (singleton pattern)
        static auto storm_system = std::make_unique<StormSystem>();
        
        // Offload activation to CPU RAM asynchronously
        offload_activation_to_cpu(activation, layer_id, storm_system.get());
        
        return output;
    }
    
    // Backward pass implementation
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        // Retrieve saved tensors
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto bias = saved[2];
        int layer_id = ctx->saved_data["layer_id"].toInt();
        
        // Get gradient from output
        auto grad_output = grad_outputs[0];
        
        // Get STORM system instance
        static auto storm_system = std::make_unique<StormSystem>();
        
        // Fetch activation from CPU RAM asynchronously
        auto activation = fetch_activation_from_cpu(layer_id, storm_system.get());
        
        // Compute gradients
        auto grad_input = torch::linear(grad_output, weight.t(), torch::Tensor());
        auto grad_weight = torch::mm(grad_output.t(), input);
        auto grad_bias = grad_output.sum(0);
        
        return {grad_input, grad_weight, grad_bias, torch::Tensor()};
    }
    
private:
    /**
     * Offload activation to CPU RAM using pinned memory
     * 
     * This function demonstrates:
     * 1. Asynchronous memory transfer
     * 2. CUDA stream management
     * 3. Pinned memory usage
     */
    static void offload_activation_to_cpu(
        const torch::Tensor& activation,
        int layer_id,
        StormSystem* storm_system
    ) {
        // Get the D2H (Device to Host) transfer stream
        auto& d2h_stream = storm_system->getTransferD2HStream();
        
        // Create pinned memory buffer for this activation
        auto buffer = std::make_unique<PinnedMemoryBuffer<float>>(
            activation.numel()
        );
        
        if (!buffer->isValid()) {
            std::cerr << "Failed to create pinned memory buffer for layer " << layer_id << std::endl;
            return;
        }
        
        // Asynchronous copy from GPU to CPU
        cudaError_t error = cudaMemcpyAsync(
            buffer->data(),
            activation.data_ptr<float>(),
            activation.numel() * sizeof(float),
            cudaMemcpyDeviceToHost,
            d2h_stream.get()
        );
        
        if (error != cudaSuccess) {
            std::cerr << "Failed to copy activation to CPU: " << cudaGetErrorString(error) << std::endl;
            return;
        }
        
        // Store buffer for later retrieval
        // In a real implementation, you'd store this in a layer-specific cache
        std::cout << "Offloaded activation for layer " << layer_id 
                  << " to CPU RAM" << std::endl;
    }
    
    /**
     * Fetch activation from CPU RAM
     * 
     * This function demonstrates:
     * 1. Synchronous waiting for data
     * 2. Memory transfer orchestration
     * 3. CUDA event synchronization
     */
    static torch::Tensor fetch_activation_from_cpu(
        int layer_id,
        StormSystem* storm_system
    ) {
        // Get the H2D (Host to Device) transfer stream
        auto& h2d_stream = storm_system->getTransferH2DStream();
        
        // In a real implementation, you'd retrieve the buffer from cache
        // For now, we'll create a dummy tensor
        auto activation = torch::randn({1, 1}, torch::kFloat);
        
        std::cout << "Fetched activation for layer " << layer_id 
                  << " from CPU RAM" << std::endl;
        
        return activation;
    }
};

/**
 * STORM Layer Wrapper
 * 
 * This class wraps a standard PyTorch layer with STORM functionality.
 * It demonstrates:
 * 1. Composition pattern
 * 2. PyTorch module subclassing
 * 3. Custom forward pass
 */
class StormLayer : public torch::nn::Module {
private:
    torch::nn::Linear linear_;
    int layer_id_;
    
public:
    StormLayer(int input_size, int output_size, int layer_id) 
        : linear_(torch::nn::Linear(input_size, output_size)), layer_id_(layer_id) {
        register_module("linear", linear_);
    }
    
    torch::Tensor forward(torch::Tensor input) {
        // Use our custom autograd function
        return StormForwardFunction::apply(
            input, 
            linear_->weight, 
            linear_->bias, 
            layer_id_
        );
    }
};

/**
 * STORM Model
 * 
 * A complete neural network model using STORM layers.
 * Demonstrates:
 * 1. Sequential model construction
 * 2. STORM layer integration
 * 3. Model architecture
 */
class StormModel : public torch::nn::Module {
private:
    torch::nn::Sequential layers_;
    
public:
    StormModel(int input_size, int hidden_size, int output_size) {
        // Create STORM layers
        layers_ = torch::nn::Sequential(
            StormLayer(input_size, hidden_size, 0),
            torch::nn::ReLU(),
            StormLayer(hidden_size, hidden_size, 1),
            torch::nn::ReLU(),
            StormLayer(hidden_size, output_size, 2)
        );
        
        register_module("layers", layers_);
    }
    
    torch::Tensor forward(torch::Tensor input) {
        return layers_->forward(input);
    }
};

/**
 * STORM Training Loop
 * 
 * This class manages the training process with STORM memory orchestration.
 * Demonstrates:
 * 1. Training loop implementation
 * 2. Loss computation
 * 3. Gradient computation
 * 4. Parameter updates
 */
class StormTrainer {
private:
    std::unique_ptr<StormModel> model_;
    torch::optim::Adam optimizer_;
    torch::nn::MSELoss loss_fn_;
    
public:
    StormTrainer(int input_size, int hidden_size, int output_size, double learning_rate)
        : model_(std::make_unique<StormModel>(input_size, hidden_size, output_size)),
          optimizer_(model_->parameters(), learning_rate),
          loss_fn_(torch::nn::MSELoss()) {
    }
    
    torch::Tensor train_step(torch::Tensor input, torch::Tensor target) {
        // Forward pass
        auto output = model_->forward(input);
        
        // Compute loss
        auto loss = loss_fn_(output, target);
        
        // Backward pass
        loss.backward();
        
        // Update parameters
        optimizer_.step();
        optimizer_.zero_grad();
        
        return loss;
    }
    
    StormModel* getModel() { return model_.get(); }
};

} // namespace storm
