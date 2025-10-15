#include <iostream>
#include <vector>
#include <memory>
#include <chrono>

// CUDA includes
#include <cuda_runtime.h>
#include <cuda.h>

// PyTorch includes
#include <torch/torch.h>
#include <torch/extension.h>

// Our STORM classes
#include "storm_core.h"
#include "storm_autograd.h"
#include "storm_orchestration.h"
#include "storm_profiling.h"

/**
 * STORM - Synchronous Transfer Orchestration for RAM Memory
 * 
 * This is the main entry point for our STORM system.
 * Let me teach you advanced C++ concepts as we build this!
 */

int main() {
    std::cout << "STORM: Synchronous Transfer Orchestration for RAM Memory" << std::endl;
    std::cout << "========================================================" << std::endl;
    
    // Initialize CUDA
    std::cout << "Initializing CUDA environment..." << std::endl;
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to initialize CUDA device!" << std::endl;
        return 1;
    }
    std::cout << "CUDA initialized successfully!" << std::endl;
    
    // Create STORM system using RAII
    std::cout << "\nCreating STORM system..." << std::endl;
    storm::StormSystem storm;
    
    if (!storm.isInitialized()) {
        std::cerr << "Failed to initialize STORM system!" << std::endl;
        return 1;
    }
    std::cout << "STORM system ready!" << std::endl;
    
    // Demonstrate memory management with smart pointers
    std::cout << "\nCreating pinned memory buffer..." << std::endl;
    auto buffer = std::make_unique<storm::PinnedMemoryBuffer<float>>(1000);
    
    if (!buffer->isValid()) {
        std::cerr << "Failed to create pinned memory buffer!" << std::endl;
        return 1;
    }
    std::cout << "Pinned memory buffer created successfully!" << std::endl;
    std::cout << "Buffer size: " << buffer->size() << " elements" << std::endl;
    
    // Demonstrate move semantics
    std::cout << "\nDemonstrating move semantics..." << std::endl;
    auto moved_buffer = std::move(buffer);  // Transfer ownership
    std::cout << "Original buffer is now: " << (buffer ? "valid" : "invalid") << std::endl;
    std::cout << "Moved buffer is: " << (moved_buffer ? "valid" : "invalid") << std::endl;
    
    // Demonstrate STORM autograd functions
    std::cout << "\n=== STORM Autograd Demonstration ===" << std::endl;
    
    try {
        // Create a simple STORM model
        std::cout << "Creating STORM model..." << std::endl;
        storm::StormModel model(10, 64, 1);  // 10 inputs, 64 hidden, 1 output
        
        // Create trainer
        std::cout << "Creating STORM trainer..." << std::endl;
        storm::StormTrainer trainer(10, 64, 1, 0.001);  // learning rate 0.001
        
        // Create sample data
        std::cout << "Creating sample data..." << std::endl;
        auto input = torch::randn({32, 10});  // batch size 32, 10 features
        auto target = torch::randn({32, 1});  // batch size 32, 1 output
        
        // Training loop demonstration
        std::cout << "\nRunning training loop..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int epoch = 0; epoch < 5; ++epoch) {
            auto loss = trainer.train_step(input, target);
            std::cout << "Epoch " << epoch + 1 << ", Loss: " << loss.item<float>() << std::endl;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "\nTraining completed in " << duration.count() << " ms" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during STORM demonstration: " << e.what() << std::endl;
        return 1;
    }
    
    // Demonstrate advanced STORM orchestration
    std::cout << "\n=== STORM Advanced Orchestration ===" << std::endl;
    
    try {
        // Create STORM orchestrator
        std::cout << "Creating STORM orchestrator..." << std::endl;
        storm::StormOrchestrator orchestrator;
        
        if (!orchestrator.initialize()) {
            std::cerr << "Failed to initialize STORM orchestrator!" << std::endl;
            return 1;
        }
        
        // Demonstrate orchestrated forward pass
        std::cout << "Running orchestrated forward pass..." << std::endl;
        auto input = torch::randn({32, 128});
        for (int layer = 0; layer < 3; ++layer) {
            auto output = orchestrator.orchestratedForward(input, layer);
            std::cout << "Layer " << layer << " output shape: " << output.sizes() << std::endl;
        }
        
        // Demonstrate orchestrated backward pass
        std::cout << "Running orchestrated backward pass..." << std::endl;
        auto grad_output = torch::randn({32, 64});
        for (int layer = 2; layer >= 0; --layer) {
            auto grad_input = orchestrator.orchestratedBackward(grad_output, layer);
            std::cout << "Layer " << layer << " gradient shape: " << grad_input.sizes() << std::endl;
        }
        
        // Print performance report
        orchestrator.printPerformanceReport();
        
    } catch (const std::exception& e) {
        std::cerr << "Error during orchestration demonstration: " << e.what() << std::endl;
        return 1;
    }
    
    // Demonstrate STORM specification verification
    std::cout << "\n=== STORM Specification Verification ===" << std::endl;
    
    try {
        // Create specification verifier
        std::cout << "Creating STORM specification verifier..." << std::endl;
        storm::StormSpecVerifier verifier;
        
        // Run comprehensive verification
        std::cout << "Running comprehensive STORM specification verification..." << std::endl;
        bool specs_met = verifier.verifyStormSpecs();
        
        if (specs_met) {
            std::cout << "\n🎉 STORM meets all specification requirements!" << std::endl;
        } else {
            std::cout << "\n⚠️ STORM needs optimization to meet all requirements" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error during specification verification: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n=== STORM Demonstration Completed Successfully! ===" << std::endl;
    std::cout << "Key C++ concepts demonstrated:" << std::endl;
    std::cout << "- RAII (Resource Acquisition Is Initialization)" << std::endl;
    std::cout << "- Move semantics and ownership transfer" << std::endl;
    std::cout << "- Smart pointers and automatic memory management" << std::endl;
    std::cout << "- Exception safety and error handling" << std::endl;
    std::cout << "- Template classes and generic programming" << std::endl;
    std::cout << "- PyTorch C++ API integration" << std::endl;
    std::cout << "- CUDA stream management and memory orchestration" << std::endl;
    
    return 0;
}
