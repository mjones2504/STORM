# PyTorch Integration Guide for STORM Development

## Advanced C++ Concepts with PyTorch Integration

### 1. PyTorch Autograd Function Subclassing

```cpp
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
        
        // Fetch activation from CPU RAM asynchronously
        auto activation = fetch_activation_from_cpu(layer_id, storm_system.get());
        
        // Compute gradients
        auto grad_input = torch::linear(grad_output, weight.t(), torch::Tensor());
        auto grad_weight = torch::mm(grad_output.t(), input);
        auto grad_bias = grad_output.sum(0);
        
        return {grad_input, grad_weight, grad_bias, torch::Tensor()};
    }
};
```

**Key C++ Concepts Demonstrated:**

#### **1. Template Specialization**
```cpp
class StormForwardFunction : public torch::autograd::Function<StormForwardFunction>
```
- **CRTP (Curiously Recurring Template Pattern)**: The class inherits from a template of itself
- **Static polymorphism**: Compile-time method resolution
- **Type safety**: Ensures correct function signatures

#### **2. Context Management**
```cpp
ctx->save_for_backward({input, weight, bias});
ctx->saved_data["layer_id"] = layer_id;
```
- **Context object**: Stores intermediate values for backward pass
- **Automatic memory management**: PyTorch handles tensor lifecycle
- **Type-safe storage**: Different data types stored safely

#### **3. Tensor Operations**
```cpp
torch::Tensor output = torch::linear(input, weight, bias);
torch::Tensor activation = output.clone();
```
- **Tensor operations**: High-level mathematical operations
- **Memory management**: Automatic tensor allocation/deallocation
- **GPU acceleration**: Operations run on GPU when available

### 2. PyTorch Module Subclassing

```cpp
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
        return StormForwardFunction::apply(
            input, 
            linear_->weight, 
            linear_->bias, 
            layer_id_
        );
    }
};
```

**Key C++ Concepts Demonstrated:**

#### **1. Module Registration**
```cpp
register_module("linear", linear_);
```
- **Parameter management**: Automatic parameter registration
- **Serialization**: Modules can be saved/loaded
- **Gradient computation**: Parameters automatically tracked

#### **2. Composition Pattern**
```cpp
torch::nn::Linear linear_;
```
- **Composition over inheritance**: Building complex modules from simpler ones
- **Encapsulation**: Internal implementation hidden
- **Reusability**: Linear layer can be reused

#### **3. Custom Forward Pass**
```cpp
torch::Tensor forward(torch::Tensor input) {
    return StormForwardFunction::apply(/* ... */);
}
```
- **Function application**: Using custom autograd functions
- **Automatic differentiation**: Gradients computed automatically
- **Memory orchestration**: STORM logic integrated seamlessly

### 3. Model Architecture and Training

```cpp
class StormModel : public torch::nn::Module {
private:
    torch::nn::Sequential layers_;
    
public:
    StormModel(int input_size, int hidden_size, int output_size) {
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
```

**Key C++ Concepts Demonstrated:**

#### **1. Sequential Module Construction**
```cpp
torch::nn::Sequential layers_;
```
- **Sequential execution**: Layers executed in order
- **Automatic forward pass**: Input flows through all layers
- **Parameter aggregation**: All parameters collected automatically

#### **2. Layer Composition**
```cpp
StormLayer(input_size, hidden_size, 0),
torch::nn::ReLU(),
StormLayer(hidden_size, hidden_size, 1),
```
- **Mixed layer types**: Custom STORM layers + standard PyTorch layers
- **Flexible architecture**: Easy to modify network structure
- **Layer identification**: Each layer has unique ID for STORM orchestration

### 4. Training Loop Implementation

```cpp
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
};
```

**Key C++ Concepts Demonstrated:**

#### **1. Smart Pointer Management**
```cpp
std::unique_ptr<StormModel> model_;
```
- **Automatic memory management**: Model destroyed when trainer destroyed
- **Exception safety**: Memory cleaned up even if exceptions occur
- **Ownership semantics**: Clear ownership of model

#### **2. Optimizer Integration**
```cpp
torch::optim::Adam optimizer_(model_->parameters(), learning_rate);
```
- **Parameter tracking**: Optimizer automatically tracks all model parameters
- **Gradient updates**: Automatic parameter updates
- **Learning rate management**: Configurable learning rates

#### **3. Loss Function Integration**
```cpp
torch::nn::MSELoss loss_fn_;
auto loss = loss_fn_(output, target);
```
- **Loss computation**: Standard loss functions
- **Gradient computation**: Automatic gradient computation
- **Backpropagation**: Gradients flow back through network

### 5. Advanced Memory Management

```cpp
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
    
    // Asynchronous copy from GPU to CPU
    cudaMemcpyAsync(
        buffer->data(),
        activation.data_ptr<float>(),
        activation.numel() * sizeof(float),
        cudaMemcpyDeviceToHost,
        d2h_stream.get()
    );
}
```

**Key C++ Concepts Demonstrated:**

#### **1. Asynchronous Memory Transfer**
```cpp
cudaMemcpyAsync(/* ... */, d2h_stream.get());
```
- **Non-blocking operations**: GPU continues computing while data transfers
- **Stream management**: Different streams for different operations
- **Concurrent execution**: Compute and transfer happen simultaneously

#### **2. Pinned Memory Usage**
```cpp
auto buffer = std::make_unique<PinnedMemoryBuffer<float>>(activation.numel());
```
- **Fast transfers**: Pinned memory enables fast CPU-GPU transfers
- **Memory alignment**: Proper memory alignment for optimal performance
- **Automatic cleanup**: Smart pointers handle memory deallocation

#### **3. Tensor Data Access**
```cpp
activation.data_ptr<float>()
activation.numel()
```
- **Raw data access**: Direct access to tensor data
- **Type safety**: Template-based type checking
- **Memory layout**: Understanding tensor memory layout

## Summary of Advanced C++ Concepts

### **1. Template Metaprogramming**
- **CRTP**: Curiously Recurring Template Pattern
- **Type traits**: Compile-time type information
- **SFINAE**: Substitution Failure Is Not An Error

### **2. Memory Management**
- **RAII**: Resource Acquisition Is Initialization
- **Smart pointers**: Automatic memory management
- **Move semantics**: Efficient resource transfer

### **3. Exception Safety**
- **Strong exception safety**: Operations either complete or leave state unchanged
- **RAII guarantees**: Resources cleaned up even if exceptions occur
- **Exception propagation**: Proper exception handling

### **4. Performance Optimization**
- **Asynchronous operations**: Non-blocking I/O
- **Memory pooling**: Efficient memory reuse
- **Stream parallelism**: Concurrent execution

### **5. API Design**
- **Composition over inheritance**: Building complex systems from simpler components
- **Interface segregation**: Small, focused interfaces
- **Dependency injection**: Flexible component configuration

This guide demonstrates how advanced C++ concepts enable the creation of sophisticated systems like STORM, combining high-performance computing with modern software engineering practices.
