#!/usr/bin/env python3
"""
STORM - Synchronous Transfer Orchestration for RAM Memory
PyTorch C++/CUDA Extension Setup

This setup.py file configures the compilation of STORM's C++/CUDA code
into a Python extension that can be imported and used with PyTorch.

Key Features:
- Automatic CUDA compilation with nvcc
- C++17 standard support
- Optional NVIDIA profiling tools integration
- Cross-platform compatibility
- Proper dependency management
"""

import os
import sys
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
from setuptools import setup, find_packages
import platform

# Check PyTorch version
if torch.__version__ < "1.9.0":
    raise RuntimeError("STORM requires PyTorch 1.9.0 or later")

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
if not CUDA_AVAILABLE:
    print("Warning: CUDA is not available. STORM will be compiled in CPU-only mode.")
    print("For full STORM functionality, please ensure CUDA is properly installed.")

# Get CUDA version
CUDA_VERSION = None
if CUDA_AVAILABLE:
    CUDA_VERSION = torch.version.cuda
    print(f"CUDA Version: {CUDA_VERSION}")

# Platform-specific settings
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"

# Compiler flags
cpp_flags = [
    "-std=c++17",
    "-O3",
    "-Wall",
    "-Wextra",
    "-Wno-unused-parameter",
    "-Wno-unused-variable",
]

# CUDA-specific flags
cuda_flags = [
    "-std=c++17",
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-Xcompiler", "-Wall",
    "-Xcompiler", "-Wextra",
    "-Xcompiler", "-Wno-unused-parameter",
    "-Xcompiler", "-Wno-unused-variable",
]

# Windows-specific adjustments
if IS_WINDOWS:
    cpp_flags = ["/std:c++17", "/O2", "/W3"]
    cuda_flags = [
        "-std=c++17",
        "-O3",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-Xcompiler", "/std:c++17",
        "-Xcompiler", "/O2",
        "-Xcompiler", "/W3",
    ]

# Include directories
include_dirs = [
    ".",
    torch.utils.cpp_extension.include_paths(),
]

# Library directories
library_dirs = []
if CUDA_AVAILABLE:
    library_dirs.extend(torch.utils.cpp_extension.library_paths())

# Libraries to link
libraries = []
if CUDA_AVAILABLE:
    libraries.extend(["cudart", "cublas", "curand"])

# Optional NVIDIA profiling tools
NVTX_AVAILABLE = False
try:
    import nvtx
    NVTX_AVAILABLE = True
    print("NVIDIA profiling tools (NVTX) detected - enabling advanced profiling")
except ImportError:
    print("NVIDIA profiling tools not available - using basic profiling mode")

if NVTX_AVAILABLE:
    cpp_flags.append("-DNVTX_ENABLED")
    cuda_flags.append("-DNVTX_ENABLED")
    libraries.append("nvToolsExt")

# Define the STORM extension
extensions = []

# Main STORM extension - using header-only implementation
storm_sources = [
    "storm_bindings.cpp",  # Python bindings (this will be the main entry point)
]

# Check if we have the bindings file
existing_sources = []
for source in storm_sources:
    if os.path.exists(source):
        existing_sources.append(source)
    else:
        print(f"Note: {source} not found - will create minimal bindings")

# If no source files exist, create a minimal one
if not existing_sources:
    print("No source files found, creating minimal bindings...")
    minimal_bindings = '''
#include <torch/extension.h>

torch::Tensor storm_test(torch::Tensor input) {
    return input * 2.0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("storm_test", &storm_test, "STORM test function");
}
'''
    with open("storm_bindings.cpp", "w") as f:
        f.write(minimal_bindings)
    existing_sources = ["storm_bindings.cpp"]

# If we have CUDA, create CUDA extension
if CUDA_AVAILABLE and existing_sources:
    storm_extension = CUDAExtension(
        name="storm_cuda",
        sources=existing_sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args={
            "cxx": cpp_flags,
            "nvcc": cuda_flags,
        },
        define_macros=[
            ("CUDA_AVAILABLE", "1"),
            ("TORCH_EXTENSION_NAME", "storm_cuda"),
        ] + ([("NVTX_ENABLED", "1")] if NVTX_AVAILABLE else []),
    )
    extensions.append(storm_extension)

# Fallback C++ extension for CPU-only mode
elif existing_sources:
    storm_extension = CppExtension(
        name="storm_cpu",
        sources=existing_sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=cpp_flags,
        define_macros=[
            ("CUDA_AVAILABLE", "0"),
            ("TORCH_EXTENSION_NAME", "storm_cpu"),
        ],
    )
    extensions.append(storm_extension)

# Ensure we have at least one extension
if not extensions:
    print("Warning: No extensions created. Creating minimal fallback...")
    minimal_extension = CppExtension(
        name="storm_minimal",
        sources=["storm_bindings.cpp"],
        include_dirs=["."],
        extra_compile_args=["-std=c++17"],
        define_macros=[("TORCH_EXTENSION_NAME", "storm_minimal")],
    )
    extensions.append(minimal_extension)

# Python package configuration
setup(
    name="storm",
    version="1.0.0",
    author="STORM Development Team",
    author_email="storm@auditve.com",
    description="Synchronous Transfer Orchestration for RAM Memory - VRAM-Free Deep Learning",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/mjones2504/STORM",
    packages=find_packages(),
    ext_modules=extensions,
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=False,  # Disable ninja for better compatibility
            no_python_abi_suffix=True,
        )
    },
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "profiling": [
            "nvtx>=0.2.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="pytorch cuda deep-learning memory-optimization gpu-computing",
    project_urls={
        "Bug Reports": "https://github.com/mjones2504/STORM/issues",
        "Source": "https://github.com/mjones2504/STORM",
        "Documentation": "https://github.com/mjones2504/STORM/blob/main/README.md",
    },
    zip_safe=False,
    include_package_data=True,
)
