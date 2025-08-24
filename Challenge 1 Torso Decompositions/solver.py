# setup_cuda.py
import os, sys, subprocess

def build_cuda_module():
    print("🚀 Building CUDA evaluation kernel...")
    compile_command = [
        "nvcc", "-Xcompiler", "-fPIC", "-shared",
        "-o", "evaluator.so", "evaluator.cu"
    ]
    try:
        subprocess.check_call(compile_command)
        print("✅ CUDA kernel built successfully.")
    except Exception as e:
        print(f"❌ FATAL: Failed to build CUDA kernel.")
        print(f"   Please ensure the NVIDIA CUDA Toolkit (nvcc) is installed and in your system's PATH.")
        print(f"   Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build_cuda_module()
