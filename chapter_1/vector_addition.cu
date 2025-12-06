#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/types.h>
#include <cuda.h>

// my first CUDA kernel
__global__ void vec_add_kernel(const float* A, const float* B, float* C, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<n){
        C[i] = A[i] + B[i];
    }
}

torch::Tensor vector_addition(torch::Tensor vector1, torch::Tensor vector2){
    const auto size = vector1.size(0);
    auto result = torch::empty_like(vector1);

    dim3 threads_per_block(256);
    dim3 number_of_blocks((size + threads_per_block.x - 1) / threads_per_block.x);

    vec_add_kernel<<<number_of_blocks, threads_per_block>>>(vector1.data_ptr<float>(), vector2.data_ptr<float>(), result.data_ptr<float>(), size);

    return result;
    }