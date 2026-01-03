#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/types.h>
#include <cuda.h>

__global__ void matmul_ker(const float* matrix_1, const float* matrix_2, float* matrix_output, int height_1, int width_2, int K){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int pos = width_2 * j + i;
    if ((i < width_2) and (j <height_1)){
        float accum = 0.0;
        for (int k = 0; k < K + 1; k++){
            accum += matrix_1[K * j + k] * matrix_2[width_2 * k + i];
        }
        matrix_output[pos] = accum;
    }

}

torch::Tensor matmul(torch::Tensor matrix_1,torch::Tensor matrix_2){
    // Input dimensions
    const auto height_1 = matrix_1.size(0);
    const auto K = matrix_2.size(0);
    const auto width_2 = matrix_2.size(0);
    // Grid Params
    int block_size = 32;
    dim3 block_dim(block_size, block_size, 1);
    dim3 grid_dim((width_2 + block_size - 1) / block_size, (height_1 + block_size - 1) / block_size);
    // Init output
    auto output = torch::empty(height_1, width_2);
    // Launch Grid
    matmul_ker<<<grid_dim, block_dim>>>(matrix_1.data_ptr<float>(), matrix_2.data_ptr<float>(), output.data_ptr<float>(), height_1, width_2, K);
    return output;
}