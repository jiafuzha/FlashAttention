#include <cuda_runtime.h>
#include <stdio.h>

#include "common.h"

__global__ void printWarpSize() {
    printf("warp size: %d\n", warpSize);

    __shared__ float data[32][32];

    for (int j=0; j<32; j++) {
        data[threadIdx.x][j] = threadIdx.x * j;
    }
    
    float4 my_data;
    float sum{0.0};
    for (int i=0; i<(32/4); i++) {
        // my_data.x = data[threadIdx.x][i*4];
        // my_data.y = data[threadIdx.x][i*4 + 1];
        // my_data.z = data[threadIdx.x][i*4 + 2];
        // my_data.w = data[threadIdx.x][i*4 + 3];
        my_data = *reinterpret_cast<const float4*>(&data[threadIdx.x][i*4]);
        sum += Qk_dot<float>::dot(my_data, my_data);
    }
    printf("sum %d: %f\n", threadIdx.x, sum);
}

__global__ void test_shuffle_float_array() {
    float arr[4];
    for (int j=0; j<4; j++) {
        arr[j] = threadIdx.x + j;
    }
    for (int j=0; j<4; j++) {
        for (int m=8/2; m>=1; m/=2)
            arr[j] += __shfl_xor_sync((uint32_t)-1, arr[j], m);
    }
    printf("threadIdx: %d, sum is %f, %f, %f, %f\n", threadIdx.x, arr[0], arr[1], arr[2], arr[3]);
}

__global__ void test_shuffle_width() {
    float arr[4];
    // float max[4];
    const int MAX_MASK=4;
    for (int j=0; j<4; j++) {
        arr[j] = threadIdx.x + j;
        // max[j] = -INFINITY;
    }
    printf("threadIdx: %d, arr is %f, %f, %f, %f\n", threadIdx.x, arr[0], arr[1], arr[2], arr[3]);
    for (int j=0; j<4; j++) {
        for (int m=MAX_MASK/2; m>=1; m/=2)
            arr[j] = fmaxf(arr[j], __shfl_xor_sync((uint32_t)-1, arr[j], m));
    }
    for (int j=0; j<4; j++) {
        arr[j] = __shfl_sync((uint32_t)-1, arr[j], 0, MAX_MASK);
    }
    printf("threadIdx: %d, max is %f, %f, %f, %f\n", threadIdx.x, arr[0], arr[1], arr[2], arr[3]);
}

int main() {
    // printWarpSize<<<1,32>>>();
    // test_shuffle_float_array<<<1,32>>>();
    test_shuffle_width<<<1,16>>>();
    cudaDeviceSynchronize();

    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("max sram size per block: %d\n", max_sram_size);

    int max_sram_size_per_sm;
    cudaDeviceGetAttribute(&max_sram_size_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
    printf("max sram size per sm: %d\n", max_sram_size_per_sm);

    int max_registers_32bit_per_block;
    cudaDeviceGetAttribute(&max_registers_32bit_per_block, cudaDevAttrMaxRegistersPerBlock, 0);
    printf("max number of registers per block: %d\n", max_registers_32bit_per_block);

    int max_registers_32bit_per_sm;
    cudaDeviceGetAttribute(&max_registers_32bit_per_sm, cudaDevAttrMaxRegistersPerMultiprocessor, 0);
    printf("max number of registers per sm: %d\n", max_registers_32bit_per_sm);

    int max_blocks_per_sm;
    cudaDeviceGetAttribute(&max_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, 0);
    printf("max number of blocks per sm: %d\n", max_blocks_per_sm);

    int max_threads_per_block;
    cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
    printf("max number of threads per block: %d\n", max_threads_per_block);

    
    
}