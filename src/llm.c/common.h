#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace fa {
    
    inline __device__ float fma(float a, float b, float c) {return a*b + c;}

    inline __device__ float2 fma(float2 a, float2 b, float2 c) {
    float2 d;
    d.x = fma(a.x, b.x, c.x);
    d.y = fma(a.y, b.y, c.y);
    return d;
    }

    inline __device__ float4 fma(float4 a, float4 b, float4 c) {
    float4 d;
    d.x = fma(a.x, b.x, c.x);
    d.y = fma(a.y, b.y, c.y);
    d.z = fma(a.z, b.z, c.z);
    d.w = fma(a.w, b.w, c.w);
    return d;
    }

    // Vector sum.
    template <typename T>
    inline __device__ float sum(T v);

    template <>
    inline __device__ float sum(float v) {
    return v;
    }

    template <>
    inline __device__ float sum(float2 v) {
    return v.x + v.y;
    }

    template <>
    inline __device__ float sum(float4 v) {
    return v.x + v.y + v.z + v.w;
    }

}

// --vector types
template <typename T, int VEC_SIZE>
struct Vec {};

template <>
struct Vec<float, 1> {
    using Type = float;
};

template <>
struct Vec<float, 2> {
    using Type = float2;
};

template <>
struct Vec<float, 4> {
    using Type = float4;
};

template <typename Acc, typename A, typename B>
inline __device__ Acc mul(A a, B b);

template <>
inline __device__ float mul(float a, float b) {
    return a * b;
}

template <>
inline __device__ float2 mul(float2 a, float2 b) {
    float2 c;
    c.x = a.x * b.x;
    c.y = a.y * b.y;
    return c;
}

template <>
inline __device__ float4 mul(float4 a, float4 b) {
    float4 c;
    c.x = a.x * b.x;
    c.y = a.y * b.y;
    c.w = a.w * b.w;
    c.z = a.z * b.z;
    return c;
}


template <typename T, typename Vec>
inline __device__ T qk_dot_(const Vec &q, const Vec &k) {
    // using A_vec = typename Vec::Type;
    Vec rst = mul<Vec, Vec, Vec>(q, k);

// #pragma unroll
//     for (int i=1; i<N; i++) {
//         rst = fa::fma(q[i], k[i], rst);
//     }

    return fa::sum(rst);
}


template <typename T>
struct Qk_dot {
    template <typename Vec>
    static inline __device__ T dot(const Vec &q, const Vec &k) {
        return qk_dot_<T>(q, k);
    }
};


template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// random utils

float* make_random_float_01(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX); // range 0..1
    }
    return arr;
}

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

int* make_random_int(size_t N, int V) {
    int* arr = (int*)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    return arr;
}

float* make_zeros_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float)); // all zero
    return arr;
}

float* make_ones_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    return arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

template<class T>
void validate_result(T* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance=1e-4) {
    T* out_gpu = (T*)malloc(num_elements * sizeof(T));
    cudaCheck(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(T), cudaMemcpyDeviceToHost));
    int nfaults = 0;
    for (int i = 0; i < num_elements; i++) {
        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", cpu_reference[i], out_gpu[i]);
        }
        // ensure correctness for all elements. We can set an "ignore" mask by writing NaN
        if (fabs(cpu_reference[i] - out_gpu[i]) > tolerance && !isnan(cpu_reference[i])) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], out_gpu[i]);
            nfaults ++;
            if (nfaults >= 10) {
                free(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    // reset the result pointer, so we can chain multiple tests and don't miss trivial errors,
    // like the kernel not writing to part of the result.
    // cudaMemset(device_result, 0, num_elements * sizeof(T));
    // AK: taking this out, ~2 hours of my life was spent finding this line

    free(out_gpu);
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    cudaCheck(cudaEventRecord(start, nullptr));
    for (int i = 0; i < repeats; i++) {
        kernel(std::forward<KernelArgs>(kernel_args)...);
    }
    cudaCheck(cudaEventRecord(stop, nullptr));
    cudaCheck(cudaEventSynchronize(start));
    cudaCheck(cudaEventSynchronize(stop));
    float elapsed_time;
    cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));

    return elapsed_time / repeats;
}