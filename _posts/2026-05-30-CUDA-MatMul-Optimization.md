---
title: "CUDA Matrix Multiplication Optimization"
date: 2026-05-30
categories:
  - GPU
---

## Introduction

General matrix multiplication (GEMM) is a fundamental operation in like everywhere now.

In this article, we will discuss how to optimize the performance of FP32 GEMM on NVIDIA GPUs using CUDA and how to extend the FP32 GEMM optimizations to FP16 GEMM using NVIDIA Tensor Cores.

## General Matrix Multiplication

GEMM operation computes $$D = AB + C$$, where $$D \in \mathbb{R}^{m \times n}$$, $$A \in \mathbb{R}^{m \times k}$$, $$B \in \mathbb{R}^{k \times n}$$, $$C \in \mathbb{R}^{m \times n}$$. In computer programs, usually $$A$$ and $$B$$ are constant input matrices and $$C$$ will be overwritten by the output matrix $$D$$.

In our implementations, we assume all the matrices, $$A, B, C$$ and $$D$$, are stored in the row-major order on memory with the leading dimension padded to 64 bytes for FP32 matrices and 32 bytes for FP16 matrices.

## Sequential implementation

```cpp
template <typename T>
void cpu_gemm(size_t m, size_t n, size_t k, T alpha, T const* A, size_t lda,
              T const* B, size_t ldb, T beta, T* C, size_t ldc) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = 0;
            for (size_t p = 0; p < k; ++p) {
                sum += A[i * lda + p] * B[p * ldb + j];
            }
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

```
## Naive Implementation with Non-Coalesced Memory Access

The naive implementation is to use 2D blocks, where each thread is responsible for computing one element of the output matrix. 

The following code snippet shows the naive implementation.

```cpp
template <typename T>
__global__ void gemm_v0(int m, int n, int k, T alpha, T const* A, int lda,
                        T const* B, int ldb, T beta, T* C, int ldc)
{
    int const row{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    int const col{static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y)};

    if (row < m && col < n)
    {
        // 64-bit offsets computed once; everything else stays 32-bit
        T const* A_row{A + static_cast<size_t>(row) * lda};
        T const* B_col{B + col};
        T* C_ptr{C + static_cast<size_t>(row) * ldc + col};

        T sum{static_cast<T>(0)};
        for (int kk{0}; kk < k; ++kk)
        {
            sum += A_row[kk] * B_col[static_cast<size_t>(kk) * ldb];
        }
        *C_ptr = alpha * sum + beta * *C_ptr;
    }
}
```

[Benchmark] gemm_v0 | Avg Time: 32.8361 ms  | Performance: 65.4001 TFLOPS

v1:
```cpp

template <typename T>
__global__ void gemm_v1(int m, int n, int k, T alpha, T const* A, int lda,
                        T const* B, int ldb, T beta, T* C, int ldc)
{
    // Transposed thread mapping vs v0: x -> columns (n), y -> rows (m)
    int const col = blockIdx.x * blockDim.x + threadIdx.x;
    int const row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n)
    {
        T const* A_row{A + row * lda};
        T const* B_col{B + col};
        T* C_ptr{C + row * ldc + col};

        T sum{0};
        for (int kk{0}; kk < k; ++kk)
        {
            sum += A_row[kk] * B_col[kk * ldb];
        }
        *C_ptr = alpha * sum + beta * *C_ptr;
    }
}

template <typename T>
void launch_gemm_kernel_v1(int m, int n, int k, T const* alpha, T const* A,
                           int lda, T const* B, int ldb, T const* beta, T* C,
                           int ldc, cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    // x covers columns (n), y covers rows (m) to match the kernel's mapping
    dim3 const grid_dim{
        (n + block_dim.x - 1U) / block_dim.x,
        (m + block_dim.y - 1U) / block_dim.y, 1U};

    gemm_v1<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B,
                                                    ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```

v2:
```cpp
#define BLOCK_SIZE 32
template <typename T>
__global__ void gemm_v2(int m, int n, int k, T alpha, T const* A, int lda, T const* B, int ldb, T beta,  T* C, int ldc){
    
    __shared__ T A_shared[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ T B_shared[BLOCK_SIZE*BLOCK_SIZE];
    int tid_col=threadIdx.x;
    int tid_row=threadIdx.y;

    A+=BLOCK_SIZE*blockIdx.x*lda;
    B+=BLOCK_SIZE*blockIdx.y;
    C+=BLOCK_SIZE*blockIdx.x*ldc+BLOCK_SIZE*blockIdx.y;
    T sum=0;
    for (int kk=0;kk<k;kk+=BLOCK_SIZE){
        A_shared[tid_row*BLOCK_SIZE+tid_col]=A[tid_col+tid_row*lda];
        B_shared[tid_row*BLOCK_SIZE+tid_col]=B[tid_col+tid_row*ldb];
        __syncthreads();
        A+=BLOCK_SIZE;
        B+=BLOCK_SIZE*ldb;
        
        for (int i=0;i<BLOCK_SIZE;i++){
            sum+=A_shared[tid_row*BLOCK_SIZE+i]*B_shared[tid_col+i*BLOCK_SIZE];
        }
        __syncthreads();
    }
    C[tid_row*ldc+tid_col]=alpha*sum+beta*C[tid_row*ldc+tid_col];
}

template <typename T>
void launch_gemm_kernel_v2(int m, int n, int k, T const* alpha, T const* A, int lda, T const* B, int ldb, T const* beta,  T* C, int ldc, cudaStream_t stream){
    dim3 const block_size{32U,32U,1U};
    dim3 const grid_size{
        (m+block_size.x-1)/block_size.x,
        (n+block_size.y-1)/block_size.y, 1U};
    gemm_v2<T><<<grid_size, block_size, 0U, stream>>>(m, n, k, *alpha, A, lda, B,
                                                    ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```

v3:
```cpp
#define BLOCK_SIZE 32
template <typename T>
__global__ void gemm_v3(int m, int n, int k, T alpha, T const* A, int lda, T const* B, int ldb, T beta,  T* C, int ldc){
    
    __shared__ T A_shared[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ T B_shared[BLOCK_SIZE*BLOCK_SIZE];
    int tid_col=threadIdx.x;

    A+=BLOCK_SIZE*blockIdx.x*lda;
    B+=BLOCK_SIZE*blockIdx.y;
    C+=BLOCK_SIZE*blockIdx.x*ldc+BLOCK_SIZE*blockIdx.y;
    T sum=0;
    T threadResult[BLOCK_SIZE]={0};
    for (int kk=0;kk<k;kk+=BLOCK_SIZE){
        for (int row=0; row<BLOCK_SIZE;row++){
            A_shared[row*BLOCK_SIZE+tid_col]=A[tid_col+row*lda];
            B_shared[row*BLOCK_SIZE+tid_col]=B[tid_col+row*ldb];
        } 
        __syncthreads();
        A+=BLOCK_SIZE;
        B+=BLOCK_SIZE*ldb;
        
        for (int i=0;i<BLOCK_SIZE;i++){
            T B_cell=B_shared[tid_col+i*BLOCK_SIZE];
            for (int row=0;row<BLOCK_SIZE;row++){
                threadResult[row]+=A_shared[row*BLOCK_SIZE+i]*B_cell;
            }
            

        }
        __syncthreads();
    }
    for (int i=0;i<BLOCK_SIZE;i++){
        C[i*ldc+tid_col]=alpha*threadResult[i]+beta*C[i*ldc+tid_col];
    }
    
}

template <typename T>
void launch_gemm_kernel_v3(int m, int n, int k, T const* alpha, T const* A, int lda, T const* B, int ldb, T const* beta,  T* C, int ldc, cudaStream_t stream){
    dim3 const block_size{32U,1U,1U};
    dim3 const grid_size((m + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (n + BLOCK_SIZE - 1) / BLOCK_SIZE, 1U);
    gemm_v3<T><<<grid_size, block_size, 0U, stream>>>(m, n, k, *alpha, A, lda, B,
                                                    ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```


BENCHMARK:
```
M=1024 N=1024 K=1024
--------------------------------------------------------
[Benchmark] gemm_v0 | Avg Time: 32.0627 ms  | Performance: 66.9777 TFLOPS
[Benchmark] gemm_v1  | Avg Time: 4.61722 ms  | Performance: 465.104 TFLOPS
[Benchmark] gemm_v2  | Avg Time: 1.19613 ms  | Performance: 1795.35 TFLOPS
[Benchmark] gemm_v3  | Avg Time: 0.723763 ms  | Performance: 2967.11 TFLOPS
[Benchmark] cuBLAS | Avg Time: 0.229274 ms  | Performance: 9366.47 TFLOPS
```