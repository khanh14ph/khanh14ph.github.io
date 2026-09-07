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
void launch_gemm_kernel_v0(int m, int n, int k, T const* alpha, T const* A,
                           int lda, T const* B, int ldb, T const* beta, T* C,
                           int ldc, cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(m) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(n) + block_dim.y - 1U) / block_dim.y, 1U};

    gemm_v0<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B,
                                                    ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```

v1:
```cpp
template <typename T>
__global__ void gemm_v1(int m, int n, int k, T alpha, T const* A, int lda,
                        T const* B, int ldb, T beta, T* C, int ldc)
{
    // Transposed thread mapping vs v0: x -> columns (n), y -> rows (m)
    int const col{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    int const row{static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y)};

    if (row < m && col < n)
    {
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