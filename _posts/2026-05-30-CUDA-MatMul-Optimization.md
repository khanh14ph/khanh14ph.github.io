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
__global__ void gemm_v00(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // Compute the row and column of C that this thread is responsible for.
    size_t const C_row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_col_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // Each thread compute
    // C[C_row_idx, C_col_idx] = alpha * A[C_row_idx, :] * B[:, C_col_idx] +
    // beta * C[C_row_idx, C_col_idx].
    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_gemm_kernel_v00(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(m) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(n) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v00<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B,
                                       `              ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```
