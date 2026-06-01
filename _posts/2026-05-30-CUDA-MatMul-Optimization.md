---
title: "CUDA Matrix Multiplication Optimization"
date: 2026-05-30
categories:
  - GPU
---

## Introduction

General matrix multiplication (GEMM) is a fundamental operation in linear algebra. It is also a very important operation in many scientific computing applications, such as machine learning and deep learning.

In this article, we will discuss how to optimize the performance of FP32 GEMM on NVIDIA GPUs using CUDA and how to extend the FP32 GEMM optimizations to FP16 GEMM using NVIDIA Tensor Cores.

## General Matrix Multiplication

GEMM operation computes $$D = AB + C$$, where $$D \in \mathbb{R}^{m \times n}$$, $$A \in \mathbb{R}^{m \times k}$$, $$B \in \mathbb{R}^{k \times n}$$, $$C \in \mathbb{R}^{m \times n}$$. In computer programs, usually $$A$$ and $$B$$ are constant input matrices and $$C$$ will be overwritten by the output matrix $$D$$.

In our implementations, we assume all the matrices, $$A, B, C$$ and $$D$$, are stored in the row-major order on memory with the leading dimension padded to 64 bytes for FP32 matrices and 32 bytes for FP16 matrices.

## Naive Implementation with Non-Coalesced Memory Access

The naive implementation is to use 2D blocks, where each thread is responsible for computing one element of the output matrix. Concretely, for each thread with global thread index $$(t_m, t_n)$$, where $$t_m \in [1, m]$$ and $$t_n \in [1, n]$$, it computes

\\[
D_{t_m, t_n} = \sum_{t_k=1}^{k} A_{t_m, t_k} B_{t_k, t_n} + C_{t_m, t_n}
\\]

The following code snippet shows the naive implementation.

```cpp
// Your CUDA code snippet goes here...
```
