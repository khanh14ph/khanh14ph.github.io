---
title: "High Performance Neural Network Implementation"
excerpt: "Built a feedforward neural network from scratch for MNIST classification in C++/CUDA, achieving 90% of cuBLAS throughput on an NVIDIA V100 GPU."
collection: portfolio
---

**Role:** Developer
**Date:** January 2026 – February 2026
**Technologies:** C++, CUDA

- Built a feedforward neural network from scratch for MNIST classification, implementing forward/backward propagation, mini-batch SGD, and cross-entropy loss.
- Engineered custom CUDA kernels for matrix multiplication operations, achieving 90% of cuBLAS throughput on an NVIDIA V100 GPU.
- Maximized GPU utilization by masking data-transfer latency, utilizing pinned memory, and multiple asynchronous CUDA streams to overlap host-to-device loading with computation.
