#pragma once

#include <cuda_runtime.h>

void test_kernel(const unsigned int blocks, const unsigned int threads_per_block, const unsigned int shared_memory_per_block, cudaStream_t stream, uchar4 *d_frames, const unsigned int buffer_count, const unsigned int buffer, const unsigned int width, const unsigned int height, void *user_pointer);