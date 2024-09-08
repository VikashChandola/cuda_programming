// This program computes the sum of two vectors of length N
// By: Nick from CoffeeBeforeArch

#include <cassert>
#include <vector>
#include <algorithm>

namespace cuda
{
namespace __impl
{
// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void vectorAdd(const int *__restrict a, const int *__restrict b,
                          int *__restrict c, int N) {
  // Calculate global thread ID
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Boundary check
  if (tid < N) c[tid] = a[tid] + b[tid];
}
} // __impl

std::vector<int> vectorAdd(const std::vector<int> &input_1,
                           const std::vector<int> &input_2){
  //validate input is coorectly sized.
  //All input vectors must be of same size
  assert(input_1.size() == input_2.size());
  // Allocate memory on the device
  size_t input_size = input_1.size();
  std::vector<int> output(input_size, 0);
  int *d_input_1, *d_input_2, *d_output;
  size_t bytes = sizeof(int) * input_size;
  cudaMalloc(&d_input_1, bytes);
  cudaMalloc(&d_input_2, bytes);
  cudaMalloc(&d_output, bytes);

  // Copy data from the host to the device (CPU -> GPU)
  cudaMemcpy(d_input_1, input_1.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_2, input_2.data(), bytes, cudaMemcpyHostToDevice);

  // Threads per CTA (1024)
  int NUM_THREADS = 1 << 10;

  // CTAs per Grid
  // We need to launch at LEAST as many threads as we have elements
  // This equation pads an extra CTA to the grid if N cannot evenly be divided
  // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
  int NUM_BLOCKS = (input_size + NUM_THREADS - 1) / NUM_THREADS;

  // Launch the kernel on the GPU
  // Kernel calls are asynchronous (the CPU program continues execution after
  // call, but no necessarily before the kernel finishes)
  __impl::vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_input_1, d_input_2, d_output, input_size);

  // Copy sum vector from device to host
  // cudaMemcpy is a synchronous operation, and waits for the prior kernel
  // launch to complete (both go to the default stream in this case).
  // Therefore, this cudaMemcpy acts as both a memcpy and synchronization
  // barrier.
  cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

  // Free memory on device
  cudaFree(d_input_1);
  cudaFree(d_input_2);
  cudaFree(d_output);
  return output;
}

std::vector<int> vectorAdd(const std::vector<std::vector<int>> &inputs){
  size_t inputs_size = inputs.size();
  assert(inputs_size > 0);
  const size_t input_size = inputs[0].size();
  for(int i = 0; i < inputs_size; i++){
      assert(inputs[i].size() == input_size);
  }
  std::vector<int> output = inputs[0];
  for(auto itr = inputs.cbegin() + 1;itr != inputs.cend(); itr++){
      output = vectorAdd(output, *itr);
  }
  return output;
}

std::vector<int> vectorAdd_O1(const std::vector<std::vector<int>> &inputs)
{
  size_t inputs_size = inputs.size();
  assert(inputs_size > 0);
  const size_t input_size = inputs[0].size();
  for(int i = 0; i < inputs_size; i++){
      assert(inputs[i].size() == input_size);
  }

  std::vector<int> output(input_size, 0);
  int *d_input_1, *d_input_2, *d_output;
  size_t bytes = sizeof(int) * input_size;
  int NUM_THREADS = 1 << 10;
  int NUM_BLOCKS = (input_size + NUM_THREADS - 1) / NUM_THREADS;

  cudaMalloc(&d_input_1, bytes);
  cudaMalloc(&d_input_2, bytes);
  cudaMalloc(&d_output, bytes);

  cudaMemcpy(d_input_1, inputs[0].data(), bytes, cudaMemcpyHostToDevice);
  for(auto itr = inputs.cbegin() + 1; itr != inputs.cend(); itr++) {
    cudaMemcpy(d_input_2, (*itr).data(), bytes, cudaMemcpyHostToDevice);
    __impl::vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_input_1, d_input_2, d_output, input_size);
    d_input_1 = d_output;
  }

  cudaMemcpy(output.data(), d_input_1, bytes, cudaMemcpyDeviceToHost);

  // Free memory on device
  cudaFree(d_input_1);
  cudaFree(d_input_2);
  cudaFree(d_output);
  return output;
}

std::vector<int> vectorAdd_O2(const std::vector<std::vector<int>> &inputs){
  size_t input_count = inputs.size();
  assert(input_count > 0);
  const size_t input_size = inputs[0].size();
  size_t bytes = sizeof(int) * input_size;
  for(int i = 0; i < input_count; i++) {
      assert(inputs[i].size() == input_size);
  }

  std::vector<int> output(input_size, 0);
  int **pd_input;
  cudaMalloc(&pd_input, input_count * sizeof(int*));
  for(int i = 0; i < input_count; i++){
    cudaMalloc(&pd_input[i], bytes);
    cudaMemcpy(pd_input[i], inputs[i].data(), bytes, cudaMemcpyHostToDevice);
  }

  int NUM_THREADS = 1 << 10;
  int NUM_BLOCKS = (input_size + NUM_THREADS - 1) / NUM_THREADS;
  int BS = 1;
  while(true)
  {
    int a = 0, b = a + BS, step = 2*BS;
    bool computed = false;
    while(a < input_count && b < input_count)
    {
      __impl::vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(pd_input[a], pd_input[b], pd_input[a], input_size);
      a += step;
      b += step;
    }
    if(!computed){
      break;
    }
    BS << 1;
  }
  cudaMemcpy(output.data(), d_input[0], bytes, cudaMemcpyDeviceToHost);
  for(int i = 0; i < input_count; i++){
    cudaFree(pd_input[i]);
  }
  cudaFree(pd_input);
  return output;
}
} //cuda
