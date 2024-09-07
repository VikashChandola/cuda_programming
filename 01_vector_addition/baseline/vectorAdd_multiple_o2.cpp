// This program computes the sum of two vectors of length N
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "cuda_kernel.hpp"

void verify_result(std::vector<std::vector<int>> &inputs, std::vector<int> &output) {
  for (int i = 0; i < inputs[0].size(); i++) {
    int total = 0;
    for(auto & input : inputs){
      total += input[i];
    }
    assert(output[i] == total);
  }
}

int main(int argc, char** argv) {
  int INPUTS_SIZE = 10;
  if(argc > 1 ){
    INPUTS_SIZE = std::stoi(argv[1]);
    if(INPUTS_SIZE < 3){
      INPUTS_SIZE = 10;
    }
    std::cout << "Using input size " <<  INPUTS_SIZE << std::endl;
  }
  // Array size of 2^16 (65536 elements)
  constexpr int N = 1 << 16;
  constexpr size_t bytes = sizeof(int) * N;

  // Vectors for holding the host-side (CPU-side) data
  std::vector<int> a(N, 0);

  std::vector<std::vector<int>> inputs;
  for(int inputs_size = 0; inputs_size < INPUTS_SIZE; inputs_size++){
    // Initialize random numbers in each array
    for (int i = 0; i < N; i++) {
      a[i] = rand() % 100;
    }
    inputs.push_back(a);
  }

  std::vector<int> output = cuda::vectorAdd_O2(inputs);
  verify_result(inputs, output);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
