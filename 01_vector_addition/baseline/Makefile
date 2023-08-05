NVCC ?= nvcc
CXX ?= g++

vectorAdd : vectorAdd_binary vectorAdd_multiple vectorAdd_multiple_optimized
	rm -f *.o

vectorAdd_binary : vectorAdd_binary.o vectorAdd_cuda.o 
	$(NVCC) vectorAdd_cuda.o vectorAdd_binary.o -o vectorAdd_binary.bin

vectorAdd_multiple : vectorAdd_multiple.o vectorAdd_cuda.o 
	$(NVCC) vectorAdd_cuda.o vectorAdd_multiple.o -o vectorAdd_multiple.bin

vectorAdd_multiple_optimized : vectorAdd_multiple_optimized.o vectorAdd_cuda.o 
	$(NVCC) vectorAdd_cuda.o vectorAdd_multiple_optimized.o -o vectorAdd_multiple_optimized.bin

vectorAdd_cuda.o : vectorAdd_cuda.cu cuda_kernel.hpp
	$(NVCC) -c vectorAdd_cuda.cu

vectorAdd_binary.o : vectorAdd_binary.cpp cuda_kernel.hpp
	$(CXX) -c vectorAdd_binary.cpp

vectorAdd_multiple.o : vectorAdd_multiple.cpp cuda_kernel.hpp
	$(CXX) -c vectorAdd_multiple.cpp

vectorAdd_multiple_optimized.o : vectorAdd_multiple_optimized.cpp cuda_kernel.hpp
	$(CXX) -c vectorAdd_multiple_optimized.cpp

clean :
	rm -f *.o *.bin

