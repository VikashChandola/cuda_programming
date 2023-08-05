NVCC ?= nvcc
CXX ?= g++

vectorAdd : vectorAdd_cuda.o vectorAdd_main.o
	nvcc vectorAdd_cuda.o vectorAdd_main.o -o vectorAdd.bin

vectorAdd_cuda.o : vectorAdd_cuda.cu cuda_kernel.hpp
	$(NVCC) -c vectorAdd_cuda.cu

vectorAdd_main.o : vectorAdd_main.cpp cuda_kernel.hpp
	$(CXX) -c vectorAdd_main.cpp

clean :
	rm -f *.o *.bin

