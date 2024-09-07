NVCC ?= nvcc
CXX ?= g++

vectorAdd : vectorAdd_binary vectorAdd_multiple vectorAdd_multiple_o1 vectorAdd_multiple_o2
	rm -f *.o

vectorAdd_binary : vectorAdd_binary.o vectorAdd_cuda.o 
	$(NVCC) vectorAdd_cuda.o vectorAdd_binary.o -o vectorAdd_binary.bin

vectorAdd_multiple : vectorAdd_multiple.o vectorAdd_cuda.o 
	$(NVCC) vectorAdd_cuda.o vectorAdd_multiple.o -o vectorAdd_multiple.bin

vectorAdd_multiple_o1 : vectorAdd_multiple_o1.o vectorAdd_cuda.o 
	$(NVCC) vectorAdd_cuda.o vectorAdd_multiple_o1.o -o vectorAdd_multiple_o1.bin

vectorAdd_multiple_o2 : vectorAdd_multiple_o2.o vectorAdd_cuda.o 
	$(NVCC) vectorAdd_cuda.o vectorAdd_multiple_o2.o -o vectorAdd_multiple_o2.bin

vectorAdd_cuda.o : vectorAdd_cuda.cu cuda_kernel.hpp
	$(NVCC) -c vectorAdd_cuda.cu

vectorAdd_binary.o : vectorAdd_binary.cpp cuda_kernel.hpp
	$(CXX) -c vectorAdd_binary.cpp

vectorAdd_multiple.o : vectorAdd_multiple.cpp cuda_kernel.hpp
	$(CXX) -c vectorAdd_multiple.cpp

vectorAdd_multiple_o1.o : vectorAdd_multiple_o1.cpp cuda_kernel.hpp
	$(CXX) -c vectorAdd_multiple_o1.cpp

vectorAdd_multiple_o2.o : vectorAdd_multiple_o2.cpp cuda_kernel.hpp
	$(CXX) -c vectorAdd_multiple_o2.cpp

clean :
	rm -f *.o *.bin

