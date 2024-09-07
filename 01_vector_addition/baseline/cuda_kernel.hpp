#ifndef __CUDA_KERNEL__
#define __CUDA_KERNEL__
namespace cuda
{
std::vector<int> vectorAdd(const std::vector<int> &,const  std::vector<int> &);
std::vector<int> vectorAdd(const std::vector<std::vector<int>> &);
std::vector<int> vectorAdd_O1(const std::vector<std::vector<int>> &);
std::vector<int> vectorAdd_O2(const std::vector<std::vector<int>> &);
}
#endif
