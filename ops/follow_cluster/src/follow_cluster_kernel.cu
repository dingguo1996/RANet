#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__global__ void FollowClusterForward(const int nthreads,
                                const scalar_t *class_cluster_table,
                                const float threshold,
                                const int height, const int width,
                                int *follow_index) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int py = index % height;                    //position end width
    int bs_ind = index / height ; //position bs in all
    int i = 0;
    for (i; i<=py; i++){
        follow_index[index] = i;
        if (class_cluster_table[bs_ind*height*width+py*width + i]>threshold){
            break;
        }
    }
  }
}

int FollowClusterForwardLaucher(const at::Tensor class_cluster_table,
                           const float threshold,
                           const int batch_size,
                           const int height, const int width,
                           at::Tensor follow_index) {
  const int output_size = batch_size * height;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      class_cluster_table.type(), "FollowClusterLaucherForward", ([&] {
        const scalar_t *class_cluster_table_data = class_cluster_table.data<scalar_t>();
        int *follow_index_data = follow_index.data<int>();

        FollowClusterForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, class_cluster_table_data, threshold, height, width, follow_index_data);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}