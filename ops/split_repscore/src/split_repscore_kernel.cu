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
__global__ void SplitRepscoreForward(const int nthreads,
                                const scalar_t *repscore_map,
                                const int *region_map,
                                const int length,
                                scalar_t *pric_table) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {

    int py = index % length;                    //position width
    int cluster_ind = index / length; //position cluster_ind
    if (region_map[py] == cluster_ind){
        scalar_t *offset_pric_table = pric_table + cluster_ind * length + py;
        atomicAdd(offset_pric_table, repscore_map[py]);
    }
  }
}

int SplitRepscoreForwardLaucher(const at::Tensor repscore_map,
                           const at::Tensor region_map,
                           const int cluster,
                           const int length,
                           at::Tensor pric_table) {
  const int output_size = cluster * length;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      repscore_map.type(), "SplitRepscoreLaucherForward", ([&] {
        const scalar_t *repscore_map_data = repscore_map.data<scalar_t>();
        const int *region_map_data = region_map.data<int>();
        scalar_t *pric_table_data = pric_table.data<scalar_t>();

        SplitRepscoreForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, repscore_map_data, region_map_data, length, pric_table_data);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}