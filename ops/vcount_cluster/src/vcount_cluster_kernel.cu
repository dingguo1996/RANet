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
__global__ void VcountClusterForward(const int nthreads,
                                const scalar_t *region_attention_table,
                                const int *region_map,
                                const int height, const int width,
                                scalar_t *pvic_table) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {

    int py = index % height;                   //position height
    int cluster_ind = index / height;          //position cluster_ind
    for (int px=0; px<width; px++){
        if (region_map[py] == cluster_ind && region_map[px] == cluster_ind){
            scalar_t *offset_pvic_table = pvic_table + cluster_ind * height + py;
            atomicAdd(offset_pvic_table, region_attention_table[py*height+px]);
        }
    }
  }
}

int VcountClusterForwardLaucher(const at::Tensor region_attention_table,
                           const at::Tensor region_map,
                           const int cluster,
                           const int height, const int width,
                           at::Tensor pvic_table) {
  const int output_size = cluster * height;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      region_attention_table.type(), "VcountClusterLaucherForward", ([&] {
        const scalar_t *region_attention_table_data = region_attention_table.data<scalar_t>();
        const int *region_map_data = region_map.data<int>();
        scalar_t *pvic_table_data = pvic_table.data<scalar_t>();

        VcountClusterForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, region_attention_table_data, region_map_data, height, width, pvic_table_data);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}


template <typename scalar_t>
__global__ void VcountClusterBackward(
    const int nthreads, const scalar_t *top_diff,
    const int *region_map,
    const int cluster,
    const int height, const int width,
    scalar_t *grad_region_attention_table) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {

    int py = index % height;                   //position height
    int cluster_ind = index / height;          //position cluster_ind
        for (int px=0; px<width; px++){
        if (region_map[py] == cluster_ind && region_map[px] == cluster_ind){
            scalar_t *offset_grad_region_attention_table = grad_region_attention_table + py*height+px;
            atomicAdd(offset_grad_region_attention_table, top_diff[cluster_ind * height + py]);
        }
    }
  }
}

int VcountClusterBackwardLaucher(const at::Tensor top_grad,
                            const at::Tensor region_map,
                            const int cluster,
                            const int height, const int width,
                            at::Tensor grad_region_attention_table) {
  const int output_size = cluster * height;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "VcountClusterLaucherBackward", ([&] {
        const scalar_t *top_diff = top_grad.data<scalar_t>();
        const int *region_map_data = region_map.data<int>();
        scalar_t *grad_region_attention_table_diff = grad_region_attention_table.data<scalar_t>();
        if (sizeof(scalar_t) == sizeof(double)) {
          fprintf(stderr, "double is not supported\n");
          exit(-1);
        }

        VcountClusterBackward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, top_diff, region_map_data, cluster, height, width,
                grad_region_attention_table_diff);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}