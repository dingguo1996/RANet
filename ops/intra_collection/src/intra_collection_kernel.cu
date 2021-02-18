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
__global__ void IntraCollectionForward(const int nthreads,
                                const scalar_t *rep_feat,
                                const scalar_t *feat,
                                const int *vtopk_table,
                                const int *region_map,
                                const int num_rep_pixels,
                                const int num_channels,
                                const int num_pixels,
                                scalar_t *collect_rep_feat) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    scalar_t sum_relation = 0.0;

    for(int i = 0; i < num_pixels; i++){
        if(region_map[vtopk_table[index]]!=region_map[i])
            continue;
        scalar_t relation_ij = 0.0;
        for(int j = 0; j < num_channels; j++){
            relation_ij += exp(rep_feat[index*num_channels+j] * feat[i*num_channels+j]);
        }
        sum_relation += relation_ij;
    }

    for(int i = 0; i < num_pixels; i++){
        if(region_map[vtopk_table[index]]!=region_map[i])
            continue;
        scalar_t relation_ij = 0.0;
        for(int j = 0; j < num_channels; j++){
            relation_ij += exp(rep_feat[index*num_channels+j] * feat[i*num_channels+j]);
        }
        relation_ij = relation_ij / sum_relation;
        for(int j = 0; j < num_channels; j++){
            collect_rep_feat[index*num_channels+j] += relation_ij * feat[i*num_channels+j];
        }
    }
  }
}

int IntraCollectionForwardLaucher(const at::Tensor rep_feat,
                           const at::Tensor feat,
                           const at::Tensor vtopk_table,
                           const at::Tensor region_map,
                           const int num_rep_pixels,
                           const int num_channels,
                           const int num_pixels,
                           at::Tensor collect_rep_feat) {
  const int output_size = num_rep_pixels;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      rep_feat.type(), "IntraCollectionLaucherForward", ([&] {
        const scalar_t *rep_feat_data = rep_feat.data<scalar_t>();
        const scalar_t *feat_data = feat.data<scalar_t>();
        const int *vtopk_table_data = vtopk_table.data<int>();
        const int *region_map_data = region_map.data<int>();
        scalar_t *collect_rep_feat_data = collect_rep_feat.data<scalar_t>();

        IntraCollectionForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, rep_feat_data, feat_data, vtopk_table_data, region_map_data,
                num_rep_pixels, num_channels, num_pixels, collect_rep_feat_data);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}


template <typename scalar_t>
__global__ void IntraCollectionBackward(
    const int nthreads, const scalar_t *top_diff,
    const scalar_t *rep_feat,
    const scalar_t *feat,
    const int *vtopk_table,
    const int *region_map,
    const int num_rep_pixels,
    const int num_channels,
    const int num_pixels,
    scalar_t *bottom_rep_feat_grad,
    scalar_t *bottom_feat_grad) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    scalar_t sum_relation = 0.0;

    for(int i = 0; i < num_pixels; i++){
        if(region_map[vtopk_table[index]]!=region_map[i])
            continue;
        scalar_t relation_ij = 0.0;
        for(int j = 0; j < num_channels; j++){
            relation_ij += exp(rep_feat[index*num_channels+j] * feat[i*num_channels+j]);
        }
        sum_relation += relation_ij;
    }
    scalar_t* relations = new scalar_t[num_pixels];
    for(int i = 0; i < num_pixels; i++){
        if(region_map[vtopk_table[index]]!=region_map[i])
            continue;
        scalar_t relation_ij = 0.0;
        for(int j = 0; j < num_channels; j++){
            relation_ij += exp(rep_feat[index*num_channels+j] * feat[i*num_channels+j]);
        }
        relations[i] = relation_ij / sum_relation;
        scalar_t grad_top = 0.0;
        scalar_t grad_rep_feat_value = 0.0;
        scalar_t grad_feat_value = 0.0;
        for(int j = 0; j < num_channels; j++){
            if (index==i){
                grad_top = top_diff[index*num_channels+j]*(relations[index]-relations[index]*relations[i]);
                grad_rep_feat_value = grad_top*feat[i*num_channels+j];
                grad_feat_value = grad_top*rep_feat[index*num_channels+j];
            }else{
                grad_top = top_diff[index*num_channels+j]*(-relations[index]*relations[i]);
                grad_rep_feat_value = grad_top*feat[i*num_channels+j];
                grad_feat_value = grad_top*rep_feat[index*num_channels+j];
            }
            scalar_t *offset_bottom_rep_feat_diff = bottom_rep_feat_grad + index*num_channels+j;
            scalar_t *offset_bottom_feat_diff = bottom_feat_grad + i*num_channels+j;
            atomicAdd(offset_bottom_rep_feat_diff, grad_rep_feat_value);
            atomicAdd(offset_bottom_feat_diff, grad_feat_value);
        }
    }
    delete[] relations;
  }
}

int IntraCollectionBackwardLaucher(const at::Tensor top_grad,
                            const at::Tensor rep_feat,
                            const at::Tensor feat,
                            const at::Tensor vtopk_table,
                            const at::Tensor region_map,
                            const int num_rep_pixels,
                            const int num_channels,
                            const int num_pixels,
                            at::Tensor bottom_rep_feat_grad,
                            at::Tensor bottom_feat_grad) {
  const int output_size = num_rep_pixels;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "IntraCollectionLaucherBackward", ([&] {
        const scalar_t *top_diff = top_grad.data<scalar_t>();
        const scalar_t *rep_feat_data = rep_feat.data<scalar_t>();
        const scalar_t *feat_data = feat.data<scalar_t>();
        const int *vtopk_table_data = vtopk_table.data<int>();
        const int *region_map_data = region_map.data<int>();
        scalar_t *bottom_rep_feat_grad_diff = bottom_rep_feat_grad.data<scalar_t>();
        scalar_t *bottom_feat_grad_diff = bottom_feat_grad.data<scalar_t>();
        if (sizeof(scalar_t) == sizeof(double)) {
          fprintf(stderr, "double is not supported\n");
          exit(-1);
        }

        IntraCollectionBackward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, top_diff, rep_feat_data, feat_data, vtopk_table_data,
                region_map_data, num_rep_pixels, num_channels, num_pixels,
                bottom_rep_feat_grad_diff, bottom_feat_grad_diff);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}
