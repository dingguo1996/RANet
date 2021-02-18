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
__global__ void MatchClassForward(const int nthreads,
                                const scalar_t *class_pred_softmax,
                                const int *class_max_prob_A_index,
                                const int channels,
                                const int height, const int width,
                                scalar_t *confidence_output_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int pex = index % width;                    //position end width
    int pey = (index / width) % height;         //position end height
    int psx = (index / width / height) % width; //position start width
    int psy = (index / width / height / width) % height;   //position start height
    int bs_ind = index / width / height / width / height; //bs_ind

    int pixelPerImage = channels * width * height;
    int pixelPerTable = width * height;
    
    int pc = class_max_prob_A_index[bs_ind * pixelPerTable + psy * width + psx];
    scalar_t prob_start_c = class_pred_softmax[bs_ind * pixelPerImage + pc * pixelPerTable + psy * width + psx];
    scalar_t prob_end_c = class_pred_softmax[bs_ind * pixelPerImage + pc * pixelPerTable + pey * width + pex];

    if (prob_start_c < 0.001)
        prob_start_c = 0.001;
    if (prob_start_c > 0.99)
        prob_start_c = 0.99;
    if (prob_end_c < 0.001)
        prob_end_c = 0.001;
    if (prob_end_c > 0.99)
        prob_end_c = 0.99;

    scalar_t prob_m_c = (prob_start_c + prob_end_c)/2;
    scalar_t confidence = 0.5*(prob_start_c*log(prob_start_c/prob_m_c)+ \
                        (1-prob_start_c)*log((1-prob_start_c)/(1-prob_m_c))+ \
                        prob_end_c*log(prob_end_c/prob_m_c)+ \
                        (1-prob_end_c)*log((1-prob_end_c)/(1-prob_m_c)));

    confidence_output_data[index] = confidence;
  }
}

int MatchClassForwardLaucher(const at::Tensor class_pred_softmax,
                           const at::Tensor class_max_prob_A_index,
                           const int batch_size, const int channels, 
                           const int height, const int width,
                           at::Tensor confidence_output) {
  const int output_size = batch_size * height * width * height * width;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      class_pred_softmax.type(), "MatchClassLaucherForward", ([&] {
        const scalar_t *class_pred_softmax_data = class_pred_softmax.data<scalar_t>();
        const int *class_max_prob_A_index_data = class_max_prob_A_index.data<int>();
        scalar_t *confidence_output_data = confidence_output.data<scalar_t>();

        MatchClassForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, class_pred_softmax_data, class_max_prob_A_index_data, channels, height, width, confidence_output_data);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}


template <typename scalar_t>
__global__ void MatchClassBackward(
    const int nthreads, const scalar_t *top_diff,
    const scalar_t *class_pred_softmax,
    const int *class_max_prob_A_index,
    const int channels,
    const int height, const int width,
    scalar_t *bottom_class_pred_softmax_grad) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int pex = index % width;                    //position end width
    int pey = (index / width) % height;         //position end height
    int psx = (index / width / height) % width; //position start width
    int psy = (index / width / height / width) % height;   //position start height
    int bs_ind = index / width / height / width / height; //position start height

    int pixelPerImage = channels * width * height;
    int pixelPerTable = width * height;

    int pc = class_max_prob_A_index[bs_ind * pixelPerTable + psy * width + psx];
    scalar_t prob_start_c = class_pred_softmax[bs_ind * pixelPerImage + pc * pixelPerTable + psy * width + psx];
    scalar_t prob_end_c = class_pred_softmax[bs_ind * pixelPerImage + pc * pixelPerTable + pey * width + pex];

    if (prob_start_c < 0.001)
        prob_start_c = 0.001;
    if (prob_start_c > 0.99)
        prob_start_c = 0.99;
    if (prob_end_c < 0.001)
        prob_end_c = 0.001;
    if (prob_end_c > 0.99)
        prob_end_c = 0.99;

    scalar_t prob_m_c = (prob_start_c + prob_end_c)/2;

    //scalar_t grad_prob_start_c_value = 0.5*(log(prob_start_c/prob_m_c)-log((1-prob_start_c)/(1-prob_m_c)))*top_diff[index];
    //scalar_t grad_prob_end_c_value = 0.5*(log(prob_end_c/prob_m_c)-log((1-prob_end_c)/(1-prob_m_c)))*top_diff[index];

    scalar_t grad_prob_start_c_value = 0.5*log(prob_start_c*(1-prob_m_c)/prob_m_c*(1-prob_start_c))*top_diff[index];
    scalar_t grad_prob_end_c_value = 0.5*log(prob_end_c*(1-prob_m_c)/prob_m_c*(1-prob_end_c))*top_diff[index];

    scalar_t *offset_bottom_start_c_diff = bottom_class_pred_softmax_grad + bs_ind * pixelPerImage + pc * pixelPerTable + psy * width + psx;
    scalar_t *offset_bottom_end_c_diff = bottom_class_pred_softmax_grad + bs_ind * pixelPerImage + pc * pixelPerTable + pey * width + pex;
    atomicAdd(offset_bottom_start_c_diff, grad_prob_start_c_value);
    atomicAdd(offset_bottom_end_c_diff, grad_prob_end_c_value);
  }
}

int MatchClassBackwardLaucher(const at::Tensor top_grad,
                            const at::Tensor class_pred_softmax,
                            const at::Tensor class_max_prob_A_index,
                            const int batch_size, const int channels,
                            const int height, const int width,
                            at::Tensor bottom_class_pred_softmax_grad) {
  const int output_size = batch_size * height * width * height * width;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "MatchClassLaucherBackward", ([&] {
        const scalar_t *top_diff = top_grad.data<scalar_t>();
        const scalar_t *class_pred_softmax_data = class_pred_softmax.data<scalar_t>();
        const int *class_max_prob_A_index_data = class_max_prob_A_index.data<int>();
        scalar_t *bottom_class_pred_softmax_grad_diff = bottom_class_pred_softmax_grad.data<scalar_t>();
        if (sizeof(scalar_t) == sizeof(double)) {
          fprintf(stderr, "double is not supported\n");
          exit(-1);
        }

        MatchClassBackward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, top_diff, class_pred_softmax_data, class_max_prob_A_index_data, channels, height, width,
                bottom_class_pred_softmax_grad_diff);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}
