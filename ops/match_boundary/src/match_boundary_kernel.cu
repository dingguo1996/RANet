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
__global__ void MatchBoundaryForward(const int nthreads, const scalar_t *prob_boundary,
                                const int height, const int width,
                                scalar_t *table_boundary_data,
                                int *index_output_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int pex = index % width;                    //position end width
    int pey = (index / width) % height;         //position end height
    int psx = (index / width / height) % width; //position start width
    int psy = (index / width / height / width) % height;   //position start height
    int bs_ind = index / width / height / width / height; //position start height

    int pixelPerImage = width * height;

    int delta_w = pex - psx;
    int delta_h = pey - psy;

    scalar_t max_confidence = 0.0;
    int max_index = -1;
    if (abs(delta_w) >= abs(delta_h) && abs(delta_w)!=0){
        scalar_t rate = scalar_t(delta_h)/scalar_t(delta_w);
        for (int ix = 1; ix < abs(delta_w); ix++) {
            scalar_t iy = ix * rate;
            int x = 0;
            if (delta_w>0){
                x = psx+ix;
            }
            else{
                iy = (-1) * iy;
                x = psx-ix;
            }
            scalar_t y = psy+iy;
            // clamp the x,y coordinate
            if (y <= 0) y = 0;
            if (x <= 0) x = 0;
            int y_low = (int)y;
            int y_high = y_low + 1;
            if (y_low >= height - 1) {
                y_high = y_low = height - 1;
            }
            if (x >= width - 1) {
                x = width - 1;
            }
            // clamp the x,y coordinate
            scalar_t ty = y - scalar_t(y_low);
            scalar_t by = scalar_t(y_high) - y;
            scalar_t prob_e = ty * prob_boundary[bs_ind * pixelPerImage + y_high * width + x] + \
                                by * prob_boundary[bs_ind * pixelPerImage + y_low * width + x];
            if (max_confidence < prob_e){
                max_confidence = prob_e;
                max_index = x;
            }
        }
    }
    else if(abs(delta_w) < abs(delta_h) && abs(delta_h)!=0){
        scalar_t rate = scalar_t(delta_w)/scalar_t(delta_h);
        for (int iy = 1; iy < abs(delta_h); iy++) {
            scalar_t ix = iy * rate;
            int y = 0;
            if (delta_h>0){
                y = psy+iy;
            }
            else{
                ix = (-1) * ix;
                y = psy-iy;
            }
            scalar_t x = psx+ix;
            if (y <= 0) y = 0;
            if (x <= 0) x = 0;
            int x_low = (int)x;
            int x_high = x_low + 1;
            if (x_low >= width - 1) {
                x_high = x_low = width - 1;
            }
            if (y >= height - 1) {
                y = height - 1;
            }
            scalar_t lx = x - scalar_t(x_low);
            scalar_t rx = scalar_t(x_high) - x;
            scalar_t prob_e = lx * prob_boundary[bs_ind * pixelPerImage + y * width + x_high] + \
                                rx * prob_boundary[bs_ind * pixelPerImage + y * width + x_low];
            if (max_confidence < prob_e){
                max_confidence = prob_e;
                max_index = y;
            }
        }
    }
    table_boundary_data[index] = max_confidence;
    index_output_data[index] = max_index;
  }
}

int MatchBoundaryForwardLaucher(const at::Tensor prob_boundary,
                           const int batch_size,
                           const int height, const int width,
                           at::Tensor table_boundary,
                           at::Tensor index_output) {
  const int output_size = batch_size * height * width * height * width;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      prob_boundary.type(), "MatchBoundaryLaucherForward", ([&] {
        const scalar_t *prob_boundary_data = prob_boundary.data<scalar_t>();
        scalar_t *table_boundary_data = table_boundary.data<scalar_t>();
        int *index_output_data = index_output.data<int>();

        MatchBoundaryForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, prob_boundary_data, height, width, table_boundary_data, index_output_data);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}


template <typename scalar_t>
__global__ void MatchBoundaryBackward(
    const int nthreads, const scalar_t *top_diff,
    const int *index_output,
    const scalar_t *prob_boundary,
    const int height, const int width,
    scalar_t *bottom_boundary_grad) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    if (index_output[index]>-1){
        int pex = index % width;                    //position end width
        int pey = (index / width) % height;         //position end height
        int psx = (index / width / height) % width; //position start width
        int psy = (index / width / height / width) % height;   //position start height
        int bs_ind = index / width / height / width / height; //position start height

        int pixelPerImage = width * height;

        int delta_w = pex - psx;
        int delta_h = pey - psy;

        int max_index = index_output[index];
        if (abs(delta_w) >= abs(delta_h) && abs(delta_w)!=0){
            scalar_t rate = scalar_t(delta_h) / scalar_t(delta_w);
            scalar_t iy = (max_index - psx) * rate;
            int x = max_index;
            scalar_t y = psy + iy;
            if (y <= 0) y = 0;
            if (y <= 0) y = 0;
            int y_low = (int)y;
            int y_high = y + 1;
            if (y_low >= height - 1) {
                y_high = y_low = height - 1;
            }
            if (x >= width - 1) {
                x = width - 1;
            }
            scalar_t ty = y - scalar_t(y_low);
            scalar_t by = scalar_t(y_high) - y;
            scalar_t *offset_bottom_bboundary_diff = bottom_boundary_grad + bs_ind * pixelPerImage + y_high * width + x;
            scalar_t *offset_bottom_tboundary_diff = bottom_boundary_grad + bs_ind * pixelPerImage + y_low * width + x;

            scalar_t offset_bottom_bboundary_diff_value = ty*top_diff[index];
            scalar_t offset_bottom_tboundary_diff_value = by*top_diff[index];
            atomicAdd(offset_bottom_bboundary_diff, offset_bottom_bboundary_diff_value);
            atomicAdd(offset_bottom_tboundary_diff, offset_bottom_tboundary_diff_value);
        }
        else if(abs(delta_w) < abs(delta_h) && abs(delta_h)!=0){
            scalar_t rate = scalar_t(delta_w) / scalar_t(delta_h);
            scalar_t ix = (max_index - psy) * rate;
            int y = max_index;
            scalar_t x = psx + ix;
            if (y <= 0) y = 0;
            if (x <= 0) x = 0;
            int x_low = (int)x;
            int x_high = x_low + 1;
            if (x_low >= width - 1) {
                x_high = x_low = width - 1;
            }
            if (y >= height - 1) {
                y = height - 1;
            }
            scalar_t lx = x - scalar_t(x_low);
            scalar_t rx = scalar_t(x_high) - x;
            scalar_t *offset_bottom_rboundary_diff = bottom_boundary_grad + bs_ind * pixelPerImage + y * width + x_high;
            scalar_t *offset_bottom_lboundary_diff = bottom_boundary_grad + bs_ind * pixelPerImage + y * width + x_low;

            scalar_t offset_bottom_rboundary_diff_value = lx*top_diff[index];
            scalar_t offset_bottom_lboundary_diff_value = rx*top_diff[index];
            atomicAdd(offset_bottom_rboundary_diff, offset_bottom_rboundary_diff_value);
            atomicAdd(offset_bottom_lboundary_diff, offset_bottom_lboundary_diff_value);
        }
    }
  }
}

int MatchBoundaryBackwardLaucher(const at::Tensor top_grad,
                            const at::Tensor index_output,
                            const at::Tensor prob_boundary,
                            const int batch_size,
                            const int height, const int width,
                            at::Tensor bottom_boundary_grad) {
  const int output_size = batch_size * height * width * height * width;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "MatchBoundaryLaucherBackward", ([&] {
        const scalar_t *top_diff = top_grad.data<scalar_t>();
        const int *index_output_data = index_output.data<int>();
        const scalar_t *prob_boundary_data = prob_boundary.data<scalar_t>();
        scalar_t *bottom_boundary_grad_diff = bottom_boundary_grad.data<scalar_t>();
        if (sizeof(scalar_t) == sizeof(double)) {
          fprintf(stderr, "double is not supported\n");
          exit(-1);
        }

        MatchBoundaryBackward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, top_diff, index_output_data, prob_boundary_data, height, width, bottom_boundary_grad_diff);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}
