#include <torch/extension.h>

#include <cmath>
#include <vector>

int MatchBoundaryForwardLaucher(const at::Tensor prob_boundary,
                           const int batch_size,
                           const int height, const int width,
                           at::Tensor table_boundary,
                           at::Tensor index_output);

int MatchBoundaryBackwardLaucher(const at::Tensor top_grad,
                            const at::Tensor index_output,
                            const at::Tensor prob_boundary,
                            const int batch_size,
                            const int height, const int width,
                            at::Tensor bottom_boundary_grad);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int match_boundary_forward_cuda(at::Tensor prob_boundary, at::Tensor table_boundary, at::Tensor index_output) {
  CHECK_INPUT(prob_boundary);
  CHECK_INPUT(table_boundary);
  CHECK_INPUT(index_output);

  int data_batch_size = prob_boundary.size(0);
  int data_height = prob_boundary.size(2);
  int data_width = prob_boundary.size(3);

  MatchBoundaryForwardLaucher(prob_boundary, data_batch_size, data_height, data_width, table_boundary, index_output);

  return 1;
}

int match_boundary_backward_cuda(at::Tensor top_grad, at::Tensor index_output,
                            at::Tensor prob_boundary, at::Tensor bottom_boundary_grad) {
  CHECK_INPUT(top_grad);
  CHECK_INPUT(index_output);
  CHECK_INPUT(prob_boundary);
  CHECK_INPUT(bottom_boundary_grad);

  int data_batch_size = bottom_boundary_grad.size(0);
  int data_height = bottom_boundary_grad.size(2);
  int data_width = bottom_boundary_grad.size(3);

  MatchBoundaryBackwardLaucher(top_grad, index_output, prob_boundary, data_batch_size, data_height, data_width, bottom_boundary_grad);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &match_boundary_forward_cuda, "Match_Boundary forward (CUDA)");
  m.def("backward", &match_boundary_backward_cuda, "Match_Boundary backward (CUDA)");
}
