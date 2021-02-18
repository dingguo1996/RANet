#include <torch/extension.h>

#include <cmath>
#include <vector>

int MatchClassForwardLaucher(const at::Tensor class_pred_softmax,
                           const at::Tensor class_max_prob_A_index,
                           const int batch_size, const int channel,
                           const int height, const int width,
                           at::Tensor confidence_output);

int MatchClassBackwardLaucher(const at::Tensor top_grad,
                            const at::Tensor class_pred_softmax,
                            const at::Tensor class_max_prob_A_index,
                            const int batch_size, const int channel,
                            const int height, const int width,
                            at::Tensor bottom_class_pred_softmax_grad);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int match_class_forward_cuda(at::Tensor class_pred_softmax, at::Tensor class_max_prob_A_index, at::Tensor confidence_output) {
  CHECK_INPUT(class_pred_softmax);
  CHECK_INPUT(class_max_prob_A_index);
  CHECK_INPUT(confidence_output);

  int data_batch_size = class_pred_softmax.size(0);
  int data_channel = class_pred_softmax.size(1);
  int data_height = class_pred_softmax.size(2);
  int data_width = class_pred_softmax.size(3);

  MatchClassForwardLaucher(class_pred_softmax, class_max_prob_A_index, data_batch_size, data_channel, data_height, data_width, confidence_output);

  return 1;
}

int match_class_backward_cuda(at::Tensor top_grad, 
                            at::Tensor class_pred_softmax,
                            at::Tensor class_max_prob_A_index,
                            at::Tensor bottom_class_pred_softmax_grad) {
  CHECK_INPUT(top_grad);
  CHECK_INPUT(class_pred_softmax);
  CHECK_INPUT(class_max_prob_A_index);
  CHECK_INPUT(bottom_class_pred_softmax_grad);

  int data_batch_size = bottom_class_pred_softmax_grad.size(0);
  int data_channel = bottom_class_pred_softmax_grad.size(1);
  int data_height = bottom_class_pred_softmax_grad.size(2);
  int data_width = bottom_class_pred_softmax_grad.size(3);

  MatchClassBackwardLaucher(top_grad, class_pred_softmax, class_max_prob_A_index, data_batch_size, data_channel, data_height, data_width,
                            bottom_class_pred_softmax_grad);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &match_class_forward_cuda, "Match_Class forward (CUDA)");
  m.def("backward", &match_class_backward_cuda, "Match_Class backward (CUDA)");
}
