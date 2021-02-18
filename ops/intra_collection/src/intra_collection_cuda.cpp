#include <torch/extension.h>

#include <cmath>
#include <vector>

int IntraCollectionForwardLaucher(const at::Tensor rep_feat,
                           const at::Tensor feat,
                           const at::Tensor vtopk_table,
                           const at::Tensor region_map,
                           const int num_rep_pixels,
                           const int num_channels,
                           const int num_pixels,
                           at::Tensor collect_rep_feat);

int IntraCollectionBackwardLaucher(const at::Tensor top_grad,
                            const at::Tensor rep_feat,
                            const at::Tensor feat,
                            const at::Tensor vtopk_table,
                            const at::Tensor region_map,
                            const int num_rep_pixels,
                            const int num_channels,
                            const int num_pixels,
                            at::Tensor bottom_rep_feat_grad,
                            at::Tensor bottom_feat_grad);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int intra_collection_forward_cuda(at::Tensor rep_feat, at::Tensor feat, at::Tensor vtopk_table,
                                   at::Tensor region_map, at::Tensor collect_rep_feat) {
  CHECK_INPUT(rep_feat);
  CHECK_INPUT(feat);
  CHECK_INPUT(vtopk_table);
  CHECK_INPUT(region_map);
  CHECK_INPUT(collect_rep_feat);

  int data_num_rep_pixels = rep_feat.size(0);
  int data_num_channels = rep_feat.size(1);
  int data_num_pixels = feat.size(0);

  IntraCollectionForwardLaucher(rep_feat, feat, vtopk_table, region_map, data_num_rep_pixels,
                                    data_num_channels, data_num_pixels, collect_rep_feat);

  return 1;
}

int intra_collection_backward_cuda(at::Tensor top_grad, 
                            at::Tensor rep_feat,
                            at::Tensor feat,
                            at::Tensor vtopk_table,
                            at::Tensor region_map,
                            at::Tensor bottom_rep_feat_grad,
                            at::Tensor bottom_feat_grad) {
  CHECK_INPUT(top_grad);
  CHECK_INPUT(rep_feat);
  CHECK_INPUT(feat);
  CHECK_INPUT(vtopk_table);
  CHECK_INPUT(region_map);
  CHECK_INPUT(bottom_rep_feat_grad);
  CHECK_INPUT(bottom_feat_grad);

  int data_num_rep_pixels = rep_feat.size(0);
  int data_num_channels = rep_feat.size(1);
  int data_num_pixels = feat.size(0);

  IntraCollectionBackwardLaucher(top_grad, rep_feat, feat, vtopk_table, region_map,
                            data_num_rep_pixels, data_num_channels, data_num_pixels,
                            bottom_rep_feat_grad, bottom_feat_grad);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &intra_collection_forward_cuda, "Intra_Collection forward (CUDA)");
  m.def("backward", &intra_collection_backward_cuda, "Intra_Collection backward (CUDA)");
}
