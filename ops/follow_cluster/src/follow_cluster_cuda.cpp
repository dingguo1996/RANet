#include <torch/extension.h>

#include <cmath>
#include <vector>

int FollowClusterForwardLaucher(const at::Tensor class_cluster_table,
                           const float threshold,
                           const int batch_size,
                           const int height, const int width,
                           at::Tensor follow_index);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int follow_cluster_forward_cuda(at::Tensor class_cluster_table, float threshold, at::Tensor follow_index) {
  CHECK_INPUT(class_cluster_table);
  CHECK_INPUT(follow_index);


  int data_batch_size = class_cluster_table.size(0);
  int data_height = class_cluster_table.size(1);
  int data_width = class_cluster_table.size(2);

  FollowClusterForwardLaucher(class_cluster_table, threshold, data_batch_size, data_height, data_width, follow_index);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &follow_cluster_forward_cuda, "Follow_Cluster forward (CUDA)");
}
