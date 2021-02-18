#include <torch/extension.h>

#include <cmath>
#include <vector>

int VcountClusterForwardLaucher(const at::Tensor region_attention_table,
                           const at::Tensor region_map,
                           const int data_cluster,
                           const int height, const int width,
                           at::Tensor pvic_table);
                           
int VcountClusterBackwardLaucher(const at::Tensor top_grad,
                            const at::Tensor region_map,
                            const int cluster,
                            const int height, const int width,
                            at::Tensor grad_region_attention_table);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int vcount_cluster_forward_cuda(at::Tensor region_attention_table, at::Tensor region_map, at::Tensor pvic_table) {
  CHECK_INPUT(region_attention_table);
  CHECK_INPUT(region_map);
  CHECK_INPUT(pvic_table);


  int data_cluster = pvic_table.size(0);
  int data_height = region_attention_table.size(0);
  int data_width = region_attention_table.size(1);

  VcountClusterForwardLaucher(region_attention_table, region_map, data_cluster, data_height, data_width, pvic_table);

  return 1;
}

int vcount_cluster_backward_cuda(at::Tensor top_grad, 
                            at::Tensor region_map,
                            at::Tensor grad_region_attention_table) {
  CHECK_INPUT(top_grad);
  CHECK_INPUT(region_map);
  CHECK_INPUT(grad_region_attention_table);

  int data_cluster = top_grad.size(0);
  int data_height = grad_region_attention_table.size(0);
  int data_width = grad_region_attention_table.size(1);

  VcountClusterBackwardLaucher(top_grad, region_map, data_cluster, data_height, data_width,
                            grad_region_attention_table);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &vcount_cluster_forward_cuda, "Vcount_Cluster forward (CUDA)");
  m.def("backward", &vcount_cluster_backward_cuda, "Vcount_Cluster backward (CUDA)");
}
