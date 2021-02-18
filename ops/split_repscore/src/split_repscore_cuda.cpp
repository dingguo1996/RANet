#include <torch/extension.h>

#include <cmath>
#include <vector>

int SplitRepscoreForwardLaucher(const at::Tensor repscore_map,
                           const at::Tensor region_map,
                           const int data_cluster,
                           const int data_length,
                           at::Tensor pric_table);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int split_repscore_forward_cuda(at::Tensor repscore_map, at::Tensor region_map, at::Tensor pric_table) {
  CHECK_INPUT(repscore_map);
  CHECK_INPUT(region_map);
  CHECK_INPUT(pric_table);


  int data_cluster = pric_table.size(0);
  int data_length = repscore_map.size(0);


  SplitRepscoreForwardLaucher(repscore_map, region_map, data_cluster, data_length, pric_table);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &split_repscore_forward_cuda, "Vcount_Cluster forward (CUDA)");
}
