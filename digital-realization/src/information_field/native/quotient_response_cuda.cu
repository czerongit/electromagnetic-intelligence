#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

template <typename scalar_t>
__global__ void quotient_response_kernel(
    const scalar_t* columns,
    const int64_t* indices,
    const scalar_t* amplitudes,
    const bool* valid,
    scalar_t* output,
    int batch,
    int relations,
    int queries,
    int support,
    int width) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * queries * width;
  if (gid >= total) return;

  int coordinate = gid % width;
  int query = (gid / width) % queries;
  int item = gid / (queries * width);
  float response = 0.0f;
  for (int slot = 0; slot < support; ++slot) {
    int incident_offset = (item * queries + query) * support + slot;
    if (!valid[incident_offset]) continue;
    int64_t relation = indices[incident_offset];
    response += static_cast<float>(amplitudes[incident_offset]) *
        static_cast<float>(
            columns[(item * relations + relation) * width + coordinate]);
  }
  output[gid] = static_cast<scalar_t>(response);
}

torch::Tensor quotient_response_cuda(
    torch::Tensor columns,
    torch::Tensor indices,
    torch::Tensor amplitudes,
    torch::Tensor valid) {
  TORCH_CHECK(columns.is_cuda(), "columns must be CUDA resident");
  TORCH_CHECK(indices.is_cuda() && amplitudes.is_cuda() && valid.is_cuda(),
              "incidents must be CUDA resident");
  TORCH_CHECK(columns.is_contiguous() && indices.is_contiguous() &&
              amplitudes.is_contiguous() && valid.is_contiguous(),
              "all inputs must be contiguous");
  TORCH_CHECK(columns.dim() == 3, "columns must have shape batch by relation by width");
  TORCH_CHECK(indices.dim() == 3, "indices must have shape batch by query by support");
  TORCH_CHECK(indices.sizes() == amplitudes.sizes() && indices.sizes() == valid.sizes(),
              "incident tensors must have one shape");
  TORCH_CHECK(indices.scalar_type() == torch::kInt64, "indices must be int64");
  TORCH_CHECK(valid.scalar_type() == torch::kBool, "valid must be boolean");
  TORCH_CHECK(columns.scalar_type() == amplitudes.scalar_type(),
              "columns and amplitudes must share one dtype");
  TORCH_CHECK(columns.size(0) == indices.size(0), "batch dimensions must agree");

  auto batch = static_cast<int>(columns.size(0));
  auto relations = static_cast<int>(columns.size(1));
  auto width = static_cast<int>(columns.size(2));
  auto queries = static_cast<int>(indices.size(1));
  auto support = static_cast<int>(indices.size(2));
  c10::cuda::CUDAGuard guard(columns.device());
  auto output = torch::empty({batch, queries, width}, columns.options());
  int total = batch * queries * width;
  constexpr int threads = 256;
  int blocks = (total + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      columns.scalar_type(), "quotient_response_cuda", [&] {
        quotient_response_kernel<scalar_t><<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            columns.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            amplitudes.data_ptr<scalar_t>(),
            valid.data_ptr<bool>(),
            output.data_ptr<scalar_t>(),
            batch, relations, queries, support, width);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &quotient_response_cuda,
             "Relation-native quotient response (CUDA)");
}
