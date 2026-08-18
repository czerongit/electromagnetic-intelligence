#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

__device__ inline void apply3(const float* matrix, float& x, float& y, float& z) {
  const float next_x = matrix[0] * x + matrix[1] * y + matrix[2] * z;
  const float next_y = matrix[3] * x + matrix[4] * y + matrix[5] * z;
  const float next_z = matrix[6] * x + matrix[7] * y + matrix[8] * z;
  x = next_x;
  y = next_y;
  z = next_z;
}

__global__ void regular_grid_temporal_response_kernel(
    const float* transitions,
    const int64_t* lengths,
    const float* initial,
    float* output,
    int batch,
    int modes,
    int events) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= batch * modes) return;
  const int mode = gid % modes;
  float x = initial[gid * 3];
  float y = initial[gid * 3 + 1];
  float z = initial[gid * 3 + 2];
  for (int event = 0; event < events; ++event) {
    const float* matrix = transitions + (event * modes + mode) * 9;
    for (int64_t step = 0; step < lengths[event]; ++step) {
      apply3(matrix, x, y, z);
    }
  }
  output[gid * 3] = x;
  output[gid * 3 + 1] = y;
  output[gid * 3 + 2] = z;
}

__global__ void event_composed_temporal_response_kernel(
    const float* powers,
    const float* initial,
    float* output,
    int batch,
    int modes,
    int events) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= batch * modes) return;
  const int mode = gid % modes;
  float x = initial[gid * 3];
  float y = initial[gid * 3 + 1];
  float z = initial[gid * 3 + 2];
  for (int event = 0; event < events; ++event) {
    const float* matrix = powers + (event * modes + mode) * 9;
    apply3(matrix, x, y, z);
  }
  output[gid * 3] = x;
  output[gid * 3 + 1] = y;
  output[gid * 3 + 2] = z;
}

void validate_common(torch::Tensor matrices, torch::Tensor initial) {
  TORCH_CHECK(matrices.is_cuda() && initial.is_cuda(), "inputs must be CUDA resident");
  TORCH_CHECK(matrices.is_contiguous() && initial.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(matrices.scalar_type() == torch::kFloat32 &&
              initial.scalar_type() == torch::kFloat32, "inputs must be float32");
  TORCH_CHECK(matrices.dim() == 4 && matrices.size(2) == 3 && matrices.size(3) == 3,
              "matrices require event by mode by 3 by 3 shape");
  TORCH_CHECK(initial.dim() == 3 && initial.size(2) == 3,
              "initial state requires batch by mode by 3 shape");
  TORCH_CHECK(matrices.size(1) == initial.size(1), "mode dimensions must agree");
}

torch::Tensor regular_forward(
    torch::Tensor transitions,
    torch::Tensor lengths,
    torch::Tensor initial) {
  validate_common(transitions, initial);
  TORCH_CHECK(lengths.is_cuda() && lengths.is_contiguous(), "lengths must be CUDA resident and contiguous");
  TORCH_CHECK(lengths.scalar_type() == torch::kInt64 && lengths.dim() == 1,
              "lengths must be an int64 vector");
  TORCH_CHECK(lengths.size(0) == transitions.size(0), "each event requires one duration");
  c10::cuda::CUDAGuard guard(initial.device());
  auto output = torch::empty_like(initial);
  const int total = static_cast<int>(initial.size(0) * initial.size(1));
  constexpr int threads = 256;
  regular_grid_temporal_response_kernel<<<(total + threads - 1) / threads, threads, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      transitions.data_ptr<float>(), lengths.data_ptr<int64_t>(),
      initial.data_ptr<float>(), output.data_ptr<float>(),
      static_cast<int>(initial.size(0)), static_cast<int>(initial.size(1)),
      static_cast<int>(transitions.size(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor event_forward(torch::Tensor powers, torch::Tensor initial) {
  validate_common(powers, initial);
  c10::cuda::CUDAGuard guard(initial.device());
  auto output = torch::empty_like(initial);
  const int total = static_cast<int>(initial.size(0) * initial.size(1));
  constexpr int threads = 256;
  event_composed_temporal_response_kernel<<<(total + threads - 1) / threads, threads, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      powers.data_ptr<float>(), initial.data_ptr<float>(), output.data_ptr<float>(),
      static_cast<int>(initial.size(0)), static_cast<int>(initial.size(1)),
      static_cast<int>(powers.size(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("regular_forward", &regular_forward, "Regular-grid temporal response (CUDA)");
  module.def("event_forward", &event_forward, "Event-composed temporal response (CUDA)");
}
