#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <vector>

__global__ void modal_step_parallel_kernel(
    float* position,
    float* velocity,
    const float* incident,
    const float* cosine,
    const float* sine_over_omega,
    const float* negative_omega_sine,
    const float* force_position,
    const float* force_velocity,
    const float* modal_incident,
    int64_t modes,
    int64_t input_dimension) {
  const auto mode = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (mode >= modes) {
    return;
  }
  float force = 0.0f;
  for (int64_t input = 0; input < input_dimension; ++input) {
    force += modal_incident[mode * input_dimension + input] * incident[input];
  }
  const auto old_position = position[mode];
  const auto old_velocity = velocity[mode];
  position[mode] = cosine[mode] * old_position +
                   sine_over_omega[mode] * old_velocity +
                   force_position[mode] * force;
  velocity[mode] = negative_omega_sine[mode] * old_position +
                   cosine[mode] * old_velocity +
                   force_velocity[mode] * force;
}

__global__ void modal_readout_parallel_kernel(
    float* output,
    const float* position,
    const float* modal_observation,
    int64_t modes,
    int64_t output_dimension) {
  const auto coordinate =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (coordinate >= output_dimension) {
    return;
  }
  float value = 0.0f;
  for (int64_t mode = 0; mode < modes; ++mode) {
    value += modal_observation[coordinate * modes + mode] * position[mode];
  }
  output[coordinate] = value;
}

void require_cuda_fp32(torch::Tensor value, const char* name) {
  TORCH_CHECK(value.is_cuda(), name, " must be on CUDA");
  TORCH_CHECK(value.scalar_type() == torch::kFloat32, name, " must be float32");
}

std::vector<torch::Tensor> modal_history_wide_cuda(
    torch::Tensor incidents,
    torch::Tensor cosine,
    torch::Tensor sine_over_omega,
    torch::Tensor negative_omega_sine,
    torch::Tensor force_position,
    torch::Tensor force_velocity,
    torch::Tensor modal_incident,
    torch::Tensor modal_observation,
    torch::Tensor initial_position,
    torch::Tensor initial_velocity) {
  require_cuda_fp32(incidents, "incidents");
  require_cuda_fp32(cosine, "cosine");
  require_cuda_fp32(sine_over_omega, "sine_over_omega");
  require_cuda_fp32(negative_omega_sine, "negative_omega_sine");
  require_cuda_fp32(force_position, "force_position");
  require_cuda_fp32(force_velocity, "force_velocity");
  require_cuda_fp32(modal_incident, "modal_incident");
  require_cuda_fp32(modal_observation, "modal_observation");
  require_cuda_fp32(initial_position, "initial_position");
  require_cuda_fp32(initial_velocity, "initial_velocity");

  auto current_incidents = incidents.contiguous();
  auto current_cosine = cosine.contiguous();
  auto current_sine_over_omega = sine_over_omega.contiguous();
  auto current_negative_omega_sine = negative_omega_sine.contiguous();
  auto current_force_position = force_position.contiguous();
  auto current_force_velocity = force_velocity.contiguous();
  auto current_modal_incident = modal_incident.contiguous();
  auto current_modal_observation = modal_observation.contiguous();
  auto position = initial_position.contiguous().clone();
  auto velocity = initial_velocity.contiguous().clone();

  TORCH_CHECK(current_incidents.dim() == 2, "incidents must be a matrix");
  TORCH_CHECK(current_modal_incident.dim() == 2,
              "modal incident port must be a matrix");
  TORCH_CHECK(current_modal_observation.dim() == 2,
              "modal observation port must be a matrix");
  const auto steps = current_incidents.size(0);
  const auto input_dimension = current_incidents.size(1);
  const auto modes = current_cosine.size(0);
  const auto output_dimension = current_modal_observation.size(0);
  TORCH_CHECK(modes > 1024 || output_dimension > 1024,
              "use one-block recurrence at narrow widths");
  TORCH_CHECK(position.numel() == modes && velocity.numel() == modes,
              "prior state has the wrong dimension");
  TORCH_CHECK(current_modal_incident.size(0) == modes &&
                  current_modal_incident.size(1) == input_dimension,
              "modal incident port has the wrong shape");
  TORCH_CHECK(current_modal_observation.size(1) == modes,
              "modal observation port has the wrong shape");

  auto output = torch::empty({steps, output_dimension}, incidents.options());
  constexpr int threads = 256;
  const auto mode_blocks = (modes + threads - 1) / threads;
  const auto output_blocks = (output_dimension + threads - 1) / threads;
  const auto stream = at::cuda::getCurrentCUDAStream();
  for (int64_t step = 0; step < steps; ++step) {
    modal_step_parallel_kernel<<<mode_blocks, threads, 0, stream>>>(
        position.data_ptr<float>(), velocity.data_ptr<float>(),
        current_incidents.data_ptr<float>() + step * input_dimension,
        current_cosine.data_ptr<float>(),
        current_sine_over_omega.data_ptr<float>(),
        current_negative_omega_sine.data_ptr<float>(),
        current_force_position.data_ptr<float>(),
        current_force_velocity.data_ptr<float>(),
        current_modal_incident.data_ptr<float>(), modes, input_dimension);
    modal_readout_parallel_kernel<<<output_blocks, threads, 0, stream>>>(
        output.data_ptr<float>() + step * output_dimension,
        position.data_ptr<float>(), current_modal_observation.data_ptr<float>(),
        modes, output_dimension);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, position, velocity};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("modal_history", &modal_history_wide_cuda,
             "Wide complete modal response history on CUDA");
}
