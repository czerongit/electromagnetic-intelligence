#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <vector>

__global__ void packed_mv_kernel(
    float* output,
    const float* matrix,
    const float* coordinates,
    int64_t rows,
    int64_t columns) {
  const auto row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  float value = 0.0f;
  for (int64_t column = 0; column < columns; ++column) {
    value += matrix[row * columns + column] * coordinates[column];
  }
  output[row] = value;
}

__global__ void weighted_columns_kernel(
    float* output,
    const float* columns,
    const int64_t* indices,
    const float* weights,
    const int64_t* offsets,
    int64_t batch,
    int64_t output_dimension) {
  const auto item = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto total = batch * output_dimension;
  if (item >= total) {
    return;
  }
  const auto row = item / output_dimension;
  const auto coordinate = item - row * output_dimension;
  float value = 0.0f;
  for (int64_t entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
    const auto feature = indices[entry];
    value += columns[feature * output_dimension + coordinate] * weights[entry];
  }
  output[item] = value;
}

__global__ void modal_history_threadgroup_kernel(
    float* output,
    float* position,
    float* velocity,
    const float* incidents,
    const float* cosine,
    const float* sine_over_omega,
    const float* negative_omega_sine,
    const float* force_position,
    const float* force_velocity,
    const float* modal_incident,
    const float* modal_observation,
    int64_t modes,
    int64_t input_dimension,
    int64_t output_dimension,
    int64_t steps) {
  const auto item = static_cast<int64_t>(threadIdx.x);
  for (int64_t step = 0; step < steps; ++step) {
    if (item < modes) {
      float force = 0.0f;
      for (int64_t input = 0; input < input_dimension; ++input) {
        force += modal_incident[item * input_dimension + input] *
                 incidents[step * input_dimension + input];
      }
      const auto old_position = position[item];
      const auto old_velocity = velocity[item];
      position[item] = cosine[item] * old_position +
                       sine_over_omega[item] * old_velocity +
                       force_position[item] * force;
      velocity[item] = negative_omega_sine[item] * old_position +
                       cosine[item] * old_velocity +
                       force_velocity[item] * force;
    }
    __syncthreads();
    if (item < output_dimension) {
      float value = 0.0f;
      for (int64_t mode = 0; mode < modes; ++mode) {
        value += modal_observation[item * modes + mode] * position[mode];
      }
      output[step * output_dimension + item] = value;
    }
    __syncthreads();
  }
}

__global__ void modal_history_serial_kernel(
    float* output,
    float* position,
    float* velocity,
    const float* incidents,
    const float* cosine,
    const float* sine_over_omega,
    const float* negative_omega_sine,
    const float* force_position,
    const float* force_velocity,
    const float* modal_incident,
    const float* modal_observation,
    int64_t modes,
    int64_t input_dimension,
    int64_t output_dimension,
    int64_t steps) {
  for (int64_t step = 0; step < steps; ++step) {
    for (int64_t mode = 0; mode < modes; ++mode) {
      float force = 0.0f;
      for (int64_t input = 0; input < input_dimension; ++input) {
        force += modal_incident[mode * input_dimension + input] *
                 incidents[step * input_dimension + input];
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
    for (int64_t coordinate = 0; coordinate < output_dimension; ++coordinate) {
      float value = 0.0f;
      for (int64_t mode = 0; mode < modes; ++mode) {
        value += modal_observation[coordinate * modes + mode] * position[mode];
      }
      output[step * output_dimension + coordinate] = value;
    }
  }
}

void require_cuda_fp32(torch::Tensor value, const char* name) {
  TORCH_CHECK(value.is_cuda(), name, " must be on CUDA");
  TORCH_CHECK(value.scalar_type() == torch::kFloat32, name, " must be float32");
}

torch::Tensor packed_mv_cuda(
    torch::Tensor matrix, torch::Tensor coordinates) {
  require_cuda_fp32(matrix, "matrix");
  require_cuda_fp32(coordinates, "coordinates");
  auto current_matrix = matrix.contiguous();
  auto current_coordinates = coordinates.contiguous();
  TORCH_CHECK(current_matrix.dim() == 2, "matrix must have rank two");
  TORCH_CHECK(current_coordinates.dim() == 1, "coordinates must be a vector");
  TORCH_CHECK(current_matrix.size(1) == current_coordinates.size(0),
              "packed coordinate width does not match the map");
  auto output = torch::empty({current_matrix.size(0)}, matrix.options());
  constexpr int threads = 256;
  const auto blocks = (current_matrix.size(0) + threads - 1) / threads;
  packed_mv_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<float>(),
      current_matrix.data_ptr<float>(),
      current_coordinates.data_ptr<float>(),
      current_matrix.size(0),
      current_matrix.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor weighted_columns_cuda(
    torch::Tensor columns,
    torch::Tensor indices,
    torch::Tensor weights,
    torch::Tensor offsets) {
  require_cuda_fp32(columns, "columns");
  require_cuda_fp32(weights, "weights");
  TORCH_CHECK(indices.is_cuda() && indices.scalar_type() == torch::kInt64,
              "indices must be CUDA int64");
  TORCH_CHECK(offsets.is_cuda() && offsets.scalar_type() == torch::kInt64,
              "offsets must be CUDA int64");
  auto current_columns = columns.contiguous();
  auto current_indices = indices.contiguous();
  auto current_weights = weights.contiguous();
  auto current_offsets = offsets.contiguous();
  const auto batch = current_offsets.size(0) - 1;
  const auto output_dimension = current_columns.size(1);
  auto output = torch::empty({batch, output_dimension}, columns.options());
  const auto total = batch * output_dimension;
  if (total) {
    constexpr int threads = 256;
    const auto blocks = (total + threads - 1) / threads;
    weighted_columns_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<float>(),
        current_columns.data_ptr<float>(),
        current_indices.data_ptr<int64_t>(),
        current_weights.data_ptr<float>(),
        current_offsets.data_ptr<int64_t>(),
        batch,
        output_dimension);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

std::vector<torch::Tensor> modal_history_cuda(
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
  const auto steps = current_incidents.size(0);
  const auto input_dimension = current_incidents.size(1);
  const auto modes = current_cosine.size(0);
  const auto output_dimension = current_modal_observation.size(0);
  auto output = torch::empty({steps, output_dimension}, incidents.options());
  if (steps) {
    const auto width = std::max(modes, output_dimension);
    if (width <= 1024) {
      modal_history_threadgroup_kernel<<<
          1, width, 0, at::cuda::getCurrentCUDAStream()>>>(
          output.data_ptr<float>(), position.data_ptr<float>(),
          velocity.data_ptr<float>(), current_incidents.data_ptr<float>(),
          current_cosine.data_ptr<float>(),
          current_sine_over_omega.data_ptr<float>(),
          current_negative_omega_sine.data_ptr<float>(),
          current_force_position.data_ptr<float>(),
          current_force_velocity.data_ptr<float>(),
          current_modal_incident.data_ptr<float>(),
          current_modal_observation.data_ptr<float>(), modes, input_dimension,
          output_dimension, steps);
    } else {
      modal_history_serial_kernel<<<
          1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
          output.data_ptr<float>(), position.data_ptr<float>(),
          velocity.data_ptr<float>(), current_incidents.data_ptr<float>(),
          current_cosine.data_ptr<float>(),
          current_sine_over_omega.data_ptr<float>(),
          current_negative_omega_sine.data_ptr<float>(),
          current_force_position.data_ptr<float>(),
          current_force_velocity.data_ptr<float>(),
          current_modal_incident.data_ptr<float>(),
          current_modal_observation.data_ptr<float>(), modes, input_dimension,
          output_dimension, steps);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output, position, velocity};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("packed_mv", &packed_mv_cuda, "Packed response map on CUDA");
  module.def("weighted_columns", &weighted_columns_cuda,
             "Weighted response columns on CUDA");
  module.def("modal_history", &modal_history_cuda,
             "Complete modal response history on CUDA");
}
