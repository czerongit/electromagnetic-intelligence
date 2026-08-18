#include <torch/extension.h>

#include <vector>

torch::Tensor weighted_columns_cpu(
    torch::Tensor columns,
    torch::Tensor indices,
    torch::Tensor weights,
    torch::Tensor offsets) {
  TORCH_CHECK(columns.device().is_cpu(), "columns must be on CPU");
  TORCH_CHECK(indices.device().is_cpu(), "indices must be on CPU");
  TORCH_CHECK(weights.device().is_cpu(), "weights must be on CPU");
  TORCH_CHECK(offsets.device().is_cpu(), "offsets must be on CPU");
  TORCH_CHECK(columns.dim() == 2, "columns must be a matrix");
  TORCH_CHECK(indices.dim() == 1, "indices must be a vector");
  TORCH_CHECK(weights.dim() == 1, "weights must be a vector");
  TORCH_CHECK(offsets.dim() == 1 && offsets.size(0) >= 1,
              "offsets must contain the final offset");
  TORCH_CHECK(indices.scalar_type() == torch::kInt64, "indices must be int64");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt64, "offsets must be int64");
  TORCH_CHECK(weights.scalar_type() == columns.scalar_type(),
              "weights and columns must have one dtype");
  TORCH_CHECK(indices.size(0) == weights.size(0),
              "indices and weights must have one length");

  auto contiguous_columns = columns.contiguous();
  auto contiguous_indices = indices.contiguous();
  auto contiguous_weights = weights.contiguous();
  auto contiguous_offsets = offsets.contiguous();
  const auto batch = contiguous_offsets.size(0) - 1;
  const auto output_dimension = contiguous_columns.size(1);
  auto output = torch::zeros({batch, output_dimension}, columns.options());

  AT_DISPATCH_FLOATING_TYPES(
      contiguous_columns.scalar_type(), "weighted_columns_cpu", [&] {
        auto column_values = contiguous_columns.accessor<scalar_t, 2>();
        auto index_values = contiguous_indices.accessor<int64_t, 1>();
        auto weight_values = contiguous_weights.accessor<scalar_t, 1>();
        auto offset_values = contiguous_offsets.accessor<int64_t, 1>();
        auto result = output.accessor<scalar_t, 2>();
        for (int64_t row = 0; row < batch; ++row) {
          const auto begin = offset_values[row];
          const auto end = offset_values[row + 1];
          for (int64_t entry = begin; entry < end; ++entry) {
            const auto feature = index_values[entry];
            TORCH_CHECK(feature >= 0 && feature < contiguous_columns.size(0),
                        "feature index is outside compiled columns");
            const auto weight = weight_values[entry];
            for (int64_t coordinate = 0; coordinate < output_dimension;
                 ++coordinate) {
              result[row][coordinate] +=
                  column_values[feature][coordinate] * weight;
            }
          }
        }
      });
  return output;
}

std::vector<torch::Tensor> modal_history_cpu(
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
  TORCH_CHECK(incidents.device().is_cpu(), "incidents must be on CPU");
  TORCH_CHECK(incidents.dim() == 2, "incidents must be a matrix");
  TORCH_CHECK(modal_incident.dim() == 2, "modal incident port must be a matrix");
  TORCH_CHECK(modal_observation.dim() == 2,
              "modal observation port must be a matrix");
  TORCH_CHECK(incidents.scalar_type() == cosine.scalar_type(),
              "recurrent tensors must have one dtype");

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
  TORCH_CHECK(position.numel() == modes && velocity.numel() == modes,
              "initial recurrent state has the wrong dimension");
  TORCH_CHECK(current_modal_incident.size(0) == modes &&
                  current_modal_incident.size(1) == input_dimension,
              "modal incident port has the wrong shape");
  TORCH_CHECK(current_modal_observation.size(1) == modes,
              "modal observation port has the wrong shape");
  auto output = torch::zeros({steps, output_dimension}, incidents.options());

  AT_DISPATCH_FLOATING_TYPES(current_cosine.scalar_type(), "modal_history_cpu", [&] {
    auto incident_values = current_incidents.accessor<scalar_t, 2>();
    auto cosine_values = current_cosine.accessor<scalar_t, 1>();
    auto sine_values = current_sine_over_omega.accessor<scalar_t, 1>();
    auto negative_sine_values =
        current_negative_omega_sine.accessor<scalar_t, 1>();
    auto force_position_values = current_force_position.accessor<scalar_t, 1>();
    auto force_velocity_values = current_force_velocity.accessor<scalar_t, 1>();
    auto modal_incident_values = current_modal_incident.accessor<scalar_t, 2>();
    auto modal_observation_values =
        current_modal_observation.accessor<scalar_t, 2>();
    auto position_values = position.accessor<scalar_t, 1>();
    auto velocity_values = velocity.accessor<scalar_t, 1>();
    auto result = output.accessor<scalar_t, 2>();
    std::vector<scalar_t> next_position(modes);
    std::vector<scalar_t> next_velocity(modes);
    for (int64_t step = 0; step < steps; ++step) {
      for (int64_t mode = 0; mode < modes; ++mode) {
        scalar_t force = 0;
        for (int64_t input = 0; input < input_dimension; ++input) {
          force += modal_incident_values[mode][input] *
                   incident_values[step][input];
        }
        const auto old_position = position_values[mode];
        const auto old_velocity = velocity_values[mode];
        next_position[mode] = cosine_values[mode] * old_position +
                              sine_values[mode] * old_velocity +
                              force_position_values[mode] * force;
        next_velocity[mode] = negative_sine_values[mode] * old_position +
                              cosine_values[mode] * old_velocity +
                              force_velocity_values[mode] * force;
      }
      for (int64_t mode = 0; mode < modes; ++mode) {
        position_values[mode] = next_position[mode];
        velocity_values[mode] = next_velocity[mode];
      }
      for (int64_t coordinate = 0; coordinate < output_dimension; ++coordinate) {
        scalar_t value = 0;
        for (int64_t mode = 0; mode < modes; ++mode) {
          value += modal_observation_values[coordinate][mode] *
                   position_values[mode];
        }
        result[step][coordinate] = value;
      }
    }
  });
  return {output, position, velocity};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("weighted_columns", &weighted_columns_cpu,
             "Weighted response columns on CPU");
  module.def("modal_history", &modal_history_cpu,
             "Complete modal response history on CPU");
}
