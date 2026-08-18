#include <metal_stdlib>
using namespace metal;

kernel void packed_mv(
    device float* output,
    const device float* matrix,
    const device float* coordinates,
    constant uint& columns,
    uint row [[thread_position_in_grid]]) {
  float value = 0.0f;
  for (uint column = 0; column < columns; ++column) {
    value += matrix[row * columns + column] * coordinates[column];
  }
  output[row] = value;
}

kernel void weighted_columns(
    device float* output,
    const device float* columns,
    const device long* indices,
    const device float* weights,
    const device long* offsets,
    constant uint& output_dimension,
    uint item [[thread_position_in_grid]]) {
  uint batch = item / output_dimension;
  uint coordinate = item - batch * output_dimension;
  float value = 0.0f;
  long begin = offsets[batch];
  long end = offsets[batch + 1];
  for (long entry = begin; entry < end; ++entry) {
    long feature = indices[entry];
    value += columns[feature * output_dimension + coordinate] * weights[entry];
  }
  output[item] = value;
}

kernel void modal_history_serial(
    device float* output,
    device float* position,
    device float* velocity,
    const device float* incidents,
    const device float* cosine,
    const device float* sine_over_omega,
    const device float* negative_omega_sine,
    const device float* force_position,
    const device float* force_velocity,
    const device float* modal_incident,
    const device float* modal_observation,
    constant uint& modes,
    constant uint& input_dimension,
    constant uint& output_dimension,
    constant uint& steps,
    uint item [[thread_position_in_grid]]) {
  if (item != 0) {
    return;
  }
  for (uint step = 0; step < steps; ++step) {
    for (uint mode = 0; mode < modes; ++mode) {
      float force = 0.0f;
      for (uint input = 0; input < input_dimension; ++input) {
        force += modal_incident[mode * input_dimension + input]
               * incidents[step * input_dimension + input];
      }
      float old_position = position[mode];
      float old_velocity = velocity[mode];
      position[mode] = cosine[mode] * old_position
                     + sine_over_omega[mode] * old_velocity
                     + force_position[mode] * force;
      velocity[mode] = negative_omega_sine[mode] * old_position
                     + cosine[mode] * old_velocity
                     + force_velocity[mode] * force;
    }
    for (uint coordinate = 0; coordinate < output_dimension; ++coordinate) {
      float value = 0.0f;
      for (uint mode = 0; mode < modes; ++mode) {
        value += modal_observation[coordinate * modes + mode] * position[mode];
      }
      output[step * output_dimension + coordinate] = value;
    }
  }
}

kernel void modal_history_threadgroup(
    device float* output,
    device float* position,
    device float* velocity,
    const device float* incidents,
    const device float* cosine,
    const device float* sine_over_omega,
    const device float* negative_omega_sine,
    const device float* force_position,
    const device float* force_velocity,
    const device float* modal_incident,
    const device float* modal_observation,
    constant uint& modes,
    constant uint& input_dimension,
    constant uint& output_dimension,
    constant uint& steps,
    uint item [[thread_position_in_threadgroup]]) {
  for (uint step = 0; step < steps; ++step) {
    if (item < modes) {
      float force = 0.0f;
      for (uint input = 0; input < input_dimension; ++input) {
        force += modal_incident[item * input_dimension + input]
               * incidents[step * input_dimension + input];
      }
      float old_position = position[item];
      float old_velocity = velocity[item];
      position[item] = cosine[item] * old_position
                     + sine_over_omega[item] * old_velocity
                     + force_position[item] * force;
      velocity[item] = negative_omega_sine[item] * old_position
                     + cosine[item] * old_velocity
                     + force_velocity[item] * force;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (item < output_dimension) {
      float value = 0.0f;
      for (uint mode = 0; mode < modes; ++mode) {
        value += modal_observation[item * modes + mode] * position[mode];
      }
      output[step * output_dimension + item] = value;
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
}
