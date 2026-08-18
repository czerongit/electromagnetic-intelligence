#include <metal_stdlib>
using namespace metal;

kernel void modal_step_parallel(
    device float* position,
    device float* velocity,
    const device float* incident,
    const device float* cosine,
    const device float* sine_over_omega,
    const device float* negative_omega_sine,
    const device float* force_position,
    const device float* force_velocity,
    const device float* modal_incident,
    constant uint& modes,
    constant uint& input_dimension,
    uint mode [[thread_position_in_grid]]) {
  if (mode >= modes) {
    return;
  }
  float force = 0.0f;
  for (uint input = 0; input < input_dimension; ++input) {
    force += modal_incident[mode * input_dimension + input] * incident[input];
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

kernel void modal_readout_parallel(
    device float* output,
    const device float* position,
    const device float* modal_observation,
    constant uint& modes,
    constant uint& output_dimension,
    uint coordinate [[thread_position_in_grid]]) {
  if (coordinate >= output_dimension) {
    return;
  }
  float value = 0.0f;
  for (uint mode = 0; mode < modes; ++mode) {
    value += modal_observation[coordinate * modes + mode] * position[mode];
  }
  output[coordinate] = value;
}
