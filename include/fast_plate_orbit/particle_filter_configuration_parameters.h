#pragma once

#include <Eigen/Dense>

namespace fast_plate_orbit {

struct particle_filter_configuration_parameters {
  float radius_prior;
  float visibility_logit_coefficient;

  float radius_prior_variance_one_plate;
  float radius_prior_variance_two_plates;
  float radius_process_variance;

  float orientation_velocity_prior_variance;
  float orientation_velocity_process_variance;

  float center_z_position_process_variance;

  float z_offset_prior;
  float z_offset_prior_variance;
  float z_offset_process_variance;

  Eigen::Vector2f center_xy_velocity_prior_diagonal_covariance;
  Eigen::Vector2f center_xy_velocity_process_diagonal_covariance;

  // Switches for A/B runs in the simulator; the defaults are the behavior before they existed.
  // radius_mode 0: radius noise only while |omega| > 2 rad/s; 1: always; 2: always, and each radius
  // relaxes toward radius_prior with time constant radius_reversion_time (s).
  int radius_mode = 0;
  float radius_reversion_time = 2.0f;
  // z_offset < 0 relabels the pairs (orientation + pi/2, radii swapped, z_offset negated) instead
  // of reflecting at 0, so a particle can change which pair it calls high.
  bool relabel_at_zero = false;
  // The particle average also tries the +-pi/2 relabelings of each particle.
  bool swap_reduction = false;
  // Plate visibility from horizontal directions only (as plate_orbit_v2), so plate heights and the
  // observer's height don't change how visible a plate looks.
  bool visibility_xy = false;
};

}  // namespace fast_plate_orbit
