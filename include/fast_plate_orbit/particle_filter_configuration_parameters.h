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

  // Fixed plate-height model: the high plate pair sits plate_height_offset above the orbit center,
  // the low pair the same amount below. This is known robot geometry, NOT estimated -- the filter
  // only uses it in the likelihood so the orientation phase locks onto which side is actually high.
  float plate_height_offset;

  Eigen::Vector2f center_xy_velocity_prior_diagonal_covariance;
  Eigen::Vector2f center_xy_velocity_process_diagonal_covariance;
};

}  // namespace fast_plate_orbit
