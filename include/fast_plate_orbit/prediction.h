#pragma once

#include <fast_plate_orbit/observation.h>
#include <fast_plate_orbit/predicted_plate.h>
#include <pf/util/device_array.h>

#include <Eigen/Dense>
#include <array>

namespace fast_plate_orbit {

namespace helper {

PF_TARGET_ATTRS [[nodiscard]] inline float to_radius(const float& radius) noexcept {
  constexpr float min_radius = 0.100f;
  constexpr float max_radius = 0.400f;
  return thrust::max(min_radius, thrust::min(max_radius, radius));
}

PF_TARGET_ATTRS [[nodiscard]] inline float to_orientation(const float& angle_radians) noexcept {
  // Period is PI (not PI/2): adjacent plate pairs sit at different heights, so a 90 degree
  // rotation is no longer a symmetry of the target -- only a 180 degree rotation is.
  const float value = fmod(angle_radians, M_PI);
  return (value < 0.0f) ? value + M_PI : value;
}

PF_TARGET_ATTRS [[nodiscard]] inline float to_z_offset(const float& z_offset) noexcept {
  constexpr float offset_limit = 0.05f;
  return thrust::max(-offset_limit, thrust::min(offset_limit, z_offset));
}

PF_TARGET_ATTRS [[nodiscard]] inline Eigen::Vector3f rpad_zero(const Eigen::Vector2f& vector) noexcept {
  return (Eigen::Vector3f{} << vector, Eigen::Matrix<float, 1, 1>::Zero()).finished();
}

PF_TARGET_ATTRS [[nodiscard]] inline Eigen::Vector3f lpad_zero(const float& scalar) noexcept {
  return (Eigen::Vector3f{} << 0.0f, 0.0f, scalar).finished();
}

}  // namespace helper

class prediction {
 private:
  static constexpr size_t number_of_plates = 4;

  float radius_;
  float z_offset_;  // half the height difference between the two alternating plate pairs

  float orientation_;
  float orientation_velocity_;

  Eigen::Vector3f center_;
  Eigen::Vector2f center_velocity_;

  struct angle_offset_and_radius {
    float angle_offset;
    float radius;
    float z_sign;  // +1 for the high pair, -1 for the low pair
  };

 public:
  [[nodiscard]] std::array<predicted_plate, number_of_plates> predicted_plates_for_host() const noexcept {
    return predicted_plates().to_host_array();
  }

  PF_TARGET_ATTRS [[nodiscard]] pf::util::device_array<predicted_plate, number_of_plates> predicted_plates() const noexcept {
    const pf::util::device_array<angle_offset_and_radius, number_of_plates> angle_offsets_and_radii = {
        angle_offset_and_radius{0.0f, radius_, +1.0f},
        angle_offset_and_radius{M_PI_2, radius_, -1.0f},
        angle_offset_and_radius{M_PI, radius_, +1.0f},
        angle_offset_and_radius{M_PI + M_PI_2, radius_, -1.0f},
    };

    return angle_offsets_and_radii.transformed([this](const angle_offset_and_radius& value) {
      const float angle = value.angle_offset + orientation_;

      const Eigen::Vector3f predicted_plate_position =
          center_ + (Eigen::Vector3f{} << value.radius * cosf(angle), value.radius * sinf(angle),
                     value.z_sign * z_offset_)
                        .finished();

      const Eigen::Vector3f predicted_plate_velocity =
          helper::rpad_zero(center_velocity_) +
          orientation_velocity_ * value.radius * (Eigen::Vector3f{} << -sinf(angle), cosf(angle), 0.0f).finished();

      return predicted_plate(predicted_plate_position, predicted_plate_velocity);
    });
  }

  PF_TARGET_ATTRS [[nodiscard]] prediction extrapolate_state(const float& time_offset_seconds) const noexcept {
    return prediction(
        radius_,
        orientation_ + time_offset_seconds * orientation_velocity_,
        orientation_velocity_,
        center_ + time_offset_seconds * helper::rpad_zero(center_velocity_),
        center_velocity_,
        z_offset_);
  }

  PF_TARGET_ATTRS void update_state(
      const float& time_offset_seconds,
      const float& radius_noise,
      const float& orientation_velocity_noise_0,
      const float& orientation_velocity_noise_1,
      const float& center_z_position_noise,
      const Eigen::Vector2f& center_xy_velocity_noise_0,
      const Eigen::Vector2f& center_xy_velocity_noise_1) noexcept {
    static constexpr float one_half = 1.0 / 2.0;
    static constexpr float one_twelfth = 1.0 / 12.0;

    const float radius_noise_scale = sqrtf(time_offset_seconds);
    const float velocity_noise_scale = radius_noise_scale;
    const float center_z_position_noise_scale = radius_noise_scale;
    const float position_noise_scale = sqrtf(one_twelfth) * powf(velocity_noise_scale, 3);

    const float d_radius = radius_noise_scale * radius_noise;
    // z_offset_ is a fixed geometry constant -- it is never noised or updated.
    const float d_center_z = center_z_position_noise_scale * center_z_position_noise;

    const float d_orientation_velocity = velocity_noise_scale * orientation_velocity_noise_1;
    const float d_orientation = time_offset_seconds * orientation_velocity_ +
                                one_half * time_offset_seconds * d_orientation_velocity +
                                position_noise_scale * orientation_velocity_noise_0;

    const Eigen::Vector2f d_center_velocity = velocity_noise_scale * center_xy_velocity_noise_1;
    const Eigen::Vector3f d_center = time_offset_seconds * helper::rpad_zero(center_velocity_) +
                                     one_half * time_offset_seconds * helper::rpad_zero(d_center_velocity) +
                                     position_noise_scale * helper::rpad_zero(center_xy_velocity_noise_0) +
                                     helper::lpad_zero(d_center_z);

    radius_ = helper::to_radius(radius_ + d_radius);

    orientation_ = helper::to_orientation(orientation_ + d_orientation);
    orientation_velocity_ = orientation_velocity_ + d_orientation_velocity;

    center_ = center_ + d_center;
    center_velocity_ = center_velocity_ + d_center_velocity;
  }

  PF_TARGET_ATTRS [[nodiscard]] const float& radius() const noexcept { return radius_; }
  PF_TARGET_ATTRS [[nodiscard]] const float& z_offset() const noexcept { return z_offset_; }
  PF_TARGET_ATTRS [[nodiscard]] const float& orientation() const noexcept { return orientation_; }
  PF_TARGET_ATTRS [[nodiscard]] const float& orientation_velocity() const noexcept { return orientation_velocity_; }
  PF_TARGET_ATTRS [[nodiscard]] const Eigen::Vector3f& center() const noexcept { return center_; }
  PF_TARGET_ATTRS [[nodiscard]] const Eigen::Vector2f& center_velocity() const noexcept { return center_velocity_; }

  PF_TARGET_ATTRS prediction() noexcept
      : radius_{0.0f},
        z_offset_{0.0f},
        orientation_{0.0f},
        orientation_velocity_{0.0f},
        center_{Eigen::Vector3f::Zero()},
        center_velocity_{Eigen::Vector2f::Zero()} {}

  PF_TARGET_ATTRS prediction(
      const float& radius,
      const float& orientation,
      const float& orientation_velocity,
      const Eigen::Vector3f& center,
      const Eigen::Vector2f& center_velocity,
      const float& z_offset) noexcept
      : radius_{helper::to_radius(radius)},
        z_offset_{helper::to_z_offset(z_offset)},
        orientation_{helper::to_orientation(orientation)},
        orientation_velocity_{orientation_velocity},
        center_{center},
        center_velocity_{center_velocity} {}
};

}  // namespace fast_plate_orbit
