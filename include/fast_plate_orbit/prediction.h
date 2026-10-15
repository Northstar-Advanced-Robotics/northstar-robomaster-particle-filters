#pragma once

#include <fast_plate_orbit/observation.h>
#include <fast_plate_orbit/predicted_plate.h>
#include <pf/util/device_array.h>
#include <pf/util/min_max.h>

#include <Eigen/Dense>
#include <array>

namespace fast_plate_orbit {

namespace helper {

PF_TARGET_ATTRS [[nodiscard]] inline float to_radius(const float& radius) noexcept {
  constexpr float min_radius = 0.100f;
  constexpr float max_radius = 0.500f;
  return pf::util::max(min_radius, pf::util::min(max_radius, radius));
}

PF_TARGET_ATTRS [[nodiscard]] inline float to_orientation(const float& angle_radians) noexcept {
  // Period is PI (not PI/2): adjacent plate pairs sit at different heights, so a 90 degree
  // rotation is no longer a symmetry of the target -- only a 180 degree rotation is.
  const float value = fmod(angle_radians, M_PI);
  return (value < 0.0f) ? value + M_PI : value;
}

PF_TARGET_ATTRS [[nodiscard]] inline float to_z_offset(const float& z_offset) noexcept {
  constexpr float offset_limit = 0.06f;
  return pf::util::min(offset_limit, fabsf(z_offset));
}

PF_TARGET_ATTRS [[nodiscard]] inline Eigen::Vector2f to_slope(const Eigen::Vector2f& slope) noexcept {
  constexpr float slope_limit = 0.6f;  // ~31 degrees
  const float norm = slope.norm();
  return norm > slope_limit ? (slope * (slope_limit / norm)).eval() : slope;
}

// The chassis tilt on ground of gradient `slope`: the rotation taking world up onto the ground's
// normal n (about the horizontal axis across the slope), applied to a chassis-frame vector v.
PF_TARGET_ATTRS [[nodiscard]] inline Eigen::Vector3f tilt_by_slope(const Eigen::Vector2f& slope, const Eigen::Vector3f& v) noexcept {
  const float inv = 1.0f / sqrtf(1.0f + slope.squaredNorm());
  const Eigen::Vector3f n{-slope[0] * inv, -slope[1] * inv, inv};
  const Eigen::Vector3f a{-n[1], n[0], 0.0f};  // up x n
  const Eigen::Vector3f av = a.cross(v);
  return v + av + a.cross(av) / (1.0f + n[2]);
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

  float radius_0_;
  float radius_1_;
  float z_offset_;  // half the height difference between the two alternating plate pairs
  float settled_;   // seconds this particle has watched the robot spin (geometry settles with it)
  float maneuver_;  // 1 while this particle's robot is maneuvering (see maneuver_velocity_variance)

  float orientation_;
  float orientation_velocity_;

  Eigen::Vector3f center_;
  Eigen::Vector2f center_velocity_;
  // Gradient (dz/dx, dz/dy) of the ground under the robot: its chassis tilts with it, so a plate
  // sits radius * slope.(its direction) higher, and driving moves the center by slope.velocity
  // up or down. Zero unless the filter's slope_process_variance turns it on.
  Eigen::Vector2f slope_;

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
        angle_offset_and_radius{0.0f, radius_0_, +1.0f},
        angle_offset_and_radius{M_PI_2, radius_1_, -1.0f},
        angle_offset_and_radius{M_PI, radius_0_, +1.0f},
        angle_offset_and_radius{M_PI + M_PI_2, radius_1_, -1.0f},
    };

    return angle_offsets_and_radii.transformed([this](const angle_offset_and_radius& value) {
      const float angle = value.angle_offset + orientation_;

      const float c = cosf(angle);
      const float s = sinf(angle);
      // In the chassis frame, then tilted with the ground under the robot (nothing on level ground).
      const Eigen::Vector3f offset{value.radius * c, value.radius * s, value.z_sign * z_offset_};
      const Eigen::Vector3f turning{-value.radius * s, value.radius * c, 0.0f};
      const bool level = slope_[0] == 0.0f && slope_[1] == 0.0f;
      const Eigen::Vector3f predicted_plate_position = center_ + (level ? offset : helper::tilt_by_slope(slope_, offset));

      const Eigen::Vector3f predicted_plate_velocity =
          (Eigen::Vector3f{} << center_velocity_, slope_.dot(center_velocity_)).finished() +
          orientation_velocity_ * (level ? turning : helper::tilt_by_slope(slope_, turning));

      return predicted_plate(predicted_plate_position, predicted_plate_velocity);
    });
  }

  PF_TARGET_ATTRS [[nodiscard]] prediction extrapolate_state(const float& time_offset_seconds) const noexcept {
    return prediction(
        radius_0_,
        radius_1_,
        orientation_ + time_offset_seconds * orientation_velocity_,
        orientation_velocity_,
        center_ + time_offset_seconds * (Eigen::Vector3f{} << center_velocity_, slope_.dot(center_velocity_)).finished(),
        center_velocity_,
        z_offset_,
        settled_,
        maneuver_,
        slope_);
  }

  PF_TARGET_ATTRS void update_state(
      const float& time_offset_seconds,
      const float& radius_0_noise,
      const float& radius_1_noise,
      const float& orientation_velocity_noise_0,
      const float& orientation_velocity_noise_1,
      const float& center_z_position_noise,
      const float& z_offset_noise,
      const Eigen::Vector2f& center_xy_velocity_noise_0,
      const Eigen::Vector2f& center_xy_velocity_noise_1,
      const int& radius_mode = 0,
      const float& radius_prior = 0.0f,
      const float& radius_reversion_time = 1.0f,
      const bool& relabel_at_zero = false,
      const float& lock_time = 0.0f,
      const float& lock_floor = 1.0f,
      const bool& settle_by_angle = false,
      const Eigen::Vector2f& slope_noise = Eigen::Vector2f::Zero(),
      const float& center_z_extra_noise = 0.0f) noexcept {
    static constexpr float one_half = 1.0 / 2.0;
    static constexpr float one_twelfth = 1.0 / 12.0;

    const float radius_noise_scale = sqrtf(time_offset_seconds);
    const float velocity_noise_scale = radius_noise_scale;
    const float center_z_position_noise_scale = radius_noise_scale;
    const float position_noise_scale = sqrtf(one_twelfth) * powf(velocity_noise_scale, 3);

    // How fast the robot turns, as a fraction of a full-information spin (1.5 rad/s): the radii and
    // the height offset only show while it turns.
    const float spin_fraction = pf::util::min(1.0f, fabsf(orientation_velocity_) / 1.5f);
    const float radius_noise_gain = radius_mode == 4 ? spin_fraction :
                                    (radius_mode != 0 || fabsf(orientation_velocity_) > 2.0f) ? 1.0f :
                                                                                               0.0f;
    // Mode 3 pulls only while the robot isn't spinning (fully below 1 rad/s, not at all above 3),
    // when the radius can't be seen; spinning, the plates pin it down without a bias toward the prior.
    const float spin_gate = pf::util::max(0.0f, pf::util::min(1.0f, (3.0f - fabsf(orientation_velocity_)) / 2.0f));
    // Mode 4 pulls only a particle that hasn't yet watched the robot turn (a fresh filter on a still
    // robot); once it has, what it learned stays when the robot stops.
    const float pull_weight = radius_mode == 2 ? 1.0f :
                              radius_mode == 3 ? spin_gate :
                              radius_mode == 4 ? (1.0f - spin_fraction) * expf(-settled_ / 6.2832f) :
                                                 0.0f;
    const float radius_pull = pull_weight * pf::util::min(1.0f, time_offset_seconds / radius_reversion_time);
    // Geometry settles as the particle watches the robot spin: its noise fades toward lock_floor.
    // settled: seconds spent spinning faster than 2 rad/s, or with settle_by_angle, radians turned.
    if (settle_by_angle) {
      if (fabsf(orientation_velocity_) > 0.5f) settled_ += fabsf(orientation_velocity_) * time_offset_seconds;
    } else if (fabsf(orientation_velocity_) > 2.0f) {
      settled_ += time_offset_seconds;
    }
    const float lock_gain = lock_time > 0.0f ? lock_floor + (1.0f - lock_floor) * expf(-settled_ / lock_time) : 1.0f;

    const float d_radius_0 =
        lock_gain * radius_noise_gain * radius_noise_scale * radius_0_noise - radius_pull * (radius_0_ - radius_prior);
    const float d_radius_1 =
        lock_gain * radius_noise_gain * radius_noise_scale * radius_1_noise - radius_pull * (radius_1_ - radius_prior);

    const float d_center_z = center_z_position_noise_scale * (lock_gain * center_z_position_noise + center_z_extra_noise);

    const float d_z_offset = lock_gain * (radius_mode == 4 ? spin_fraction : 1.0f) * radius_noise_scale * z_offset_noise;

    const float d_orientation_velocity = velocity_noise_scale * orientation_velocity_noise_1;
    const float d_orientation = time_offset_seconds * orientation_velocity_ +
                                one_half * time_offset_seconds * d_orientation_velocity +
                                position_noise_scale * orientation_velocity_noise_0;

    const Eigen::Vector2f d_center_velocity = velocity_noise_scale * center_xy_velocity_noise_1;
    const Eigen::Vector3f d_center = time_offset_seconds * (Eigen::Vector3f{} << center_velocity_, slope_.dot(center_velocity_)).finished() +
                                     one_half * time_offset_seconds * helper::rpad_zero(d_center_velocity) +
                                     position_noise_scale * helper::rpad_zero(center_xy_velocity_noise_0) +
                                     helper::lpad_zero(d_center_z);

    radius_0_ = helper::to_radius(radius_0_ + d_radius_0);
    radius_1_ = helper::to_radius(radius_1_ + d_radius_1);

    orientation_ = helper::to_orientation(orientation_ + d_orientation);
    orientation_velocity_ = orientation_velocity_ + d_orientation_velocity;

    center_ = center_ + d_center;

    z_offset_ = z_offset_ + d_z_offset;
    if (relabel_at_zero && z_offset_ < 0.0f) {
      // Pair 1 has become the higher one: call it pair 0. Same plates, so nothing jumps.
      z_offset_ = -z_offset_;
      orientation_ = helper::to_orientation(orientation_ + static_cast<float>(M_PI_2));
      const float radius_0 = radius_0_;
      radius_0_ = radius_1_;
      radius_1_ = radius_0;
    }
    z_offset_ = helper::to_z_offset(z_offset_);

    center_velocity_ = center_velocity_ + d_center_velocity;
    slope_ = helper::to_slope(slope_ + slope_noise);
  }

  PF_TARGET_ATTRS [[nodiscard]] const float& radius_0() const noexcept { return radius_0_; }
  PF_TARGET_ATTRS [[nodiscard]] const float& radius_1() const noexcept { return radius_1_; }
  PF_TARGET_ATTRS [[nodiscard]] const float& z_offset() const noexcept { return z_offset_; }
  PF_TARGET_ATTRS [[nodiscard]] const float& settled() const noexcept { return settled_; }
  PF_TARGET_ATTRS [[nodiscard]] const float& maneuver() const noexcept { return maneuver_; }
  PF_TARGET_ATTRS void set_maneuver(const float& maneuver) noexcept { maneuver_ = maneuver; }
  PF_TARGET_ATTRS [[nodiscard]] const float& orientation() const noexcept { return orientation_; }
  PF_TARGET_ATTRS [[nodiscard]] const float& orientation_velocity() const noexcept { return orientation_velocity_; }
  PF_TARGET_ATTRS [[nodiscard]] const Eigen::Vector3f& center() const noexcept { return center_; }
  PF_TARGET_ATTRS [[nodiscard]] const Eigen::Vector2f& center_velocity() const noexcept { return center_velocity_; }
  PF_TARGET_ATTRS [[nodiscard]] const Eigen::Vector2f& slope() const noexcept { return slope_; }

  PF_TARGET_ATTRS prediction() noexcept
      : radius_0_{0.0f},
        radius_1_{0.0f},
        z_offset_{0.0f},
        settled_{0.0f},
        maneuver_{0.0f},
        orientation_{0.0f},
        orientation_velocity_{0.0f},
        center_{Eigen::Vector3f::Zero()},
        center_velocity_{Eigen::Vector2f::Zero()},
        slope_{Eigen::Vector2f::Zero()} {}

  PF_TARGET_ATTRS prediction(
      const float& radius_0,
      const float& radius_1,
      const float& orientation,
      const float& orientation_velocity,
      const Eigen::Vector3f& center,
      const Eigen::Vector2f& center_velocity,
      const float& z_offset,
      const float& settled = 0.0f,
      const float& maneuver = 0.0f,
      const Eigen::Vector2f& slope = Eigen::Vector2f::Zero()) noexcept
      : radius_0_{helper::to_radius(radius_0)},
        radius_1_{helper::to_radius(radius_1)},
        z_offset_{helper::to_z_offset(z_offset)},
        settled_{settled},
        maneuver_{maneuver},
        orientation_{helper::to_orientation(orientation)},
        orientation_velocity_{orientation_velocity},
        center_{center},
        center_velocity_{center_velocity},
        slope_{helper::to_slope(slope)} {}
};

}  // namespace fast_plate_orbit
