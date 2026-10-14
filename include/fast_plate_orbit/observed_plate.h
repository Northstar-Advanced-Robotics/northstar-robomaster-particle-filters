#pragma once

#include <pf/config/target_config.h>
#include <thrust/execution_policy.h>

#include <Eigen/Dense>

namespace fast_plate_orbit {

class observed_plate {
 private:
  Eigen::Vector3f position_;
  Eigen::Vector3f position_diagonal_covariance_;
  float yaw_;  // direction the plate faces (its outward normal), radians in the xy plane
  bool has_yaw_;
  float yaw_variance_;  // > 0: how far to trust this plate's yaw (rad^2), else the filter's yaw_variance

 public:
  PF_TARGET_ATTRS [[nodiscard]] const Eigen::Vector3f& position() const noexcept { return position_; }

  PF_TARGET_ATTRS [[nodiscard]] const Eigen::Vector3f& position_diagonal_covariance() const noexcept {
    return position_diagonal_covariance_;
  }

  PF_TARGET_ATTRS [[nodiscard]] const float& yaw() const noexcept { return yaw_; }
  PF_TARGET_ATTRS [[nodiscard]] const bool& has_yaw() const noexcept { return has_yaw_; }
  PF_TARGET_ATTRS [[nodiscard]] const float& yaw_variance() const noexcept { return yaw_variance_; }

  PF_TARGET_ATTRS
  observed_plate(const Eigen::Vector3f& position, const Eigen::Vector3f& position_diagonal_covariance) noexcept
      : position_{position}, position_diagonal_covariance_{position_diagonal_covariance}, yaw_{0.0f}, has_yaw_{false}, yaw_variance_{0.0f} {}

  PF_TARGET_ATTRS
  observed_plate(
      const Eigen::Vector3f& position,
      const Eigen::Vector3f& position_diagonal_covariance,
      const float& yaw,
      const float& yaw_variance = 0.0f) noexcept
      : position_{position},
        position_diagonal_covariance_{position_diagonal_covariance},
        yaw_{yaw},
        has_yaw_{true},
        yaw_variance_{yaw_variance} {}
};

}  // namespace fast_plate_orbit
