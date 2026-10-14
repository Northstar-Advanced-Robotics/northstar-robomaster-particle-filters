#pragma once

#include <fast_plate_orbit/observation.h>
#include <fast_plate_orbit/observed_plate.h>
#include <fast_plate_orbit/observed_plate_orbit.h>
#include <fast_plate_orbit/observed_plate_orbit_builder.h>
#include <fast_plate_orbit/particle_filter_configuration_parameters.h>
#include <fast_plate_orbit/predicted_plate.h>
#include <fast_plate_orbit/prediction.h>
#include <pf/config/target_config.h>
#include <pf/filter/particle_reduction_state.h>
#include <pf/util/device_array.h>
#include <thrust/random.h>
#include <util/random_variable_sampler.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace fast_plate_orbit {

namespace helper {

PF_TARGET_ONLY_ATTRS [[nodiscard]] inline float log_sigmoid(const float& x) noexcept { return -logf(1.0f + expf(-x)); }

}  // namespace helper

struct most_likely_particle_reduction_impl {
  using state_type = pf::filter::particle_reduction_state<prediction>;
  static constexpr float pi = M_PI;

  bool swap_labelings = false;

  struct labeling {
    float orientation;
    float radius_0;
    float radius_1;
    float z_offset;  // signed here: negative means pair 1 is the high one
    bool relabeled;
  };

  PF_TARGET_ONLY_ATTRS [[nodiscard]] inline state_type operator()(const state_type& a, const state_type& b) const noexcept {
    // this does not necessarily provide the correct mean in the MLE sense,
    // though it gives a reasonable approximation for most inputs.

    const prediction a_particle = a.most_likely_particle();
    const prediction b_particle = b.most_likely_particle();

    // Rotating by pi maps each plate onto its same-height opposite: the same labeling. Rotating by
    // +-pi/2 relabels the pairs (old pair 1 becomes pair 0), so radii swap and z_offset changes sign.
    const float a_r0 = a_particle.radius_0();
    const float a_r1 = a_particle.radius_1();
    const float a_dz = a_particle.z_offset();
    const float a_o = a_particle.orientation();
    const pf::util::device_array<labeling, 5> candidates = {
        labeling{a_o, a_r0, a_r1, a_dz, false},
        labeling{a_o + pi, a_r0, a_r1, a_dz, false},
        labeling{a_o - pi, a_r0, a_r1, a_dz, false},
        labeling{a_o + 0.5f * pi, a_r1, a_r0, -a_dz, true},
        labeling{a_o - 0.5f * pi, a_r1, a_r0, -a_dz, true},
    };

    const auto target = b_particle.orientation();
    const float relabel_penalty = swap_labelings ? 0.0f : 1.0e9f;
    const labeling a_l = *candidates.minimum_by([target, relabel_penalty](const labeling& l) {
      return abs(target - l.orientation) + (l.relabeled ? relabel_penalty : 0.0f);
    });

    const float alpha = static_cast<float>(b.count()) / static_cast<float>(a.count() + b.count());
    const float c_alpha = 1.0f - alpha;

    float radius_0 = c_alpha * a_l.radius_0 + alpha * b_particle.radius_0();
    float radius_1 = c_alpha * a_l.radius_1 + alpha * b_particle.radius_1();
    float orientation = c_alpha * a_l.orientation + alpha * b_particle.orientation();
    float z_offset = c_alpha * a_l.z_offset + alpha * b_particle.z_offset();
    if (z_offset < 0.0f) {  // back to "pair 0 is the high one"
      orientation += 0.5f * pi;
      const float r = radius_0;
      radius_0 = radius_1;
      radius_1 = r;
      z_offset = -z_offset;
    }

    const float orientation_velocity = c_alpha * a_particle.orientation_velocity() + alpha * b_particle.orientation_velocity();

    const Eigen::Vector3f center = c_alpha * a_particle.center() + alpha * b_particle.center();
    const Eigen::Vector2f center_velocity = c_alpha * a_particle.center_velocity() + alpha * b_particle.center_velocity();

    const float settled = c_alpha * a_particle.settled() + alpha * b_particle.settled();
    const float maneuver = c_alpha * a_particle.maneuver() + alpha * b_particle.maneuver();

    const auto state = prediction(
        radius_0, radius_1, orientation, orientation_velocity, center, center_velocity, z_offset, settled, maneuver);
    return state_type{state, a.count() + b.count()};
  }
};

class particle_filter_configuration {
 private:
  particle_filter_configuration_parameters params_;

 public:
  using observation_type = observation;
  using prediction_type = prediction;
  using sampler_type = util::default_rv_sampler;

  [[nodiscard]] most_likely_particle_reduction_impl most_likely_particle_reduction() const noexcept {
    return most_likely_particle_reduction_impl{params_.swap_reduction};
  }

  PF_TARGET_ONLY_ATTRS [[nodiscard]] float conditional_log_likelihood(
      const util::default_rv_sampler& sampler,
      const observation& state,
      const prediction& given) const noexcept {
    const auto predicted_plates = given.predicted_plates().selection_sort_by(
        [state](const predicted_plate& a) { return (state.observer_position() - a.position()).squaredNorm(); });

    const auto [pred_0, pred_1, pred_2, pred_3] = predicted_plates;
    const Eigen::Vector3f visibility_mask{1.0f, 1.0f, params_.visibility_xy ? 0.0f : 1.0f};
    const Eigen::Vector3f observer_delta = (state.observer_position() - given.center()).cwiseProduct(visibility_mask);

    const auto [logit_visibility_0, logit_visibility_1, logit_visibility_2, logit_visibility_3] =
        predicted_plates.transformed([&, this](const predicted_plate& plate) {
          const Eigen::Vector3f plate_delta = (plate.position() - given.center()).cwiseProduct(visibility_mask);
          const float similarity = observer_delta.dot(plate_delta) / (observer_delta.norm() * plate_delta.norm());
          return params_.visibility_logit_coefficient * (similarity - params_.visibility_threshold);
        });

    auto log_p_of = [&, this](const observed_plate& x, const predicted_plate& y) {
      const auto error = (x.position() - y.position()).eval();
      if (params_.obs_model == 0) {
        return sampler.unnormalized_normal_log_density(x.position_diagonal_covariance(), error);
      }
      // Along and across the camera ray (horizontally); along it, the less reliable the further
      // this particle's plate is turned away from the camera.
      const Eigen::Vector2f ray = (x.position() - state.observer_position()).template head<2>().normalized();
      const Eigen::Vector2f facing = (y.position() - given.center()).template head<2>().normalized();
      const float cos_angle = -facing.dot(ray);
      const float sin2_angle = pf::util::max(0.0f, 1.0f - cos_angle * cos_angle);
      const float range_variance = params_.obs_range_variance + params_.obs_range_angle_variance * sin2_angle;
      const float along = error[0] * ray[0] + error[1] * ray[1];
      const float across = -error[0] * ray[1] + error[1] * ray[0];
      return -0.5f * (along * along / range_variance + across * across / params_.obs_lateral_variance +
                      error[2] * error[2] / x.position_diagonal_covariance()[2]) -
             0.5f * logf(range_variance);
    };

    if (!state.plate_two().has_value()) {
      const float pr_visibility = helper::log_sigmoid(logit_visibility_0) + helper::log_sigmoid(-logit_visibility_1) +
                                  helper::log_sigmoid(-logit_visibility_2) + helper::log_sigmoid(-logit_visibility_3);

      if (!params_.assoc_best) return pr_visibility + log_p_of(state.plate_one(), pred_0);
      const float pr_visibility_1 = helper::log_sigmoid(-logit_visibility_0) + helper::log_sigmoid(logit_visibility_1) +
                                    helper::log_sigmoid(-logit_visibility_2) + helper::log_sigmoid(-logit_visibility_3);
      return pf::util::max(pr_visibility + log_p_of(state.plate_one(), pred_0),
                           pr_visibility_1 + log_p_of(state.plate_one(), pred_1));
    }

    const float log_pr_visibility = helper::log_sigmoid(logit_visibility_0) + helper::log_sigmoid(logit_visibility_1) +
                                    helper::log_sigmoid(-logit_visibility_2) + helper::log_sigmoid(-logit_visibility_3);

    const float assignment_one = log_p_of(state.plate_one(), pred_0) + log_p_of(*state.plate_two(), pred_1);
    const float assignment_two = log_p_of(state.plate_one(), pred_1) + log_p_of(*state.plate_two(), pred_0);
    return log_pr_visibility + std::max(assignment_one, assignment_two);
  }

  PF_TARGET_ONLY_ATTRS [[nodiscard]] prediction sample_from(util::default_rv_sampler& sampler, const observation& state)
      const noexcept {
    for (int i = 0; i < params_.seed; ++i) (void)sampler.normal_sample(1.0f);
    // Remembered geometry (radius_prior_high/_low > 0): each pair starts at its own radius.
    const bool remembered = params_.radius_prior_high > 0.0f && params_.radius_prior_low > 0.0f;
    const observed_plate_orbit_builder builder(params_.radius_prior, state.observer_position());
    observed_plate_orbit orbit;
    float z_offset;
    if (state.plate_two().has_value()) {
      orbit = builder.from_two_plates(state.plate_one(), *state.plate_two());
      z_offset = helper::to_z_offset(orbit.z_offset + sampler.normal_sample(params_.z_offset_prior_variance * 0.1f));
      if (remembered) {
        z_offset = helper::to_z_offset(params_.z_offset_prior + sampler.normal_sample(params_.z_offset_prior_variance));
      }
    } else {
      z_offset = helper::to_z_offset(params_.z_offset_prior + sampler.normal_sample(params_.z_offset_prior_variance));
      bool seen_plate_is_high = sampler.normal_sample(1.0f) > 0.0f;  // per particle: half each way
      if (params_.center_z_prior_variance > 0.0f) {
        // Remembered center height: P(high) from how well each choice puts the center there.
        const float plate_z = state.plate_one().position()[2];
        const float miss_high = plate_z - z_offset - params_.center_z_prior;
        const float miss_low = plate_z + z_offset - params_.center_z_prior;
        const float logit = 0.5f * (miss_low * miss_low - miss_high * miss_high) / params_.center_z_prior_variance;
        seen_plate_is_high = sampler.normal_sample(1.0f) < logit * 0.6266f;  // probit(x) ~ logistic(1.6 x)
      }
      const float seen_radius =
          !remembered ? params_.radius_prior : (seen_plate_is_high ? params_.radius_prior_high : params_.radius_prior_low);
      orbit = observed_plate_orbit_builder(seen_radius, state.observer_position())
                  .from_one_plate(state.plate_one(), z_offset, seen_plate_is_high);
    }

    const float radius_variance = remembered                       ? params_.radius_prior_variance_memory :
                                  state.plate_two().has_value()    ? params_.radius_prior_variance_two_plates :
                                                                     params_.radius_prior_variance_one_plate;

    const float radius_0 = (remembered ? params_.radius_prior_high : orbit.radius) + sampler.normal_sample(radius_variance);
    const float radius_1 = (remembered ? params_.radius_prior_low : orbit.radius) + sampler.normal_sample(radius_variance);

    const float orientation = orbit.orientation;
    const float orientation_velocity = sampler.normal_sample(params_.orientation_velocity_prior_variance);

    const Eigen::Vector3f center = orbit.center + sampler.normal_sample(state.plate_one().position_diagonal_covariance());
    const Eigen::Vector2f center_velocity = sampler.normal_sample(params_.center_xy_velocity_prior_diagonal_covariance);

    const float maneuver = params_.maneuver_velocity_variance > 0.0f && sampler.normal_sample(1.0f) > 0.0f ? 1.0f : 0.0f;
    return prediction(
        radius_0,
        radius_1,
        orientation,
        orientation_velocity,
        center,
        center_velocity,
        z_offset,
        params_.settled_prior,
        maneuver);
  }

  PF_TARGET_ONLY_ATTRS void apply_process(const float& time_offset_seconds, util::default_rv_sampler& sampler, prediction& state)
      const noexcept {
    const float radius_0_noise = sampler.normal_sample(params_.radius_process_variance);
    const float radius_1_noise = sampler.normal_sample(params_.radius_process_variance);
    const float orientation_velocity_noise_0 = sampler.normal_sample(params_.orientation_velocity_process_variance);
    const float orientation_velocity_noise_1 = sampler.normal_sample(params_.orientation_velocity_process_variance);

    const float center_z_position_noise = sampler.normal_sample(params_.center_z_position_process_variance);

    const float z_offset_noise = sampler.normal_sample(params_.z_offset_process_variance);

    Eigen::Vector2f center_xy_velocity_noise_0 =
        sampler.normal_sample(params_.center_xy_velocity_process_diagonal_covariance);

    Eigen::Vector2f center_xy_velocity_noise_1 =
        sampler.normal_sample(params_.center_xy_velocity_process_diagonal_covariance);

    if (params_.maneuver_velocity_variance > 0.0f) {
      // Calm or maneuvering, switching now and then (uniform from a normal through its CDF).
      const float u = 0.5f * erfcf(-sampler.normal_sample(1.0f) * static_cast<float>(M_SQRT1_2));
      if (u < params_.maneuver_switch_rate * time_offset_seconds) state.set_maneuver(1.0f - state.maneuver());
      if (state.maneuver() > 0.5f) {
        const float gain =
            sqrtf(params_.maneuver_velocity_variance / params_.center_xy_velocity_process_diagonal_covariance[0]);
        center_xy_velocity_noise_0 *= gain;
        center_xy_velocity_noise_1 *= gain;
      }
    }

    state.update_state(
        time_offset_seconds,
        radius_0_noise,
        radius_1_noise,
        orientation_velocity_noise_0,
        orientation_velocity_noise_1,
        center_z_position_noise,
        z_offset_noise,
        center_xy_velocity_noise_0,
        center_xy_velocity_noise_1,
        params_.radius_mode,
        params_.radius_prior,
        params_.radius_reversion_time,
        params_.relabel_at_zero,
        params_.lock_time,
        params_.lock_floor,
        params_.settle_by_angle);
  }

  particle_filter_configuration(const particle_filter_configuration_parameters& params) noexcept : params_{params} {}
};

}  // namespace fast_plate_orbit
