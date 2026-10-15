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
  // seed > 0 offsets every particle's random sequence, for independent repeats (the filter is
  // otherwise seeded the same way every time).
  int seed = 0;
  // radius_mode 0: radius noise only while |omega| > 2 rad/s; 1: always; 2: always, and each radius
  // relaxes toward radius_prior with time constant radius_reversion_time (s); 3: like 2, but the pull
  // fades out as the robot spins faster (none above 3 rad/s); 4: radius and height-offset noise in
  // proportion to the spin (none when still), and the pull only for particles that haven't yet
  // watched the robot turn (settled), so a robot that stops keeps its learned geometry.
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
  // Visibility logit = visibility_logit_coefficient * (cos(plate angle to the camera) -
  // visibility_threshold). With a large coefficient and a threshold near the angle the detector
  // stops seeing plates, the term no longer prefers plates square to the camera; it only rules out
  // plates that couldn't be seen (or that must have been).
  float visibility_threshold = 0.0f;
  // obs_model 1: a plate's measured position is precise across the line of sight
  // (obs_lateral_variance) and less so along it (obs_range_variance), the more so the further the
  // plate is turned from the camera (+ obs_range_angle_variance * sin^2 of that angle, from the
  // particle's own plate). Height keeps the observation's variance. 0: the observation's diagonal.
  int obs_model = 0;
  // assoc_best: a lone plate is matched to whichever of the two plates nearest the camera fits it
  // (with that one visible, the others not), not always to the nearest; around a plate switch the
  // plate in view is often the other one.
  bool assoc_best = false;
  float obs_lateral_variance = 0.0001f;
  float obs_range_variance = 0.001f;
  float obs_range_angle_variance = 0.006f;
  // lock_time > 0: the radii, height offset and center height settle. Their process noise is
  // scaled by lock_floor + (1 - lock_floor) * exp(-settled / lock_time), where settled is the
  // time the particle has watched the robot spin (faster than 2 rad/s, when they show).
  float lock_time = 0.0f;
  float lock_floor = 0.1f;
  // settle_by_angle: settled counts radians the robot has turned (above 0.5 rad/s), not seconds
  // spun above 2 rad/s, and lock_time is then in radians. Slow spinners settle too, in proportion
  // to what they show.
  bool settle_by_angle = false;
  // What an earlier filter learned about this robot: radius_prior_high / _low > 0 start the high
  // and low pairs there (radius_prior_variance_memory), and the particles start settled_prior
  // seconds settled. z_offset_prior / z_offset_prior_variance carry its height offset.
  float radius_prior_high = 0.0f;
  float radius_prior_low = 0.0f;
  float radius_prior_variance_memory = 0.0001f;
  float settled_prior = 0.0f;
  // center_z_prior_variance > 0: the robot's center was at height center_z_prior. A lone first
  // plate then says which pair it belongs to (its height minus or plus z_offset_prior, whichever is
  // nearer that center), instead of half the particles guessing each way.
  float center_z_prior = 0.0f;
  float center_z_prior_variance = 0.0f;
  // center_z_prior_gate > 0: ignore that remembered height when the plate is further than this
  // from both places it could put the center (the robot has since gone up or down a level).
  float center_z_prior_gate = 0.0f;
  // maneuver_velocity_variance > 0: each particle is either calm (center velocity noise
  // center_xy_velocity_process_diagonal_covariance) or maneuvering (this variance, per axis), and
  // switches at maneuver_switch_rate per second. A robot spinning in place is followed by calm
  // particles that don't chase measurement noise; a dodging one by maneuvering ones.
  float maneuver_velocity_variance = 0.0f;
  float maneuver_switch_rate = 2.0f;
  // spin_calm_variance > 0 (with the maneuver modes on): calm particles' spin rate changes with this
  // process variance, maneuvering ones with orientation_velocity_process_variance. A robot holding
  // its spin (or still) then isn't given spin-rate noise to chase, which the aim's lead turns into
  // jitter; one changing its spin is followed by the maneuvering particles.
  float spin_calm_variance = 0.0f;
  // yaw_variance > 0: plates measured with a facing direction (PnP's) are also scored on it,
  // against the particle's plate (rad^2, + yaw_variance_angle * sin^2 of the plate's angle to the
  // camera; PnP's yaw gets worse as the plate turns away). yaw_outlier_weight keeps the rare wild
  // PnP yaw from overruling everything. yaw_init: a filter started from one plate takes its
  // orientation and center from that plate's yaw rather than assuming it faces the camera.
  float yaw_variance = 0.0f;
  float yaw_variance_angle = 0.0f;
  float yaw_outlier_weight = 0.05f;
  bool yaw_init = false;
  // yaw_slow_factor > 1: a particle spinning slower than yaw_slow_spin (rad/s) trusts each plate's
  // yaw less, up to this factor on the variance when still (linearly in between): a still robot's
  // orientation then comes from many frames' yaw at once rather than following each frame's noise.
  float yaw_slow_factor = 1.0f;
  float yaw_slow_spin = 1.0f;
  // position_outlier_weight > 0: a plate's position is scored log(w + (1 - w) * Gaussian), so a
  // wild PnP position (far plates' range has a heavy tail) can't drag the whole filter.
  float position_outlier_weight = 0.0f;
  // position_outlier_split: that floor applies to the horizontal position and the height
  // separately, so a robot whose height the filter has wrong (it went up a ramp) still pins its
  // horizontal position.
  bool position_outlier_split = false;
  // center_z_maneuver_variance > 0 (with the maneuver modes on): maneuvering particles' center
  // height also moves with this process variance (m^2/s), not damped by the lock: a robot dropping
  // off a ledge, or one whose height the slope doesn't explain.
  float center_z_maneuver_variance = 0.0f;
  // slope_process_variance > 0: each particle also carries the gradient of the ground under the
  // robot (prediction::slope), which tilts its plate ring and lifts or lowers its center as it
  // drives. The slope's noise is this variance per meter driven, plus slope_time_variance per second
  // (a parked robot's slope barely moves; one driving onto a ramp finds it within ~0.3 m). It starts
  // at 0 with slope_prior_variance.
  float slope_process_variance = 0.0f;
  float slope_time_variance = 0.0f;
  float slope_prior_variance = 0.0f;
  // With the maneuver modes on, calm particles' slope noise is this fraction of it (a robot driving
  // across level ground keeps its slope at 0; one driving onto a ramp is followed by maneuvering ones).
  float slope_calm_fraction = 1.0f;
};

}  // namespace fast_plate_orbit
