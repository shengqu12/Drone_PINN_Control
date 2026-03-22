# ------- proportional-integral-derivative (PID) controller for quadrotor -------

"""
PID Controller for Quadrotor
A cascade PID structure — standard in real quadrotor firmware (e.g. PX4, Betaflight).

Structure:
    Outer loop (position)  →  generates desired velocity
    Inner loop (velocity)  →  generates desired acceleration → thrust + attitude
    Attitude loop          →  generates torques [tau_x, tau_y, tau_z]

Why cascade PID:
    A single PID from position directly to thrust doesn't work well because
    the drone's response to thrust is highly nonlinear (depends on attitude).
    Splitting into position → velocity → attitude → torque makes each loop
    nearly linear and much easier to tune.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import MASS, G, DT
from CONTROLLERS.base import BaseController
import matplotlib.pyplot as plt


class PIDController(BaseController):
    """
    Cascade PID controller (redesigned).

    Key design decisions vs naive cascade PID:

    1. Position D-term uses velocity error (vel_ref - vel) directly instead of
       finite-differencing pos_err.  Both are mathematically equivalent but the
       direct form avoids one extra integration of noise.

    2. Attitude D-term uses the measured angular velocity omega directly.
       omega IS the time-derivative of the Euler angles (at small angles),
       so using -omega as the D-term is exact, noise-free, and standard in
       real firmware (PX4, ArduCopter).

    3. No attitude integral.  The position loop already has an integrator that
       drives steady-state position error to zero; adding another integrator
       in the attitude loop creates a double-integrator structure that tends
       to oscillate.

    4. max_tilt raised to 35 deg.  At 10 m/s wind, the required lean angle to
       balance the wind force is atan2(10 N, 9.81 N) ≈ 45 deg.  The old 20 deg
       cap prevented the drone from ever counteracting strong wind.
    """

    def __init__(self):
        # ── Position loop: pos_err → desired acceleration ─────────────────
        # Kd acts on velocity error (vel_ref - vel), not on finite-diff of pos_err
        # Tuning guide (lemniscate: max_vel≈1.4 m/s, max_acc≈0.5 m/s²):
        #   ω_n_pos ≈ sqrt(Kp_pos)  →  Kp=5 gives ω_n=2.24 rad/s (period 2.8s < traj period 12.6s)
        #   Kd_pos acts on vel_err: good value ≈ 2*sqrt(Kp_pos) - 2*sqrt(Kd_pos) for critical damping
        #   Ki rule: keep Ki ≤ Kp/30 to avoid windup  (Ki=0.5 was 10× too high → oscillation!)
        self.Kp_pos = np.array([5.0,  5.0,  7.0])    # ω_n ≈ 2.24 rad/s (was too low at 0.5–1.0) 
        self.Ki_pos = np.array([0.20, 0.20, 0.25])   # slow DC correction only (Ki/Kp ≈ 0.02)
        self.Kd_pos = np.array([10.0,  10.0,  10.0])    # velocity damping

        # ── Attitude loop: att_err → torques ──────────────────────────────
        # Kd acts on measured omega directly (no finite-diff, no integral)
        self.Kp_att = np.array([12.0, 12.0, 4.0])
        self.Kd_att = np.array([ 4.0,  4.0, 1.5])

        # internal state — only position integrator needed
        self._pos_integral  = np.zeros(3)
        self._pos_int_limit = 1.5    # m·s  anti-windup clamp

    def reset(self):
        self._pos_integral = np.zeros(3)

    @property
    def name(self):
        return "PID"

    def compute_control(self, state, ref, wind_force=None):
        """
        Cascade PID: position error → desired attitude → torques + thrust.

        Args:
            state      : (12,) [x,y,z, phi,theta,psi, vx,vy,vz, p,q,r]
            ref        : (6,)  [x_ref, y_ref, z_ref, vx_ref, vy_ref, vz_ref]
            wind_force : ignored — PID has no feedforward wind compensation

        Returns:
            u : (4,) [T, tau_x, tau_y, tau_z]
        """
        pos   = state[0:3]
        euler = state[3:6]    # [phi, theta, psi]
        vel   = state[6:9]
        omega = state[9:12]   # [p, q, r] — measured angular velocity

        pos_ref = ref[0:3]
        vel_ref = ref[3:6]

        # ── Step 1: position PID → desired acceleration ──────────────────────
        pos_err = pos_ref - pos
        vel_err = vel_ref - vel    # velocity error used as D-term directly

        self._pos_integral += pos_err * DT
        self._pos_integral  = np.clip(self._pos_integral,
                                      -self._pos_int_limit, self._pos_int_limit)

        a_des = (self.Kp_pos * pos_err
               + self.Ki_pos * self._pos_integral
               + self.Kd_pos * vel_err)

        # ── Step 2: desired acceleration → thrust + attitude targets ─────────
        # Total desired force including gravity compensation
        F_des = MASS * (a_des + np.array([0.0, 0.0, G]))

        T = np.linalg.norm(F_des)
        T = np.clip(T, 0.0, 4 * MASS * G)

        # Extract desired roll/pitch from the force direction.
        # 45 deg: covers atan2(10N, 9.81N)=45.5° needed at 10 m/s wind.
        # 35 deg was too small — drone couldn't lean enough against strong wind.
        max_tilt  = np.deg2rad(45)
        phi_des   = np.clip(-np.arctan2(F_des[1], F_des[2]), -max_tilt, max_tilt)
        theta_des = np.clip( np.arctan2(F_des[0], F_des[2]), -max_tilt, max_tilt)
        psi_des   = euler[2]    # hold current yaw

        att_des = np.array([phi_des, theta_des, psi_des])

        # ── Step 3: attitude PD → torques ─────────────────────────────────────
        att_err    = att_des - euler
        att_err[2] = (att_err[2] + np.pi) % (2 * np.pi) - np.pi  # wrap yaw

        # D-term: use measured angular velocity directly.
        # omega ≈ d(euler)/dt at small angles, so -omega damps attitude oscillations.
        tau = self.Kp_att * att_err - self.Kd_att * omega
        tau = np.clip(tau, -0.5, 0.5)

        return np.array([T, tau[0], tau[1], tau[2]])


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from config import N_STEPS, T_TOTAL
    from SIMULATION.quad_model import QuadrotorModel
    from SIMULATION.trajectory import make_trajectory
    from SIMULATION.wind import make_wind

    print("=== PID Controller Sanity Check ===\n")

    quad = QuadrotorModel()
    pid  = PIDController()
    traj = make_trajectory("circle")     # circle first — easier than lemniscate
    wind = make_wind("none")

    # set initial state close to trajectory start
    init_state = np.zeros(12)
    init_state[0:3] = traj.get(0)[0]    # start at reference position
    init_state[2]   = 2.0               # z = 2m
    quad.reset(init_state)
    pid.reset()
    wind.reset()

    t_arr    = np.arange(N_STEPS) * DT
    pos_log  = np.zeros((N_STEPS, 3))
    ref_log  = np.zeros((N_STEPS, 3))
    err_log  = np.zeros(N_STEPS)

    for i, t in enumerate(t_arr):
        ref        = traj.get_full(t)
        wind_force = wind.step()
        u          = pid.compute_control(quad.state, ref, wind_force)
        state, done = quad.step(u, wind_force)

        pos_log[i] = state[0:3]
        ref_log[i] = ref[0:3]
        err_log[i] = np.linalg.norm(state[0:3] - ref[0:3])

        if done:
            print(f"Crashed at t={t:.2f}s")
            break

    rmse = np.sqrt(np.mean(err_log**2))
    print(f"Trajectory: circle, Wind: none")
    print(f"RMSE position error : {rmse:.4f} m")
    print(f"Max  position error : {err_log.max():.4f} m")
    print(f"Final position error: {err_log[-1]:.4f} m")

    # plot XY trajectory
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(ref_log[:, 0], ref_log[:, 1],
                 '--', color='#888780', linewidth=1.5, label='reference')
    axes[0].plot(pos_log[:, 0], pos_log[:, 1],
                 color='#534AB7', linewidth=1.5, label='PID')
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title("XY trajectory")
    axes[0].legend()
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_arr, err_log, color='#E24B4A', linewidth=1.2)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("position error (m)")
    axes[1].set_title(f"Tracking error  (RMSE = {rmse:.3f} m)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("./SANITY_CHECK/windcheck", exist_ok=True)
    plt.savefig("./SANITY_CHECK/windcheck/pid_check.png", dpi=120)
    print("\nPlot saved to ./SANITY_CHECK/windcheck/pid_check.png")