"""
LQR (Linear Quadratic Regulator) Controller for Quadrotor

Key idea:
    Linearize the nonlinear quadrotor dynamics around the hover equilibrium,
    then solve the algebraic Riccati equation to find the optimal gain matrix K.

    u = -K * (state - state_ref)

    K is computed once at initialization by solving:
        minimize  integral( x'Qx + u'Ru ) dt
    subject to the linearized dynamics.

    Q penalizes state error, R penalizes control effort.
    Tuning LQR = choosing Q and R (much easier than tuning PID gains).

Why LQR is better than PID:
    - Handles all 12 state dimensions simultaneously (no manual cascade design)
    - Mathematically optimal for the linearized system
    - Guaranteed stable for the linear system

Why LQR still has limits:
    - Linearized around hover — degrades when far from hover (large angles)
    - No feedforward wind compensation — reacts only after error appears
    - This is exactly where PINN will outperform it
"""

import numpy as np
from scipy.linalg import solve_continuous_are
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import MASS, G, I_XX, I_YY, I_ZZ, DT
from CONTROLLERS.base import BaseController


# ─── Linearized dynamics matrices ─────────────────────────────────────────────

def get_hover_linearization():
    """
    Linearize quadrotor dynamics around hover equilibrium:
        state_eq  = [0,0,z0, 0,0,0, 0,0,0, 0,0,0]
        control_eq = [mg, 0, 0, 0]

    Returns A (12x12) and B (12x4) matrices such that:
        dx/dt ≈ A*x + B*u   (where x and u are deviations from equilibrium)

    Derivation sketch (hover, phi=theta=psi=0):
        Position:   dp/dt = v                      → A[0:3, 6:9] = I
        Euler:      dη/dt = W*ω ≈ ω at hover       → A[3:6, 9:12] = I
        Velocity:   dv/dt = R*[0,0,T]/m - g        →
                    d(vx)/dT=0, but d(vx)/d(theta) = g (tilt → horizontal accel)
                    d(vy)/d(phi)  = -g
                    d(vz)/dT      = 1/m
        Angular:    dω/dt = I⁻¹ * τ               → B[9:12, 1:4] = diag(1/Ixx, 1/Iyy, 1/Izz)
    """
    A = np.zeros((12, 12))
    B = np.zeros((12, 4))

    # dp/dt = v
    A[0:3, 6:9] = np.eye(3)

    # dη/dt = ω  (valid at hover where W ≈ I)
    A[3:6, 9:12] = np.eye(3)

    # dv/dt: gravity coupling through pitch/roll
    # tilting by theta causes vx acceleration = g * theta
    # tilting by phi   causes vy acceleration = -g * phi
    A[6, 4] =  G    # dvx/d(theta) = g
    A[7, 3] = -G    # dvy/d(phi)   = -g

    # dv/dt: thrust → vz
    B[8, 0] = 1.0 / MASS        # dvz/dT = 1/m

    # dω/dt: torques → angular acceleration
    B[9,  1] = 1.0 / I_XX       # dp/d(tau_x)
    B[10, 2] = 1.0 / I_YY       # dq/d(tau_y)
    B[11, 3] = 1.0 / I_ZZ       # dr/d(tau_z)

    return A, B


# ─── LQR gain computation ──────────────────────────────────────────────────────

def compute_lqr_gain(A, B, Q, R):
    """
    Solve the continuous-time algebraic Riccati equation (CARE):
        A'P + PA - PBR⁻¹B'P + Q = 0

    Then the optimal gain matrix:
        K = R⁻¹ B' P

    scipy's solve_continuous_are does the heavy lifting.
    """
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K


# ─── LQR Controller ───────────────────────────────────────────────────────────

class LQRController(BaseController):
    """
    LQR controller with hover linearization.

    Q matrix: penalizes state deviations
        - Higher Q[i,i] → controller cares more about keeping state[i] small
        - We weight position and velocity errors most heavily

    R matrix: penalizes control effort
        - Higher R[i,i] → controller uses less of control input i
        - We weight torques higher than thrust (torques are more sensitive)
    """

    def __init__(self):
        A, B = get_hover_linearization()

        # ── Q matrix: state cost ──────────────────────────────────────────
        # state order: [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
        q_pos   = 10.0    # position error cost
        q_euler = 5.0     # attitude error cost
        q_vel   = 4.0     # velocity error cost
        q_omega = 1.0     # angular rate error cost

        Q = np.diag([
            q_pos,   q_pos,   q_pos,     # x, y, z
            q_euler, q_euler, q_euler,   # phi, theta, psi
            q_vel,   q_vel,   q_vel,     # vx, vy, vz
            q_omega, q_omega, q_omega    # p, q, r
        ])

        # ── R matrix: control cost ────────────────────────────────────────
        # control order: [T, tau_x, tau_y, tau_z]
        R = np.diag([0.1, 1.0, 1.0, 2.0])
        # thrust is cheap (R small), torques are expensive (R large)
        # yaw torque gets highest penalty — we don't need aggressive yaw

        # solve for optimal K
        self.K = compute_lqr_gain(A, B, Q, R)

        # equilibrium control (hover thrust to cancel gravity)
        self.u_eq = np.array([MASS * G, 0.0, 0.0, 0.0])

        print(f"LQR gain matrix K shape: {self.K.shape}")
        print(f"K norm: {np.linalg.norm(self.K):.3f}")

    @property
    def name(self):
        return "LQR"

    def compute_control(self, state, ref, wind_force=None):
        """
        u = u_eq - K * (state - state_ref)

        state_ref is built from ref (6,) by padding zeros for
        euler angles and angular rates (we don't track those directly).

        wind_force: ignored — LQR has no wind feedforward.
        Error accumulates until position feedback corrects it.
        """
        # build full 12-dim reference state
        # we only have position + velocity reference
        # attitude reference = hover (zeros), angular rate reference = zeros
        state_ref = np.zeros(12)
        state_ref[0:3] = ref[0:3]    # position reference
        state_ref[6:9] = ref[3:6]    # velocity reference

        # state error (deviation from reference)
        err = state - state_ref

        # wrap euler angle errors to [-pi, pi]
        err[3:6] = (err[3:6] + np.pi) % (2 * np.pi) - np.pi

        # LQR control law
        u = self.u_eq - self.K @ err

        # clip to physical limits
        u[0] = np.clip(u[0], 0.0, 4 * MASS * G)
        u[1:] = np.clip(u[1:], -0.5, 0.5)

        return u


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from config import N_STEPS, T_TOTAL
    from SIMULATION.quad_model import QuadrotorModel
    from SIMULATION.trajectory import make_trajectory
    from SIMULATION.wind import make_wind

    print("=== LQR Controller Sanity Check ===\n")

    quad = QuadrotorModel()
    lqr  = LQRController()
    traj = make_trajectory("circle")
    wind = make_wind("none")

    # start at trajectory initial position
    init_state = np.zeros(12)
    init_state[0:3] = traj.get(0)[0]
    quad.reset(init_state)
    wind.reset()

    t_arr   = np.arange(N_STEPS) * DT
    pos_log = np.zeros((N_STEPS, 3))
    ref_log = np.zeros((N_STEPS, 3))
    err_log = np.zeros(N_STEPS)

    for i, t in enumerate(t_arr):
        ref        = traj.get_full(t)
        wind_force = wind.step()
        u          = lqr.compute_control(quad.state, ref, wind_force)
        state, done = quad.step(u, wind_force)

        pos_log[i] = state[0:3]
        ref_log[i] = ref[0:3]
        err_log[i] = np.linalg.norm(state[0:3] - ref[0:3])

        if done:
            print(f"Crashed at t={t:.2f}s")
            break

    rmse = np.sqrt(np.mean(err_log**2))
    print(f"\nTrajectory: circle,  Wind: none")
    print(f"RMSE position error : {rmse:.4f} m")
    print(f"Max  position error : {err_log.max():.4f} m")
    print(f"Final position error: {err_log[-1]:.4f} m")

    # compare with and without wind
    print("\n--- Testing with constant wind (5 m/s) ---")
    quad.reset(init_state)
    wind_on = make_wind("constant", wind_speed=5.0)
    err_wind = np.zeros(N_STEPS)

    for i, t in enumerate(t_arr):
        ref        = traj.get_full(t)
        wind_force = wind_on.step()
        u          = lqr.compute_control(quad.state, ref, wind_force)
        state, done = quad.step(u, wind_force)
        err_wind[i] = np.linalg.norm(state[0:3] - ref[0:3])
        if done:
            print(f"Crashed at t={t:.2f}s")
            break

    rmse_wind = np.sqrt(np.mean(err_wind**2))
    print(f"RMSE position error : {rmse_wind:.4f} m")
    print(f"Max  position error : {err_wind.max():.4f} m")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(ref_log[:, 0], ref_log[:, 1],
                 '--', color='#888780', linewidth=1.5, label='reference')
    axes[0].plot(pos_log[:, 0], pos_log[:, 1],
                 color='#1D9E75', linewidth=1.5, label='LQR (no wind)')
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title("XY trajectory")
    axes[0].legend()
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_arr, err_log,  color='#1D9E75', linewidth=1.2, label='no wind')
    axes[1].plot(t_arr, err_wind, color='#E24B4A', linewidth=1.2, label='wind 5 m/s')
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("position error (m)")
    axes[1].set_title(f"Tracking error comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("./SANITY_CHECK/windcheck", exist_ok=True)
    plt.savefig("./SANITY_CHECK/windcheck/lqr_check.png", dpi=120)
    print("\nPlot saved to ./SANITY_CHECK/windcheck/lqr_check.png")