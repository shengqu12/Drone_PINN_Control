"""
LQR with Analytical Wind Feedforward (LQR+FF)

This is the ANALYTICAL baseline that PINN(Free) is trained to imitate.
The teacher labels used in train_pinn_free.py are exactly what this
controller computes. Comparing PINN vs LQR+FF shows how much the
neural network approximation adds (or loses) beyond the analytical formula.

Control law:
    phi_eq   = atan2(+Fy/m, g)      # lean right to cancel +y wind
    theta_eq = atan2(-Fx/m, g)      # lean back  to cancel +x wind
    T_eq     = m * sqrt(ax² + ay² + g²)
    x_ref_wind = [p_r, phi_eq, theta_eq, 0, v_r, 0]
    u = [T_eq, 0, 0, 0] - K * (x - x_ref_wind)
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import MASS, G, I_XX, I_YY, I_ZZ
from CONTROLLERS.base import BaseController
from CONTROLLERS.lqr import compute_lqr_gain, get_hover_linearization

# maximum tilt allowed (safety clip, same as PINN(Free) training)
MAX_TILT = 0.8   # rad ≈ 45.8°


class LQRFFController(BaseController):
    """
    Feedforward LQR: the closed-form version of what PINN(Free) learns.

    Differences from plain LQR:
      - Modifies the attitude reference to include the wind-equilibrium tilt
      - Adjusts the equilibrium thrust to account for the tilt
      - Responds IMMEDIATELY to wind (no error accumulation needed)

    Differences from PINN(Free):
      - Purely analytical, no neural network approximation
      - Based on linearized hover dynamics (breaks down at large tilt angles)
      - Does not capture interactions between dynamic manoeuvre states
        and wind compensation (e.g., fast turns + strong crosswind)
    """

    def __init__(self):
        A, B = get_hover_linearization()

        Q = np.diag([
            10.0, 10.0, 10.0,    # x, y, z
            5.0,  5.0,  5.0,     # phi, theta, psi
            4.0,  4.0,  4.0,     # vx, vy, vz
            1.0,  1.0,  1.0      # p, q, r
        ])
        R = np.diag([0.1, 1.0, 1.0, 2.0])

        self.K    = compute_lqr_gain(A, B, Q, R)
        self.u_eq = np.array([MASS * G, 0.0, 0.0, 0.0])

    @property
    def name(self):
        return "LQR+FF"

    def compute_control(self, state, ref, wind_force=None):
        if wind_force is None:
            wind_force = np.zeros(3)

        Fx, Fy = wind_force[0], wind_force[1]
        ax, ay = Fx / MASS, Fy / MASS

        # Wind-equilibrium attitude
        phi_eq   = np.arctan2(+ay, G)
        theta_eq = np.arctan2(-ax, G)

        # Clip to safe tilt range
        phi_eq   = np.clip(phi_eq,   -MAX_TILT, MAX_TILT)
        theta_eq = np.clip(theta_eq, -MAX_TILT, MAX_TILT)

        # Equilibrium thrust (must overcome gravity + horizontal wind)
        T_eq = MASS * np.sqrt(ax**2 + ay**2 + G**2)
        T_eq = np.clip(T_eq, 0.0, 4 * MASS * G)

        # Build wind-adjusted reference
        state_ref = np.zeros(12)
        state_ref[0:3] = ref[0:3]    # position reference
        state_ref[3]   = phi_eq      # wind-equilibrium roll
        state_ref[4]   = theta_eq    # wind-equilibrium pitch
        state_ref[6:9] = ref[3:6]    # velocity reference

        err = state - state_ref
        err[3:6] = (err[3:6] + np.pi) % (2 * np.pi) - np.pi  # wrap angles

        u = np.array([T_eq, 0.0, 0.0, 0.0]) - self.K @ err

        u[0] = np.clip(u[0], 0.0, 4 * MASS * G)
        u[1:] = np.clip(u[1:], -0.5, 0.5)

        return u
