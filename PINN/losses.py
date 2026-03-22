"""
PINN Loss Functions

Total loss = L_track + lambda * L_physics + beta * L_smooth + gamma * L_bc

Each term serves a different purpose:
  L_track   : supervised signal  — stay close to reference trajectory
  L_physics : physics constraint — obey Newton-Euler equations
  L_smooth  : regularization     — avoid jerky control inputs
  L_bc      : boundary condition — satisfy initial state
"""

import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (MASS, G, I_XX, I_YY, I_ZZ,
                    LAMBDA_PHYSICS, BETA_SMOOTH, GAMMA_BC)
import torch.nn as nn


# ─── Rotation matrix (torch) ──────────────────────────────────────────────────

def rotation_matrix_torch(phi, theta, psi):
    """
    ZYX Euler → rotation matrix R (body → inertial), batched.

    Args:
        phi, theta, psi : (batch,) tensors

    Returns:
        R : (batch, 3, 3)
    """
    B = phi.shape[0]

    cos_phi   = torch.cos(phi)
    sin_phi   = torch.sin(phi)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_psi   = torch.cos(psi)
    sin_psi   = torch.sin(psi)

    # ZYX convention: R = Rz @ Ry @ Rx
    # Each entry of R as a function of euler angles
    R = torch.zeros(B, 3, 3, dtype=phi.dtype, device=phi.device)

    R[:, 0, 0] =  cos_psi * cos_theta
    R[:, 0, 1] =  cos_psi * sin_theta * sin_phi - sin_psi * cos_phi
    R[:, 0, 2] =  cos_psi * sin_theta * cos_phi + sin_psi * sin_phi

    R[:, 1, 0] =  sin_psi * cos_theta
    R[:, 1, 1] =  sin_psi * sin_theta * sin_phi + cos_psi * cos_phi
    R[:, 1, 2] =  sin_psi * sin_theta * cos_phi - cos_psi * sin_phi

    R[:, 2, 0] = -sin_theta
    R[:, 2, 1] =  cos_theta * sin_phi
    R[:, 2, 2] =  cos_theta * cos_phi

    return R


# ─── Physics residual ─────────────────────────────────────────────────────────

def physics_residual(state, u_pred, wind_force, state_dot_pred):
    """
    Compute Newton-Euler equation residuals.

    The idea:
        Given current state and control u_pred,
        the physics equations tell us what state_dot SHOULD be.
        We compare that against state_dot_pred (from autograd or finite diff).
        The difference is the physics residual — it should be zero.

    Args:
        state          : (batch, 12) [x,y,z, phi,theta,psi, vx,vy,vz, p,q,r]
        u_pred         : (batch, 4)  [T, tau_x, tau_y, tau_z]
        wind_force     : (batch, 3)  [Fx, Fy, Fz]
        state_dot_pred : (batch, 12) predicted state derivative (from network)

    Returns:
        residual : (batch, 12) physics equation residuals
    """
    B = state.shape[0]

    # unpack state
    phi   = state[:, 3]
    theta = state[:, 4]
    psi   = state[:, 5]
    vx    = state[:, 6]
    vy    = state[:, 7]
    vz    = state[:, 8]
    p     = state[:, 9]
    q     = state[:, 10]
    r     = state[:, 11]

    # unpack control
    T     = u_pred[:, 0]
    tau_x = u_pred[:, 1]
    tau_y = u_pred[:, 2]
    tau_z = u_pred[:, 3]

    # ── Translation: what physics says state_dot should be ────────────────

    # dp/dt = v  (exact, no approximation)
    dp_dt = state[:, 6:9]   # [vx, vy, vz]

    # dv/dt = R * [0,0,T] / m - [0,0,g] + F_wind / m
    R = rotation_matrix_torch(phi, theta, psi)

    thrust_body    = torch.zeros(B, 3, dtype=state.dtype, device=state.device)
    thrust_body[:, 2] = T                           # thrust along body z-axis
    thrust_inertial = torch.bmm(R, thrust_body.unsqueeze(-1)).squeeze(-1)

    gravity = torch.zeros(B, 3, dtype=state.dtype, device=state.device)
    gravity[:, 2] = -G * MASS

    dv_dt = (thrust_inertial + gravity + wind_force) / MASS

    # ── Rotation: what physics says state_dot should be ───────────────────

    # deta/dt = W(phi, theta) * omega
    # W matrix (Euler angle rate transformation)
    cos_phi   = torch.cos(phi)
    sin_phi   = torch.sin(phi)
    cos_theta = torch.cos(theta)
    tan_theta = torch.tan(theta)

    # W * omega, computed row by row
    phi_dot   = p + sin_phi * tan_theta * q + cos_phi * tan_theta * r
    theta_dot =     cos_phi               * q - sin_phi             * r
    psi_dot   =     sin_phi / cos_theta   * q + cos_phi / cos_theta * r

    deta_dt = torch.stack([phi_dot, theta_dot, psi_dot], dim=1)

    # domega/dt = I^-1 * (tau - omega x (I*omega))
    omega   = state[:, 9:12]                  # [p, q, r]
    I       = torch.tensor([I_XX, I_YY, I_ZZ],
                            dtype=state.dtype, device=state.device)
    I_omega = I * omega                        # element-wise (diagonal I)

    # cross product omega x (I*omega)
    gyro = torch.cross(omega, I_omega, dim=1)

    tau     = u_pred[:, 1:4]                  # [tau_x, tau_y, tau_z]
    domega_dt = (tau - gyro) / I              # element-wise

    # ── Assemble physics-predicted state_dot ──────────────────────────────
    state_dot_physics = torch.cat([dp_dt, deta_dt, dv_dt, domega_dt], dim=1)

    # residual = physics prediction - network prediction
    # (should be zero if network obeys physics)
    residual = state_dot_physics - state_dot_pred

    return residual


# ─── Individual loss terms ────────────────────────────────────────────────────

def l_track(state, ref):
    """
    Tracking loss: penalize deviation from reference trajectory.

    L_track = ||pos - pos_ref||^2 + 0.1 * ||vel - vel_ref||^2

    Position error weighted higher than velocity error because
    our final metric (RMSE) is position-based.

    Args:
        state : (batch, 12)
        ref   : (batch, 6)  [x_ref, y_ref, z_ref, vx_ref, vy_ref, vz_ref]
    """
    pos_err = state[:, 0:3] - ref[:, 0:3]    # position error
    vel_err = state[:, 6:9] - ref[:, 3:6]    # velocity error

    return (pos_err ** 2).mean() + 0.1 * (vel_err ** 2).mean()


def l_physics(state, u_pred, wind_force, state_dot_pred):
    """
    Physics loss: penalize violation of Newton-Euler equations.

    This is the core of PINN — it enforces that the network's
    behavior is physically consistent with the quadrotor dynamics.

    Wind force F_wind appears explicitly here, which is why PINN
    can do feedforward compensation (it knows F_wind at training time).

    Args:
        state          : (batch, 12)
        u_pred         : (batch, 4)
        wind_force     : (batch, 3)
        state_dot_pred : (batch, 12)  predicted state derivative
    """
    residual = physics_residual(state, u_pred, wind_force, state_dot_pred)
    return (residual ** 2).mean()


def l_smooth(u_pred, u_prev):
    """
    Smoothness loss: penalize large changes in control input between steps.

    Prevents chattering — rapid oscillation of control inputs that
    would damage motors in real hardware and cause instability.

    Args:
        u_pred : (batch, 4)  current control
        u_prev : (batch, 4)  previous control
    """
    delta_u = u_pred - u_prev
    return (delta_u ** 2).mean()


def l_bc(state, state0):
    """
    Boundary condition loss: enforce initial state constraint.

    Ensures the trajectory starts from the correct initial condition.
    Critical for quadrotor because a wrong initial state
    (e.g., starting tilted or with wrong velocity) leads to crashes.

    Args:
        state  : (batch, 12)  predicted initial states
        state0 : (batch, 12)  true initial states
    """
    return ((state - state0) ** 2).mean()


# ─── Total loss ───────────────────────────────────────────────────────────────

class PINNLoss:
    """
    Combines all loss terms with configurable weights.

    Usage:
        loss_fn = PINNLoss()
        loss, info = loss_fn(state, u_pred, wind, ref, u_prev, state_dot_pred, state0)
    """

    def __init__(self,
                 lambda_physics = LAMBDA_PHYSICS,
                 beta_smooth    = BETA_SMOOTH,
                 gamma_bc       = GAMMA_BC):
        self.lambda_physics = lambda_physics
        self.beta_smooth    = beta_smooth
        self.gamma_bc       = gamma_bc

    def __call__(self, state, u_pred, wind_force, ref,
                 u_prev, state_dot_pred, state0):
        """
        Compute total loss and return breakdown.

        Returns:
            total : scalar tensor (for backprop)
            info  : dict with individual loss values (for logging)
        """
        lt  = l_track(state, ref)
        lp  = l_physics(state, u_pred, wind_force, state_dot_pred)
        ls  = l_smooth(u_pred, u_prev)
        lbc = l_bc(state, state0)

        total = (lt
                 + self.lambda_physics * lp
                 + self.beta_smooth    * ls
                 + self.gamma_bc       * lbc)

        info = {
            'total'   : total.item(),
            'track'   : lt.item(),
            'physics' : lp.item(),
            'smooth'  : ls.item(),
            'bc'      : lbc.item(),
        }

        return total, info
# ─── Add this to the bottom of PINN/losses.py ────────────────────────────────
# (paste before the "if __name__ == '__main__'" block)

import torch.nn as nn

class AutoWeightedLoss(nn.Module):
    """
    Automatic loss weighting via homoscedastic uncertainty.
    Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"
    CVPR 2018.

    Learns log(sigma_i^2) for each loss term:
        total = sum_i [ exp(-log_var_i) * L_i  +  0.5 * log_var_i ]

    Interpretation:
        log_var large  ->  low weight  (high uncertainty, de-emphasize)
        log_var small  ->  high weight (low uncertainty, emphasize)
    """

    def __init__(self, n_losses=4):
        super().__init__()
        # initialize at 0: sigma=1, effective weight=1
        self.log_vars = nn.Parameter(torch.zeros(n_losses))

    def forward(self, losses):
        """
        losses  : list of n_losses scalar tensors
                  [L_track, L_physics, L_imitation, L_smooth+bc]
        returns : (total scalar tensor, list of float weights)
        """
        total   = 0.0
        weights = []
        for i, loss in enumerate(losses):
            weight = torch.exp(-self.log_vars[i])
            total  = total + weight * loss + 0.5 * self.log_vars[i]
            weights.append(weight.item())
        return total, weights


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== PINN Loss Sanity Check ===\n")

    B = 32    # batch size

    # fake batch data
    state          = torch.randn(B, 12)
    u_pred         = torch.randn(B, 4)
    wind_force     = torch.randn(B, 3)
    ref            = torch.randn(B, 6)
    u_prev         = torch.randn(B, 4)
    state_dot_pred = torch.randn(B, 12)
    state0         = torch.randn(B, 12)

    loss_fn = PINNLoss()
    total, info = loss_fn(state, u_pred, wind_force, ref,
                          u_prev, state_dot_pred, state0)

    print("Loss breakdown:")
    for k, v in info.items():
        print(f"  {k:8s}: {v:.4f}")

    # check gradients flow through
    state.requires_grad_(True)
    u_pred_grad = torch.randn(B, 4, requires_grad=True)
    total2, _ = loss_fn(state, u_pred_grad, wind_force, ref,
                        u_prev, state_dot_pred, state0)
    total2.backward()

    assert u_pred_grad.grad is not None, "No gradient on u_pred!"
    assert state.grad is not None, "No gradient on state!"
    print(f"\nGradient check:")
    print(f"  u_pred grad norm : {u_pred_grad.grad.norm():.4f} ✓")
    print(f"  state  grad norm : {state.grad.norm():.4f} ✓")

    # check: perfect tracking should give zero L_track
    state_perfect = torch.zeros(B, 12)
    ref_perfect   = torch.zeros(B, 6)
    lt_zero = l_track(state_perfect, ref_perfect)
    assert lt_zero.item() < 1e-8, "L_track should be 0 for perfect tracking!"
    print(f"\nL_track = 0 when state matches ref ✓")

    # check: same control as previous step → L_smooth = 0
    u_same = torch.ones(B, 4)
    ls_zero = l_smooth(u_same, u_same)
    assert ls_zero.item() < 1e-8, "L_smooth should be 0 for constant control!"
    print(f"L_smooth = 0 when control is constant ✓")

    print("\nAll checks passed!")