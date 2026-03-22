"""
Reference Trajectory Generator
Returns the reference state x_ref(t) at any given time t.
x = [pos, euler, vel, omega]  (12-dim)
ref = [pos_ref, vel_ref]      (6-dim)

Three trajectory types:
  lemniscate  - figure-8 shape, tests both axes simultaneously (default)
  helix       - ascending spiral, tests 3D tracking
  circle      - simple circle, good for initial debugging
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import TRAJ_TYPE, TRAJ_A, TRAJ_OMEGA, TRAJ_HEIGHT


# ─── Base class ───────────────────────────────────────────────────────────────

class Trajectory:
    """
    Base class for reference trajectories.
    All trajectories return a 6-dim reference state:
        [x_ref, y_ref, z_ref, vx_ref, vy_ref, vz_ref]

    Velocity is computed analytically (exact derivative),
    not by finite difference — this matters for PINN training accuracy.
    """

    def get(self, t):
        """
        Returns reference state at time t.
        pos : (3,)  [x, y, z]       reference position
        vel : (3,)  [vx, vy, vz]    reference velocity (analytical derivative)
        """
        raise NotImplementedError

    def get_full(self, t):
        """Returns flat (6,) array [x, y, z, vx, vy, vz] for PINN input."""
        pos, vel = self.get(t)
        return np.concatenate([pos, vel])


# ─── Lemniscate (figure-8) ────────────────────────────────────────────────────

class LemniscateTrajectory(Trajectory):
    """
    Lemniscate of Bernoulli — figure-8 in the XY plane at fixed altitude.

    Parametric form:
        x(t) = A * cos(w*t) / (1 + sin²(w*t))
        y(t) = A * sin(w*t) * cos(w*t) / (1 + sin²(w*t))
        z(t) = z0  (constant)

    Why this shape:
        - Tests tracking in both X and Y simultaneously
        - Continuously changing curvature challenges the controller
        - Periodic so we can run multiple laps and average metrics
    """

    def __init__(self, A=TRAJ_A, omega=TRAJ_OMEGA, z0=TRAJ_HEIGHT):
        self.A     = A        # amplitude (m)
        self.omega = omega    # angular frequency (rad/s)
        self.z0    = z0       # fixed altitude (m)

    def get(self, t):
        w  = self.omega
        wt = w * t

        sin_wt  = np.sin(wt)
        cos_wt  = np.cos(wt)
        denom   = 1 + sin_wt ** 2       # denominator term

        # position
        x = self.A * cos_wt / denom
        y = self.A * sin_wt * cos_wt / denom
        z = self.z0

        # velocity — analytical derivatives (via quotient rule)
        d_denom = 2 * sin_wt * cos_wt * w    # d(denom)/dt

        vx = self.A * (-sin_wt * w * denom - cos_wt * d_denom) / denom**2
        vy = self.A * (
            (cos_wt**2 - sin_wt**2) * w * denom - sin_wt * cos_wt * d_denom
        ) / denom**2
        vz = 0.0

        pos = np.array([x,  y,  z])
        vel = np.array([vx, vy, vz])
        return pos, vel


# ─── Helix ────────────────────────────────────────────────────────────────────

class HelixTrajectory(Trajectory):
    """
    Ascending helix — tests full 3D tracking including vertical control.

    x(t) = A * cos(w*t)
    y(t) = A * sin(w*t)
    z(t) = z0 + climb_rate * t
    """

    def __init__(self, A=TRAJ_A, omega=TRAJ_OMEGA,
                 z0=TRAJ_HEIGHT, climb_rate=0.2):
        self.A          = A
        self.omega      = omega
        self.z0         = z0
        self.climb_rate = climb_rate    # m/s

    def get(self, t):
        w = self.omega

        pos = np.array([
            self.A * np.cos(w * t),
            self.A * np.sin(w * t),
            self.z0 + self.climb_rate * t
        ])
        vel = np.array([
            -self.A * w * np.sin(w * t),
             self.A * w * np.cos(w * t),
             self.climb_rate
        ])
        return pos, vel


# ─── Circle ───────────────────────────────────────────────────────────────────

class CircleTrajectory(Trajectory):
    """
    Horizontal circle at fixed altitude.
    Simplest trajectory — good for initial debugging and sanity checks.

    x(t) = A * cos(w*t)
    y(t) = A * sin(w*t)
    z(t) = z0
    """

    def __init__(self, A=TRAJ_A, omega=TRAJ_OMEGA, z0=TRAJ_HEIGHT):
        self.A     = A
        self.omega = omega
        self.z0    = z0

    def get(self, t):
        w = self.omega

        pos = np.array([
            self.A * np.cos(w * t),
            self.A * np.sin(w * t),
            self.z0
        ])
        vel = np.array([
            -self.A * w * np.sin(w * t),
             self.A * w * np.cos(w * t),
             0.0
        ])
        return pos, vel


# ─── Factory function ─────────────────────────────────────────────────────────

def make_trajectory(traj_type=TRAJ_TYPE, **kwargs):
    """
    Factory function -- create a trajectory from a string identifier.

    Usage:
        traj = make_trajectory("lemniscate")
        pos, vel = traj.get(t)
        ref = traj.get_full(t)   # (6,) flat array for PINN input
    """
    traj_type = traj_type.lower()
    if traj_type == "lemniscate":
        return LemniscateTrajectory(**kwargs)
    elif traj_type == "helix":
        return HelixTrajectory(**kwargs)
    elif traj_type == "circle":
        return CircleTrajectory(**kwargs)
    else:
        raise ValueError(f"Unknown trajectory type: '{traj_type}'. "
                         f"Choose from: lemniscate, helix, circle")


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from config import DT, T_TOTAL

    print("=== Trajectory Sanity Check ===\n")

    N = int(T_TOTAL / DT)
    t = np.arange(N) * DT

    trajs = {
        "lemniscate" : make_trajectory("lemniscate"),
        "helix"      : make_trajectory("helix"),
        "circle"     : make_trajectory("circle"),
    }

    fig = plt.figure(figsize=(14, 4))

    for i, (name, traj) in enumerate(trajs.items()):
        positions = np.array([traj.get(ti)[0] for ti in t])
        velocities = np.array([traj.get(ti)[1] for ti in t])

        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                linewidth=1.5, color='#534AB7')
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")

        # check: numerical velocity vs analytical velocity
        if N > 1:
            dt = DT
            num_vel = np.diff(positions, axis=0) / dt
            ana_vel = velocities[:-1]
            max_err = np.abs(num_vel - ana_vel).max()
            print(f"{name:12s}: max velocity error (numerical vs analytical) = {max_err:.5f} m/s")

        # check: trajectory stays within reasonable bounds
        pos_range = positions.max(axis=0) - positions.min(axis=0)
        print(f"{'':12s}  XYZ range = [{pos_range[0]:.2f}, {pos_range[1]:.2f}, {pos_range[2]:.2f}] m\n")

    plt.tight_layout()
    os.makedirs("./SANITY_CHECK/windcheck", exist_ok=True)
    plt.savefig("./SANITY_CHECK/windcheck/trajectory_check.png", dpi=120)
    print("Plot saved to ./SANITY_CHECK/windcheck/trajectory_check.png")
    print("All checks passed!")