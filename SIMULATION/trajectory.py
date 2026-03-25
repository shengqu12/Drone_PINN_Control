"""
Reference Trajectory Generator
Returns the reference state x_ref(t) at any given time t.
x = [pos, euler, vel, omega]  (12-dim)
ref = [pos_ref, vel_ref]      (6-dim)

Training trajectories (used in PINN dataset generation):
  lemniscate  - figure-8 shape, tests both axes simultaneously (default)
  helix       - ascending spiral, tests 3D tracking
  circle      - simple circle, good for initial debugging

Test-only trajectories (UNSEEN during training — for generalization evaluation):
  tilted_circle     - circle in a 45°-tilted plane (true 3D motion)
  lissajous_3d      - 3D Lissajous figure with 1:2:3 frequency ratios
  rising_lemniscate - figure-8 with sinusoidal altitude variation
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

    _ACC_DT = 1e-4   # step for finite-difference acceleration

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

    def get_acceleration(self, t):
        """
        Reference acceleration (3,) at time t via central finite difference.
        Accurate to O(dt²) — sufficient for training label generation.
        """
        dt = self._ACC_DT
        _, v_plus  = self.get(t + dt)
        _, v_minus = self.get(t - dt)
        return (v_plus - v_minus) / (2.0 * dt)


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


# ─── TiltedCircle (TEST ONLY) ─────────────────────────────────────────────────

class TiltedCircleTrajectory(Trajectory):
    """
    Circle lying in a 45°-tilted plane — generates true 3D motion in all axes.

    The tilt is about the x-axis by angle alpha (default π/4 = 45°):
        x(t) = A * cos(ω*t)
        y(t) = A * sin(ω*t) * cos(α)
        z(t) = z0 + A * sin(ω*t) * sin(α)

    Unlike Helix (monotonically ascending) this oscillates in z, requiring
    the controller to continuously adjust thrust — qualitatively different
    from all three training trajectories.

    NOTE: UNSEEN during training. Use only for generalization evaluation.
    """

    def __init__(self, A=TRAJ_A, omega=TRAJ_OMEGA,
                 z0=TRAJ_HEIGHT, tilt_angle=np.pi / 4):
        self.A          = A
        self.omega      = omega
        self.z0         = z0
        self.tilt_angle = tilt_angle    # radians (default 45°)

    def get(self, t):
        w     = self.omega
        alpha = self.tilt_angle

        wt = w * t
        cos_wt = np.cos(wt)
        sin_wt = np.sin(wt)

        pos = np.array([
            self.A * cos_wt,
            self.A * sin_wt * np.cos(alpha),
            self.z0 + self.A * sin_wt * np.sin(alpha)
        ])
        vel = np.array([
            -self.A * w * sin_wt,
             self.A * w * cos_wt * np.cos(alpha),
             self.A * w * cos_wt * np.sin(alpha)
        ])
        return pos, vel


# ─── Lissajous3D (TEST ONLY) ──────────────────────────────────────────────────

class Lissajous3DTrajectory(Trajectory):
    """
    3D Lissajous figure with frequency ratios 1:2:3 — highly complex shape
    with self-intersections and rapid direction changes in all three axes.

        x(t) = A   * cos(ω*t)
        y(t) = A   * sin(2*ω*t)
        z(t) = z0  + A_z * sin(3*ω*t)

    The 1:2:3 ratio produces a non-repeating-looking pattern over short
    windows, very different from any training trajectory's structure.

    NOTE: UNSEEN during training. Use only for generalization evaluation.
    """

    def __init__(self, A=TRAJ_A, omega=TRAJ_OMEGA,
                 z0=TRAJ_HEIGHT, A_z=None):
        self.A     = A
        self.omega = omega
        self.z0    = z0
        self.A_z   = A_z if A_z is not None else A * 0.5  # half amplitude in z

    def get(self, t):
        w  = self.omega
        wt = w * t

        pos = np.array([
            self.A   * np.cos(wt),
            self.A   * np.sin(2 * wt),
            self.z0  + self.A_z * np.sin(3 * wt)
        ])
        vel = np.array([
            -self.A   * w     * np.sin(wt),
             self.A   * 2 * w * np.cos(2 * wt),
             self.A_z * 3 * w * np.cos(3 * wt)
        ])
        return pos, vel


# ─── RisingLemniscate (TEST ONLY) ─────────────────────────────────────────────

class RisingLemniscateTrajectory(Trajectory):
    """
    Figure-8 (Lemniscate of Bernoulli) with sinusoidal altitude variation.

    The XY part is identical to the training Lemniscate, but the altitude
    oscillates at twice the base frequency:
        x(t) = A * cos(ω*t) / (1 + sin²(ω*t))
        y(t) = A * sin(ω*t)*cos(ω*t) / (1 + sin²(ω*t))
        z(t) = z0 + A_z * sin(2*ω*t)

    Tests whether the controller generalises the learned figure-8 pattern to
    the case where altitude is no longer constant — a minimal 3D extension
    of a shape the PINN has seen (in 2D) during training.

    NOTE: UNSEEN during training (training Lemniscate has z=const).
          Use only for generalization evaluation.
    """

    def __init__(self, A=TRAJ_A, omega=TRAJ_OMEGA,
                 z0=TRAJ_HEIGHT, A_z=None):
        self.A     = A
        self.omega = omega
        self.z0    = z0
        self.A_z   = A_z if A_z is not None else A * 0.4

    def get(self, t):
        w  = self.omega
        wt = w * t

        sin_wt  = np.sin(wt)
        cos_wt  = np.cos(wt)
        denom   = 1.0 + sin_wt ** 2
        d_denom = 2.0 * sin_wt * cos_wt * w    # d(denom)/dt

        # XY: same as standard Lemniscate
        x = self.A * cos_wt / denom
        y = self.A * sin_wt * cos_wt / denom

        vx = self.A * (-sin_wt * w * denom - cos_wt * d_denom) / denom ** 2
        vy = self.A * (
            (cos_wt ** 2 - sin_wt ** 2) * w * denom - sin_wt * cos_wt * d_denom
        ) / denom ** 2

        # Z: sinusoidal at 2ω
        z  = self.z0 + self.A_z * np.sin(2 * wt)
        vz = self.A_z * 2 * w * np.cos(2 * wt)

        return np.array([x, y, z]), np.array([vx, vy, vz])


# ─── Factory function ─────────────────────────────────────────────────────────

# Trajectories used during PINN training (do NOT add test-only types here)
TRAIN_TRAJECTORIES = ["lemniscate", "circle", "helix"]

# Trajectories reserved for generalization testing (never seen during training)
TEST_TRAJECTORIES  = ["tilted_circle", "lissajous_3d", "rising_lemniscate"]


def make_trajectory(traj_type=TRAJ_TYPE, **kwargs):
    """
    Factory function -- create a trajectory from a string identifier.

    Usage:
        traj = make_trajectory("lemniscate")
        pos, vel = traj.get(t)
        ref = traj.get_full(t)   # (6,) flat array for PINN input

    Training trajectories  : lemniscate, helix, circle
    Test-only trajectories : tilted_circle, lissajous_3d, rising_lemniscate
    """
    traj_type = traj_type.lower()
    if traj_type == "lemniscate":
        return LemniscateTrajectory(**kwargs)
    elif traj_type == "helix":
        return HelixTrajectory(**kwargs)
    elif traj_type == "circle":
        return CircleTrajectory(**kwargs)
    elif traj_type == "tilted_circle":
        return TiltedCircleTrajectory(**kwargs)
    elif traj_type == "lissajous_3d":
        return Lissajous3DTrajectory(**kwargs)
    elif traj_type == "rising_lemniscate":
        return RisingLemniscateTrajectory(**kwargs)
    else:
        raise ValueError(
            f"Unknown trajectory type: '{traj_type}'. "
            f"Training: {TRAIN_TRAJECTORIES}  |  "
            f"Test-only: {TEST_TRAJECTORIES}"
        )


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
        # training
        "lemniscate"       : make_trajectory("lemniscate"),
        "helix"            : make_trajectory("helix"),
        "circle"           : make_trajectory("circle"),
        # test-only
        "tilted_circle"    : make_trajectory("tilted_circle"),
        "lissajous_3d"     : make_trajectory("lissajous_3d"),
        "rising_lemniscate": make_trajectory("rising_lemniscate"),
    }

    fig = plt.figure(figsize=(21, 4))

    for i, (name, traj) in enumerate(trajs.items()):
        positions = np.array([traj.get(ti)[0] for ti in t])
        velocities = np.array([traj.get(ti)[1] for ti in t])

        ax = fig.add_subplot(1, 6, i+1, projection='3d')
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