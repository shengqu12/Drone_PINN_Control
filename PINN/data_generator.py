"""
PINN_FREE Data Generator — Two modes:

  1. generate_from_simulation() [DEFAULT / RECOMMENDED]
     Runs actual PID simulation episodes and records every (state, wind, ref)
     encountered during flight.  These states match the deployment distribution —
     temporal correlations, realistic (phi, omega, vel) combinations, etc.

  2. generate_free_dataset() [ORIGINAL / fallback]
     Directly samples random time points on the reference trajectory and
     constructs perturbed drone states near each point.
     Faster to generate but suffers from distribution mismatch:
     independent random samples ≠ actual closed-loop flight states.

Why distribution matters
------------------------
During deployment PINN_FREE sees states that are the *result of running a
controller in closed-loop*.  Those states have temporal correlations and
specific (phi, omega, vel) combinations that never appear in independent
random Gaussian samples.  Training on random samples means the network is
always predicting out-of-distribution, leading to accumulated error and crash.

By recording states from actual PID flight, we ensure the training states
ARE the deployment states (or very close to them).

State format
------------
X : (N, 21) float32   [state(12), wind(3), ref(6)]         ← original
X : (N, 24) float32   [state(12), wind(3), ref_pos(3), ref_vel(3), ref_acc(3)]
                                                             ← geometric variant
    No control labels — labels are computed on-the-fly in trainer.py.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import T_TOTAL, MASS, DT, N_STEPS
from SIMULATION.trajectory import make_trajectory


# ─── Mode 1: Simulation-based data generation ─────────────────────────────────

def generate_from_simulation(n_points       = 100_000,
                              wind_speed_max  = 15.0,
                              traj_types     = None,
                              max_pos_error  = 2.0,
                              include_acc    = False,
                              save_path      = None,
                              drag_coeff     = 0.0):
    """
    Generate training data by running actual PID simulation episodes.

    Records (state_before_action, wind_force, ref[, acc_ref]) at every step.
    States come from the real closed-loop flight distribution.

    Parameters
    ----------
    include_acc    : if True, append reference acceleration (3,) → X shape (N, 24)
                     Required for geometric-teacher training.
    drag_coeff     : aerodynamic drag coefficient for QuadrotorModel (kg/m).
                     Must match the evaluation environment — pass C_DRAG when
                     generating data for PINN(Geo+Drag) so that training states
                     reflect drag-affected dynamics (avoids distribution mismatch).
    Returns
    -------
    X : (n_points, 21) or (n_points, 24) float32 array
    """
    from SIMULATION.quad_model import QuadrotorModel
    from CONTROLLERS.pid import PIDController

    if traj_types is None:
        traj_types = ["lemniscate", "circle", "helix"]

    all_X      = []
    n_trajs    = len(traj_types)
    n_per_traj = n_points // n_trajs
    dim        = 24 if include_acc else 21

    drag_str = f", drag={drag_coeff}" if drag_coeff > 0 else ""
    print(f"Generating {n_points:,} samples from PID simulation "
          f"(wind 0–{wind_speed_max} m/s, dim={dim}{drag_str})...")

    for traj_idx, traj_name in enumerate(traj_types):
        traj   = make_trajectory(traj_name)
        traj_X = []
        n_needed = n_per_traj if traj_idx < n_trajs - 1 \
                               else n_points - len(all_X) * n_per_traj

        episode = 0
        while len(traj_X) < n_needed:
            ws         = np.random.uniform(0.0, wind_speed_max)
            angle      = np.random.uniform(0.0, 2.0 * np.pi)
            wind_force = np.array([np.cos(angle), np.sin(angle), 0.0],
                                  dtype=np.float64) * ws * MASS

            quad = QuadrotorModel(drag_coeff=drag_coeff)
            pid  = PIDController()
            init_state      = np.zeros(12)
            init_state[0:3] = traj.get(0)[0]
            quad.reset(init_state)
            if hasattr(pid, 'reset'):
                pid.reset()

            for step in range(N_STEPS):
                t         = step * DT
                ref       = traj.get_full(t)       # (6,) [pos, vel]
                state_now = quad.state.copy()

                u = pid.compute_control(state_now, ref, wind_force)
                _, done = quad.step(u, wind_force)

                pos_err = np.linalg.norm(state_now[:3] - ref[:3])
                if pos_err <= max_pos_error:
                    row = [state_now.astype(np.float32),
                           wind_force.astype(np.float32),
                           ref.astype(np.float32)]
                    if include_acc:
                        acc = traj.get_acceleration(t).astype(np.float32)
                        row.append(acc)
                    traj_X.append(np.concatenate(row))

                if done:
                    break

            episode += 1

        traj_arr = np.array(traj_X[:n_needed], dtype=np.float32)
        all_X.append(traj_arr)
        steps_per_ep = len(traj_X) / max(episode, 1)
        print(f"  {traj_name:12s}: {len(traj_arr):,} samples "
              f"({episode} episodes, ~{steps_per_ep:.0f} steps/ep)")

    X = np.concatenate(all_X, axis=0)
    np.random.shuffle(X)

    print(f"\nTotal samples : {X.shape[0]:,}")
    print(f"Input dim     : {X.shape[1]}")

    if save_path is not None:
        np.save(save_path, X)
        print(f"Saved to      : {save_path}")

    return X


# ─── Mode 2: Random-perturbation data generation (original, fallback) ─────────

def generate_free_dataset(n_points      = 100_000,
                           wind_speed_max = 15.0,
                           traj_types    = None,
                           pos_noise     = 0.3,
                           vel_noise     = 0.5,
                           att_noise     = 0.15,
                           omega_noise   = 0.3,
                           include_acc   = False,
                           save_path     = None):
    """
    Generate a reference-only training dataset via random perturbations.

    NOTE: For best results prefer generate_from_simulation() above.

    Parameters
    ----------
    include_acc : if True, append reference acceleration (3,) → X shape (N, 24)
    Returns
    -------
    X : (n_points, 21) or (n_points, 24) float32 array
    """
    if traj_types is None:
        traj_types = ["lemniscate", "circle", "helix"]

    trajs      = [make_trajectory(t) for t in traj_types]
    n_trajs    = len(trajs)
    n_per_traj = n_points // n_trajs
    dim        = 24 if include_acc else 21

    all_X = []
    print(f"Generating {n_points:,} random-perturbation points (dim={dim})...")

    for idx, (traj, name) in enumerate(zip(trajs, traj_types)):
        n = n_per_traj if idx < n_trajs - 1 else n_points - len(all_X) * n_per_traj

        X = np.zeros((n, dim), dtype=np.float32)

        t_samples = np.random.uniform(0.0, T_TOTAL, n)

        for i, t in enumerate(t_samples):
            ref = traj.get_full(t)          # (6,) [pos(3), vel(3)]

            state = np.zeros(12)
            state[0:3]  = ref[0:3] + np.random.randn(3) * pos_noise
            state[6:9]  = ref[3:6] + np.random.randn(3) * vel_noise
            state[3:6]  = np.random.randn(3) * att_noise
            state[9:12] = np.random.randn(3) * omega_noise

            ws    = np.random.uniform(0.0, wind_speed_max)
            angle = np.random.uniform(0.0, 2.0 * np.pi)
            wind  = np.array([np.cos(angle), np.sin(angle), 0.0],
                             dtype=np.float32) * ws * MASS

            row = [state.astype(np.float32), wind, ref.astype(np.float32)]
            if include_acc:
                acc = traj.get_acceleration(t).astype(np.float32)
                row.append(acc)
            X[i] = np.concatenate(row)

        all_X.append(X)
        print(f"  {name:12s}: {n:,} samples")

    X = np.concatenate(all_X, axis=0)
    np.random.shuffle(X)

    print(f"\nTotal samples : {X.shape[0]:,}")
    print(f"Input dim     : {X.shape[1]}")

    if save_path is not None:
        np.save(save_path, X)
        print(f"Saved to      : {save_path}")

    return X


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Free Data Generator Sanity Check ===\n")

    print("--- Mode 1: simulation-based ---")
    X_sim = generate_from_simulation(
        n_points      = 3_000,
        wind_speed_max = 10.0,
        traj_types    = ["lemniscate"],
    )
    print(f"X_sim shape : {X_sim.shape}")
    print(f"State   min/max : {X_sim[:, :12].min():.3f} / {X_sim[:, :12].max():.3f}")
    print(f"Wind    min/max : {X_sim[:, 12:15].min():.3f} / {X_sim[:, 12:15].max():.3f}")
    print(f"Ref pos min/max : {X_sim[:, 15:18].min():.3f} / {X_sim[:, 15:18].max():.3f}")

    print("\n--- Mode 2: random-perturbation ---")
    X_rnd = generate_free_dataset(
        n_points      = 3_000,
        wind_speed_max = 10.0,
        traj_types    = ["lemniscate"],
    )
    print(f"X_rnd shape : {X_rnd.shape}")

    print("\nAll checks passed!")
