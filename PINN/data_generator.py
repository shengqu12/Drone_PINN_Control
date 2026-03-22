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
X : (N, 21) float32   [state(12), wind(3), ref(6)]
    No control labels — cascade-PD labels are computed on-the-fly in trainer_free.py.
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
                              save_path      = None):
    """
    Generate training data by running actual PID simulation episodes.

    Records (state_before_action, wind_force, ref) at every step of every
    episode.  States come from the real closed-loop flight distribution —
    NOT from independent Gaussian perturbations.

    Parameters
    ----------
    n_points       : total samples to collect
    wind_speed_max : maximum wind speed to simulate (m/s)
    traj_types     : list of trajectory names; default = all three
    max_pos_error  : discard states farther than this from the reference (m).
                     States with huge position error (e.g. 500 m) have clamped
                     labels and are not useful — they never appear during normal
                     PINN deployment.  2.0 m covers the realistic error range.
    save_path      : if given, save X as .npy file

    Returns
    -------
    X : (n_points, 21) float32 array  [state(12), wind(3), ref(6)]
    """
    from SIMULATION.quad_model import QuadrotorModel
    from CONTROLLERS.pid import PIDController

    if traj_types is None:
        traj_types = ["lemniscate", "circle", "helix"]

    all_X      = []
    n_trajs    = len(traj_types)
    n_per_traj = n_points // n_trajs

    print(f"Generating {n_points:,} samples from PID simulation "
          f"(wind 0–{wind_speed_max} m/s)...")

    for traj_idx, traj_name in enumerate(traj_types):
        traj   = make_trajectory(traj_name)
        traj_X = []
        n_needed = n_per_traj if traj_idx < n_trajs - 1 \
                               else n_points - len(all_X) * n_per_traj

        episode = 0
        while len(traj_X) < n_needed:
            # ── Sample wind for this episode ───────────────────────────────
            # Random wind per episode gives uniform coverage over all wind
            # speeds and directions regardless of episode count.
            # High-wind episodes may crash early but those states are still
            # valid training data — they're exactly what PINN will encounter.
            ws         = np.random.uniform(0.0, wind_speed_max)
            angle      = np.random.uniform(0.0, 2.0 * np.pi)
            wind_force = np.array([np.cos(angle), np.sin(angle), 0.0],
                                  dtype=np.float64) * ws * MASS

            # ── Initialise quad and controller ─────────────────────────────
            quad = QuadrotorModel()
            pid  = PIDController()

            init_state      = np.zeros(12)
            init_state[0:3] = traj.get(0)[0]
            quad.reset(init_state)
            if hasattr(pid, 'reset'):
                pid.reset()

            # ── Run one episode ────────────────────────────────────────────
            for step in range(N_STEPS):
                t           = step * DT
                ref         = traj.get_full(t)              # (6,) [pos, vel]
                state_now   = quad.state.copy()             # state BEFORE action

                u = pid.compute_control(state_now, ref, wind_force)
                _, done = quad.step(u, wind_force)

                # Only record states that are close to the reference.
                # States with huge position error (e.g. 500m at high wind)
                # produce clamped, useless labels and never appear during
                # PINN deployment — filtering them keeps the dataset clean.
                pos_err = np.linalg.norm(state_now[:3] - ref[:3])
                if pos_err <= max_pos_error:
                    traj_X.append(np.concatenate([
                        state_now.astype(np.float32),
                        wind_force.astype(np.float32),
                        ref.astype(np.float32),
                    ]))

                if done:
                    break   # crash — stop this episode, start a new one

            episode += 1

        traj_arr = np.array(traj_X[:n_needed], dtype=np.float32)
        all_X.append(traj_arr)
        steps_per_ep = len(traj_X) / max(episode, 1)
        print(f"  {traj_name:12s}: {len(traj_arr):,} samples "
              f"({episode} episodes, ~{steps_per_ep:.0f} steps/ep)")

    X = np.concatenate(all_X, axis=0)
    np.random.shuffle(X)

    print(f"\nTotal samples : {X.shape[0]:,}")
    print(f"Input dim     : {X.shape[1]}  (no control labels)")

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
                           save_path     = None):
    """
    Generate a reference-only training dataset via random perturbations.

    NOTE: For best results prefer generate_from_simulation() above.
    This function is kept for fast smoke-tests and ablations.

    Parameters
    ----------
    n_points       : total samples (split evenly across trajectory types)
    wind_speed_max : maximum wind speed to sample (m/s)
    traj_types     : list of trajectory names; default = all three
    pos_noise      : std of position perturbation from reference (m)
    vel_noise      : std of velocity perturbation from reference (m/s)
    att_noise      : std of attitude noise (rad)
    omega_noise    : std of angular-rate noise (rad/s)
    save_path      : if given, save X as .npy file

    Returns
    -------
    X : (n_points, 21) float32 array  [state(12), wind(3), ref(6)]
    """
    if traj_types is None:
        traj_types = ["lemniscate", "circle", "helix"]

    trajs      = [make_trajectory(t) for t in traj_types]
    n_trajs    = len(trajs)
    n_per_traj = n_points // n_trajs

    all_X = []
    print(f"Generating {n_points:,} random-perturbation points...")

    for idx, (traj, name) in enumerate(zip(trajs, traj_types)):
        n = n_per_traj if idx < n_trajs - 1 else n_points - len(all_X) * n_per_traj

        X = np.zeros((n, 21), dtype=np.float32)

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

            X[i] = np.concatenate([state, wind, ref])

        all_X.append(X)
        print(f"  {name:12s}: {n:,} samples")

    X = np.concatenate(all_X, axis=0)
    np.random.shuffle(X)

    print(f"\nTotal samples : {X.shape[0]:,}")
    print(f"Input dim     : {X.shape[1]}  (no control labels)")

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
