"""
train_pinn_free.py — Reference-Only PINN Training (No LQR Labels)

This is the "label-free" alternative to train_pinn.py.

Key difference
--------------
  train_pinn.py      : supervises ONLY with trajectory reference position/velocity


The network self-discovers what control to apply by minimising the cascade-PD
physics loss (Step 1–4 in trainer_free.py). Gradients flow directly through the
analytical label computation — no RK4 rollout required for the primary loss.

Data generation
---------------
  DEFAULT (--sim_data, recommended):
    States are recorded from actual PID simulation episodes across multiple
    wind speeds.  This ensures training distribution = deployment distribution.

  FALLBACK (--rand_data):
    States are sampled near the reference trajectory with random Gaussian noise.
    Faster to generate but suffers from distribution mismatch.

Checkpoint
----------
  Saved to CHECKPOINTS/pinn_free.pt
  Can be loaded by PINNController:
      ctrl = PINNController(ckpt_path="CHECKPOINTS/pinn_free.pt")

Usage
-----
    python train_pinn_free.py               # full training (5000 epochs, sim data)
    python train_pinn_free.py --rand_data   # use random perturbations instead
    python train_pinn_free.py --fast        # quick test  (200 epochs, 5k pts)
    python train_pinn_free.py --epochs 3000
    python train_pinn_free.py --n_points 150000 --wind_max 15
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PINN without LQR control labels")
    parser.add_argument("--epochs",    type=int,   default=5000,
                        help="Number of training epochs (default: 5000)")
    parser.add_argument("--n_points",  type=int,   default=100_000,
                        help="Dataset size (default: 100000)")
    parser.add_argument("--wind_max",  type=float, default=15.0,
                        help="Max wind speed in training data (default: 15 m/s)")
    parser.add_argument("--batch",     type=int,   default=512,
                        help="Batch size (default: 512)")
    parser.add_argument("--fast",      action="store_true",
                        help="Quick smoke-test: 200 epochs, 5000 points")
    parser.add_argument("--rand_data", action="store_true",
                        help="Use random-perturbation data instead of simulation data")
    parser.add_argument("--save_data", action="store_true",
                        help="Cache generated dataset to disk for reuse")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Import modules ────────────────────────────────────────────────────────
    try:
        from PINN.data_generator import (generate_from_simulation,
                                         generate_free_dataset)
        from PINN.trainer        import PINNFreeTrainer, PINN_FREE_CKPT
    except ImportError as e:
        print(f"Import error: {e}")
        sys.exit(1)

    # ── Fast mode overrides ───────────────────────────────────────────────────
    if args.fast:
        args.epochs   = 200
        args.n_points = 5_000
        print("[FAST MODE] epochs=200, n_points=5000\n")

    if args.rand_data:
        data_mode = "random-perturbation"
    else:
        data_mode = "combined (PID-sim + large-perturbation)"

    print("=" * 62)
    print("  PINN Free Trainer — Reference-Only (No LQR Labels)")
    print("=" * 62)
    print(f"  Epochs      : {args.epochs}")
    print(f"  Dataset     : {args.n_points:,} points ({data_mode})")
    print(f"  Wind range  : 0 – {args.wind_max} m/s")
    print(f"  Batch size  : {args.batch}")
    print(f"  Checkpoint  : {PINN_FREE_CKPT}")
    print("=" * 62 + "\n")

    # ── Generate (or load) dataset ────────────────────────────────────────────
    cache_suffix = "rand" if args.rand_data else "combined"
    data_cache   = os.path.join("CHECKPOINTS", f"free_dataset_{cache_suffix}.npy")

    if args.save_data and os.path.exists(data_cache):
        print(f"Loading cached dataset from {data_cache}...")
        X = np.load(data_cache)
        print(f"Loaded {len(X):,} samples.\n")
    elif args.rand_data:
        X = generate_free_dataset(
            n_points      = args.n_points,
            wind_speed_max = args.wind_max,
            traj_types    = ["lemniscate", "circle", "helix"],
            save_path     = data_cache if args.save_data else None,
        )
    else:
        # ── DEFAULT: combined simulation + large-perturbation data ─────────
        # Simulation data covers normal-flight states (drone near trajectory).
        # Large-perturbation data covers large-error recovery states that PINN
        # will encounter in closed loop but PID never reaches at low wind.
        # Without large-perturbation data the network outputs near-zero torques
        # when the drone has drifted 0.5+ m from the trajectory at 0 wind,
        # because PID always corrects before reaching such errors.
        n_sim  = args.n_points // 2
        n_rand = args.n_points - n_sim
        print(f"Generating combined dataset: {n_sim:,} sim + {n_rand:,} large-perturbation...")
        X_sim = generate_from_simulation(
            n_points       = n_sim,
            wind_speed_max = args.wind_max,
            traj_types     = ["lemniscate", "circle", "helix"],
        )
        X_rnd = generate_free_dataset(
            n_points       = n_rand,
            wind_speed_max = args.wind_max,
            traj_types     = ["lemniscate", "circle", "helix"],
            pos_noise      = 1.5,    # covers up to 1.5m position error (was 0.3)
            vel_noise      = 2.0,    # covers large velocity mismatches (was 0.5)
            att_noise      = 0.4,    # covers up to ±0.4 rad attitude errors (was 0.15)
            omega_noise    = 1.5,    # covers large angular rates (was 0.3)
        )
        X = np.concatenate([X_sim, X_rnd], axis=0)
        np.random.shuffle(X)
        print(f"Combined dataset: {X.shape[0]:,} samples total\n")
        if args.save_data:
            np.save(data_cache, X)
            print(f"Saved to {data_cache}")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = PINNFreeTrainer()
    trainer.train(
        X,
        epochs     = args.epochs,
        batch_size = args.batch,
        log_every  = 100,
        ckpt_path  = PINN_FREE_CKPT,
    )

    # ── Loss plot ──────────────────────────────────────────────────────────────
    try:
        os.makedirs("RESULTS", exist_ok=True)
        trainer.plot_loss(os.path.join("RESULTS", "pinn_free_loss.png"))
    except Exception as e:
        print(f"(Loss plot skipped: {e})")

    print("\n" + "=" * 62)
    print("  Training complete.")
    print(f"  Checkpoint : {PINN_FREE_CKPT}")
    print()
    print("  To evaluate alongside other controllers:")
    print("    from CONTROLLERS.pinn_controller import PINNController")
    print(f'    ctrl = PINNController(ckpt_path="{PINN_FREE_CKPT}")')
    print("=" * 62)


if __name__ == "__main__":
    main()
