"""
train_pinn_geo.py — Geometric PINN Training (Differential Flatness Labels)

Key difference from train_pinn.py
-----------------------------------
  train_pinn.py     : LQR+FF teacher  (21-dim input, no trajectory acceleration)
  train_pinn_geo.py : Geometric teacher (24-dim input, includes a_ref)

The geometric teacher adds reference acceleration (centripetal forces on curves)
to the label computation, giving the PINN a genuine advantage over LQR+FF on
curved trajectories like the lemniscate.

Data generation
---------------
  generate_from_simulation(include_acc=True) → (N, 24) dataset
  Extra 3 dimensions = reference acceleration at each timestep

Checkpoint
----------
  Saved to CHECKPOINTS/pinn_geo.pt
  Can be loaded by PINNGeoController:
      ctrl = PINNGeoController()  # auto-loads CHECKPOINTS/pinn_geo.pt

Usage
-----
    python train_pinn_geo.py               # full training (5000 epochs, 100k pts)
    python train_pinn_geo.py --fast        # quick test (200 epochs, 5k pts)
    python train_pinn_geo.py --epochs 3000
    python train_pinn_geo.py --n_points 150000 --wind_max 15
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Geometric PINN with differential flatness labels")
    parser.add_argument("--epochs",   type=int,   default=5000,
                        help="Training epochs (default: 5000)")
    parser.add_argument("--n_points", type=int,   default=100_000,
                        help="Dataset size (default: 100000)")
    parser.add_argument("--wind_max", type=float, default=15.0,
                        help="Max wind speed in training data (default: 15 m/s)")
    parser.add_argument("--batch",    type=int,   default=512,
                        help="Batch size (default: 512)")
    parser.add_argument("--fast",     action="store_true",
                        help="Quick smoke-test: 200 epochs, 5000 points")
    parser.add_argument("--save_data", action="store_true",
                        help="Cache generated dataset to disk")
    return parser.parse_args()


def main():
    args = parse_args()

    from PINN.data_generator import generate_from_simulation, generate_free_dataset
    from PINN.trainer_geo    import PINNGeoTrainer
    from config              import PINN_GEO_CKPT

    if args.fast:
        args.epochs   = 200
        args.n_points = 5_000
        print("[FAST MODE] epochs=200, n_points=5000\n")

    print("=" * 65)
    print("  Geometric PINN Trainer — Differential Flatness Labels")
    print("=" * 65)
    print(f"  Epochs      : {args.epochs}")
    print(f"  Dataset     : {args.n_points:,} points (24-dim, with acc_ref)")
    print(f"  Wind range  : 0 – {args.wind_max} m/s")
    print(f"  Batch size  : {args.batch}")
    print(f"  Checkpoint  : {PINN_GEO_CKPT}")
    print(f"  Label       : f_des = m*(a_ref + Kp*err + Kv*vel_err + g) - F_wind")
    print(f"                T = ||f_des||, att from exact trig, tau from inner PD")
    print("=" * 65 + "\n")

    # ── Generate / load dataset ────────────────────────────────────────────────
    data_cache = os.path.join("CHECKPOINTS", "geo_dataset_combined.npy")

    if args.save_data and os.path.exists(data_cache):
        print(f"Loading cached dataset from {data_cache}...")
        X = np.load(data_cache)
        print(f"Loaded {len(X):,} samples (dim={X.shape[1]}).\n")
    else:
        # Combined: simulation data (near-ref states) + large-perturbation data
        # Large-perturbation covers recovery scenarios that PID sim never generates
        n_sim  = args.n_points // 2
        n_rand = args.n_points - n_sim

        print(f"Generating combined dataset: {n_sim:,} sim + {n_rand:,} large-perturbation...")

        X_sim = generate_from_simulation(
            n_points       = n_sim,
            wind_speed_max = args.wind_max,
            traj_types     = ["lemniscate", "circle", "helix"],
            include_acc    = True,     # ← 24-dim output
        )
        X_rnd = generate_free_dataset(
            n_points       = n_rand,
            wind_speed_max = args.wind_max,
            traj_types     = ["lemniscate", "circle", "helix"],
            pos_noise      = 1.5,
            vel_noise      = 2.0,
            att_noise      = 0.4,
            omega_noise    = 1.5,
            include_acc    = True,     # ← 24-dim output
        )
        X = np.concatenate([X_sim, X_rnd], axis=0)
        np.random.shuffle(X)
        print(f"Combined dataset: {X.shape[0]:,} samples, dim={X.shape[1]}\n")

        if args.save_data:
            os.makedirs("CHECKPOINTS", exist_ok=True)
            np.save(data_cache, X)
            print(f"Saved to {data_cache}")

    # ── Train ──────────────────────────────────────────────────────────────────
    trainer = PINNGeoTrainer()
    trainer.train(
        X,
        epochs     = args.epochs,
        batch_size = args.batch,
        log_every  = 100,
        ckpt_path  = PINN_GEO_CKPT,
    )

    # ── Loss plot ──────────────────────────────────────────────────────────────
    try:
        os.makedirs("RESULTS", exist_ok=True)
        trainer.plot_loss(os.path.join("RESULTS", "pinn_geo_loss.png"))
    except Exception as e:
        print(f"(Loss plot skipped: {e})")

    print("\n" + "=" * 65)
    print("  Training complete.")
    print(f"  Checkpoint : {PINN_GEO_CKPT}")
    print()
    print("  To evaluate alongside other controllers:")
    print("    from CONTROLLERS.pinn_geo_controller import PINNGeoController")
    print("    ctrl = PINNGeoController()   # auto-loads pinn_geo.pt")
    print()
    print("  Or run the full comparison:")
    print("    python run_simulation.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
