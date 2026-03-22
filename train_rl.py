"""
train_rl.py - with LQR pre-training

Usage:
    python train_rl.py                          # 1M steps + LQR pre-training
    python train_rl.py --no_pretrain            # pure RL from scratch
    python train_rl.py --fast                   # 100k steps quick test
    python train_rl.py --continue               # continue from existing checkpoint
    python train_rl.py --continue --steps 2000000  # continue with custom steps
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RL_CKPT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",       type=int, default=1_000_000)
    parser.add_argument("--envs",        type=int, default=8)
    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--fast",        action="store_true")
    parser.add_argument("--continue",    dest="cont", action="store_true",
                        help="Continue training from existing checkpoint "
                             "(no LQR pre-training, step counter preserved)")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from CONTROLLERS.rl_agent import train_rl, continue_rl
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Run: pip install stable-baselines3 gymnasium")
        return

    os.makedirs("RESULTS", exist_ok=True)
    steps = 100_000 if args.fast else args.steps

    if args.cont:
        # ── Continue from existing checkpoint ─────────────────────────────
        print("=" * 60)
        print("RL (PPO) — Continuing from checkpoint")
        print("=" * 60)
        print(f"Additional steps: {steps:,}")
        print(f"Parallel envs   : {args.envs}")
        print(f"Checkpoint      : {RL_CKPT}.zip")
        print("=" * 60 + "\n")

        model = continue_rl(
            additional_timesteps=steps,
            n_envs=args.envs,
            ckpt_path=RL_CKPT,
        )
    else:
        # ── Fresh training ─────────────────────────────────────────────────
        pretrain = not args.no_pretrain

        print("=" * 60)
        print("RL (PPO) Training — LQR Pre-trained")
        print("=" * 60)
        print(f"Timesteps    : {steps:,}")
        print(f"Parallel envs: {args.envs}")
        print(f"LQR pre-train: {'YES (recommended)' if pretrain else 'NO'}")
        print("=" * 60 + "\n")

        model = train_rl(
            total_timesteps=steps,
            n_envs=args.envs,
            save_path=RL_CKPT,
            pretrain=pretrain,
        )

    if model is not None:
        print("\n" + "=" * 60)
        print("Done.")
        print(f"  Checkpoint: {RL_CKPT}.zip")
        print("\nNext: python run_simulation.py")
        print("=" * 60)


if __name__ == "__main__":
    main()