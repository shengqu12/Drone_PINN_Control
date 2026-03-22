"""
run_simulation.py — Step 3 of the pipeline

Runs all controllers under various wind conditions and saves results.

Usage:
    python run_simulation.py                  # default: all controllers, wind sweep
    python run_simulation.py --traj circle    # change trajectory
    python run_simulation.py --wind constant  # single wind type
    python run_simulation.py --quick          # fast test run

Output:
    RESULTS/simulation_results.npz   — raw trajectory + error data
    RESULTS/trajectory_*.png         — XY trajectory plots per wind speed
    RESULTS/error_timeseries.png     — error over time for each controller
    RESULTS/wind_sweep.png           — RMSE vs wind speed (the key comparison plot)
"""

import numpy as np
import os
import sys
import argparse
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (DT, N_STEPS, T_TOTAL, WIND_SPEEDS,
                    RESULTS_DIR, PINN_CKPT)
from SIMULATION.quad_model import QuadrotorModel
from SIMULATION.wind import make_wind
from SIMULATION.trajectory import make_trajectory
from CONTROLLERS.pid import PIDController
from CONTROLLERS.lqr import LQRController
from CONTROLLERS.pinn_controller import PINNController
from CONTROLLERS.rl_agent import RLController, SB3_AVAILABLE
from PINN.trainer import PINN_FREE_CKPT


# ─── Single episode runner ────────────────────────────────────────────────────

def run_episode(controller, traj, wind_model):
    """
    Run one full simulation episode.

    Returns:
        dict with:
            pos      : (N, 3)  actual positions
            ref      : (N, 3)  reference positions
            err      : (N,)    position error at each step
            ctrl     : (N, 4)  control inputs
            crashed  : bool
            rmse     : float
    """
    quad = QuadrotorModel()

    # start at trajectory initial position
    init_state = np.zeros(12)
    init_state[0:3] = traj.get(0)[0]
    quad.reset(init_state)
    wind_model.reset()

    if hasattr(controller, 'reset'):
        controller.reset()

    N = N_STEPS
    pos_log  = np.zeros((N, 3))
    ref_log  = np.zeros((N, 3))
    err_log  = np.zeros(N)
    ctrl_log = np.zeros((N, 4))

    crashed     = False
    crash_step  = N

    for i in range(N):
        t          = i * DT
        ref        = traj.get_full(t)
        wind_force = wind_model.step()
        u          = controller.compute_control(quad.state, ref, wind_force)
        state, done = quad.step(u, wind_force)

        pos_log[i]  = state[0:3]
        ref_log[i]  = ref[0:3]
        err_log[i]  = np.linalg.norm(state[0:3] - ref[0:3])
        ctrl_log[i] = u

        if done:
            crashed    = True
            crash_step = i
            # fill remaining steps with last error (for fair RMSE comparison)
            err_log = err_log[:i+1]
            pos_log = pos_log[:i+1]
            break

    rmse = np.sqrt(np.mean(err_log ** 2))

    return dict(
        pos=pos_log, ref=ref_log, err=err_log, ctrl=ctrl_log,
        crashed=crashed, crash_step=crash_step, rmse=rmse
    )


# ─── Wind sweep ───────────────────────────────────────────────────────────────

def run_wind_sweep(controllers, traj_type, wind_type, wind_speeds):
    """
    Run all controllers across a range of wind speeds.

    Returns:
        results[controller_name][wind_speed] = episode_dict
    """
    traj = make_trajectory(traj_type)
    results = {name: {} for name in controllers}

    for ws in wind_speeds:
        print(f"\n  wind = {ws:5.1f} m/s", end="")
        for name, ctrl in controllers.items():
            wind = make_wind(wind_type, wind_speed=ws)
            ep   = run_episode(ctrl, traj, wind)
            results[name][ws] = ep
            status = "CRASH" if ep['crashed'] else f"RMSE={ep['rmse']:.3f}m"
            print(f"  |  {name}: {status}", end="", flush=True)

    print()
    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

COLORS = {
    "PID"         : "#888780",
    "LQR"         : "#1D9E75",
    "RL(PPO)"     : "#E24B4A",
    "PINN(Free)"  : "#F5A623",
    "PINN(Outer)" : "#4BA3E2",
}


def plot_wind_sweep(results, wind_speeds, save_path):
    """
    The key figure: RMSE vs wind speed for all controllers.
    This is the plot that shows PINN's advantage at high wind.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, wind_results in results.items():
        rmses = [wind_results[ws]['rmse'] for ws in wind_speeds]
        color = COLORS.get(name, '#333333')
        ax.plot(wind_speeds, rmses,
                color=color, linewidth=2, marker='o',
                markersize=6, label=name)

        # mark crashes with X
        for ws, ep in wind_results.items():
            if ep['crashed']:
                ax.scatter(ws, ep['rmse'], marker='x',
                          color=color, s=100, zorder=5)

    ax.set_xlabel("Wind speed (m/s)", fontsize=12)
    ax.set_ylabel("Position RMSE (m)", fontsize=12)
    ax.set_title("Controller comparison under wind disturbance", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_trajectories(results, wind_speeds_to_plot, save_dir):
    """
    XY trajectory plots for selected wind speeds.
    """
    for ws in wind_speeds_to_plot:
        fig, ax = plt.subplots(figsize=(7, 7))

        # reference (same for all controllers)
        first = next(iter(results.values()))
        ax.plot(first[ws]['ref'][:, 0], first[ws]['ref'][:, 1],
                '--', color='#AAAAAA', linewidth=1.5, label='reference', zorder=1)

        for name, wind_results in results.items():
            ep    = wind_results[ws]
            color = COLORS.get(name, '#333333')
            end   = ep['crash_step'] if ep['crashed'] else N_STEPS
            ax.plot(ep['pos'][:end, 0], ep['pos'][:end, 1],
                    color=color, linewidth=1.5,
                    label=f"{name} (RMSE={ep['rmse']:.2f}m)")

        ax.set_xlabel("x (m)", fontsize=11)
        ax.set_ylabel("y (m)", fontsize=11)
        ax.set_title(f"XY trajectory — wind {ws} m/s", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = os.path.join(save_dir, f"trajectory_wind{int(ws)}ms.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")


def plot_error_timeseries(results, wind_speed, save_path):
    """
    Error over time for a specific wind speed.
    Shows how quickly each controller recovers from disturbance.
    """
    t_arr = np.arange(N_STEPS) * DT
    fig, ax = plt.subplots(figsize=(10, 4))

    for name, wind_results in results.items():
        ep    = wind_results[wind_speed]
        color = COLORS.get(name, '#333333')
        ax.plot(t_arr[:len(ep['err'])], ep['err'],
                color=color, linewidth=1.3,
                label=f"{name} (RMSE={ep['rmse']:.3f}m)")

    ax.set_xlabel("time (s)", fontsize=11)
    ax.set_ylabel("position error (m)", fontsize=11)
    ax.set_title(f"Tracking error over time — wind {wind_speed} m/s", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def print_summary_table(results, wind_speeds):
    """Print RMSE table to console."""
    ctrl_names = list(results.keys())
    col_w      = 12

    print("\n" + "=" * 60)
    print("RMSE Summary (m)")
    print("=" * 60)

    # header
    header = f"{'Wind (m/s)':<12}" + "".join(f"{n:>{col_w}}" for n in ctrl_names)
    print(header)
    print("-" * len(header))

    for ws in wind_speeds:
        row = f"{ws:<12.1f}"
        for name in ctrl_names:
            ep    = results[name][ws]
            rmse  = ep['rmse']
            crash = " *" if ep['crashed'] else ""
            row  += f"{rmse:>{col_w-2}.4f}{crash:>2}"
        print(row)

    print("-" * len(header))
    print("* = crashed")
    print("=" * 60)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj",  default="lemniscate",
                        choices=["lemniscate", "circle", "helix"])
    parser.add_argument("--wind",  default="constant",
                        choices=["constant", "gust", "turbulence"])
    parser.add_argument("--quick", action="store_true",
                        help="quick test: fewer wind speeds, no RL")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    wind_speeds = [0, 2, 4, 6, 8, 10] if not args.quick else [0, 5, 10]

    print("=" * 60)
    print("Running simulation comparison")
    print("=" * 60)
    print(f"Trajectory : {args.traj}")
    print(f"Wind type  : {args.wind}")
    print(f"Wind speeds: {wind_speeds} m/s")
    print("=" * 60)

    # ── Initialize controllers ────────────────────────────────────────────
    print("\nInitializing controllers...")
    controllers = {
        # "PID"     : PIDController(),
        "LQR"     : LQRController(),
        # "RL(PPO)" : RLController(),
    }
    if os.path.exists(PINN_FREE_CKPT):
        controllers["PINN(Free)"] = PINNController(ckpt_path=PINN_FREE_CKPT)
        print(f"  Loaded PINN(Free) from {PINN_FREE_CKPT}")
    else:
        print(f"  [SKIP] PINN(Free) checkpoint not found: {PINN_FREE_CKPT}")



    # ── Run wind sweep ────────────────────────────────────────────────────
    print("\nRunning wind sweep...")
    results = run_wind_sweep(controllers, args.traj, args.wind, wind_speeds)

    # ── Print summary ─────────────────────────────────────────────────────
    print_summary_table(results, wind_speeds)

    # ── Save plots ────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    # 1. RMSE vs wind speed (key figure)
    plot_wind_sweep(results, wind_speeds,
                    os.path.join(RESULTS_DIR, "wind_sweep.png"))

    # 2. XY trajectories at 0, 5, 10 m/s
    plot_to_show = [ws for ws in [0, 2,4,6,8, 10] if ws in wind_speeds]
    plot_trajectories(results, plot_to_show, RESULTS_DIR)

    # 3. Error timeseries at medium wind
    mid_wind = wind_speeds[len(wind_speeds) // 2]
    plot_error_timeseries(results, mid_wind,
                          os.path.join(RESULTS_DIR, "error_timeseries.png"))

    # ── Save raw results ──────────────────────────────────────────────────
    save_path = os.path.join(RESULTS_DIR, "simulation_results.npz")
    np.savez(save_path, results=results, wind_speeds=wind_speeds)
    print(f"Saved: {save_path}")

    print("\n" + "=" * 60)
    print("Done. Results saved to RESULTS/")
    print("Next step: python evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    main()