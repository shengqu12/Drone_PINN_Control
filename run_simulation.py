"""
run_simulation.py — Step 3 of the pipeline

Runs all controllers under various wind conditions and saves results.
Includes IMU-based wind estimation comparison (vs perfect wind knowledge).

Usage:
    python run_simulation.py                  # default: all controllers, wind sweep
    python run_simulation.py --traj circle    # change trajectory
    python run_simulation.py --wind constant  # single wind type
    python run_simulation.py --quick          # fast test run (no robustness sweep)

Output:
    RESULTS/wind_sweep.png           — RMSE vs wind speed (main comparison)
    RESULTS/robustness.png           — RMSE vs IMU estimation noise (NEW)
    RESULTS/trajectory_*.png         — XY trajectory plots per wind speed
    RESULTS/error_timeseries.png     — error over time for each controller
    RESULTS/simulation_results.npz   — raw data
"""

import numpy as np
import os
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (DT, N_STEPS, T_TOTAL, WIND_SPEEDS, MASS, G,
                    RESULTS_DIR, PINN_FREE_CKPT)
from SIMULATION.quad_model import QuadrotorModel, WindEstimator
from SIMULATION.wind import make_wind
from SIMULATION.trajectory import make_trajectory
from CONTROLLERS.pid import PIDController
from CONTROLLERS.lqr import LQRController
from CONTROLLERS.pinn_controller import PINNController
from CONTROLLERS.rl_agent import RLController, SB3_AVAILABLE


# ─── Single episode runner ────────────────────────────────────────────────────

def run_episode(controller, traj, wind_model, wind_estimator=None):
    """
    Run one full simulation episode.

    Parameters
    ----------
    controller     : any controller with .compute_control(state, ref, wind) -> u
    traj           : trajectory object
    wind_model     : wind source (provides TRUE wind force for dynamics)
    wind_estimator : WindEstimator or None.
                     None     → controller receives true wind (perfect knowledge).
                     not None → controller receives IMU-estimated wind.

    Returns
    -------
    dict with pos, ref, err, ctrl, crashed, crash_step, rmse
    """
    quad = QuadrotorModel()

    init_state = np.zeros(12)
    init_state[0:3] = traj.get(0)[0]
    quad.reset(init_state)
    wind_model.reset()

    if hasattr(controller, 'reset'):
        controller.reset()
    if wind_estimator is not None:
        wind_estimator.reset()

    N = N_STEPS
    pos_log  = np.zeros((N, 3))
    ref_log  = np.zeros((N, 3))
    err_log  = np.zeros(N)
    ctrl_log = np.zeros((N, 4))

    crashed    = False
    crash_step = N

    # Previous state/control — needed for IMU wind estimation
    state_prev = init_state.copy()
    u_prev     = np.array([MASS * G, 0., 0., 0.])

    for i in range(N):
        t          = i * DT
        ref        = traj.get_full(t)
        wind_force = wind_model.step()    # true wind — always drives dynamics

        # Wind input to controller:
        #   i=0  : no previous data → use true wind for the first step
        #   i>0  : use IMU residual estimate (or true wind if no estimator)
        if wind_estimator is not None and i > 0:
            wind_input = wind_estimator.update(quad.state, state_prev, u_prev[0], DT)
        else:
            wind_input = wind_force

        u = controller.compute_control(quad.state, ref, wind_input)

        # Save state/control BEFORE stepping (used in next iteration's estimate)
        state_prev = quad.state.copy()
        u_prev     = u.copy()

        state, done = quad.step(u, wind_force)   # dynamics always use true wind

        pos_log[i]  = state[0:3]
        ref_log[i]  = ref[0:3]
        err_log[i]  = np.linalg.norm(state[0:3] - ref[0:3])
        ctrl_log[i] = u

        if done:
            crashed    = True
            crash_step = i
            err_log    = err_log[:i+1]
            pos_log    = pos_log[:i+1]
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

    Parameters
    ----------
    controllers : dict[str, (controller, estimator_or_None)]
                  estimator=None     → receives true wind (perfect knowledge)
                  estimator=<obj>    → receives IMU-estimated wind

    Returns
    -------
    results[controller_name][wind_speed] = episode_dict
    """
    traj    = make_trajectory(traj_type)
    results = {name: {} for name in controllers}

    for ws in wind_speeds:
        print(f"\n  wind = {ws:5.1f} m/s", end="")
        for name, (ctrl, estimator) in controllers.items():
            wind = make_wind(wind_type, wind_speed=ws)
            ep   = run_episode(ctrl, traj, wind, wind_estimator=estimator)
            results[name][ws] = ep
            status = "CRASH" if ep['crashed'] else f"RMSE={ep['rmse']:.3f}m"
            print(f"  |  {name}: {status}", end="", flush=True)

    print()
    return results


# ─── Robustness sweep (RMSE vs estimation noise) ──────────────────────────────

NOISE_LEVELS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]


def run_robustness_sweep(pinn_ctrl, traj_type, wind_type, wind_speed,
                         noise_levels=None):
    """
    Test PINN(Free) + IMU estimator across increasing noise levels.
    Wind speed is fixed; only the estimation noise σ varies.

    Returns
    -------
    results[sigma] = episode_dict
    """
    if noise_levels is None:
        noise_levels = NOISE_LEVELS

    traj    = make_trajectory(traj_type)
    results = {}

    print(f"\nRobustness sweep at wind = {wind_speed} m/s ...")
    for sigma in noise_levels:
        estimator = WindEstimator(noise_std=sigma)
        wind      = make_wind(wind_type, wind_speed=wind_speed)
        ep        = run_episode(pinn_ctrl, traj, wind, wind_estimator=estimator)
        results[sigma] = ep
        status = "CRASH" if ep['crashed'] else f"RMSE={ep['rmse']:.3f}m"
        print(f"  σ = {sigma:.1f} N  →  {status}")

    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

COLORS = {
    "PID"                  : "#888780",
    "LQR"                  : "#1D9E75",
    "RL(PPO)"              : "#E24B4A",
    "PINN (perfect wind)"  : "#F5A623",
    "PINN (IMU, σ=0)"      : "#4BA3E2",
    "PINN (IMU, σ=0.3)"    : "#A855D4",
}


def plot_wind_sweep(results, wind_speeds, save_path):
    """RMSE vs wind speed for all controllers."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, wind_results in results.items():
        rmses = [wind_results[ws]['rmse'] for ws in wind_speeds]
        color = COLORS.get(name, '#333333')
        ls    = '--' if 'IMU' in name else '-'
        ax.plot(wind_speeds, rmses,
                color=color, linewidth=2, marker='o', linestyle=ls,
                markersize=6, label=name)

        for ws, ep in wind_results.items():
            if ep['crashed']:
                ax.scatter(ws, ep['rmse'], marker='x',
                           color=color, s=100, zorder=5)

    ax.set_xlabel("Wind speed (m/s)", fontsize=12)
    ax.set_ylabel("Position RMSE (m)", fontsize=12)
    ax.set_title("Controller comparison under wind disturbance", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_robustness_curve(rob_results, noise_levels, wind_speed,
                          perfect_rmse, save_path):
    """RMSE vs IMU estimation noise σ at a fixed wind speed."""
    rmses   = [rob_results[s]['rmse'] for s in noise_levels]
    crashed = [rob_results[s]['crashed'] for s in noise_levels]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(noise_levels, rmses,
            color='#4BA3E2', linewidth=2, marker='o', markersize=6,
            label='PINN (IMU estimated)')

    if perfect_rmse is not None:
        ax.axhline(perfect_rmse, color='#F5A623', linewidth=1.5, linestyle='--',
                   label=f'PINN (perfect wind, RMSE={perfect_rmse:.3f}m)')

    crash_labeled = False
    for s, c, r in zip(noise_levels, crashed, rmses):
        if c:
            lbl = 'crash' if not crash_labeled else None
            ax.scatter(s, r, marker='x', color='#E24B4A', s=120,
                       zorder=5, label=lbl)
            crash_labeled = True

    ax.set_xlabel("Estimation noise σ (N)", fontsize=12)
    ax.set_ylabel("Position RMSE (m)", fontsize=12)
    ax.set_title(
        f"PINN(Free) robustness to IMU wind estimation noise"
        f"  (wind = {wind_speed} m/s)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_trajectories(results, wind_speeds_to_plot, save_dir):
    """XY trajectory plots for selected wind speeds."""
    for ws in wind_speeds_to_plot:
        fig, ax = plt.subplots(figsize=(7, 7))

        first = next(iter(results.values()))
        ax.plot(first[ws]['ref'][:, 0], first[ws]['ref'][:, 1],
                '--', color='#AAAAAA', linewidth=1.5, label='reference', zorder=1)

        for name, wind_results in results.items():
            ep    = wind_results[ws]
            color = COLORS.get(name, '#333333')
            ls    = '--' if 'IMU' in name else '-'
            end   = ep['crash_step'] if ep['crashed'] else N_STEPS
            ax.plot(ep['pos'][:end, 0], ep['pos'][:end, 1],
                    color=color, linewidth=1.5, linestyle=ls,
                    label=f"{name} (RMSE={ep['rmse']:.2f}m)")

        ax.set_xlabel("x (m)", fontsize=11)
        ax.set_ylabel("y (m)", fontsize=11)
        ax.set_title(f"XY trajectory — wind {ws} m/s", fontsize=12)
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = os.path.join(save_dir, f"trajectory_wind{int(ws)}ms.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")


def plot_error_timeseries(results, wind_speed, save_path):
    """Error over time for a specific wind speed."""
    t_arr = np.arange(N_STEPS) * DT
    fig, ax = plt.subplots(figsize=(10, 4))

    for name, wind_results in results.items():
        ep    = wind_results[wind_speed]
        color = COLORS.get(name, '#333333')
        ls    = '--' if 'IMU' in name else '-'
        ax.plot(t_arr[:len(ep['err'])], ep['err'],
                color=color, linewidth=1.3, linestyle=ls,
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
    col_w      = 16

    sep = "=" * (12 + col_w * len(ctrl_names))
    print("\n" + sep)
    print("RMSE Summary (m)   * = crashed")
    print(sep)

    header = f"{'Wind (m/s)':<12}" + "".join(f"{n:>{col_w}}" for n in ctrl_names)
    print(header)
    print("-" * len(header))

    for ws in wind_speeds:
        row = f"{ws:<12.1f}"
        for name in ctrl_names:
            ep    = results[name][ws]
            crash = " *" if ep['crashed'] else "  "
            row  += f"{ep['rmse']:>{col_w-2}.4f}{crash}"
        print(row)

    print("-" * len(header))
    print(sep)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj",  default="lemniscate",
                        choices=["lemniscate", "circle", "helix"])
    parser.add_argument("--wind",  default="constant",
                        choices=["constant", "gust", "turbulence"])
    parser.add_argument("--quick", action="store_true",
                        help="fast test: fewer wind speeds, skip robustness sweep")
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
    pinn = PINNController(ckpt_path=PINN_FREE_CKPT)

    controllers = {
        "LQR"                 : (LQRController(),               None),
        "PINN (perfect wind)" : (pinn,                          None),
        "PINN (IMU, σ=0)"     : (pinn,  WindEstimator(noise_std=0.0)),
        "PINN (IMU, σ=0.3)"   : (pinn,  WindEstimator(noise_std=0.3)),
    }
    if not args.quick and SB3_AVAILABLE:
        controllers["RL(PPO)"] = (RLController(), None)

    # ── Wind sweep ────────────────────────────────────────────────────────
    print("\nRunning wind sweep...")
    results = run_wind_sweep(controllers, args.traj, args.wind, wind_speeds)
    print_summary_table(results, wind_speeds)

    # ── Robustness sweep: PINN(IMU) at varying noise, fixed wind ─────────
    rob_results  = None
    perfect_rmse = None
    if not args.quick:
        rob_wind    = 6
        rob_results = run_robustness_sweep(
            pinn, args.traj, args.wind, wind_speed=rob_wind)
        perfect_rmse = results["PINN (perfect wind)"].get(rob_wind, {}).get('rmse')

    # ── Save plots ────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    plot_wind_sweep(results, wind_speeds,
                    os.path.join(RESULTS_DIR, "wind_sweep.png"))

    plot_to_show = [ws for ws in [0, 2, 4, 6, 8, 10] if ws in wind_speeds]
    plot_trajectories(results, plot_to_show, RESULTS_DIR)

    mid_wind = wind_speeds[len(wind_speeds) // 2]
    plot_error_timeseries(results, mid_wind,
                          os.path.join(RESULTS_DIR, "error_timeseries.png"))

    if rob_results is not None:
        plot_robustness_curve(
            rob_results, NOISE_LEVELS, rob_wind, perfect_rmse,
            os.path.join(RESULTS_DIR, "robustness.png"))

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
