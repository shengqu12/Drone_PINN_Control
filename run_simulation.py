"""
run_simulation.py — Step 3 of the pipeline

Runs all controllers under various wind conditions and saves results.

Usage:
    python run_simulation.py                  # default: all controllers, wind sweep
    python run_simulation.py --traj circle    # change trajectory
    python run_simulation.py --wind constant  # single wind type
    python run_simulation.py --quick          # fast test run
    python run_simulation.py --generalization # also test on unseen 3D trajectories

Output:
    RESULTS/wind_sweep.png           — RMSE vs wind speed (main comparison)
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
from SIMULATION.trajectory import make_trajectory, TEST_TRAJECTORIES, LemniscateTrajectory
from CONTROLLERS.lqr_ff import LQRFFController
from CONTROLLERS.pinn_controller import PINNController
from CONTROLLERS.pinn_geo_controller import PINNGeoController


# ─── Single episode runner ────────────────────────────────────────────────────

def run_episode(controller, traj, wind_model, wind_estimator=None, drag_coeff=0.0):
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
    drag_coeff     : float  aerodynamic drag coefficient for QuadrotorModel.
                     0.0 = no drag (default); C_DRAG = 0.5 for drag experiment.

    Returns
    -------
    dict with pos, ref, err, ctrl, crashed, crash_step, rmse
    """
    quad = QuadrotorModel(drag_coeff=drag_coeff)

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

        if isinstance(controller, PINNGeoController):
            acc_ref = traj.get_acceleration(t)
            u = controller.compute_control(quad.state, ref, wind_input, acc_ref)
        else:
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

def run_wind_sweep(controllers, traj_type, wind_type, wind_speeds, traj=None):
    """
    Run all controllers across a range of wind speeds.

    Parameters
    ----------
    controllers : dict[str, (controller, estimator_or_None)]
                  estimator=None  → receives true wind (perfect knowledge)
                  estimator=<obj> → receives IMU-estimated wind
    traj        : pre-built trajectory object, or None to build from traj_type

    Returns
    -------
    results[controller_name][wind_speed] = episode_dict
    """
    if traj is None:
        traj = make_trajectory(traj_type)
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


# ─── Plotting ─────────────────────────────────────────────────────────────────

COLORS = {
    "LQR+FF"    : "#2ECC71",   # green — analytical wind feedforward baseline
    "PINN(Free)": "#F5A623",   # amber — neural imitation of LQR+FF teacher
    "PINN(Geo)" : "#E2684B",   # orange-red — geometric teacher with acc_ref
}


def plot_wind_sweep(results, wind_speeds, save_path):
    """RMSE vs wind speed for all controllers."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, wind_results in results.items():
        rmses = [wind_results[ws]['rmse'] for ws in wind_speeds]
        color = COLORS.get(name, '#333333')
        ax.plot(wind_speeds, rmses,
                color=color, linewidth=2, marker='o', linestyle='-',
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
            end   = ep['crash_step'] if ep['crashed'] else N_STEPS
            ax.plot(ep['pos'][:end, 0], ep['pos'][:end, 1],
                    color=color, linewidth=1.5, linestyle='-',
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
        ax.plot(t_arr[:len(ep['err'])], ep['err'],
                color=color, linewidth=1.3, linestyle='-',
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


# ─── Generalization test (unseen trajectories) ────────────────────────────────

def run_generalization_test(controllers, wind_type, wind_speeds):
    """
    Evaluate all controllers on the three TEST-ONLY trajectories that were
    never seen during PINN training.  This mirrors PI-WAN's evaluation
    protocol: train on {circle, lemniscate, helix}, test on unseen shapes.

    Parameters
    ----------
    controllers : same dict as run_wind_sweep  (name → (ctrl, estimator))
    wind_type   : str
    wind_speeds : list of wind speeds to test (usually a subset, e.g. [0, 4, 8])

    Returns
    -------
    gen_results[traj_name][ctrl_name][wind_speed] = episode_dict
    """
    gen_results = {}

    for traj_name in TEST_TRAJECTORIES:
        print(f"\n  Trajectory: {traj_name}")
        traj = make_trajectory(traj_name)
        gen_results[traj_name] = {name: {} for name in controllers}

        for ws in wind_speeds:
            print(f"    wind = {ws:4.1f} m/s", end="")
            for name, (ctrl, estimator) in controllers.items():
                wind = make_wind(wind_type, wind_speed=ws)
                ep   = run_episode(ctrl, traj, wind, wind_estimator=estimator)
                gen_results[traj_name][name][ws] = ep
                status = "CRASH" if ep['crashed'] else f"{ep['rmse']:.3f}m"
                print(f"  |  {name}: {status}", end="", flush=True)
        print()

    return gen_results


def plot_generalization_results(gen_results, wind_speeds, save_path):
    """
    One subplot per test trajectory — RMSE vs wind speed for all controllers.
    Visualises how well each controller generalises to unseen 3D shapes.
    """
    n_trajs = len(TEST_TRAJECTORIES)
    fig, axes = plt.subplots(1, n_trajs, figsize=(6 * n_trajs, 5), sharey=False)

    traj_display = {
        "tilted_circle"    : "Tilted Circle\n(test-only)",
        "lissajous_3d"     : "3D Lissajous\n(test-only)",
        "rising_lemniscate": "Rising Lemniscate\n(test-only)",
    }

    for ax, traj_name in zip(axes, TEST_TRAJECTORIES):
        ctrl_results = gen_results[traj_name]
        for name, wind_results in ctrl_results.items():
            rmses = [wind_results[ws]['rmse'] for ws in wind_speeds]
            color = COLORS.get(name, '#333333')
            ax.plot(wind_speeds, rmses,
                    color=color, linewidth=2, marker='o', linestyle='-',
                    markersize=6, label=name)

            for ws, ep in wind_results.items():
                if ep['crashed']:
                    ax.scatter(ws, ep['rmse'], marker='x',
                               color=color, s=100, zorder=5)

        ax.set_title(traj_display.get(traj_name, traj_name), fontsize=12)
        ax.set_xlabel("Wind speed (m/s)", fontsize=11)
        ax.set_ylabel("Position RMSE (m)", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Generalization to Unseen 3D Trajectories", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def print_generalization_table(gen_results, wind_speeds):
    """Print per-trajectory RMSE tables."""
    for traj_name in TEST_TRAJECTORIES:
        ctrl_results = gen_results[traj_name]
        ctrl_names   = list(ctrl_results.keys())
        col_w        = 16

        sep = "=" * (14 + col_w * len(ctrl_names))
        print(f"\n{sep}")
        print(f"Generalization — {traj_name}   * = crashed")
        print(sep)
        header = f"{'Wind (m/s)':<14}" + "".join(f"{n:>{col_w}}" for n in ctrl_names)
        print(header)
        print("-" * len(header))
        for ws in wind_speeds:
            row = f"{ws:<14.1f}"
            for name in ctrl_names:
                ep   = ctrl_results[name][ws]
                flag = " *" if ep['crashed'] else "  "
                row += f"{ep['rmse']:>{col_w - 2}.4f}{flag}"
            print(row)
        print("-" * len(header))
        print(sep)


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
                        help="fast test: fewer wind speeds")
    parser.add_argument("--generalization", action="store_true",
                        help="also run generalization test on 3 unseen 3D trajectories")
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
    pinn     = PINNController(ckpt_path=PINN_FREE_CKPT)
    pinn_geo = PINNGeoController()

    controllers = {
        "LQR+FF"    : (LQRFFController(), None),
        "PINN(Free)": (pinn,              None),
        "PINN(Geo)" : (pinn_geo,          None),
    }

    sim_traj = LemniscateTrajectory(omega=0.5) if args.traj == "lemniscate" \
               else make_trajectory(args.traj)

    # ── Wind sweep ────────────────────────────────────────────────────────
    print("\nRunning wind sweep...")
    results = run_wind_sweep(controllers, args.traj, args.wind, wind_speeds,
                             traj=sim_traj)
    print_summary_table(results, wind_speeds)

    # ── Save plots ────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    plot_wind_sweep(results, wind_speeds,
                    os.path.join(RESULTS_DIR, "wind_sweep.png"))

    plot_to_show = [ws for ws in [0, 2, 4, 6, 8, 10] if ws in wind_speeds]
    plot_trajectories(results, plot_to_show, RESULTS_DIR)

    mid_wind = wind_speeds[len(wind_speeds) // 2]
    plot_error_timeseries(results, mid_wind,
                          os.path.join(RESULTS_DIR, "error_timeseries.png"))

    # ── Generalization test (unseen 3D trajectories) ──────────────────────
    if args.generalization:
        print("\n" + "=" * 60)
        print("Generalization test — 3 unseen 3D trajectories")
        print("(None of these shapes were used during PINN training)")
        print("=" * 60)
        gen_wind_speeds = [0, 4, 8] if not args.quick else [0, 8]
        gen_results = run_generalization_test(
            controllers, args.wind, gen_wind_speeds)
        print_generalization_table(gen_results, gen_wind_speeds)
        plot_generalization_results(
            gen_results, gen_wind_speeds,
            os.path.join(RESULTS_DIR, "generalization.png"))

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
