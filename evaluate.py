"""
evaluate.py — Step 4 of the pipeline

Generates publication-quality figures and performance tables.

Usage:
    python evaluate.py                    # uses existing simulation_results.npz
    python evaluate.py --rerun            # re-runs simulation first
    python evaluate.py --wind_type gust   # evaluate under gust wind

Output:
    RESULTS/fig1_wind_sweep.png       — RMSE vs wind speed (key figure)
    RESULTS/fig2_trajectory_*.png     — XY trajectory comparison
    RESULTS/fig3_error_timeseries.png — error over time
    RESULTS/fig4_improvement.png      — PINN improvement over LQR (%)
    RESULTS/performance_table.csv     — full results table
"""

import numpy as np
import os
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_DIR, DT, N_STEPS
from CONTROLLERS.pid import PIDController
from CONTROLLERS.lqr import LQRController
from CONTROLLERS.pinn_controller import PINNController
from CONTROLLERS.rl_agent import RLController, train_rl
from PINN.trainer import PINN_FREE_CKPT


# ─── Style ────────────────────────────────────────────────────────────────────

COLORS = {
    "PID"       : "#888780",
    "LQR"       : "#1D9E75",
    "PINN"      : "#534AB7",
    "RL(PPO)"   : "#E24B4A",
    "PINN(Free)": "#F5A623",
}
MARKERS = {"PID": "s", "LQR": "o", "PINN": "D", "RL(PPO)": "X", "PINN(Free)": "^"}

plt.rcParams.update({
    'font.size'        : 11,
    'axes.linewidth'   : 0.8,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'grid.alpha'       : 0.3,
    'grid.linewidth'   : 0.5,
})


# ─── Load or run simulation ───────────────────────────────────────────────────

def load_results(results_path):
    data = np.load(results_path, allow_pickle=True)
    results     = data['results'].item()
    wind_speeds = data['wind_speeds'].tolist()
    return results, wind_speeds


def run_simulation(wind_speeds, wind_type, traj_type):
    """Run full simulation and return results dict."""
    from SIMULATION.quad_model import QuadrotorModel
    from SIMULATION.wind import make_wind
    from SIMULATION.trajectory import make_trajectory
    from CONTROLLERS.pid import PIDController
    from CONTROLLERS.lqr import LQRController
    from CONTROLLERS.pinn_controller import PINNController

    traj = make_trajectory(traj_type)
    controllers = {
        "PID"     : PIDController(),
        "LQR"     : LQRController(),
        "PINN"    : PINNController(),
        "RL(PPO)" : RLController(),
    }
    if os.path.exists(PINN_FREE_CKPT):
        controllers["PINN(Free)"] = PINNController(ckpt_path=PINN_FREE_CKPT)

    results = {name: {} for name in controllers}
    t_arr   = np.arange(N_STEPS) * DT

    for ws in wind_speeds:
        print(f"  wind = {ws:5.1f} m/s", end="")
        for name, ctrl in controllers.items():
            quad = QuadrotorModel()
            wind = make_wind(wind_type, wind_speed=ws)

            init_state = np.zeros(12)
            init_state[0:3] = traj.get(0)[0]
            quad.reset(init_state)
            wind.reset()
            if hasattr(ctrl, 'reset'):
                ctrl.reset()

            pos_log  = np.zeros((N_STEPS, 3))
            ref_log  = np.zeros((N_STEPS, 3))
            err_log  = np.zeros(N_STEPS)
            crashed  = False
            crash_step = N_STEPS

            for i in range(N_STEPS):
                t          = i * DT
                ref        = traj.get_full(t)
                wind_force = wind.step()
                u          = ctrl.compute_control(quad.state, ref, wind_force)
                state, done = quad.step(u, wind_force)

                pos_log[i]  = state[0:3]
                ref_log[i]  = ref[0:3]
                err_log[i]  = np.linalg.norm(state[0:3] - ref[0:3])

                if done:
                    crashed    = True
                    crash_step = i
                    err_log[i:] = err_log[i]
                    pos_log[i:] = pos_log[i]
                    break

            rmse = np.sqrt(np.mean(err_log**2))
            results[name][ws] = dict(
                pos=pos_log, ref=ref_log, err=err_log,
                crashed=crashed, crash_step=crash_step, rmse=rmse)

            status = "CRASH" if crashed else f"RMSE={rmse:.3f}m"
            print(f"  |  {name}: {status}", end="", flush=True)
        print()

    return results, t_arr


# ─── Figure 1: RMSE vs wind speed ─────────────────────────────────────────────

def plot_wind_sweep(results, wind_speeds, save_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    plot_order = ["PID", "LQR", "PINN", "PINN(Free)", "RL(PPO)"]
    for name in plot_order:
        if name not in results:
            continue
        rmses  = [results[name][ws]['rmse'] for ws in wind_speeds]
        color  = COLORS.get(name, '#333333')
        marker = MARKERS.get(name, 'o')

        ax.plot(wind_speeds, rmses,
                color=color, linewidth=2,
                marker=marker, markersize=7,
                label=name, zorder=3)

        # mark crashes with X
        for ws, ep in results[name].items():
            if ep['crashed']:
                ax.scatter(ws, ep['rmse'],
                          marker='x', color=color,
                          s=120, linewidths=2, zorder=5)

    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Position RMSE (m)")
    ax.set_title("Controller performance under wind disturbance")
    ax.legend(framealpha=0.9, fontsize=10)
    ax.grid(True)
    ax.set_xlim(left=-0.5)
    ax.set_ylim(bottom=0)

    # annotate PINN < LQR crossover
    if "LQR" in results and "PINN" in results:
        lqr_rmses  = [results["LQR"][ws]['rmse']  for ws in wind_speeds]
        pinn_rmses = [results["PINN"][ws]['rmse'] for ws in wind_speeds]
        for i, ws in enumerate(wind_speeds):
            if pinn_rmses[i] < lqr_rmses[i] and not results["PINN"][ws]['crashed']:
                ax.annotate("PINN < LQR",
                            xy=(ws, pinn_rmses[i]),
                            xytext=(ws + 0.5, pinn_rmses[i] + 0.2),
                            fontsize=9, color=COLORS["PINN"],
                            arrowprops=dict(arrowstyle="->",
                                           color=COLORS["PINN"], lw=1))
                break

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ─── Figure 2: XY trajectories ────────────────────────────────────────────────

def plot_trajectories(results, wind_speeds_to_plot, save_dir):
    for ws in wind_speeds_to_plot:
        fig, ax = plt.subplots(figsize=(6, 6))

        # reference
        first = next(iter(results.values()))
        ax.plot(first[ws]['ref'][:, 0], first[ws]['ref'][:, 1],
                '--', color='#AAAAAA', linewidth=1.5,
                label='Reference', zorder=1)

        for name in ["LQR", "PINN", "PINN(Free)", "PID", "RL(PPO)"]:
            if name not in results:
                continue
            ep    = results[name][ws]
            color = COLORS.get(name, '#333333')
            end   = ep['crash_step'] if ep['crashed'] else N_STEPS

            crash_str = " (crash)" if ep['crashed'] else ""
            ax.plot(ep['pos'][:end, 0], ep['pos'][:end, 1],
                    color=color, linewidth=1.5,
                    label=f"{name}  RMSE={ep['rmse']:.2f}m{crash_str}",
                    zorder=2)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"XY trajectory — wind {ws} m/s (constant)")
        ax.legend(fontsize=9, framealpha=0.9)
        ax.set_aspect('equal')
        ax.grid(True)

        plt.tight_layout()
        path = os.path.join(save_dir, f"fig2_trajectory_wind{int(ws)}ms.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")


# ─── Figure 3: Error timeseries ───────────────────────────────────────────────

def plot_error_timeseries(results, wind_speeds_to_plot, save_path):
    n  = len(wind_speeds_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    t_arr = np.arange(N_STEPS) * DT

    for ax, ws in zip(axes, wind_speeds_to_plot):
        for name in ["LQR", "PINN", "PINN(Free)", "PID", "RL(PPO)"]:
            if name not in results:
                continue
            ep    = results[name][ws]
            color = COLORS.get(name, '#333333')
            ax.plot(t_arr[:len(ep['err'])], ep['err'],
                    color=color, linewidth=1.3,
                    label=f"{name} ({ep['rmse']:.2f}m)")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position error (m)")
        ax.set_title(f"Wind = {ws} m/s")
        ax.legend(fontsize=9)
        ax.grid(True)

    fig.suptitle("Tracking error over time", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ─── Figure 4: PINN improvement over LQR ─────────────────────────────────────

def plot_improvement(results, wind_speeds, save_path):
    """
    Show PINN and PINN(Free) RMSE reduction vs LQR as percentage.
    Grouped bars — one group per wind speed.
    Positive = controller better than LQR, Negative = worse.
    """
    if "LQR" not in results:
        print("LQR results not found, skipping improvement plot")
        return

    variants = [n for n in ["PINN", "PINN(Free)"] if n in results]
    if not variants:
        print("No PINN variants found for improvement plot")
        return

    lqr_rmses = {ws: results["LQR"][ws]['rmse'] for ws in wind_speeds}

    # collect per-variant improvement series
    data = {}
    for name in variants:
        imps = []
        for ws in wind_speeds:
            ep = results[name][ws]
            if ep['crashed'] or lqr_rmses[ws] == 0:
                imps.append(None)
            else:
                imps.append((lqr_rmses[ws] - ep['rmse']) / lqr_rmses[ws] * 100)
        data[name] = imps

    fig, ax = plt.subplots(figsize=(8, 4))
    n_variants = len(variants)
    bar_w  = 0.35
    offset = np.linspace(-(n_variants - 1) / 2, (n_variants - 1) / 2, n_variants) * bar_w
    x      = np.array(wind_speeds, dtype=float)

    for i, name in enumerate(variants):
        imps    = data[name]
        heights = [v if v is not None else 0.0 for v in imps]
        hatch   = [None if v is not None else '///' for v in imps]
        color   = COLORS.get(name, '#333333')

        bars = ax.bar(x + offset[i], heights, width=bar_w,
                      color=color, edgecolor='white', linewidth=0.5,
                      label=name)

        # hatching for crashed bars
        for bar, h in zip(bars, hatch):
            if h:
                bar.set_hatch(h)
                bar.set_alpha(0.4)

        # value labels
        for bar, val in zip(bars, imps):
            if val is not None:
                ypos = bar.get_height() + (0.8 if val >= 0 else -2.5)
                ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                        f"{val:+.1f}%", ha='center', va='bottom',
                        fontsize=8, color=color)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, 1.0,
                        "CRASH", ha='center', va='bottom',
                        fontsize=7, color=color, rotation=90)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("RMSE improvement over LQR (%)")
    ax.set_title("PINN variants — improvement relative to LQR\n(positive = better than LQR)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(ws)) for ws in wind_speeds])
    ax.legend(framealpha=0.9, fontsize=10)
    ax.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ─── Performance table ────────────────────────────────────────────────────────

def save_performance_table(results, wind_speeds, save_path):
    import csv

    ctrl_names = [n for n in ["PID", "LQR", "PINN", "PINN(Free)", "RL(PPO)"]
                  if n in results]

    rows = []
    for ws in wind_speeds:
        row = {"wind_speed_ms": ws}
        for name in ctrl_names:
            ep = results[name][ws]
            key = name.replace("(", "").replace(")", "")
            row[f"{key}_rmse"]    = round(ep['rmse'], 4)
            row[f"{key}_crashed"] = ep['crashed']

        # PINN variants vs LQR improvement
        if "LQR" in results:
            lqr_rmse = results["LQR"][ws]['rmse']
            for pinn_name in ["PINN", "PINN(Free)"]:
                if pinn_name not in results:
                    continue
                key = pinn_name.replace("(", "").replace(")", "")
                if not results[pinn_name][ws]['crashed'] and lqr_rmse > 0:
                    pct = (lqr_rmse - results[pinn_name][ws]['rmse']) / lqr_rmse * 100
                    row[f"{key}_vs_LQR_pct"] = round(pct, 1)
                else:
                    row[f"{key}_vs_LQR_pct"] = "CRASH"
        rows.append(row)

    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {save_path}")

    # print to console
    col_w = 12
    header_names = ctrl_names
    sep_w = 10 + col_w * len(header_names)
    print("\n" + "=" * sep_w)
    print("Performance Summary (RMSE in metres, * = crashed)")
    print("=" * sep_w)
    hdr = f"{'Wind':>8}" + "".join(f"{n:>{col_w}}" for n in header_names)
    print(hdr)
    print("-" * sep_w)
    for row in rows:
        line = f"{row['wind_speed_ms']:>7.1f} "
        for name in header_names:
            key = name.replace("(", "").replace(")", "")
            rmse    = row.get(f"{key}_rmse", float('nan'))
            crashed = row.get(f"{key}_crashed", False)
            tag     = "*" if crashed else " "
            line   += f"{rmse:>{col_w-1}.4f}{tag}"
        print(line)
    print("-" * sep_w)

    # improvement rows
    for pinn_name in ["PINN", "PINN(Free)"]:
        if pinn_name not in results:
            continue
        key = pinn_name.replace("(", "").replace(")", "")
        imps = [str(row.get(f"{key}_vs_LQR_pct", "N/A")) for row in rows]
        print(f"  {pinn_name} vs LQR: " + "  ".join(f"{v:>8}" for v in imps))
    print("=" * sep_w)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun",     action="store_true",
                        help="re-run simulation before evaluating")
    parser.add_argument("--wind_type", default="constant",
                        choices=["constant", "gust", "turbulence"])
    parser.add_argument("--traj",      default="lemniscate",
                        choices=["lemniscate", "circle", "helix"])
    parser.add_argument("--wind_speeds", nargs="+", type=float,
                        default=[0, 2, 4, 6, 8, 10, 12])
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_path = os.path.join(RESULTS_DIR, "simulation_results.npz")

    if args.rerun or not os.path.exists(results_path):
        print(f"Running simulation ({args.wind_type} wind)...")
        results, t_arr = run_simulation(
            args.wind_speeds, args.wind_type, args.traj)
        np.savez(results_path, results=results,
                 wind_speeds=args.wind_speeds)
    else:
        print(f"Loading existing results from {results_path}")
        results, wind_speeds = load_results(results_path)
        args.wind_speeds = wind_speeds

    print("\nGenerating figures...")

    # Fig 1: wind sweep
    plot_wind_sweep(results, args.wind_speeds,
                    os.path.join(RESULTS_DIR, "fig1_wind_sweep.png"))

    # Fig 2: trajectories at 0, 5, 10 m/s
    plot_to_show = [ws for ws in [0, 5, 10] if ws in args.wind_speeds]
    plot_trajectories(results, plot_to_show, RESULTS_DIR)

    # Fig 3: error timeseries at 3 wind speeds
    ts_winds = [ws for ws in [0, 5, 10] if ws in args.wind_speeds][:3]
    plot_error_timeseries(results, ts_winds,
                          os.path.join(RESULTS_DIR, "fig3_error_timeseries.png"))

    # Fig 4: improvement over LQR
    plot_improvement(results, args.wind_speeds,
                     os.path.join(RESULTS_DIR, "fig4_improvement.png"))

    # Performance table
    save_performance_table(results, args.wind_speeds,
                           os.path.join(RESULTS_DIR, "performance_table.csv"))

    print(f"\nAll outputs saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()