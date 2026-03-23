"""
PINN Controller

Wraps the trained PINNNetwork as a controller with the same interface
as PID and LQR: compute_control(state, ref, wind_force) -> u (4,)

This is the only controller that receives wind_force as a real input
(not just ignoring it). That feedforward path is what makes PINN
potentially outperform LQR under strong wind disturbances.
"""

import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import PINN_FREE_CKPT, MASS, G
from CONTROLLERS.base import BaseController
from PINN.network import PINNNetwork, InputNormalizer, build_network_input


class PINNController(BaseController):
    """
    PINN-based controller.

    At inference time, each call to compute_control() does:
        1. Concatenate [state(12), wind(3), ref(6)] → input (21,)
        2. Normalize input using fitted normalizer
        3. One forward pass through the network
        4. Return [T, tau_x, tau_y, tau_z]

    No optimization, no planning — just one matrix multiply chain.
    This is why inference is orders of magnitude faster than MPC.
    """

    def __init__(self, ckpt_path=None):
        self.net        = PINNNetwork()
        self.normalizer = InputNormalizer()
        self.loaded     = False

        ckpt_path = ckpt_path or PINN_FREE_CKPT
        if os.path.exists(ckpt_path):
            self._load(ckpt_path)
        else:
            print(f"[PINNController] No checkpoint found at '{ckpt_path}'. "
                  f"Using random weights — train first with train_pinn.py")

        self.net.eval()    # disable dropout / batchnorm if any

    def _load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.net.load_state_dict(ckpt['net_state_dict'])
        self.normalizer.mean   = ckpt['normalizer_mean']
        self.normalizer.std    = ckpt['normalizer_std']
        self.normalizer.fitted = True
        self.loaded = True
        print(f"[PINNController] Loaded checkpoint from '{ckpt_path}'")

    @property
    def name(self):
        return "PINN"

    def compute_control(self, state, ref, wind_force=None):
        """
        Args:
            state      : (12,) current drone state
            ref        : (6,)  reference [pos(3), vel(3)]
            wind_force : (3,)  wind force in N  ← PINN actually uses this!

        Returns:
            u : (4,) [T, tau_x, tau_y, tau_z]
        """
        if wind_force is None:
            wind_force = np.zeros(3)

        # build 21-dim input vector
        x = build_network_input(state, wind_force, ref)

        # normalize if normalizer is fitted
        if self.normalizer.fitted:
            x = self.normalizer.transform(x)

        # forward pass (no grad needed at inference)
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1,21)
            u_t = self.net(x_t)                                        # (1,4)
            u   = u_t.squeeze(0).numpy()                               # (4,)

        return u


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from config import N_STEPS, DT
    from SIMULATION.quad_model import QuadrotorModel
    from SIMULATION.trajectory import make_trajectory
    from SIMULATION.wind import make_wind
    from CONTROLLERS.lqr import LQRController

    sanity_dir = os.path.join(os.path.dirname(__file__), '..', 'SANITY_CHECK')
    os.makedirs(sanity_dir, exist_ok=True)

    print("=== PINN Controller Sanity Check ===\n")

    # ── Try to load trained checkpoint ────────────────────────────────────
    pinn = PINNController()

    if not pinn.loaded:
        print("No trained checkpoint found.")
        print("Running with random weights to verify pipeline works.\n")

    # ── Run simulation: PINN vs LQR side by side ──────────────────────────
    traj     = make_trajectory("circle")
    wind_5   = make_wind("constant", wind_speed=5.0)

    controllers = {
        "LQR (no wind)"  : (LQRController(), make_wind("none")),
        "LQR (wind 5m/s)": (LQRController(), make_wind("constant", wind_speed=5.0)),
        "PINN (wind 5m/s)": (pinn,           make_wind("constant", wind_speed=5.0)),
    }

    results = {}
    t_arr   = np.arange(N_STEPS) * DT

    for label, (ctrl, wind) in controllers.items():
        quad = QuadrotorModel()
        init_state = np.zeros(12)
        init_state[0:3] = traj.get(0)[0]
        quad.reset(init_state)
        wind.reset()

        pos_log = np.zeros((N_STEPS, 3))
        ref_log = np.zeros((N_STEPS, 3))
        err_log = np.zeros(N_STEPS)

        for i, t in enumerate(t_arr):
            ref        = traj.get_full(t)
            wind_force = wind.step()
            u          = ctrl.compute_control(quad.state, ref, wind_force)
            state, done = quad.step(u, wind_force)

            pos_log[i] = state[0:3]
            ref_log[i] = ref[0:3]
            err_log[i] = np.linalg.norm(state[0:3] - ref[0:3])
            if done:
                print(f"  {label}: crashed at t={t:.2f}s")
                err_log[i:] = err_log[i]
                break

        rmse = np.sqrt(np.mean(err_log**2))
        results[label] = dict(pos=pos_log, ref=ref_log, err=err_log, rmse=rmse)
        print(f"{label:25s}: RMSE = {rmse:.4f} m")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ['#1D9E75', '#E24B4A', '#534AB7']

    axes[0].plot(results['LQR (no wind)']['ref'][:, 0],
                 results['LQR (no wind)']['ref'][:, 1],
                 '--', color='#888780', linewidth=1.5, label='reference')

    for (label, res), color in zip(results.items(), colors):
        axes[0].plot(res['pos'][:, 0], res['pos'][:, 1],
                     color=color, linewidth=1.2, label=label)

    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title("XY trajectory")
    axes[0].legend(fontsize=8)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    for (label, res), color in zip(results.items(), colors):
        axes[1].plot(t_arr, res['err'],
                     color=color, linewidth=1.2,
                     label=f"{label} (RMSE={res['rmse']:.3f}m)")

    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("position error (m)")
    axes[1].set_title("Tracking error comparison")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(sanity_dir, "pinn_controller_check.png")
    plt.savefig(out, dpi=120)
    print(f"\nPlot saved to {out}")
    print("\nPipeline verified. Train PINN with train_pinn.py to get real results.")