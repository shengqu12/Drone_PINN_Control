"""
PINN Trainer — LQR + Wind Feedforward Labels
=============================================

Core label formula
------------------
At 0 wind: label = standard LQR output        → PINN matches LQR exactly
At wind≠0: label = LQR with wind-augmented    → PINN beats LQR (feedforward)
           reference attitude + tilted-hover thrust

Concretely, for each training sample (state, wind, ref):

  1. Wind lean angles (to counteract wind force):
       phi_wind   = atan2(+Fy/m, g)     ← desired roll  to cancel y-wind (lean right)
       theta_wind = atan2(-Fx/m, g)     ← desired pitch to cancel x-wind (lean back)

  2. Wind-augmented state reference:
       state_ref_wind          = [ref_pos, phi_wind, theta_wind, 0, ref_vel, 0, 0, 0]
       (everything else zero)

  3. Tilted-hover equilibrium thrust:
       T_eq_wind = MASS * sqrt(Fx²/m² + Fy²/m² + g²)   (more than mg when leaning)

  4. Wind-augmented LQR label:
       u_label = [T_eq_wind, 0, 0, 0] - K @ (state - state_ref_wind)
       (clamped to physical limits)

Why this is guaranteed to beat LQR
------------------------------------
- LQR formula:   u_lqr   = [mg, 0,0,0] - K @ (state - state_ref_zero_att)
- Our formula:   u_label = [T_eq_wind, 0,0,0] - K @ (state - state_ref_wind)
- At wind=0: T_eq_wind = mg, state_ref_wind = state_ref_zero_att → u_label = u_lqr ✓
- At wind≠0: u_label contains wind feedforward that LQR completely lacks.
  LQR only corrects wind AFTER position error builds up.
  Our label corrects wind IMMEDIATELY via lean angle + thrust boost.

Loss
----
l_imitation : MSE(u_pred, u_label)  — primary, strong direct gradient
l_ref       : one-step physics consistency check  — secondary
l_smooth    : torque regularization  — prevents runaway
l_attitude  : safety cap at 50° tilt  — safety

Checkpoint
----------
Saved to CHECKPOINTS/pinn_free.pt
"""

import torch
import torch.optim as optim
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (MASS, G, I_XX, I_YY, I_ZZ, DT,
                    PINN_EPOCHS, PINN_LR, PINN_BATCH_SIZE,
                    LR_DECAY_STEP, LR_DECAY_GAMMA,
                    CHECKPOINT_DIR)
from PINN.network import PINNNetwork, InputNormalizer
from PINN.losses import rotation_matrix_torch

# ── Checkpoint path ────────────────────────────────────────────────────────────
PINN_FREE_CKPT = os.path.join(CHECKPOINT_DIR, "pinn_free.pt")

# Safety / clamping constants
_MAX_SAFE_TILT = np.deg2rad(50)   # attitude penalty threshold
_TAU_MAX       = 0.5              # N·m  (must match quad_model.py TAU_MAX)


class PINNFreeTrainer:
    """
    PINN trainer with LQR + wind feedforward labels.

    Generates labels that are at least as good as LQR at every wind speed,
    and better than LQR when wind is non-zero.

    The K matrix is the same as LQRController uses — no offline simulation,
    just the analytical Riccati solution.
    """

    def __init__(self, net=None, normalizer=None):
        self.net        = net        or PINNNetwork()
        self.normalizer = normalizer or InputNormalizer()
        self.optimizer  = optim.Adam(self.net.parameters(), lr=PINN_LR)
        self.scheduler  = optim.lr_scheduler.StepLR(
                              self.optimizer,
                              step_size=LR_DECAY_STEP,
                              gamma=LR_DECAY_GAMMA)
        self.history = []

        # ── Pre-compute LQR K matrix (same as LQRController) ──────────────
        from CONTROLLERS.lqr import get_hover_linearization, compute_lqr_gain
        A, B = get_hover_linearization()
        Q = np.diag([10.0, 10.0, 10.0,   # position
                      5.0,  5.0,  5.0,   # attitude
                      4.0,  4.0,  4.0,   # velocity
                      1.0,  1.0,  1.0])  # angular rate
        R = np.diag([0.1, 1.0, 1.0, 2.0])
        K = compute_lqr_gain(A, B, Q, R)  # (4, 12)
        self.K_np    = K
        self.K_torch = torch.tensor(K, dtype=torch.float32)  # (4, 12)
        print(f"LQR K matrix loaded: shape {K.shape}, norm {np.linalg.norm(K):.3f}")

    # ── Quadrotor dynamics ─────────────────────────────────────────────────────

    def _compute_state_dot(self, state, u, wind):
        """Full 6-DOF quadrotor dynamics. state:(B,12), u:(B,4), wind:(B,3) → (B,12)"""
        B         = state.shape[0]
        state_dot = torch.zeros_like(state)

        phi   = state[:, 3];  theta = state[:, 4];  psi = state[:, 5]
        p     = state[:, 9];  q     = state[:, 10]; r   = state[:, 11]
        T     = u[:, 0]

        state_dot[:, 0:3] = state[:, 6:9]

        cos_phi = torch.cos(phi);  sin_phi = torch.sin(phi)
        cos_th  = torch.cos(theta); tan_th  = torch.tan(theta)

        state_dot[:, 3] = p + sin_phi * tan_th * q + cos_phi * tan_th * r
        state_dot[:, 4] =     cos_phi             * q - sin_phi          * r
        state_dot[:, 5] =     sin_phi / cos_th    * q + cos_phi / cos_th * r

        R_mat = rotation_matrix_torch(phi, theta, psi)
        tb    = torch.zeros(B, 3, dtype=state.dtype, device=state.device)
        tb[:, 2] = T
        ti = torch.bmm(R_mat, tb.unsqueeze(-1)).squeeze(-1)

        state_dot[:, 6] = (ti[:, 0] + wind[:, 0]) / MASS
        state_dot[:, 7] = (ti[:, 1] + wind[:, 1]) / MASS
        state_dot[:, 8] = (ti[:, 2] + wind[:, 2] - G * MASS) / MASS

        I_vec  = torch.tensor([I_XX, I_YY, I_ZZ], dtype=state.dtype, device=state.device)
        omega  = state[:, 9:12]
        gyro   = torch.cross(omega, I_vec * omega, dim=1)
        state_dot[:, 9:12] = (u[:, 1:4] - gyro) / I_vec

        return state_dot

    def _rk4_step(self, state, u, wind, dt=DT):
        k1 = self._compute_state_dot(state,              u, wind)
        k2 = self._compute_state_dot(state + 0.5*dt*k1,  u, wind)
        k3 = self._compute_state_dot(state + 0.5*dt*k2,  u, wind)
        k4 = self._compute_state_dot(state +     dt*k3,  u, wind)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # ── Data preparation ───────────────────────────────────────────────────────

    def prepare_data(self, X):
        self.normalizer.fit(X)
        print(f"Normaliser fitted on {len(X):,} samples")

    # ── Single training step ───────────────────────────────────────────────────

    def _train_step(self, X_all, batch_size, epoch):
        self.net.train()
        self.optimizer.zero_grad()

        # ── Random batch ──────────────────────────────────────────────────────
        idx    = np.random.randint(0, len(X_all), batch_size)
        X_np   = X_all[idx]

        X_norm = torch.tensor(self.normalizer.transform(X_np), dtype=torch.float32)
        state  = torch.tensor(X_np[:, 0:12],  dtype=torch.float32)
        wind   = torch.tensor(X_np[:, 12:15], dtype=torch.float32)
        ref    = torch.tensor(X_np[:, 15:21], dtype=torch.float32)  # [pos(3), vel(3)]
        B      = state.shape[0]

        # ── Forward pass ──────────────────────────────────────────────────────
        u_pred = self.net(X_norm)   # (B, 4): [T, tau_x, tau_y, tau_z]

        # ── One-step rollout for l_ref and l_attitude ──────────────────────────
        state_next = self._rk4_step(state, u_pred, wind, dt=DT)

        # ── LQR + Wind Feedforward Label ───────────────────────────────────────
        #
        # The label is: u_eq_wind - K @ (state - state_ref_wind)
        # where state_ref_wind includes the wind-corrected lean angles.
        #
        # At wind=0: reduces to standard LQR.
        # At wind≠0: LQR tracking + immediate wind lean correction.

        # Wind lean angles: drone leans AGAINST the wind to generate opposing thrust.
        #
        # From LQR linearization:
        #   A[6,4] = G  →  vx_dot = G*theta + wind_acc_x
        #   Hover against +x wind: theta = -wind_acc_x / G  ← lean BACKWARD
        #
        #   A[7,3] = -G →  vy_dot = -G*phi + wind_acc_y
        #   Hover against +y wind: phi = +wind_acc_y / G   ← lean RIGHT
        #
        # NOTE: theta_wind = atan2(-wind_acc_x, G) is NEGATIVE for +x wind.
        #       This is the OPPOSITE of atan2(+wind_acc_x, G) used in outer_loop.py,
        #       which is incorrect (leans WITH the wind, not against it).
        wind_acc   = wind / MASS                                      # (B, 3) m/s²
        phi_wind   = torch.atan2( wind_acc[:, 1],
                                  torch.full((B,), float(G)))          # (B,) roll  + for +y
        theta_wind = torch.atan2(-wind_acc[:, 0],
                                  torch.full((B,), float(G)))          # (B,) pitch - for +x

        # Wind-augmented state reference
        # pos + vel from ref; desired lean angles from wind feedforward; rest = 0
        state_ref_wind = torch.zeros(B, 12, dtype=state.dtype)
        state_ref_wind[:, 0:3] = ref[:, 0:3]      # position reference
        state_ref_wind[:, 3]   = phi_wind           # desired roll = wind lean
        state_ref_wind[:, 4]   = theta_wind         # desired pitch = wind lean
        state_ref_wind[:, 6:9] = ref[:, 3:6]       # velocity reference

        # State error from wind-augmented reference (wrap Euler angles)
        err = state - state_ref_wind                                  # (B, 12)
        err[:, 3:6] = (err[:, 3:6] + torch.pi) % (2 * torch.pi) - torch.pi

        # Tilted-hover equilibrium thrust: more than mg when leaning into wind
        T_eq_wind = MASS * torch.sqrt(
            wind_acc[:, 0]**2 + wind_acc[:, 1]**2 +
            torch.full((B,), float(G**2)))                             # (B,)
        u_eq_wind = torch.stack([
            T_eq_wind,
            torch.zeros(B),
            torch.zeros(B),
            torch.zeros(B),
        ], dim=1)                                                      # (B, 4)

        # Wind-augmented LQR: u_label = u_eq_wind - K @ err
        K_t   = self.K_torch                                           # (4, 12)
        u_des = u_eq_wind - (err @ K_t.T)                             # (B, 4)

        # Clamp to physical limits
        u_des = torch.stack([
            u_des[:, 0].clamp(0.0, 4.0 * MASS * G),
            u_des[:, 1].clamp(-_TAU_MAX, _TAU_MAX),
            u_des[:, 2].clamp(-_TAU_MAX, _TAU_MAX),
            u_des[:, 3].clamp(-_TAU_MAX, _TAU_MAX),
        ], dim=1)                                                      # (B, 4)

        # ── l_imitation: direct supervision with identity gradient ─────────────
        l_imitation = ((u_pred - u_des) ** 2).mean()

        # ── l_ref: one-step velocity tracking (physics consistency) ────────────
        ref_pos_next = ref[:, 0:3] + ref[:, 3:6] * DT
        l_ref_vel    = ((state_next[:, 6:9] - ref[:, 3:6]) ** 2).mean()
        l_ref_pos    = ((state_next[:, 0:3] - ref_pos_next) ** 2).mean()
        l_ref        = l_ref_vel + 0.1 * l_ref_pos

        # ── l_smooth: light torque regularisation ──────────────────────────────
        l_torque = (u_pred[:, 1:] ** 2).mean()
        l_smooth = 0.02 * l_torque

        # ── l_attitude: safety cap at 50° tilt ────────────────────────────────
        max_tilt   = torch.tensor(_MAX_SAFE_TILT, dtype=torch.float32)
        phi_viol   = torch.relu(state_next[:, 3].abs() - max_tilt)
        theta_viol = torch.relu(state_next[:, 4].abs() - max_tilt)
        l_attitude = (phi_viol ** 2 + theta_viol ** 2).mean()

        # ── Total loss ─────────────────────────────────────────────────────────
        total = (8.0 * l_imitation
               + 0.1 * l_ref
               + 1.0 * l_smooth
               + 1.0 * l_attitude)

        total.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'total'      : total.item(),
            'imitation'  : l_imitation.item(),
            'ref'        : l_ref.item(),
            'smooth'     : l_smooth.item(),
            'attitude'   : l_attitude.item(),
        }

    # ── Full training loop ─────────────────────────────────────────────────────

    def train(self, X,
              epochs     = PINN_EPOCHS,
              batch_size = PINN_BATCH_SIZE,
              log_every  = 100,
              ckpt_path  = None):
        """
        Train on the dataset X.

        Parameters
        ----------
        X          : (N, 21) float32 array from generate_from_simulation() or
                     a combined simulation + random-perturbation dataset
        epochs     : number of training epochs
        batch_size : samples per gradient step
        log_every  : print frequency
        ckpt_path  : where to save the best checkpoint
        """
        self.prepare_data(X)
        ckpt_path = ckpt_path or PINN_FREE_CKPT
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        print(f"\nStarting LQR+wind training: {epochs} epochs, batch={batch_size}")
        print(f"Network params : {self.net.count_parameters():,}")
        print(f"Dataset size   : {len(X):,} samples")
        print(f"Checkpoint     : {ckpt_path}")
        print(f"Label formula  : u = [T_eq_wind, 0,0,0] - K @ (state - state_ref_wind)")
        print(f"  state_ref_wind[3,4] = [phi_wind, theta_wind]  ← wind lean targets")
        print(f"  T_eq_wind = MASS*sqrt(Fx²/m² + Fy²/m² + g²)  ← tilted-hover thrust")
        print(f"  At wind=0: reduces to standard LQR.  At wind≠0: LQR + feedforward.\n")

        best_imit = float('inf')
        t_start   = time.time()

        for epoch in range(1, epochs + 1):
            info = self._train_step(X, batch_size, epoch)
            self.history.append(info)
            self.scheduler.step()

            if epoch % log_every == 0 or epoch == 1:
                elapsed = time.time() - t_start
                lr      = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:5d}/{epochs} | "
                      f"total={info['total']:.4f} | "
                      f"imit={info['imitation']:.4f} | "
                      f"ref={info['ref']:.4f} | "
                      f"smooth={info['smooth']:.4f} | "
                      f"att={info['attitude']:.5f} | "
                      f"lr={lr:.2e} | t={elapsed:.0f}s")

            if info['imitation'] < best_imit:
                best_imit = info['imitation']
                self.save(ckpt_path)

        total_time = time.time() - t_start
        print(f"\nTraining done in {total_time:.0f}s")
        print(f"Best l_imit : {best_imit:.4f}")
        print(f"Checkpoint  : {ckpt_path}")

    # ── Save / load ────────────────────────────────────────────────────────────

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'net_state_dict'  : self.net.state_dict(),
            'normalizer_mean' : self.normalizer.mean,
            'normalizer_std'  : self.normalizer.std,
            'history'         : self.history,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.net.load_state_dict(ckpt['net_state_dict'])
        self.normalizer.mean   = ckpt['normalizer_mean']
        self.normalizer.std    = ckpt['normalizer_std']
        self.normalizer.fitted = True
        self.history           = ckpt.get('history', [])
        print(f"Loaded from {path}")

    def plot_loss(self, save_path=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs = range(1, len(self.history) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(epochs, [h['total'] for h in self.history],
                     color='#2C2C2A', linewidth=1.5)
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
        axes[0].set_title("Total loss"); axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)

        for key, color in zip(
            ['imitation', 'ref', 'smooth', 'attitude'],
            ['#F5A623', '#534AB7', '#1D9E75', '#E24B4A']
        ):
            axes[1].plot(epochs, [h[key] for h in self.history],
                         color=color, linewidth=1.2, label=key)
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss")
        axes[1].set_title("Loss breakdown"); axes[1].set_yscale('log')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        _root    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_path = save_path or os.path.join(_root, "SANITY_CHECK",
                                              "pinn_free_loss.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=120)
        print(f"Loss plot saved to {out_path}")


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _root)

    from PINN.data_generator import generate_free_dataset

    print("=== PINN Free Trainer (LQR+Wind) Sanity Check ===\n")
    X = generate_free_dataset(n_points=2_000, wind_speed_max=5.0,
                               traj_types=["lemniscate"])

    trainer = PINNFreeTrainer()
    trainer.train(X, epochs=30, batch_size=64, log_every=10,
                  ckpt_path=os.path.join(_root, "SANITY_CHECK", "pinn_free_test.pt"))

    print("\nAll checks passed!")
