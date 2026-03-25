"""
PINN Geometric Trainer — Differential Flatness Labels
======================================================

Key advantage over LQR+FF teacher
-----------------------------------
LQR+FF computes a static wind-compensation lean angle:
    phi_wind   = atan2(Fy/m, g)
    theta_wind = atan2(-Fx/m, g)

This works for hover but misses two effects on curved trajectories:

  1. Trajectory acceleration feedforward (a_ref):
     On a lemniscate at A=2m, ω=0.5 rad/s, centripetal acc ≈ 0.5 m/s².
     LQR+FF completely ignores this — the drone always leans ONLY into wind,
     never into the centripetal direction.
     Geometric control adds a_ref explicitly, so the label says:
       "lean into wind AND lean into the curve at the same time"

  2. Exact nonlinear force decomposition (not linearized):
     LQR+FF uses linear approximation: T_eq ≈ mg*sqrt(1 + (ax/g)² + (ay/g)²)
     Geometric control uses: T = ||f_des|| with full nonlinear trig.
     The difference is small at low wind, significant (>5%) at high wind.

Label formula (differential flatness inverse dynamics)
-------------------------------------------------------
Given desired acceleration:
    a_des = a_ref - Kp_pos * (pos - pos_ref) - Kv_pos * (vel - vel_ref)

Desired force (world frame), solving Newton's 2nd law:
    m*a_des = R@[0,0,T] + F_wind - [0,0,m*g]
    f_des   = m*(a_des + [0,0,g]) - F_wind      ← body z-axis direction × T

Thrust magnitude and desired attitude:
    T         = ||f_des||
    z_des     = f_des / T
    phi_des   = atan2(-z_des[1], z_des[2])      ← exact roll from desired z
    theta_des = atan2(z_des[0], sqrt(z_des[1]²+z_des[2]²))  ← exact pitch

Inner attitude PD (same as LQR+FF):
    tau = Kp_att * (att_des - att) - Kd_att * omega

Input dim:  24 = state(12) + wind(3) + ref(6) + acc_ref(3)
Output dim:  4 = [T, tau_x, tau_y, tau_z]
Checkpoint: CHECKPOINTS/pinn_geo.pt
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
                    CHECKPOINT_DIR, PINN_GEO_CKPT)
from PINN.network import PINNNetwork, InputNormalizer
from PINN.losses import rotation_matrix_torch

# ── Constants ──────────────────────────────────────────────────────────────────
_GEO_INPUT_DIM = 24        # 12 state + 3 wind + 6 ref + 3 acc_ref
_MAX_SAFE_TILT = np.deg2rad(50)
_MAX_TILT      = 0.8       # rad — hard clamp on desired attitude
_TAU_MAX       = 0.5       # N·m

# Cascade PD gains for label generation
# Outer loop (position → desired acceleration)
# Kp_pos matches PID (5,5,7) — keeps position correction from demanding extreme tilt
# at high wind (where wind compensation alone uses most of the tilt budget).
_KP_POS = 5.0     # position error gain  (m/s² per m)  — matches PID Kp_pos
_KV_POS = 4.0     # velocity error gain  (m/s² per m/s)

# Soft saturation on the error-correction term to prevent tilt budget overflow.
# At 8 m/s wind the drone needs ~39° lean just for compensation.
# Without this cap, large position errors push total demand above MAX_TILT.
_A_CORR_MAX = 5.0  # m/s² — max acceleration from position+velocity correction

# Inner loop (attitude → torques)
_KP_ATT = 15.0    # attitude error gain  (N·m per rad)
_KD_ATT =  4.0    # angular rate gain    (N·m per rad/s)


class PINNGeoTrainer:
    """
    PINN trainer with geometric (differential flatness) labels.

    The 24-dim input includes reference acceleration (a_ref) which allows
    the label to include trajectory centripetal forces — the key advantage
    over the 21-dim LQR+FF teacher.
    """

    def __init__(self, net=None, normalizer=None):
        self.net        = net        or PINNNetwork(input_dim=_GEO_INPUT_DIM)
        self.normalizer = normalizer or InputNormalizer()
        self.optimizer  = optim.Adam(self.net.parameters(), lr=PINN_LR)
        self.scheduler  = optim.lr_scheduler.StepLR(
                              self.optimizer,
                              step_size=LR_DECAY_STEP,
                              gamma=LR_DECAY_GAMMA)
        self.history = []
        print(f"PINNGeoTrainer: input_dim={_GEO_INPUT_DIM}, "
              f"Kp_pos={_KP_POS}, Kv_pos={_KV_POS}, a_corr_max={_A_CORR_MAX}, "
              f"Kp_att={_KP_ATT}, Kd_att={_KD_ATT}")

    # ── Quadrotor dynamics (same as PINNFreeTrainer) ───────────────────────────

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

    # ── Data preparation ────────────────────────────────────────────────────────

    def prepare_data(self, X):
        self.normalizer.fit(X)
        print(f"Normaliser fitted on {len(X):,} samples (dim={X.shape[1]})")

    # ── Geometric label computation ─────────────────────────────────────────────

    def _compute_geo_labels(self, state, wind, ref, acc_ref):
        """
        Compute geometric (differential flatness) control labels.

        Args:
            state   : (B, 12) — [pos(3), euler(3), vel(3), omega(3)]
            wind    : (B, 3)  — wind force in N
            ref     : (B, 6)  — [pos_ref(3), vel_ref(3)]
            acc_ref : (B, 3)  — reference acceleration (centripetal + tangential)

        Returns:
            u_label : (B, 4) — [T, tau_x, tau_y, tau_z], clamped to physical limits
        """
        B = state.shape[0]

        # ── Position and velocity errors ───────────────────────────────────────
        pos_err = state[:, 0:3] - ref[:, 0:3]    # (B, 3)
        vel_err = state[:, 6:9] - ref[:, 3:6]    # (B, 3)

        # ── Desired acceleration: trajectory feedforward + error correction ─────
        # a_des = a_ref - Kp*pos_err - Kv*vel_err
        # This is the key: a_ref includes centripetal/tangential forces along
        # the reference path that LQR+FF completely ignores.
        #
        # Soft saturation on the correction term prevents tilt budget overflow
        # at high wind: when strong wind already demands a large lean angle,
        # unclamped position error would push total demand above MAX_TILT.
        a_corr     = _KP_POS * pos_err + _KV_POS * vel_err   # (B, 3)
        corr_norm  = torch.norm(a_corr, dim=1, keepdim=True).clamp(min=1e-8)
        scale      = torch.clamp(corr_norm, max=_A_CORR_MAX) / corr_norm
        a_corr     = a_corr * scale                            # soft-clamped correction

        a_des = acc_ref - a_corr                               # (B, 3)

        # ── Desired force in world frame (inverse dynamics) ────────────────────
        # From Newton: m*a = R@[0,0,T] + F_wind - [0,0,m*g]
        # So:  R@[0,0,T] = m*(a_des + [0,0,g]) - F_wind = f_des
        g_vec   = torch.tensor([0.0, 0.0, float(G)], dtype=state.dtype)
        f_des   = MASS * (a_des + g_vec) - wind   # (B, 3)

        # ── Thrust magnitude ──────────────────────────────────────────────────
        T_mag = torch.norm(f_des, dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1)

        # ── Desired attitude from desired z-body direction ─────────────────────
        # For ZYX Euler (psi≈0): R@[0,0,T] = T*[sin_theta*cos_phi, -sin_phi, cos_theta*cos_phi]
        # So z_des = f_des / T defines the desired body z-axis.
        z_des   = f_des / T_mag                   # (B, 3), unit vector

        phi_des   = torch.atan2(-z_des[:, 1],
                                 z_des[:, 2])      # (B,) exact roll
        theta_des = torch.atan2( z_des[:, 0],
                                 torch.sqrt(z_des[:, 1]**2 + z_des[:, 2]**2 + 1e-12))  # (B,)

        # Hard clamp on desired attitude (physical safety)
        phi_des   = phi_des.clamp(-_MAX_TILT, _MAX_TILT)
        theta_des = theta_des.clamp(-_MAX_TILT, _MAX_TILT)

        # ── Inner PD attitude loop → torques ───────────────────────────────────
        att     = state[:, 3:6]                    # (B, 3) [phi, theta, psi]
        omega   = state[:, 9:12]                   # (B, 3) [p, q, r]

        att_des = torch.stack([phi_des, theta_des,
                               torch.zeros(B, dtype=state.dtype)], dim=1)  # (B, 3)

        att_err = att_des - att                    # (B, 3)
        att_err = (att_err + torch.pi) % (2 * torch.pi) - torch.pi   # wrap to [-π, π]

        tau_des = _KP_ATT * att_err - _KD_ATT * omega   # (B, 3)

        # ── Assemble and clamp ─────────────────────────────────────────────────
        u_label = torch.cat([T_mag, tau_des], dim=1)   # (B, 4)

        u_label = torch.stack([
            u_label[:, 0].clamp(0.0, 4.0 * MASS * G),
            u_label[:, 1].clamp(-_TAU_MAX, _TAU_MAX),
            u_label[:, 2].clamp(-_TAU_MAX, _TAU_MAX),
            u_label[:, 3].clamp(-_TAU_MAX, _TAU_MAX),
        ], dim=1)

        return u_label

    # ── Single training step ────────────────────────────────────────────────────

    def _train_step(self, X_all, batch_size, epoch):
        self.net.train()
        self.optimizer.zero_grad()

        # ── Random batch ──────────────────────────────────────────────────────
        idx    = np.random.randint(0, len(X_all), batch_size)
        X_np   = X_all[idx]

        X_norm  = torch.tensor(self.normalizer.transform(X_np), dtype=torch.float32)
        state   = torch.tensor(X_np[:, 0:12],  dtype=torch.float32)
        wind    = torch.tensor(X_np[:, 12:15], dtype=torch.float32)
        ref     = torch.tensor(X_np[:, 15:21], dtype=torch.float32)  # [pos(3), vel(3)]
        acc_ref = torch.tensor(X_np[:, 21:24], dtype=torch.float32)
        B       = state.shape[0]

        # ── Forward pass ──────────────────────────────────────────────────────
        u_pred = self.net(X_norm)   # (B, 4)

        # ── One-step rollout for physics and safety losses ─────────────────────
        state_next = self._rk4_step(state, u_pred, wind, dt=DT)

        # ── Geometric label ────────────────────────────────────────────────────
        u_des = self._compute_geo_labels(state, wind, ref, acc_ref)

        # ── l_imitation: direct supervision ───────────────────────────────────
        l_imitation = ((u_pred - u_des) ** 2).mean()

        # ── l_ref: one-step tracking consistency ──────────────────────────────
        # After one step, velocity should be close to reference
        ref_pos_next = ref[:, 0:3] + ref[:, 3:6] * DT
        l_ref_vel    = ((state_next[:, 6:9] - ref[:, 3:6]) ** 2).mean()
        l_ref_pos    = ((state_next[:, 0:3] - ref_pos_next) ** 2).mean()
        l_ref        = l_ref_vel + 0.1 * l_ref_pos

        # ── l_smooth: torque regularisation ───────────────────────────────────
        l_smooth = 0.02 * (u_pred[:, 1:] ** 2).mean()

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

    # ── Full training loop ──────────────────────────────────────────────────────

    def train(self, X,
              epochs     = PINN_EPOCHS,
              batch_size = PINN_BATCH_SIZE,
              log_every  = 100,
              ckpt_path  = None):
        """
        Train on dataset X.

        Parameters
        ----------
        X          : (N, 24) float32 — from generate_from_simulation(include_acc=True)
        epochs     : training epochs
        batch_size : batch size per gradient step
        log_every  : print frequency
        ckpt_path  : checkpoint save path
        """
        assert X.shape[1] == _GEO_INPUT_DIM, (
            f"Expected 24-dim input (state+wind+ref+acc_ref), got {X.shape[1]}.\n"
            f"Generate data with include_acc=True.")

        self.prepare_data(X)
        ckpt_path = ckpt_path or PINN_GEO_CKPT
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        print(f"\nStarting Geometric PINN training: {epochs} epochs, batch={batch_size}")
        print(f"Network params  : {self.net.count_parameters():,}")
        print(f"Dataset size    : {len(X):,} samples (24-dim)")
        print(f"Checkpoint      : {ckpt_path}")
        print(f"Label: a_corr = clip(Kp*err+Kv*vel_err, max={_A_CORR_MAX} m/s²)")
        print(f"       f_des = m*(a_des+g) - F_wind")
        print(f"       T = ||f_des||,  att from exact trig decomp")
        print(f"       tau = {_KP_ATT}*att_err - {_KD_ATT}*omega\n")

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

    # ── Save / load ─────────────────────────────────────────────────────────────

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'net_state_dict'  : self.net.state_dict(),
            'normalizer_mean' : self.normalizer.mean,
            'normalizer_std'  : self.normalizer.std,
            'history'         : self.history,
            'input_dim'       : _GEO_INPUT_DIM,
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
        axes[0].set_title("Geo PINN — Total loss"); axes[0].set_yscale('log')
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
        out_path = save_path or os.path.join(_root, "RESULTS", "pinn_geo_loss.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=120)
        print(f"Loss plot saved to {out_path}")


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== PINNGeoTrainer Sanity Check ===\n")

    from PINN.data_generator import generate_from_simulation

    X = generate_from_simulation(
        n_points      = 2_000,
        wind_speed_max = 5.0,
        traj_types    = ["lemniscate"],
        include_acc   = True,          # ← 24-dim output
    )
    print(f"Dataset shape: {X.shape}  (should be (2000, 24))\n")

    trainer = PINNGeoTrainer()
    trainer.train(X, epochs=30, batch_size=64, log_every=10,
                  ckpt_path="CHECKPOINTS/pinn_geo_test.pt")

    print("\nSanity check passed!")
