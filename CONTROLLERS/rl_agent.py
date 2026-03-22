"""
RL (PPO) Controller - Improved with LQR warm-start

Key insight: instead of learning from scratch (random exploration),
we initialize the PPO policy to mimic LQR first, then fine-tune
with RL rewards. This is called "imitation pre-training" and
dramatically reduces the number of steps needed to converge.

Pipeline:
  1. Pre-train policy via behavior cloning on LQR data (supervised)
  2. Fine-tune with PPO using environment rewards
  3. Result: stable flight from step 1, then gradually improve
"""

import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (DT, N_STEPS, MASS, G, CHECKPOINT_DIR, RL_CKPT)
from CONTROLLERS.base import BaseController

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.policies import ActorCriticPolicy
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    import types
    gym = types.ModuleType('gym')
    gym.Env = object
    spaces = None
    print("[RL] stable-baselines3 not installed.")


# ─── Environment ──────────────────────────────────────────────────────────────

class QuadrotorEnv(gym.Env):
    """
    Quadrotor environment with dense reward shaping.

    Observation (21-dim): [state(12), wind(3), ref(6)]  — same as PINN
    Action (4-dim):       normalized [-1, 1]
    """

    metadata = {'render_modes': []}

    def __init__(self,
                 wind_speed_min=0.0,
                 wind_speed_max=10.0,
                 traj_type="lemniscate"):
        super().__init__()

        from SIMULATION.quad_model import QuadrotorModel
        from SIMULATION.trajectory import make_trajectory

        self.quad           = QuadrotorModel()
        self.traj           = make_trajectory(traj_type)
        self.wind_speed_min = wind_speed_min
        self.wind_speed_max = wind_speed_max

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Increased bounds: at 15 m/s wind the drone can drift several meters
        # before recovering, so ±50 would clip valid observations.
        self.observation_space = spaces.Box(
            low=-200.0, high=200.0, shape=(21,), dtype=np.float32)

        self.T_max   = 4 * MASS * G
        self.tau_max = 0.5
        self._step   = 0
        self._wind_force = np.zeros(3)

    def _sample_wind(self):
        ws    = np.random.uniform(self.wind_speed_min, self.wind_speed_max)
        angle = np.random.uniform(0, 2 * np.pi)
        d     = np.array([np.cos(angle), np.sin(angle), 0.0])
        self._wind_force = d * ws * MASS

    def _get_obs(self):
        t   = self._step * DT
        ref = self.traj.get_full(t)
        return np.concatenate([
            self.quad.state, self._wind_force, ref
        ]).astype(np.float32)

    def _scale_action(self, action):
        T     = (action[0] + 1) / 2 * self.T_max
        tau_x = action[1] * self.tau_max
        tau_y = action[2] * self.tau_max
        tau_z = action[3] * self.tau_max
        return np.array([T, tau_x, tau_y, tau_z])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._sample_wind()

        # ── Random starting time on the trajectory ────────────────────────
        # Sampling t_0 ∈ [0, 0.7×T_TOTAL] exposes the agent to all phases of
        # the trajectory while guaranteeing at least 600 steps per episode.
        from config import T_TOTAL
        t0           = np.random.uniform(0.0, T_TOTAL * 0.7)
        self._step   = int(t0 / DT)

        ref_pos, _ = self.traj.get(t0)

        # ── Larger initial perturbation to train high-wind recovery ───────
        # Old value: 0.03 m (never sees the 0.5–2 m errors from high wind).
        # New: 0.3 m position + small attitude noise so agent learns recovery.
        init_state      = np.zeros(12)
        init_state[0:3] = ref_pos + np.random.randn(3) * 0.3   # position noise
        init_state[3:5] = np.random.randn(2) * 0.05            # small attitude noise
        self.quad.reset(init_state)
        self._prev_pos_err = np.linalg.norm(
            init_state[:3] - ref_pos)

        return self._get_obs(), {}

    def step(self, action):
        u          = self._scale_action(action)
        state, crashed = self.quad.step(u, self._wind_force)
        self._step += 1

        t   = self._step * DT
        ref = self.traj.get_full(t)

        pos_err = np.linalg.norm(state[0:3] - ref[0:3])
        vel_err = np.linalg.norm(state[6:9] - ref[3:6])
        phi, theta = state[3], state[4]

        # dense reward
        r_track    = -(pos_err + 0.1 * vel_err)

        # ── Adaptive attitude penalty ────────────────────────────────────
        # At high wind the drone MUST tilt significantly to counteract the
        # force.  A fixed -0.5*(phi²+theta²) at 10 m/s wind produces
        # ~-0.43/step — nearly cancelling the survival bonus — so the agent
        # never learns to lean into the wind.
        # FIX: scale the penalty by how much the current tilt is needed.
        #   Desired tilt to cancel wind = asin(|F_wind| / (m*g)) ≈ F_wind/mg
        #   Allow that tilt for free; penalise only tilt BEYOND what is needed.
        wind_acc    = np.linalg.norm(self._wind_force) / MASS   # m/s²
        tilt_needed = min(wind_acc / G, 0.95)                   # rad (approx)
        tilt_excess = max(0.0,
                          np.sqrt(phi**2 + theta**2) - tilt_needed)
        r_attitude  = -0.1 * tilt_excess**2    # only penalise excess tilt

        r_smooth   = -0.01 * np.sum(u[1:]**2)
        r_alive    = 0.5                             # survival bonus
        r_progress = 1.0 * (self._prev_pos_err - pos_err)  # progress signal

        reward = r_track + r_attitude + r_smooth + r_alive + r_progress
        self._prev_pos_err = pos_err

        if crashed:
            reward -= 100.0

        terminated = crashed
        truncated  = self._step >= N_STEPS

        return self._get_obs(), reward, terminated, truncated, {}


# ─── LQR pre-training (behavior cloning) ─────────────────────────────────────

def pretrain_on_lqr(model, n_samples=50000, n_epochs=20):
    """
    Pre-train the PPO policy network to imitate LQR.
    This gives RL a stable starting point instead of random exploration.

    Uses the same LQR rollout data as PINN training.
    """
    print(f"\nPre-training PPO policy on LQR data ({n_samples:,} samples)...")

    from PINN.data_generator import generate_supervised_data

    # generate LQR data
    X, Y = generate_supervised_data(
        wind_speeds=[0.0, 2.0, 5.0, 8.0, 10.0],
        wind_types=["none", "constant"],
        traj_types=["lemniscate"],
        n_directions=4,
    )

    # subsample
    idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    X, Y = X[idx], Y[idx]

    # normalize actions to [-1, 1] (same as env)
    T_max   = 4 * MASS * G
    tau_max = 0.5

    Y_norm = np.zeros_like(Y)
    Y_norm[:, 0] = 2 * Y[:, 0] / T_max - 1       # T: [0, T_max] → [-1, 1]
    Y_norm[:, 1] = Y[:, 1] / tau_max              # tau_x
    Y_norm[:, 2] = Y[:, 2] / tau_max              # tau_y
    Y_norm[:, 3] = Y[:, 3] / tau_max              # tau_z
    Y_norm = np.clip(Y_norm, -1, 1).astype(np.float32)

    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y_norm, dtype=torch.float32)

    # directly optimize the policy network's action output
    policy    = model.policy
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    dataset   = torch.utils.data.TensorDataset(X_t, Y_t)
    loader    = torch.utils.data.DataLoader(
                    dataset, batch_size=256, shuffle=True)

    policy.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            # get policy's mean action
            obs_t = xb
            with torch.no_grad():
                features = policy.extract_features(obs_t,
                               policy.features_extractor)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            mean_actions = policy.action_net(latent_pi)
            loss = ((mean_actions - yb) ** 2).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  BC epoch {epoch+1:3d}/{n_epochs} | loss={avg_loss:.5f}")

    print("Pre-training done.\n")
    return model


# ─── Training ─────────────────────────────────────────────────────────────────

def train_rl(total_timesteps=1_000_000,
             n_envs=8,
             save_path=None,
             pretrain=True):
    """
    Train PPO with optional LQR pre-training.

    pretrain=True:  BC on LQR data first → much faster convergence
    pretrain=False: learn from scratch (slow, often fails)

    Training wind range is 0–12 m/s (not 10) so the agent has seen slightly
    beyond the 10 m/s evaluation point and won't freeze at the distribution edge.
    """
    if not SB3_AVAILABLE:
        print("Cannot train: stable-baselines3 not installed.")
        return None

    save_path = save_path or RL_CKPT
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Training PPO: {total_timesteps:,} steps, {n_envs} parallel envs")
    print(f"LQR pre-training: {'ON' if pretrain else 'OFF'}")

    def make_env():
        # wind_speed_max=15 — trains beyond the 10 m/s evaluation point so
        # the agent has seen the high-wind regime, not just its edge.
        return QuadrotorEnv(wind_speed_min=0.0, wind_speed_max=15.0)

    env = DummyVecEnv([make_env for _ in range(n_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256, 128],
                vf=[256, 256, 128],
            ),
        ),
    )

    # pre-train on LQR data
    if pretrain:
        model = pretrain_on_lqr(model, n_samples=50000, n_epochs=30)

    # RL fine-tuning
    print(f"RL fine-tuning for {total_timesteps:,} steps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(save_path)
    print(f"RL model saved to {save_path}.zip")
    return model


def continue_rl(additional_timesteps=1_000_000,
                n_envs=8,
                ckpt_path=None):
    """
    Continue training an existing PPO checkpoint.

    Loads the saved policy and runs additional RL steps without
    re-doing LQR pre-training.  The step counter continues from
    where it left off (reset_num_timesteps=False).

    Usage:
        python train_rl.py --continue --steps 2000000
    """
    if not SB3_AVAILABLE:
        print("Cannot train: stable-baselines3 not installed.")
        return None

    ckpt_path = ckpt_path or RL_CKPT
    ckpt_file = ckpt_path + ".zip"
    if not os.path.exists(ckpt_file):
        print(f"[continue_rl] No checkpoint found at '{ckpt_file}'. "
              f"Run train_rl first.")
        return None

    def make_env():
        return QuadrotorEnv(wind_speed_min=0.0, wind_speed_max=15.0)

    env = DummyVecEnv([make_env for _ in range(n_envs)])

    print(f"Loading checkpoint: {ckpt_file}")
    model = PPO.load(ckpt_path, env=env)

    print(f"Continuing training for {additional_timesteps:,} additional steps...")
    model.learn(total_timesteps=additional_timesteps,
                reset_num_timesteps=False,   # keeps global step counter
                progress_bar=True)
    model.save(ckpt_path)
    print(f"Updated checkpoint saved to {ckpt_file}")
    return model


# ─── Controller wrapper ───────────────────────────────────────────────────────

class RLController(BaseController):

    def __init__(self, ckpt_path=None):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 not installed.")

        self.model  = None
        self.loaded = False

        ckpt_path = ckpt_path or RL_CKPT
        if os.path.exists(ckpt_path + ".zip"):
            self.model  = PPO.load(ckpt_path)
            self.loaded = True
            print(f"[RLController] Loaded checkpoint from '{ckpt_path}'")
        else:
            print(f"[RLController] No checkpoint at '{ckpt_path}'.")

    @property
    def name(self):
        return "RL (PPO)"

    def compute_control(self, state, ref, wind_force=None):
        if not self.loaded:
            return np.array([MASS * G, 0., 0., 0.])

        if wind_force is None:
            wind_force = np.zeros(3)

        obs = np.concatenate([state, wind_force, ref]).astype(np.float32)
        action, _ = self.model.predict(obs, deterministic=True)

        T     = (action[0] + 1) / 2 * (4 * MASS * G)
        tau_x = action[1] * 0.5
        tau_y = action[2] * 0.5
        tau_z = action[3] * 0.5

        return np.array([T, tau_x, tau_y, tau_z])


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not SB3_AVAILABLE:
        print("Install: pip install stable-baselines3 gymnasium")
        sys.exit(1)

    print("=== RL Agent (with LQR pre-training) Sanity Check ===\n")
    env = QuadrotorEnv()
    check_env(env, warn=True)
    print("Environment check passed ✓")
    print("\nRun train_rl.py to start training.")