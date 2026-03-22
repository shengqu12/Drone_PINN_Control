"""
PINN Network Architecture

A standard MLP with Tanh activations.
Tanh is chosen over ReLU because:
  1. Smooth and infinitely differentiable — required for physics loss
     (L_physics needs gradients of the output w.r.t. inputs)
  2. Bounded output — prevents exploding activations during training
  3. Symmetric around zero — good for control signals that can be +/-

Input  (21-dim): [state(12), wind_force(3), reference(6)]
Output  (4-dim): [T, tau_x, tau_y, tau_z]
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (PINN_INPUT_DIM, PINN_OUTPUT_DIM,
                    PINN_HIDDEN_DIM, PINN_N_LAYERS,
                    MASS, G, T_MIN, T_MAX, TAU_MAX)


# ─── Input normalizer ─────────────────────────────────────────────────────────

class InputNormalizer:
    """
    Normalize raw inputs to zero mean, unit variance before feeding to network.
    Neural networks train much faster and more stably on normalized inputs.

    Stats are computed from training data and fixed after that.
    """

    def __init__(self):
        self.mean = None
        self.std  = None
        self.fitted = False

    def fit(self, X):
        """
        Compute normalization stats from training data X (N, 21).
        Call this once before training.
        """
        self.mean = X.mean(axis=0)
        self.std  = X.std(axis=0) + 1e-8    # avoid division by zero
        self.fitted = True

    def transform(self, X):
        """Normalize X using stored stats. X can be numpy or torch tensor."""
        if isinstance(X, torch.Tensor):
            mean = torch.tensor(self.mean, dtype=X.dtype, device=X.device)
            std  = torch.tensor(self.std,  dtype=X.dtype, device=X.device)
            return (X - mean) / std
        return (X - self.mean) / self.std

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path):
        data = np.load(path)
        self.mean   = data['mean']
        self.std    = data['std']
        self.fitted = True


# ─── PINN Network ─────────────────────────────────────────────────────────────

class PINNNetwork(nn.Module):
    """
    MLP: 21 → [256 → Tanh] × 6 → 4

    Output activation:
        T      : Softplus  → always positive (thrust can't be negative)
        tau_xyz: Tanh      → bounded torques in [-TAU_MAX, TAU_MAX]

    Why not just clip the output?
        Clipping creates zero gradients at the boundary,
        which stops the network from learning to stay within bounds.
        Smooth activations (Softplus, Tanh) maintain gradients everywhere.
    """

    def __init__(self,
                 input_dim=PINN_INPUT_DIM,
                 output_dim=PINN_OUTPUT_DIM,
                 hidden_dim=PINN_HIDDEN_DIM,
                 n_layers=PINN_N_LAYERS):
        super().__init__()

        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        # output scaling constants
        self.T_scale   = T_MAX              # Softplus output scaled to [0, T_MAX]
        self.tau_scale = TAU_MAX            # Tanh output scaled to [-TAU_MAX, TAU_MAX]

        # ── Build MLP layers ─────────────────────────────────────────────
        layers = []

        # input → first hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # hidden → hidden  (n_layers - 1 times)
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        self.backbone = nn.Sequential(*layers)

        # separate output heads for thrust and torques
        self.head_T   = nn.Linear(hidden_dim, 1)     # thrust
        self.head_tau = nn.Linear(hidden_dim, 3)     # [tau_x, tau_y, tau_z]

        # weight initialization — Xavier for Tanh networks
        self._init_weights()

    def _init_weights(self):
        """
        Xavier uniform initialization.
        Keeps activations in the linear region of Tanh at the start,
        which avoids vanishing gradients during early training.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Args:
            x : (batch, 21) normalized input tensor

        Returns:
            u : (batch, 4) control output [T, tau_x, tau_y, tau_z]
        """
        features = self.backbone(x)

        # thrust: Softplus ensures T > 0, scale to physical range
        # Softplus(x) ≈ max(0, x) but smooth
        T = self.T_scale * torch.nn.functional.softplus(self.head_T(features))

        # torques: Tanh bounds to [-1, 1], scale to physical range
        tau = self.tau_scale * torch.tanh(self.head_tau(features))

        return torch.cat([T, tau], dim=-1)

    def predict_numpy(self, x_np, normalizer=None):
        """
        Convenience wrapper for numpy input (used at inference time).

        Args:
            x_np       : (21,) or (N, 21) numpy array
            normalizer : InputNormalizer (optional)

        Returns:
            u_np : (4,) or (N, 4) numpy array
        """
        scalar = x_np.ndim == 1
        if scalar:
            x_np = x_np[np.newaxis, :]     # add batch dim

        if normalizer is not None:
            x_np = normalizer.transform(x_np)

        with torch.no_grad():
            x_t = torch.tensor(x_np, dtype=torch.float32)
            u_t = self.forward(x_t)
            u_np = u_t.numpy()

        return u_np[0] if scalar else u_np

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Input builder ────────────────────────────────────────────────────────────

def build_network_input(state, wind_force, ref):
    """
    Concatenate state (12,), wind_force (3,), ref (6,) → (21,) input vector.
    Used both during training and inference.
    """
    return np.concatenate([state, wind_force, ref])


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== PINN Network Sanity Check ===\n")

    net = PINNNetwork()
    print(f"Architecture: {PINN_INPUT_DIM} → [{PINN_HIDDEN_DIM} × Tanh] × {PINN_N_LAYERS} → {PINN_OUTPUT_DIM}")
    print(f"Total trainable parameters: {net.count_parameters():,}")

    # test forward pass
    batch_size = 32
    x = torch.randn(batch_size, PINN_INPUT_DIM)
    u = net(x)

    print(f"\nForward pass: input {tuple(x.shape)} → output {tuple(u.shape)}")
    print(f"Output range:")
    print(f"  T     : [{u[:,0].min():.3f},  {u[:,0].max():.3f}]  (should be > 0)")
    print(f"  tau_x : [{u[:,1].min():.3f}, {u[:,1].max():.3f}]  (should be in [-{TAU_MAX}, {TAU_MAX}])")
    print(f"  tau_y : [{u[:,2].min():.3f}, {u[:,2].max():.3f}]")
    print(f"  tau_z : [{u[:,3].min():.3f}, {u[:,3].max():.3f}]")

    # test that T is always positive
    assert (u[:, 0] > 0).all(), "T should always be positive!"
    print(f"\nThrust always positive: ✓")

    # test that torques are bounded
    assert (u[:, 1:].abs() <= TAU_MAX + 1e-5).all(), "Torques should be bounded!"
    print(f"Torques bounded in [-{TAU_MAX}, {TAU_MAX}]: ✓")

    # test normalizer
    print(f"\nTesting InputNormalizer...")
    norm = InputNormalizer()
    X_fake = np.random.randn(1000, PINN_INPUT_DIM).astype(np.float32)
    norm.fit(X_fake)
    X_normed = norm.transform(X_fake)
    print(f"  Mean after normalization: {X_normed.mean():.4f} (should be ~0)")
    print(f"  Std  after normalization: {X_normed.std():.4f}  (should be ~1)")

    # test build_network_input
    state      = np.zeros(12)
    wind_force = np.array([1.0, 0.0, 0.0])
    ref        = np.ones(6)
    inp = build_network_input(state, wind_force, ref)
    assert inp.shape == (21,)
    print(f"\nbuild_network_input output shape: {inp.shape} ✓")

    # test predict_numpy
    u_np = net.predict_numpy(inp, normalizer=None)
    print(f"predict_numpy output shape: {u_np.shape} ✓")

    print("\nAll checks passed!")