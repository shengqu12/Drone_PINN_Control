"""
PINN Geometric Controller

Uses the 24-dim geometric-trained network: [state(12), wind(3), ref(6), acc_ref(3)] → [T, tau×3]

The key difference from PINNController (21-dim):
  - Adds reference acceleration acc_ref(3) to the input
  - This was in the training data, so the network learned to use centripetal forces
  - Must be used with a checkpoint trained by PINNGeoTrainer (train_pinn_geo.py)

Usage:
    ctrl = PINNGeoController()           # loads CHECKPOINTS/pinn_geo.pt
    u = ctrl.compute_control(state, ref, wind_force, acc_ref)
"""

import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import PINN_GEO_CKPT
from CONTROLLERS.base import BaseController
from PINN.network import PINNNetwork, InputNormalizer, build_network_input_geo

_GEO_INPUT_DIM = 24


class PINNGeoController(BaseController):
    """
    Geometric PINN controller (24-dim input).

    At inference time:
        1. [state(12), wind(3), ref(6), acc_ref(3)] → (24,) input
        2. Normalize with fitted normalizer
        3. One forward pass through the network
        4. Return [T, tau_x, tau_y, tau_z]
    """

    def __init__(self, ckpt_path=None):
        self.net        = PINNNetwork(input_dim=_GEO_INPUT_DIM)
        self.normalizer = InputNormalizer()
        self.loaded     = False

        ckpt_path = ckpt_path or PINN_GEO_CKPT
        if os.path.exists(ckpt_path):
            self._load(ckpt_path)
        else:
            print(f"[PINNGeoController] No checkpoint at '{ckpt_path}'. "
                  f"Using random weights — run train_pinn_geo.py first.")

        self.net.eval()

    def _load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.net.load_state_dict(ckpt['net_state_dict'])
        self.normalizer.mean   = ckpt['normalizer_mean']
        self.normalizer.std    = ckpt['normalizer_std']
        self.normalizer.fitted = True
        self.loaded = True
        print(f"[PINNGeoController] Loaded checkpoint from '{ckpt_path}'")

    @property
    def name(self):
        return "PINN(Geo)"

    def compute_control(self, state, ref, wind_force=None, acc_ref=None):
        """
        Args:
            state      : (12,) current drone state
            ref        : (6,)  [pos(3), vel(3)]
            wind_force : (3,)  wind force in N
            acc_ref    : (3,)  reference acceleration (centripetal + tangential)

        Returns:
            u : (4,) [T, tau_x, tau_y, tau_z]
        """
        if wind_force is None:
            wind_force = np.zeros(3)
        if acc_ref is None:
            acc_ref = np.zeros(3)

        x = build_network_input_geo(state, wind_force, ref, acc_ref)

        if self.normalizer.fitted:
            x = self.normalizer.transform(x)

        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, 24)
            u_t = self.net(x_t)                                        # (1, 4)
            u   = u_t.squeeze(0).numpy()                               # (4,)

        return u
