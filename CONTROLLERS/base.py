"""
Base Controller Interface
All controllers must implement compute_control(state, ref, wind_force).
This unified interface lets run_simulation.py loop over all controllers
without any controller-specific logic.
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseController(ABC):
    """
    Abstract base class for all controllers.

    Every controller receives the same three inputs:
        state      : (12,) current drone state [x,y,z, phi,theta,psi, vx,vy,vz, p,q,r]
        ref        : (6,)  reference state     [x_ref, y_ref, z_ref, vx_ref, vy_ref, vz_ref]
        wind_force : (3,)  current wind force  [Fx, Fy, Fz] in Newtons

    And returns the same output:
        u : (4,) control input [T, tau_x, tau_y, tau_z]

    Note: not all controllers use wind_force (e.g. PID, LQR don't).
    It is passed in anyway so run_simulation.py never needs to branch.
    Only PINN uses it explicitly for feedforward compensation.
    """

    @abstractmethod
    def compute_control(self, state, ref, wind_force):
        """
        Compute control input for the current timestep.

        Args:
            state      : (12,) numpy array
            ref        : (6,)  numpy array
            wind_force : (3,)  numpy array

        Returns:
            u : (4,) numpy array [T, tau_x, tau_y, tau_z]
        """
        pass

    def reset(self):
        """Reset any internal state (integrators, history, etc.)."""
        pass

    @property
    def name(self):
        """Controller name used in plots and logs."""
        return self.__class__.__name__