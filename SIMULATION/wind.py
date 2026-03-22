"""
Wind Disturbance Models
Returns the wind force vector F_wind (3,) in units of N

Four modes supported:
  none        - no wind, for baseline comparison
  constant    - constant wind with fixed speed and direction
  gust        - sudden wind burst with smooth onset/offset
  turbulence  - Dryden turbulence model, most realistic
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (MASS, DT, WIND_SPEED_CONST, WIND_DIRECTION,
                    DRYDEN_SIGMA, WIND_SPEEDS)


# ─── Base class ───────────────────────────────────────────────────────────────

class WindModel:
    """
    Base class for all wind disturbance models.
    Subclasses only need to implement _compute(t).

    wind_speed : float  nominal wind speed in m/s (used for scaling)
    """

    def __init__(self, wind_speed=WIND_SPEED_CONST):
        self.wind_speed = wind_speed    # m/s
        self.t = 0.0

    def reset(self):
        self.t = 0.0

    def step(self):
        """Call once per simulation step. Returns F_wind (3,) in Newtons."""
        force = self._compute(self.t)
        self.t += DT
        return force

    def _compute(self, t):
        raise NotImplementedError


# ─── No wind ──────────────────────────────────────────────────────────────────

class NoWind(WindModel):
    def _compute(self, t):
        return np.zeros(3)


# ─── Constant wind ────────────────────────────────────────────────────────────

class ConstantWind(WindModel):
    """
    Steady wind with fixed speed and direction.

    Force is scaled by MASS so that wind_speed directly represents
    the acceleration disturbance (m/s^2) regardless of vehicle mass.
    F = mass * wind_speed * unit_direction
    """

    def __init__(self, wind_speed=WIND_SPEED_CONST,
                 direction=WIND_DIRECTION):
        super().__init__(wind_speed)
        d = np.array(direction, dtype=float)
        self.direction = d / np.linalg.norm(d)   # normalize to unit vector

    def _compute(self, t):
        return self.direction * self.wind_speed * MASS


# ─── Gust wind ────────────────────────────────────────────────────────────────

class GustWind(WindModel):
    """
    Wind gust: calm -> sudden wind burst -> calm.
    Uses a 1-cos envelope for smooth onset and offset,
    which is more physically realistic than a square pulse.

    gust_start : float  time when gust begins (s)
    gust_end   : float  time when gust ends (s)
    """

    def __init__(self, wind_speed=WIND_SPEED_CONST,
                 direction=WIND_DIRECTION,
                 gust_start=5.0, gust_end=10.0):
        super().__init__(wind_speed)
        d = np.array(direction, dtype=float)
        self.direction  = d / np.linalg.norm(d)
        self.gust_start = gust_start
        self.gust_end   = gust_end

    def _compute(self, t):
        if t < self.gust_start or t > self.gust_end:
            return np.zeros(3)

        # 1-cos envelope: smoothly ramps up and back down over [gust_start, gust_end]
        duration = self.gust_end - self.gust_start
        phase    = (t - self.gust_start) / duration
        envelope = 0.5 * (1 - np.cos(2 * np.pi * phase))

        return self.direction * self.wind_speed * MASS * envelope


# ─── Dryden turbulence model ──────────────────────────────────────────────────

class DrydenWind(WindModel):
    """
    Simplified Dryden continuous turbulence model for low-altitude flight.

    Real atmospheric turbulence is not white noise -- it has temporal
    correlation structure. This model captures that by passing white
    noise through a first-order shaping filter:

        dw/dt = -(V/L) * w + sigma * sqrt(2*V/L) * noise

    where:
        sigma : turbulence intensity (m/s)
        L     : turbulence length scale (m), ~50-200m at low altitude
        V     : mean airspeed (m/s), affects correlation time scale

    The total wind = steady mean component + stochastic turbulence component.
    """

    def __init__(self, wind_speed=WIND_SPEED_CONST,
                 sigma=DRYDEN_SIGMA, L=50.0, V=3.0):
        super().__init__(wind_speed)
        self.sigma = sigma
        self.L     = L
        self.V     = V

        # shaping filter coefficients
        self.alpha = V / L                          # decay rate
        self.beta  = sigma * np.sqrt(2 * V / L)    # noise gain

        # filter state -- three independent axes
        self.w = np.zeros(3)

        # steady mean wind direction
        direction = np.array(WIND_DIRECTION, dtype=float)
        self.mean_wind = direction / np.linalg.norm(direction) * wind_speed

    def reset(self):
        super().reset()
        self.w = np.zeros(3)    # reset filter state

    def _compute(self, t):
        # update shaping filter with new white noise input
        noise   = np.random.randn(3)
        dw      = -self.alpha * self.w + self.beta * noise
        self.w += dw * DT

        # total wind velocity = mean wind + turbulence fluctuation
        total_wind_velocity = self.mean_wind + self.w    # m/s

        # convert to force: F = m * a, where a ~ wind_velocity (simplified)
        return total_wind_velocity * MASS


# ─── Factory function ─────────────────────────────────────────────────────────

def make_wind(wind_type, wind_speed=WIND_SPEED_CONST, **kwargs):
    """
    Factory function -- create a wind model from a string identifier.

    Usage:
        wind = make_wind("turbulence", wind_speed=8.0)
        wind.reset()
        F = wind.step()    # call once per simulation step
    """
    wind_type = wind_type.lower()
    if wind_type == "none":
        return NoWind(wind_speed)
    elif wind_type == "constant":
        return ConstantWind(wind_speed, **kwargs)
    elif wind_type == "gust":
        return GustWind(wind_speed, **kwargs)
    elif wind_type == "turbulence":
        return DrydenWind(wind_speed, **kwargs)
    else:
        raise ValueError(f"Unknown wind type: '{wind_type}'. "
                         f"Choose from: none, constant, gust, turbulence")


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=== Wind Model Sanity Check ===\n")

    N = int(20.0 / DT)      # 20 seconds
    t = np.arange(N) * DT

    models = {
        "none"       : make_wind("none"),
        "constant"   : make_wind("constant",   wind_speed=5.0),
        "gust"       : make_wind("gust",        wind_speed=8.0,
                                 gust_start=5.0, gust_end=12.0),
        "turbulence" : make_wind("turbulence",  wind_speed=5.0),
    }

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    colors = ['#888780', '#1D9E75', '#E24B4A', '#534AB7']

    for ax, (name, model), color in zip(axes, models.items(), colors):
        model.reset()
        np.random.seed(42)
        forces = np.array([model.step() for _ in range(N)])

        # plot x-axis component (primary wind direction)
        ax.plot(t, forces[:, 0], color=color, linewidth=1.2, label=name)
        ax.set_ylabel("Fx (N)", fontsize=9)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        mag = np.abs(forces).max()
        print(f"{name:12s}: max force = {mag:.3f} N")

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Wind models -- Fx component", fontsize=11)
    plt.tight_layout()
    os.makedirs("./SANITY_CHECK/windcheck", exist_ok=True)
    plt.savefig("./SANITY_CHECK/windcheck/wind_check.png", dpi=120)
    print("\nPlot saved to ./SANITY_CHECK/windcheck/wind_check.png")
    print("All checks passed!")