"""
CMU 12-787 Final Project
PINN-based Quadrotor Control under Wind Disturbance
"""

# ─── Quadrotor physical parameters ───────────────────────────────────────────
MASS        = 1.0       # kg
G           = 9.81      # m/s²
ARM_LENGTH  = 0.2       # m  (motor to center)
I_XX        = 0.0049    # kg·m²  (roll inertia)
I_YY        = 0.0049    # kg·m²  (pitch inertia)
I_ZZ        = 0.0069    # kg·m²  (yaw inertia)
K_THRUST    = 1.0       # thrust coefficient (normalized)
K_TORQUE    = 0.016     # torque-to-thrust ratio

# ─── Simulation parameters ───────────────────────────────────────────────────
DT          = 0.01      # s   (100 Hz)
T_TOTAL     = 20.0      # s
N_STEPS     = int(T_TOTAL / DT)

# ─── Reference trajectory ────────────────────────────────────────────────────
TRAJ_TYPE   = "lemniscate"   # "lemniscate" | "helix" | "circle"
TRAJ_A      = 2.0       # m  (amplitude)
TRAJ_OMEGA  = 0.5       # rad/s
TRAJ_HEIGHT = 2.0       # m  (hover height)

# ─── Wind disturbance ────────────────────────────────────────────────────────
WIND_TYPE       = "constant"     # "none" | "constant" | "gust" | "turbulence"
WIND_SPEED_CONST = 5.0           # m/s
WIND_SPEEDS      = [0, 2, 4, 6, 8, 10, 12, 15]   # sweep list
WIND_DIRECTION   = [1.0, 0.0, 0.0]               # unit vector (x-axis)
DRYDEN_SIGMA     = 1.5           # m/s  turbulence intensity

# ─── PINN network architecture ───────────────────────────────────────────────
PINN_INPUT_DIM   = 21    # 12 state + 3 wind + 6 reference
PINN_OUTPUT_DIM  = 4     # [T, tau_x, tau_y, tau_z]
PINN_HIDDEN_DIM  = 256
PINN_N_LAYERS    = 6

# ─── PINN loss weights ───────────────────────────────────────────────────────
LAMBDA_PHYSICS  = 0.01    # λ  physics residual
BETA_SMOOTH     = 0.001   # β  control smoothness
GAMMA_BC        = 0.1   # γ  boundary condition (initial state)

# ─── Training ────────────────────────────────────────────────────────────────
PINN_EPOCHS     = 5000
PINN_LR         = 1e-3
PINN_BATCH_SIZE = 512
LR_DECAY_STEP   = 2000
LR_DECAY_GAMMA  = 0.5

# ─── RL ──────────────────────────────────────────────────────────────────────
RL_TOTAL_TIMESTEPS = 500_000

# ─── Control output limits ───────────────────────────────────────────────────
T_MIN   = 0.0            # N   minimum total thrust
T_MAX   = 4 * MASS * G  # N   maximum total thrust (4× hover)
TAU_MAX = 0.5            # N·m maximum torque per axis

# ─── Paths ───────────────────────────────────────────────────────────────────
import os
_ROOT = os.path.dirname(os.path.abspath(__file__))  # project root

CHECKPOINT_DIR  = os.path.join(_ROOT, "CHECKPOINTS")
RESULTS_DIR     = os.path.join(_ROOT, "RESULTS")
SANITY_DIR      = os.path.join(_ROOT, "SANITY_CHECK")
PINN_FREE_CKPT  = os.path.join(CHECKPOINT_DIR, "pinn_free.pt")
PINN_GEO_CKPT      = os.path.join(CHECKPOINT_DIR, "pinn_geo.pt")
PINN_GEO_DRAG_CKPT = os.path.join(CHECKPOINT_DIR, "pinn_geo_drag.pt")
RL_CKPT            = os.path.join(CHECKPOINT_DIR, "rl_ppo")

# ─── Aerodynamic drag ─────────────────────────────────────────────────────────
# Quadratic drag model: F_drag = -C_DRAG * ||v|| * v  (opposes motion)
# At v=2 m/s: F_drag = 0.5 * 2 * 2 = 2 N  (~20% of gravity, significant at high speed)
# LQR+FF is unaware of this term; PINN(Geo+Drag) teacher explicitly compensates it.
C_DRAG = 0.5     # kg/m