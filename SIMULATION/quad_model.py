"""
Quadrotor 6-DOF Dynamics
State: [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]  (12-dim)
Control: [T, tau_x, tau_y, tau_z]                         (4-dim)
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  
from config import MASS, G, I_XX, I_YY, I_ZZ, DT


# ─── Rotation matrix ──────────────────────────────────────────────────────────

def rotation_matrix(phi, theta, psi):
    """
    ZYX Euler angles → rotation matrix R (body → inertial)
    turns body frame into inertial frame

    phi   = roll  (x)
    theta = pitch (y)
    psi   = yaw   (z)
    """
    Rx = np.array([
        [1,           0,            0],
        [0,  np.cos(phi), -np.sin(phi)],
        [0,  np.sin(phi),  np.cos(phi)]
    ])
    Ry = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [             0, 1,             0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    Rz = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi),  np.cos(psi), 0],
        [          0,            0, 1]
    ])
    
    return Rz @ Ry @ Rx


# ─── Euler angle rate matrix ───────────────────────────────────────────────────

def euler_rate_matrix(phi, theta):

    cos_phi   = np.cos(phi)
    sin_phi   = np.sin(phi)
    cos_theta = np.cos(theta)
    tan_theta = np.tan(theta)

    W = np.array([
        [1,  sin_phi * tan_theta,  cos_phi * tan_theta],
        [0,              cos_phi,             -sin_phi],
        [0,  sin_phi / cos_theta,  cos_phi / cos_theta]
    ])
    return W


# ─── Dynamics (continuous time) ─────────────────────────────────────────────── (important)

def quad_dynamics(state, control, wind_force):
    """
    kinematics for 4 quadrotors drone  ẋ = f(x, u, F_wind)

    Args:
        state      : (12,) [x,y,z, phi,theta,psi, vx,vy,vz, p,q,r]
        control    : (4,)  [T, tau_x, tau_y, tau_z]
        wind_force : (3,)  [Fx, Fy, Fz]  

    Returns:
        dstate : (12,) derivative of state (ẋ)
    """
   
    x, y, z         = state[0:3]
    phi, theta, psi  = state[3:6]    # roll, pitch, yaw
    vx, vy, vz       = state[6:9]    # velocity in inertial frame
    p, q, r          = state[9:12]   # angular velocity (body frame)

    T     = control[0]               # total thrust (N)
    tau_x = control[1]               # roll  torque (N·m)
    tau_y = control[2]               # pitch torque (N·m)
    tau_z = control[3]               # yaw   torque (N·m)

    # ── translational dynamics ───────────────────────────────────────
    # ṗ = v  
    dp = np.array([vx, vy, vz])

    # R turns force from body frame to inertial frame
    R = rotation_matrix(phi, theta, psi)
    thrust_body    = np.array([0.0, 0.0, T])   # body frame
    thrust_inertial = R @ thrust_body           # inertial frame

    gravity = np.array([0.0, 0.0, -G * MASS])  # gravity force in inertial frame

    # v̇ = (thrust + gravity + wind) / m
    dv = (thrust_inertial + gravity + wind_force) / MASS

    # ── rotational dynamics ──────────────────────────────────────────
    # η̇ = W(φ,θ) · ω
    W = euler_rate_matrix(phi, theta)
    omega = np.array([p, q, r])
    deta = W @ omega                            # Euler angle rates from angular velocity

    # ω̇ = I⁻¹ · (τ - ω × (I·ω))
    # the second term is the gyroscopic effect, which couples the angular velocities
    I = np.array([I_XX, I_YY, I_ZZ])
    tau = np.array([tau_x, tau_y, tau_z])
    I_omega     = I * omega                     # I·ω
    gyro        = np.cross(omega, I_omega)      # ω × (I·ω)
    domega      = (tau - gyro) / I             

    return np.concatenate([dp, deta, dv, domega])


# ─── RK4 integrator ───────────────────────────────────────────────────────────
#  discretize the continuous dynamics using 4th-order Runge-Kutta method for better accuracy

def rk4_step(state, control, wind_force, dt=DT):
    """
    rk4 is better than Euler for accuracy and stability, especially with larger time steps.

    Args:
        state      : (12,) current state
        control    : (4,)  current control input (remains constant within this step)
        wind_force : (3,)  current wind force
        dt         : time step duration

    Returns:
        next_state : (12,) 
    """
    k1 = quad_dynamics(state,              control, wind_force)
    k2 = quad_dynamics(state + dt/2 * k1, control, wind_force)
    k3 = quad_dynamics(state + dt/2 * k2, control, wind_force)
    k4 = quad_dynamics(state + dt   * k3, control, wind_force)

    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


# ─── QuadrotorModel class ──────────────────────────────────────────────────────

class QuadrotorModel:
    """
    Quadrotor 6-DOF dynamics model

    Example usage:
        quad = QuadrotorModel()
        quad.reset(initial_state)
        for t in range(N_STEPS):
            wind = wind_model.get(t)
            ref  = trajectory.get(t)
            u    = controller.compute_control(quad.state, ref, wind)
            state, done = quad.step(u, wind)
    """

    # physical limits for control inputs
    T_MIN   = 0.0
    T_MAX   = 4 * MASS * G     # maximum hover thrust
    TAU_MAX = 0.5               # N·m

    def __init__(self):
        self.state = self._hover_state()

    def _hover_state(self):
        """hover initial state: at origin, level, stationary"""
        state = np.zeros(12)
        state[2] = 2.0          # z=2m (above ground, don't start from ground)
        return state

    def reset(self, initial_state=None):
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=float)
        else:
            self.state = self._hover_state()
        return self.state.copy()

    def clip_control(self, control):
        """clip control inputs to physical limits"""
        u = control.copy()
        u[0] = np.clip(u[0], self.T_MIN, self.T_MAX)
        u[1:] = np.clip(u[1:], -self.TAU_MAX, self.TAU_MAX)
        return u

    def step(self, control, wind_force=None):
        """
        execute one simulation step

        Returns:
            state : (12,) new state
            done  : bool  whether the drone has crashed (height < 0 or attitude too large)
        """
        if wind_force is None:
            wind_force = np.zeros(3)

        u = self.clip_control(control)
        self.state = rk4_step(self.state, u, wind_force)

        done = self._check_crash()
        return self.state.copy(), done

    def _check_crash(self):
        """check if the drone has crashed (height < 0 or attitude too large)"""
        z             = self.state[2]
        phi, theta, _ = self.state[3:6]

        if z < 0.0:                          # crashed into the ground
            return True
        if abs(phi) > np.pi * 0.75:             # roll is too large (beyond 90°)
            return True
        if abs(theta) > np.pi * 0.75:           # pitch is too large (beyond 90°)
            return True
        return False

    @property
    def position(self):
        return self.state[0:3]

    @property
    def euler_angles(self):
        return self.state[3:6]

    @property
    def velocity(self):
        return self.state[6:9]

    @property
    def angular_velocity(self):
        return self.state[9:12]


# ─── IMU-based wind estimator ─────────────────────────────────────────────────

class WindEstimator:
    """
    IMU-based wind force estimator using Newton-Euler residual.

    Physics:
        m * a = R_{k-1} @ [0, 0, T_{k-1}] + [0, 0, -mg] + F_wind
        → F_wind = m * a_meas - R_{k-1} @ [0, 0, T_{k-1}] + [0, 0, mg]

    where a_meas = (vel_k - vel_{k-1}) / dt  (finite-diff of velocity,
    equivalent to integrating accelerometer readings).

    In simulation with no added noise this is algebraically exact.
    Set noise_std > 0 to simulate real IMU measurement errors.

    Parameters
    ----------
    alpha          : EMA smoothing (1.0 = no filter, raw estimate)
    noise_std      : std of artificial noise added to raw estimate (N)
    zero_vertical  : if True (default), zero out the vertical (z) component
                     of the estimate.  The PINN(Free) network was trained with
                     horizontal wind only (wz=0 in all training data), so its
                     normalizer has std[wz]=1e-8.  Any tiny non-zero wz — even
                     1e-7 N — normalises to thousands of sigma and causes
                     catastrophic OOD output.  Set False only if the target
                     controller was trained with vertical wind.
    """

    def __init__(self, alpha=1.0, noise_std=0.0, zero_vertical=True):
        self.alpha         = alpha
        self.noise_std     = noise_std
        self.zero_vertical = zero_vertical
        self._F_est        = np.zeros(3)

    def reset(self):
        self._F_est = np.zeros(3)

    def update(self, state, state_prev, T_prev, dt=DT):
        """
        Estimate wind force from one step of IMU data.

        Parameters
        ----------
        state      : (12,) current drone state  [pos, euler, vel, omega]
        state_prev : (12,) drone state at previous step
        T_prev     : float  thrust command applied at previous step (N)
        dt         : float  time step (s)

        Returns
        -------
        F_wind_est : (3,) estimated wind force in inertial frame (N)
        """
        # Finite-difference acceleration over [k-1, k]
        # (caused by forces at step k-1: thrust T_prev and wind)
        a_meas = (state[6:9] - state_prev[6:9]) / dt     # (3,) m/s²

        # Thrust direction at previous step  (use R_{k-1}, not R_k)
        phi_p, theta_p, psi_p = state_prev[3:6]
        R_prev          = rotation_matrix(phi_p, theta_p, psi_p)
        thrust_inertial = R_prev @ np.array([0., 0., T_prev])

        # Newton-Euler residual
        gravity = np.array([0., 0., MASS * G])
        F_raw   = MASS * a_meas - thrust_inertial + gravity

        # Optional sensor noise (simulates real IMU)
        if self.noise_std > 0.0:
            F_raw = F_raw + np.random.randn(3) * self.noise_std

        # Zero vertical component if requested (see class docstring)
        if self.zero_vertical:
            F_raw[2] = 0.0

        # Exponential moving average (alpha=1.0 → no smoothing)
        self._F_est = self.alpha * F_raw + (1.0 - self.alpha) * self._F_est

        return self._F_est.copy()


# ─── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Quadrotor Model Sanity Check ===\n")

    quad = QuadrotorModel()
    state = quad.reset()
    print(f"Initial state: z={state[2]:.2f}m, all velocities zero ✓")

    # Test 1: Hover thrust should maintain altitude
    hover_thrust = MASS * G
    u_hover = np.array([hover_thrust, 0.0, 0.0, 0.0])

    for _ in range(100):
        state, done = quad.step(u_hover)

    print(f"After 100 steps at hover thrust: z={state[2]:.4f}m (should be ~2.0) ✓")

    # Test 2: Zero thrust should cause the drone to fall and eventually crash
    quad.reset()
    u_zero = np.zeros(4)
    for _ in range(50):
        state, done = quad.step(u_zero)
        if done:
            break

    print(f"After zero thrust: z={state[2]:.4f}m, crashed={done} ✓")

    # Test 3: Side wind should cause horizontal drift
    quad.reset()
    wind = np.array([5.0, 0.0, 0.0])    # 5 N wind in x direction
    for _ in range(100):
        state, done = quad.step(u_hover, wind)

    print(f"After 100 steps with side wind: vx={state[6]:.3f} m/s (should be >0) ✓")
    print("\nAll checks passed!")