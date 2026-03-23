# Drone_PINN_Control
---
## Overview  
Robust control of UAVs under wind disturbances remains a fundamental challenge in autonomous systems.  
This project proposes a **Physics-Informed Neural Network (PINN)-based controller** that enables **feedforward disturbance compensation**, going beyond traditional pure feedback control strategies.

By incorporating physical dynamics into the learning process, the controller achieves improved robustness and generalization under uncertain environmental conditions.

---

##  File Structure

### `train_pinn.py`  
Train the Physics-Informed Neural Network (PINN) model for disturbance-aware control.

### `train_rl.py`  
Train a reinforcement learning (RL) policy for drone control under uncertainty and stochastic wind disturbances.

### `run_simulation.py`  
Run simulations and generate comparison plots of different control methods under varying wind speeds, evaluated using RMSE metrics.

---

## Result
![trajectory_wind_8ms](Results/trajectory_wind8ms.png)
![trajectory_wind_6ms](Results/trajectory_wind6ms.png)

---
It shows very promising result compared to classical controllers (PID,LQR)

