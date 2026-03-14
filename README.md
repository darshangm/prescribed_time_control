# Constricting Tubes for Fixed-Time Control Synthesis

Simulation code accompanying the paper:

> **Constricting Tubes for Fixed-Time Control Synthesis**
> 
> Darshan Gadginmath, Ahmed Allibhoy, Fabio Pasqualetti

---

## Overview

This repository contains numerical experiments demonstrating the **constricting barrier function (CBF)** framework for prescribed-time recovery.

The core idea: given a safe set $\mathcal{C} = \{x : h(x) \geq 0\}$ and a system starting outside it, we synthesize a time-varying barrier

$$\tilde{h}(x,t) = h(x) + r(x(0),t), \quad r(x(0),t) \to 0 \text{ as } t \to T$$

This reduces prescribed-time entry into $\mathcal{C}$ to a standard forward invariance problem, avoiding the diverging control gains of prescribed-time methods. This is also highly scalable: despite coordinating N agents, control is synthesized via a single QP with one CBF constraint over the m-dimensional input.

---

## Repository Structure

```
prescribed_time_control/
├── formation.py          # Experiment 2: 16D stacked linear system
├── unicycle_nmpc.py      # Experiment 3: Unicycle NMPC with obstacle avoidance
├── data_formation/       # Output data for Experiment 2 (auto-created)
├── data/                 # Output data for Experiment 3 (auto-created)
└── figs/                 # Output figures for Experiment 3 (auto-created)
```

---

## Experiments

### `formation.py` — Stacked linear system (Experiment 2)

- **System:** 8 agents with 2D linear dynamics, stacked into a 16D system with single joint barrier $h(X) = c - \|X\|^2$
- **Controller:** Joint CBF-QP solved via CVXPY/OSQP with linear constriction schedule $r(t) = r_0(1 - t/T)$
- **Baselines:** Garg & Panagou (prescribed-time Sontag CLF)
- **Outputs:** Phase portrait, joint barrier value over time, aggregate control effort

### `unicycle_nmpc.py` — Unicycle with obstacle avoidance (Experiment 3)

- **System:** Unicycle with 3D state $(p_x, p_y, \theta)$ and nonlinear dynamics
- **Controller:** Receding-horizon NMPC ($N=40$, $\Delta t = 0.1$ s) via CasADi/IPOPT, enforcing both a constricting reach tube $h_1(x,t) \geq 0$ and a static obstacle constraint $h_2(x) \geq 0$
- **Schedule:** Shifted quadratic $r_1(t) = (r_0 + \delta)(1 - t/T)^2 - \delta$, ensuring the tube closes strictly inside $\mathcal{C}$
- **Outputs:** Phase portrait, barrier value trajectories, control inputs (`experiment3.pdf/.png`, `exp3_trajectory_mpc.dat`, `exp3_trajectory_nom.dat`)

---

## Dependencies

```bash
pip install numpy scipy matplotlib cvxpy casadi
```

- `formation.py` uses **OSQP** as the CVXPY backend (installed automatically with `cvxpy`)
- `unicycle_nmpc.py` uses **IPOPT** via CasADi (included in the standard `casadi` pip package)

Tested with Python 3.10+.

---

## Usage

```bash
python formation.py       # Experiment 2 (~1 min) — outputs to data_formation/
python unicycle_nmpc.py   # Experiment 3 (~2 min) — outputs to data/ and figs/
```

---

## Citation

If you use this code, please cite the arxiv version for now:

---

## License

MIT
