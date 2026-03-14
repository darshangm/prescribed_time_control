"""
Experiment: Constricting tubes — 12D stacked system
=====================================================
6 agents, each with 2D linear dynamics:
    x_dot_i = A x_i + B u_i

Full stacked system (12D state, 6D input):
    X_dot = A_big X + B_big U
    A_big = blkdiag(A, ..., A)  in R^{12x12}
    B_big = blkdiag(B, ..., B)  in R^{12x6}

Single joint barrier for the full system:
    h(X) = c - X^T X = c - sum_i ||x_i||^2
    C = { X : h(X) >= 0 }

Constricting barrier:
    h_tilde(X, t) = h(X) + r(t),   r(t) = r0 * (1 - t/T) for t <= T

Single CBF-QP over the full 6D input (solved via CVXPY/OSQP):
    min_{U}  ||U - U_nom||^2
    s.t.     L_f h + L_g h @ U + r_dot >= -alpha * h_tilde
             ||U||_inf <= u_max

T_min via Corollary 3 (P = I_{12}, block structure):
    sigma_min = 2c * (||B|| * u_max - lambda_max(A))
    r0 = max(0, -h(X0))   [single scalar for the joint system]
    T_min = r0 / sigma_min

All methods operate on the full 12D state with the joint barrier
h(X) = c - ||X||^2, ensuring a fair comparison.
"""

import numpy as np
import scipy.linalg
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'lines.linewidth': 1.6,
})
os.makedirs('data_formation', exist_ok=True)

# ── Per-subsystem matrices ────────────────────────────────────────────
A = np.array([[-0.1,  1.0],
              [ 0.0, 0.1]])
B = np.array([[1.0],
              [0.5]])        # (2, 1)

N_SYS = 8
n_sub = 2
m_sub = 1
n     = N_SYS * n_sub       # 12
m     = N_SYS * m_sub       # 6

u_max = 10.0
alpha = 0.999
c_val = 0.5

# ── Stacked system ────────────────────────────────────────────────────
A_big = np.kron(np.eye(N_SYS), A)   # (12, 12)
B_big = np.kron(np.eye(N_SYS), B)   # (12,  6)

print(f"Stacked system: n={n}, m={m}")

# ── Initial conditions ────────────────────────────────────────────────
rng = np.random.default_rng(42)
init_angles = np.linspace(0, 2*np.pi, N_SYS, endpoint=False) \
              + rng.uniform(-0.20, 0.20, N_SYS)
init_radii  = rng.uniform(1.8, 3.2, N_SYS)
X0 = np.concatenate([
    np.array([init_radii[i]*np.cos(init_angles[i]),
              init_radii[i]*np.sin(init_angles[i])])
    for i in range(N_SYS)])   # (12,)

print("Initial states:")
for i in range(N_SYS):
    xi = X0[2*i:2*i+2]
    print(f"  Agent {i+1}: ({xi[0]:+.3f}, {xi[1]:+.3f})  ||x_i||={np.linalg.norm(xi):.3f}")

# ── Joint barrier h(X) = c - X^T X ───────────────────────────────────
def h(X):
    return c_val - float(X @ X)

def Lf_h(X):
    return -2.0 * float(X @ (A_big @ X))

def Lg_h(X):
    return -2.0 * (X @ B_big)   # (6,)

# ── T_min — Corollary 3 ───────────────────────────────────────────────
sv_B      = np.linalg.norm(B)
lam_max_A = np.max(np.real(np.linalg.eigvals(A)))
sigma_min = 2.0 * c_val * (sv_B * u_max - lam_max_A)

delta  = 0.05
r0     = max(0.0, -h(X0)) + delta   # absorb delta so h_tilde(X0,0) = 0
T_min  = r0 / sigma_min
T      = 1.4 * T_min
t_end  = 1.3 * T
t_star = T #* (1.0 - delta / r0)     # r(t) reaches 0 here, before T

print(f"\nCorollary 3 (P=I_12, c={c_val}):")
print(f"  h(X0)={h(X0):.4f},  delta={delta:.4f},  r0={r0:.4f}")
print(f"  sigma_min={sigma_min:.4f},  T_min={T_min:.4f} s,  T={T:.4f} s")
print(f"  t_star (tube closes) = {t_star:.4f} s")

# ── Constriction schedule ─────────────────────────────────────────────
# r(t) = r0*(1 - t/T) - delta, clipped at 0
# Reaches 0 at t_star < T; controller then runs plain CBF until T
def r_s(t):
    return max(delta, r0 * (1.0 - t / T))

def r_dot(t):
    # Zero once r_s has clipped to 0 — this eliminates the spike at T
    return -r0 / T if t < t_star else 0.0

# ── Stacked LQR nominal ───────────────────────────────────────────────
P_lqr   = scipy.linalg.solve_continuous_are(A, B, np.eye(n_sub), 0.7*np.eye(m_sub))
K_sub   = B.T @ P_lqr                      # (1, 2)
K_big   = np.kron(np.eye(N_SYS), K_sub)    # (6, 12)
_ATP_PA = A.T @ P_lqr + P_lqr @ A          # for Garg

def U_nom(X):
    return np.clip(-K_big @ X, -u_max, u_max)

# ── Our controller: CVXPY/OSQP joint CBF-QP ──────────────────────────
_U_var = cp.Variable(m)

def U_ours(X, t):
    lf = Lf_h(X)
    lg = Lg_h(X)
    ht = h(X) + r_s(t)
    rd = r_dot(t)
    un = U_nom(X)
    # objective   = cp.Minimize(cp.sum_squares(_U_var - un))
    objective   = cp.Minimize(cp.sum_squares(_U_var))
    constraints = [
        lg @ _U_var + lf + rd >= -alpha * ht,
        _U_var >= -u_max,
        _U_var <=  u_max,
    ]
    cp.Problem(objective, constraints).solve(
        solver=cp.OSQP, warm_starting=True, verbose=False,polish = False)
    if _U_var.value is None:
        print("QP failed to solve; using nominal control")
        return np.clip(un, -u_max, u_max)
    return _U_var.value.copy()

# ── Baseline 1: Garg & Panagou — joint Sontag CLF on V0 = -h(X) ─────
# V0(X) = -h(X) = ||X||^2 - c  is a valid CLF for the origin.
# Prescribed-time CLF: V(X,t) = V0(X) / theta(t)^2, theta = 1 - t/T.
# Condition V_dot <= 0:  Lf V0 + Lg V0 @ U + (2/theta)*V0 <= 0
# Sontag formula with vector b = Lg V0 in R^6:
#   a = Lf V0 + (2/theta)*V0
#   U = -(a + sqrt(a^2 + ||b||^4)) / ||b||^2 * b^T
def U_garg(X, t):
    eps   = 1e-6
    theta = max(1.0 - min(t, T-eps)/T, eps)
    V0    = -h(X)                        # scalar: ||X||^2 - c
    LfV0  = -Lf_h(X)                     # scalar: 2 X^T A_big X
    LgV0  = -Lg_h(X)                     # (6,):   2 X^T B_big
    a     = LfV0 + (2.0/theta)*V0
    b_sq  = float(LgV0 @ LgV0)           # ||b||^2
    if b_sq < 1e-12:
        return np.zeros(m)
    u_scale = -(a + np.sqrt(a**2 + b_sq**2)) / b_sq
    return np.clip(u_scale * LgV0, -u_max, u_max)

# ── Baseline 2: Li & Krstic — per-agent mu(t)*K ──────────────────────
m_k = 2

def U_krstic(X, t):
    eps   = 1e-6
    theta = max(T - min(t, T - eps*T), eps*T)
    mu    = (T/theta)**m_k
    return np.clip(-mu*(K_big @ X), -u_max, u_max)

# ── RK4 simulation ────────────────────────────────────────────────────
dt    = 0.005
t_arr = np.arange(0.0, t_end+dt, dt)
Nt    = len(t_arr)

def simulate(ctrl_fn, label=''):
    X_hist  = np.zeros((Nt, n))
    U_hist  = np.zeros((Nt, m))
    H_hist  = np.zeros(Nt)
    HT_hist = np.zeros(Nt)
    X_hist[0] = X0.copy()
    for k, tk in enumerate(t_arr):
        Xk = X_hist[k]
        Uk = ctrl_fn(Xk, tk)
        U_hist[k]  = Uk
        H_hist[k]  = h(Xk)
        HT_hist[k] = h(Xk) + r_s(tk)
        if k < Nt-1:
            k1 = A_big @ Xk              + B_big @ Uk
            k2 = A_big @ (Xk+.5*dt*k1)  + B_big @ Uk
            k3 = A_big @ (Xk+.5*dt*k2)  + B_big @ Uk
            k4 = A_big @ (Xk+dt*k3)     + B_big @ Uk
            X_hist[k+1] = Xk + (dt/6.0)*(k1+2*k2+2*k3+k4)
        if label and k % 50 == 0:
            print(f"    {label}: t={tk:.2f}  h={H_hist[k]:.4f}", end='\r')
    if label: print()
    return X_hist, U_hist, H_hist, HT_hist

print("\nSimulating...")
print("  Ours (joint CVXPY/OSQP QP)...")
X_c, U_c, H_c, HT_c = simulate(U_ours,   label='Ours')
print("  Garg (per-agent Sontag)...")
X_g, U_g, H_g, HT_g = simulate(U_garg,   label='Garg')
print("  Krstic (per-agent mu(t)*K)...")
X_k, U_k, H_k, HT_k = simulate(U_krstic, label='Krstic')

U_norm_c = np.linalg.norm(U_c, axis=1)
U_norm_g = np.linalg.norm(U_g, axis=1)
U_norm_k = np.linalg.norm(U_k, axis=1)

# ── Results ───────────────────────────────────────────────────────────
idx_T = np.argmin(np.abs(t_arr - T))
print(f"\n{'='*65}")
print(f"  t=T={T:.4f} s  (T_min={T_min:.4f} s)")
print(f"{'='*65}")
print(f"  h(X(T)):  Ours={H_c[idx_T]:.4f}  Garg={H_g[idx_T]:.4f}  Krstic={H_k[idx_T]:.4f}")
print(f"  In C:     Ours={'✓' if H_c[idx_T]>=0 else '✗'}  "
      f"Garg={'✓' if H_g[idx_T]>=0 else '✗'}  "
      f"Krstic={'✓' if H_k[idx_T]>=0 else '✗'}")
print(f"  ||U(T)||: Ours={U_norm_c[idx_T]:.4f}  "
      f"Garg={U_norm_g[idx_T]:.4f}  Krstic={U_norm_k[idx_T]:.4f}")
print(f"{'='*65}")

# ── Per-agent barriers (for phase portrait colouring) ─────────────────
def agent_barriers(X_hist):
    return np.array([
        c_val - np.sum(X_hist[:, 2*i:2*i+2]**2, axis=1)
        for i in range(N_SYS)])   # (N_SYS, Nt)

HA_c = agent_barriers(X_c)

# ── Colours ───────────────────────────────────────────────────────────
col_ours   = '#1565C0'
col_garg   = '#C62828'
col_krstic = '#E65100'
agent_cols = plt.cm.tab10(np.linspace(0, 0.65, N_SYS))
theta_circ = np.linspace(0, 2*np.pi, 300)
r_floor    = np.array([-r_s(t) for t in t_arr])

# ── Figure: 3 panels ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
fig.subplots_adjust(wspace=0.38)
ax_phase, ax_h, ax_u = axes

# Panel 1 — Phase portrait
ax_phase.fill(np.cos(theta_circ), np.sin(theta_circ),
              color='#C8E6C9', alpha=0.35, zorder=0)
ax_phase.plot(np.cos(theta_circ), np.sin(theta_circ),
              'k-', lw=1.5, alpha=0.7, zorder=1)
for i in range(N_SYS):
    clr = agent_cols[i]
    xi_c = X_c[:, 2*i:2*i+2]
    xi_g = X_g[:, 2*i:2*i+2]
    xi_k = X_k[:, 2*i:2*i+2]
    ax_phase.plot(xi_c[:idx_T+1,0], xi_c[:idx_T+1,1],
                  color=clr, lw=1.8, alpha=0.95, zorder=4)
    ax_phase.plot(xi_c[idx_T:,0],   xi_c[idx_T:,1],
                  color=clr, lw=0.9, alpha=0.25, zorder=2)
    ax_phase.plot(xi_g[:idx_T+1,0], xi_g[:idx_T+1,1],
                  color=clr, lw=1.3, ls='--', alpha=0.6, zorder=3)
    # ax_phase.plot(xi_k[:idx_T+1,0], xi_k[:idx_T+1,1],
    #               color=clr, lw=1.3, ls=':', alpha=0.6, zorder=3)
    ax_phase.scatter(*X0[2*i:2*i+2], color=clr, s=50, marker='o',
                     zorder=6, edgecolors='k', linewidths=0.6)
    ax_phase.scatter(*xi_c[idx_T],   color=clr, s=75, marker='*',
                     zorder=7, edgecolors='k', linewidths=0.4)

ax_phase.legend(handles=[
    Line2D([0],[0], color='gray', lw=1.8, ls='-',  label='Ours'),
    Line2D([0],[0], color='gray', lw=1.3, ls='--', label='Garg'),
    # Line2D([0],[0], color='gray', lw=1.3, ls=':',  label='Krstic'),
    Line2D([0],[0], color='k', marker='o', lw=0, ms=5, label='$x_i(0)$'),
    Line2D([0],[0], color='k', marker='*', lw=0, ms=7, label='$x_i(T)$'),
    Line2D([0],[0], color='k', lw=1.5, label=r'$\partial\mathcal{C}$'),
], fontsize=7.5, loc='upper right', ncol=2, handlelength=1.5)
ax_phase.set_xlabel('$x_{i,1}$'); ax_phase.set_ylabel('$x_{i,2}$')
ax_phase.set_title('Phase portrait (6 agents)')
ax_phase.set_aspect('equal'); ax_phase.grid(True, alpha=0.2)

# Panel 2 — Joint barrier h(X)
ax_h.axhline(0, color='k', lw=0.9, ls='--', alpha=0.5)
ax_h.axvline(T,     color='dimgray', lw=1.0, ls=':',  alpha=0.8)
ax_h.axvline(T_min, color='dimgray', lw=0.8, ls='-.', alpha=0.6)
ymin = min(H_c.min(), H_g.min(), H_k.min()) * 1.15
ax_h.text(T     + 0.006*t_end, ymin*0.5, '$T$',      fontsize=8, color='dimgray')
ax_h.text(T_min + 0.006*t_end, ymin*0.5, '$T_{\\min}$', fontsize=8, color='dimgray')

ax_h.plot(t_arr, H_c, color=col_ours,   lw=2.0, ls='-',  label=r'$h(X)$ ours')
ax_h.plot(t_arr, H_g, color=col_garg,   lw=1.8, ls='--', label=r'$h(X)$ Garg')
# ax_h.plot(t_arr, H_k, color=col_krstic, lw=1.8, ls=':',  label=r'$h(X)$ Krstic')
ax_h.plot(t_arr, r_floor, color='#42A5F5', lw=1.5, alpha=0.85,
          label=r'$-r(t)$ (tube floor)')

ax_h.set_xlabel('Time (s)'); ax_h.set_ylabel('$h(X(t))$')
ax_h.set_title('Joint barrier value')
ax_h.legend(fontsize=7.5, loc='lower right')
ax_h.grid(True, alpha=0.2); ax_h.set_xlim(0, t_end)

# Panel 3 — Aggregate ||U(t)||
U_max_agg = u_max * np.sqrt(N_SYS)
ax_u.axhline(U_max_agg, color='k', lw=0.8, ls=':', alpha=0.45,
             label=f'$\\sqrt{{N}}u_{{\\max}}={U_max_agg:.2f}$')
ax_u.axvline(T,     color='dimgray', lw=1.0, ls=':',  alpha=0.8)
ax_u.axvline(T_min, color='dimgray', lw=0.8, ls='-.', alpha=0.6)
ax_u.text(T     + 0.006*t_end, U_max_agg*0.04, '$T$',      fontsize=8, color='dimgray')
ax_u.text(T_min + 0.006*t_end, U_max_agg*0.04, '$T_{\\min}$', fontsize=8, color='dimgray')

ax_u.plot(t_arr, U_norm_c, color=col_ours,   lw=2.0, ls='-',  label='Ours')
ax_u.plot(t_arr, U_norm_g, color=col_garg,   lw=1.8, ls='--', label='Garg')
# ax_u.plot(t_arr, U_norm_k, color=col_krstic, lw=1.8, ls=':',  label='Krstic')

ax_u.set_xlabel('Time (s)')
ax_u.set_ylabel(r'$\|U(t)\| = \|[u_i(t)]_{i=1}^N\|$')
ax_u.set_title('Aggregate control effort')
ax_u.legend(fontsize=8); ax_u.grid(True, alpha=0.2)
ax_u.set_xlim(0, t_end); ax_u.set_ylim(0, U_max_agg*1.2)

fig.suptitle(
    rf'12D system ($N={N_SYS}$, joint $h(X)=c-\|X\|^2$): '
    rf'$T_{{\min}}={T_min:.2f}$ s, $T={T:.2f}$ s, '
    rf'$r_0={r0:.3f}$, $\sigma_{{\min}}={sigma_min:.3f}$',
    fontsize=9, y=1.01)
plt.savefig('data_formation/formation_12d.pdf', bbox_inches='tight', dpi=200)
plt.savefig('data_formation/formation_12d.png', bbox_inches='tight', dpi=200)
plt.close()
print("\nSaved: formation_12d.pdf/.png")

# ── Export .dat ───────────────────────────────────────────────────────
with open('data_formation/joint_barrier.dat', 'w') as f:
    f.write('t h_ours ht_ours tube_floor h_garg h_krstic '
            'U_norm_ours U_norm_garg U_norm_krstic\n')
    for k in range(Nt):
        f.write(f'{t_arr[k]:.5f} {H_c[k]:.6f} {HT_c[k]:.6f} '
                f'{r_floor[k]:.6f} {H_g[k]:.6f} {H_k[k]:.6f} '
                f'{U_norm_c[k]:.6f} {U_norm_g[k]:.6f} {U_norm_k[k]:.6f}\n')

for i in range(N_SYS):
    with open(f'data_formation/traj_agent{i+1}.dat', 'w') as f:
        f.write('t x1_ours x2_ours x1_garg x2_garg x1_krstic x2_krstic\n')
        for k in range(Nt):
            xi_c = X_c[k, 2*i:2*i+2]
            xi_g = X_g[k, 2*i:2*i+2]
            xi_k = X_k[k, 2*i:2*i+2]
            f.write(f'{t_arr[k]:.5f} '
                    f'{xi_c[0]:.6f} {xi_c[1]:.6f} '
                    f'{xi_g[0]:.6f} {xi_g[1]:.6f} '
                    f'{xi_k[0]:.6f} {xi_k[1]:.6f}\n')

print("Saved: .dat files\nDone.")