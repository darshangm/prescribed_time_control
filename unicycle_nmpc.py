
import numpy as np
import casadi as ca
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'lines.linewidth': 1.8,
})

# ──────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────
eps    = 0.5
delta  = 0.2
cx, cy = 2.0, 1.5
rho    = 0.6
v_max  = 1.5
w_max  = 2.0
T      = 20.0
N      = 40       # MPC horizon steps
dt     = 0.1      # MPC integration step (s)
dt_sim = 0.02     # simulation step (s)
alpha  = 10.0     # terminal position cost
t_end  = T + 3.0

x0 = np.array([4.0, 3.0, np.arctan2(-3.0, -4.0)])

# ──────────────────────────────────────────────
# Barriers (numpy)
# ──────────────────────────────────────────────
def h1_np(x): return eps**2 - x[0]**2 - x[1]**2
def h2_np(x): return (x[0]-cx)**2 + (x[1]-cy)**2 - rho**2

r0_1 = max(0.0, -h1_np(x0))
R1   = r0_1 + delta

def r1_np(t):
    return R1 * max(0.0, 1.0 - t/T)**2 - delta

print("=" * 50)
print(f"  x0       = ({x0[0]}, {x0[1]}),  theta0 = {np.degrees(x0[2]):.1f} deg")
print(f"  h1(x0)   = {h1_np(x0):.4f},  r0 = {r0_1:.4f},  R1 = {R1:.4f}")
print(f"  h2(x0)   = {h2_np(x0):.4f}")
print(f"  h2(0,0)  = {h2_np(np.zeros(3)):.4f}")
print(f"  T = {T} s,  N = {N},  dt = {dt} s")
print("=" * 50)

# ──────────────────────────────────────────────
# Build NMPC solver (CasADi / IPOPT)
# ──────────────────────────────────────────────
def build_solver():
    # States: X[3 x (N+1)], Controls: U[2 x N]
    X = ca.MX.sym('X', 3, N+1)
    U = ca.MX.sym('U', 2, N)
    P = ca.MX.sym('P', 4)   # [px0, py0, theta0, t_now]

    obj = ca.MX(0)
    g   = []
    lbg = []
    ubg = []

    # Initial state match
    g   += [X[:, 0] - P[:3]]
    lbg += [0.0]*3
    ubg += [0.0]*3

    for k in range(N):
        px, py, th = X[0,k], X[1,k], X[2,k]
        v,  w      = U[0,k], U[1,k]
        t_k        = P[3] + k*dt

        # Running cost
        obj += v**2 + w**2

        # Euler dynamics
        x_next = ca.vertcat(
            px + dt * v * ca.cos(th),
            py + dt * v * ca.sin(th),
            th + dt * w,
        )
        g   += [X[:, k+1] - x_next]
        lbg += [0.0]*3
        ubg += [0.0]*3

        # h1: constricting tube
        tau_k = ca.fmax(0.0, 1.0 - t_k/T)
        r1_k  = R1 * tau_k**2 - delta
        g.append(eps**2 - px**2 - py**2 + r1_k)
        lbg.append(0.0); ubg.append(ca.inf)

        # h2: obstacle avoidance (small margin for inter-step drift)
        g.append((px-cx)**2 + (py-cy)**2 - rho**2)
        lbg.append(0.01); ubg.append(ca.inf)

    # Terminal node constraints + cost
    px_N, py_N = X[0,N], X[1,N]
    t_N  = P[3] + N*dt
    tau_N = ca.fmax(0.0, 1.0 - t_N/T)
    g.append(eps**2 - px_N**2 - py_N**2 + R1*tau_N**2 - delta)
    lbg.append(0.0); ubg.append(ca.inf)
    g.append((px_N-cx)**2 + (py_N-cy)**2 - rho**2)
    lbg.append(0.0); ubg.append(ca.inf)

    obj += alpha * (px_N**2 + py_N**2)

    n_X = 3*(N+1)
    n_U = 2*N
    opt = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

    nlp  = {'x': opt, 'f': obj, 'g': ca.vertcat(*g), 'p': P}
    opts = {'ipopt.print_level': 0, 'ipopt.max_iter': 300,
            'ipopt.tol': 1e-5, 'print_time': 0}
    slvr = ca.nlpsol('mpc', 'ipopt', nlp, opts)

    lbx = [-ca.inf]*n_X + [-v_max, -w_max]*N
    ubx = [ ca.inf]*n_X + [ v_max,  w_max]*N

    return slvr, lbx, ubx, lbg, ubg, n_X, n_U

print("\nBuilding solver (first call compiles JIT)...")
solver, lbx, ubx, lbg, ubg, n_X, n_U = build_solver()
print("Done.")

# ──────────────────────────────────────────────
# Warm-start helper: straight-line in state space
# ──────────────────────────────────────────────
def warmstart(x_now):
    xs = np.zeros(n_X)
    for k in range(N+1):
        frac = k / N
        xs[3*k]   = x_now[0] * (1 - frac)
        xs[3*k+1] = x_now[1] * (1 - frac)
        xs[3*k+2] = x_now[2]
    us = np.zeros(n_U)
    us[0::2] = v_max * 0.3
    return np.concatenate([xs, us])

# ──────────────────────────────────────────────
# NMPC simulation loop
# ──────────────────────────────────────────────
def simulate_mpc(x0, t_end):
    t_arr  = np.arange(0.0, t_end + dt_sim, dt_sim)
    N_sim  = len(t_arr)
    x_arr  = np.zeros((N_sim, 3))
    u_arr  = np.zeros((N_sim, 2))
    h1_arr = np.zeros(N_sim)
    h2_arr = np.zeros(N_sim)
    fl_arr = np.zeros(N_sim)

    x_arr[0] = x0.copy()
    u_applied = np.array([0.0, 0.0])
    opt_prev  = None
    steps_per_mpc = max(1, int(round(dt / dt_sim)))

    for i, t_now in enumerate(t_arr):
        h1_arr[i] = h1_np(x_arr[i])
        h2_arr[i] = h2_np(x_arr[i])
        fl_arr[i] = -r1_np(t_now)

        # Re-solve every MPC step
        if i % steps_per_mpc == 0:
            p_val = np.array([x_arr[i,0], x_arr[i,1], x_arr[i,2], t_now])
            opt0  = opt_prev if opt_prev is not None else warmstart(x_arr[i])

            try:
                sol = solver(x0=opt0, p=p_val,
                             lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
                sol_vec   = np.array(sol['x']).flatten()
                u_applied = sol_vec[n_X:n_X+2]
                # Shift warm-start: drop first control, repeat last
                u_shifted = np.concatenate([sol_vec[n_X+2:], sol_vec[-2:]])
                x_shifted = sol_vec[:n_X]   # reuse state trajectory
                opt_prev  = np.concatenate([x_shifted, u_shifted])
            except Exception as e:
                print(f"  t={t_now:.2f}: solver exception: {e}")

        u_arr[i] = u_applied

        if i < N_sim - 1:
            px, py, th = x_arr[i]
            v, w = u_applied
            x_arr[i+1] = x_arr[i] + dt_sim * np.array([
                v*np.cos(th), v*np.sin(th), w
            ])

    return t_arr, x_arr, u_arr, h1_arr, h2_arr, fl_arr

# ──────────────────────────────────────────────
# Nominal simulation
# ──────────────────────────────────────────────
def simulate_nominal(x0, t_end):
    t_arr  = np.arange(0.0, t_end + dt_sim, dt_sim)
    x_arr  = np.zeros((len(t_arr), 3))
    h1_arr = np.zeros(len(t_arr))
    h2_arr = np.zeros(len(t_arr))
    x_arr[0] = x0.copy()
    kv, kw = 0.8, 2.0

    def adiff(a, b):
        d = a - b; return (d + np.pi) % (2*np.pi) - np.pi

    for i in range(len(t_arr)-1):
        px, py, th = x_arr[i]
        pn = np.hypot(px, py)
        td = np.arctan2(-py, -px)
        v  = np.clip(kv*pn, -v_max, v_max)
        w  = np.clip(kw*adiff(td, th), -w_max, w_max)
        h1_arr[i] = h1_np(x_arr[i])
        h2_arr[i] = h2_np(x_arr[i])
        x_arr[i+1] = x_arr[i] + dt_sim*np.array([v*np.cos(th), v*np.sin(th), w])
    h1_arr[-1] = h1_np(x_arr[-1]); h2_arr[-1] = h2_np(x_arr[-1])
    return t_arr, x_arr, h1_arr, h2_arr

# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
print("\nSimulating NMPC (this takes ~1-2 min)...")
t_arr, x_arr, u_arr, h1_arr, h2_arr, fl_arr = simulate_mpc(x0, t_end)

print("Simulating nominal...")
t_nom, x_nom, h1_nom, h2_nom = simulate_nominal(x0, t_end)

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
idx_T  = np.argmin(np.abs(t_arr - T))
u_norm = np.linalg.norm(u_arr, axis=1)
print("\n" + "="*50)
print(f"  NMPC h1(x(T)) = {h1_arr[idx_T]:+.4f}  (need >= {delta})")
print(f"  NMPC h2 min   = {h2_arr.min():+.4f}  (need >= 0)")
print(f"  NMPC ||p(T)|| = {np.linalg.norm(x_arr[idx_T,:2]):.4f}")
print(f"  Nom  h2 min   = {h2_nom.min():+.4f}  ({'COLLISION' if h2_nom.min()<0 else 'safe'})")
t_ne = next((t_nom[i] for i in range(len(t_nom)) if h1_nom[i]>=0), None)
print(f"  Nom  enters C1= {f'{t_ne:.1f} s' if t_ne else f'>{t_end:.0f} s'}")
print("="*50)

# ──────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────
col_mpc = '#2196F3'
col_nom = 'gray'
col_obs = '#E53935'
col_tgt = '#4CAF50'

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.subplots_adjust(wspace=0.38)
ax1, ax2, ax3 = axes
th_c = np.linspace(0, 2*np.pi, 400)

# Panel 1: Phase portrait
ax1.fill(eps*np.cos(th_c), eps*np.sin(th_c), color=col_tgt, alpha=0.2)
ax1.plot(eps*np.cos(th_c), eps*np.sin(th_c), color=col_tgt, lw=1.5,
         label=r'$\partial\mathcal{C}_1$ (target)')
ax1.fill(cx+rho*np.cos(th_c), cy+rho*np.sin(th_c), color=col_obs, alpha=0.25)
ax1.plot(cx+rho*np.cos(th_c), cy+rho*np.sin(th_c), color=col_obs, lw=1.5,
         label=r'Obstacle')
ax1.plot(x_nom[:,0], x_nom[:,1], color=col_nom, lw=1.5, ls='--',
         label='Nominal (collides)')
ax1.plot(x_arr[:,0], x_arr[:,1], color=col_mpc, lw=2.0,
         label='NMPC (ours)')
ax1.plot(*x0[:2], 'ko', ms=6)
ax1.annotate(r'$x(0)$', xy=x0[:2], xytext=(x0[0]+0.1, x0[1]+0.1), fontsize=9)
ax1.plot(x_arr[idx_T,0], x_arr[idx_T,1], 'o', color=col_mpc,
         ms=7, markeredgecolor='k', markeredgewidth=0.8, label=r'$x(T)$')
ax1.set_xlabel(r'$p_x$ (m)'); ax1.set_ylabel(r'$p_y$ (m)')
ax1.set_title('Phase portrait')
ax1.legend(loc='upper right', framealpha=0.9, fontsize=7.5)
ax1.set_aspect('equal'); ax1.grid(True, alpha=0.25)

# Panel 2: Barrier values
ax2.axhline(0,     color='k',       lw=0.8, ls='--', alpha=0.4)
ax2.axhline(delta, color=col_tgt,   lw=0.8, ls=':',  alpha=0.7)
ax2.axvline(T,     color='gray',    lw=1.0, ls=':',  alpha=0.7)
ax2.text(T+0.2, -1.5, f'$T={int(T)}$', fontsize=8, color='gray')
ax2.text(0.3, delta+0.3, r'$h_1=\delta$', fontsize=8, color=col_tgt)
ax2.plot(t_arr, fl_arr,  color=col_mpc, lw=1.0, ls=':', alpha=0.5,
         label=r'$-r_1(t)$')
ax2.plot(t_nom, h1_nom, color=col_nom, lw=1.2, ls='--',
         label=r'$h_1$ nom.')
ax2.plot(t_arr, h1_arr, color=col_mpc, lw=2.0,
         label=r'$h_1$ NMPC')
ax2.plot(t_arr, h2_arr, color=col_obs, lw=1.8, ls='-.',
         label=r'$h_2$ NMPC')
ax2.plot(t_nom, h2_nom, color=col_obs, lw=1.0, ls=':', alpha=0.5,
         label=r'$h_2$ nom.')
ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Barrier value')
ax2.set_title('Barrier values')
ax2.legend(loc='upper right', framealpha=0.9, fontsize=7)
ax2.grid(True, alpha=0.25); ax2.set_xlim(0, t_end)

# Panel 3: Control inputs
ax3.axvline(T, color='gray', lw=1.0, ls=':', alpha=0.7, label=f'$T={int(T)}$ s')
ax3.axhline( v_max, color='k', lw=0.7, ls=':', alpha=0.3)
ax3.axhline(-v_max, color='k', lw=0.7, ls=':', alpha=0.3)
ax3.plot(t_arr, u_arr[:,0], color=col_mpc, lw=2.0, label=r'$v(t)$ (m/s)')
ax3.plot(t_arr, u_arr[:,1], color=col_mpc, lw=1.5, ls='--',
         label=r'$\omega(t)$ (rad/s)')
ax3.set_xlabel('Time (s)'); ax3.set_ylabel(r'Control input')
ax3.set_title('Control inputs')
ax3.legend(loc='upper right', framealpha=0.9, fontsize=7.5)
ax3.grid(True, alpha=0.25); ax3.set_xlim(0, t_end)

fig.suptitle(
    r'Experiment 3: Unicycle NMPC — Constricting reach + obstacle avoidance'
    f'\n$\\epsilon={eps}$, $\\delta={delta}$, $T={int(T)}$ s, '
    f'obstacle $c=({cx},{cy})$, $\\rho={rho}$, $N={N}$, $\\Delta t={dt}$ s',
    fontsize=9, y=1.01
)

os.makedirs('figs', exist_ok=True)
plt.savefig('figs/experiment3.pdf', bbox_inches='tight', dpi=200)
plt.savefig('figs/experiment3.png', bbox_inches='tight', dpi=200)
plt.close()
print("\nSaved: figs/experiment3.pdf / .png")

# ──────────────────────────────────────────────
# Data export
# ──────────────────────────────────────────────
dat_dir = 'data'
os.makedirs(dat_dir, exist_ok=True)

with open(f'{dat_dir}/exp3_trajectory_mpc.dat', 'w') as f:
    f.write('t px py theta h1 h2 tube_floor v omega\n')
    for i in range(len(t_arr)):
        f.write(f'{t_arr[i]:.4f} {x_arr[i,0]:.6f} {x_arr[i,1]:.6f} '
                f'{x_arr[i,2]:.6f} {h1_arr[i]:.6f} {h2_arr[i]:.6f} '
                f'{fl_arr[i]:.6f} {u_arr[i,0]:.6f} {u_arr[i,1]:.6f}\n')
print("Saved: exp3_trajectory_mpc.dat")

with open(f'{dat_dir}/exp3_trajectory_nom.dat', 'w') as f:
    f.write('t px py theta h1 h2\n')
    for i in range(len(t_nom)):
        f.write(f'{t_nom[i]:.4f} {x_nom[i,0]:.6f} {x_nom[i,1]:.6f} '
                f'{x_nom[i,2]:.6f} {h1_nom[i]:.6f} {h2_nom[i]:.6f}\n')
print("Saved: exp3_trajectory_nom.dat")

th_arr = np.linspace(0, 2*np.pi, 400)
with open(f'{dat_dir}/exp3_target.dat', 'w') as f:
    f.write('x y\n')
    for th in th_arr:
        f.write(f'{eps*np.cos(th):.6f} {eps*np.sin(th):.6f}\n')
with open(f'{dat_dir}/exp3_obstacle.dat', 'w') as f:
    f.write('x y\n')
    for th in th_arr:
        f.write(f'{cx+rho*np.cos(th):.6f} {cy+rho*np.sin(th):.6f}\n')
print("Saved: exp3_target.dat, exp3_obstacle.dat")
