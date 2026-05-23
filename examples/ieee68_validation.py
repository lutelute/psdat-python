#!/usr/bin/env python3
"""IEEE 68-bus (NETS/NYPS) validation — PSDAT Program 2.

Validates:
1. Power flow convergence and bus voltages
2. Generator initialization (all 16 generators)
3. Classical swing model — modal analysis (inter-area modes)
4. Fault simulation (classical model)
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.makedirs('output', exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from psdat.data.ieee68 import get_system, BUS, BRANCH, GEN, MD, ED, TD, WS, BASEMVA, N_GEN
from psdat.models.powerflow import build_ybus, run_powerflow
from psdat.models.machine import MachineParams, init_from_powerflow
from psdat.simulation.classical import simulate_classical, compute_internal_emf

m = N_GEN  # 16 generators

# ── 1. Power flow ─────────────────────────────────────────────────────────
print("=" * 60)
print("IEEE 68-bus (NETS/NYPS) — PSDAT Validation")
print("=" * 60)
print("\n[1] Power flow ...")

pf = run_powerflow(BUS, BRANCH, GEN, baseMVA=BASEMVA)
conv_ok = pf.get('converged', False)
V_all   = pf['V']           # complex (68,)
Vm      = np.abs(V_all)
Va      = np.degrees(np.angle(V_all))

print(f"  Converged: {conv_ok}  (iterations: {pf.get('iterations','?')})")
print(f"  Voltage range: {Vm.min():.4f} – {Vm.max():.4f} pu")
print(f"  Angle  range:  {Va.min():.2f}° – {Va.max():.2f}°")

checks = [(1, 1.045), (2, 0.98), (9, 1.025), (16, 1.0)]
print()
for bus_1, vg_ref in checks:
    v = Vm[bus_1 - 1]
    ok = "✓" if abs(v - vg_ref) < 0.02 else "✗"
    print(f"  Bus {bus_1:2d}: |V|={v:.4f} pu  (ref {vg_ref:.3f})  {ok}")

# ── 2. Generator initialization ───────────────────────────────────────────
print("\n[2] Generator initialization (all 16 generators) ...")

gen_bus_0idx = list(range(m))   # generators at 0-indexed buses 0..15
Vg_gen    = Vm[:m]              # voltage magnitudes at gen buses
theta_gen = np.angle(V_all[:m]) # voltage angles at gen buses

# Actual PG/QG from power flow (not GEN-array specifications).
# QG is a free variable for PV buses; slack-bus PG is also determined by PF.
_Ybus_pf  = np.asarray(pf['Ybus'])
_S_bus    = V_all * np.conj(_Ybus_pf @ V_all)
PG_pu     = _S_bus[:m].real      # actual generated real power [pu]
QG_pu     = _S_bus[:m].imag      # actual generated reactive power [pu]

# Build MachineParams list
machine_params = [
    MachineParams(
        H=MD[0,i], Xd=MD[1,i], Xdp=MD[2,i], Xdpp=MD[3,i],
        Xq=MD[4,i], Xqp=MD[5,i], Xqpp=MD[6,i],
        Td0p=MD[7,i], Td0pp=MD[8,i], Tq0p=MD[9,i], Tq0pp=MD[10,i],
        Rs=MD[11,i], Xls=MD[12,i], Dm=MD[13,i]*0.005,
        KA=ED[0,i], TA=ED[1,i], KE=ED[2,i], TE=ED[3,i],
        KF=ED[4,i], TF=ED[5,i], Ax=ED[6,i], Bx=ED[7,i],
        TCH=TD[0,i], TSV=TD[1,i], RD=TD[2,i], ws=WS,
    )
    for i in range(m)
]

# Multi-machine init call (PSDAT style)
try:
    ic = init_from_powerflow(Vg_gen, theta_gen, PG_pu, QG_pu, machine_params)
    delta0 = ic['delta']      # (m,) rotor angles
    omega0_arr = ic['omega']  # (m,) speeds
    Efd0   = ic['Efd']
    Pm0    = ic['TM']         # mechanical torque ≈ Pm at steady state
    Eqp0   = ic['Eqp']
    Edp0   = ic['Edp']

    print(f"  δ₀ range:  {np.degrees(delta0.min()):.2f}° – {np.degrees(delta0.max()):.2f}°")
    print(f"  ω₀ range:  {omega0_arr.min():.6f} – {omega0_arr.max():.6f} pu (≈ 1.0)")
    print(f"  Efd₀:      {Efd0.min():.4f} – {Efd0.max():.4f} pu")
    print(f"  TM₀:       {Pm0.min():.4f} – {Pm0.max():.4f} pu")
    print(f"  ω₀ error:  {np.abs(omega0_arr - WS).max():.2e} rad/s (expect < 1e-10)")
    init_ok = True
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  Init FAILED: {e}")
    # Fallback: use power flow angles as delta0
    delta0   = Va[:m] * np.pi / 180
    omega0_arr = np.ones(m) * WS
    Pm0      = PG_pu
    Eqp0     = np.ones(m)
    Edp0     = np.zeros(m)
    init_ok  = False

# ── 3. Classical swing modal analysis ─────────────────────────────────────
print("\n[3] Classical swing modal analysis ...")

# Build original Y-bus (unmodified — also used in section [4] fault simulation)
Ybus_sp   = build_ybus(BUS, BRANCH, baseMVA=BASEMVA)
Ybus_orig = (Ybus_sp.toarray().copy() if hasattr(Ybus_sp, 'toarray')
             else np.array(Ybus_sp, dtype=complex))

# Classical internal EMF: E' = Vt + jXq*I  (Anderson & Fouad classical model)
# Consistent with simulate_classical / compute_internal_emf convention
E_prime_cl, delta0_cl = compute_internal_emf(
    Vg_gen, theta_gen, PG_pu, QG_pu, machine_params)
E_p = np.abs(E_prime_cl)
E_p = np.where(E_p < 0.5, 1.05, E_p)

# Extended Y-bus: [68 terminal buses | 16 internal EMF buses]  (Anderson & Fouad §2.5)
# Loads modelled as constant-Z at power-flow voltages; generators behind Xd' (transient)
Xq_arr = MD[2, :]  # Xd' (MD row 2), consistent with compute_internal_emf
n_ext  = 68 + m
Y_ext  = np.zeros((n_ext, n_ext), dtype=complex)
Y_ext[:68, :68] = Ybus_orig.copy()
for i in range(68):
    Vm2 = max(abs(V_all[i])**2, 1e-6)
    Y_ext[i, i] += complex(BUS[i, 2]/BASEMVA, -BUS[i, 3]/BASEMVA) / Vm2

for i in range(m):
    y_g = 1.0 / complex(0.0, max(Xq_arr[i], 1e-6))
    Y_ext[i,    i]    += y_g
    Y_ext[68+i, 68+i] += y_g
    Y_ext[i,    68+i] -= y_g
    Y_ext[68+i, i]    -= y_g

# Kron reduce: eliminate ALL 68 terminal buses → 16×16 Y_red (internal EMF buses)
elim = np.arange(68)
keep = np.arange(68, n_ext)
Ykk  = Y_ext[np.ix_(keep, keep)]
Ykl  = Y_ext[np.ix_(keep, elim)]
Yll  = Y_ext[np.ix_(elim, elim)]
Ylk  = Y_ext[np.ix_(elim, keep)]
try:
    Y_red = Ykk - Ykl @ np.linalg.solve(Yll, Ylk)
except np.linalg.LinAlgError:
    Y_red = Ykk - Ykl @ np.linalg.lstsq(Yll, Ylk, rcond=None)[0]

Gred = Y_red.real
Bred = Y_red.imag

# Synchronising torque matrix K[i,j] = dPe_i/dδ_j using rotor angles
# K[i,j] = E_i*E_j * (G_ij*sin(δ_ij) - B_ij*cos(δ_ij))  for j≠i
# K[i,i] = -sum_{j≠i} K[i,j]  (row-sum = 0 ensures COI invariance)
K = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        if i != j:
            dij = delta0_cl[i] - delta0_cl[j]
            K[i, j] = E_p[i]*E_p[j]*(Gred[i,j]*np.sin(dij) - Bred[i,j]*np.cos(dij))
    K[i, i] = -np.sum(K[i, :])

H_arr = MD[0, :]
D_arr = MD[13, :] * 0.005

# State matrix A (2m × 2m), x = [Δδ (rad), Δω (rad/s)]
# dΔδ/dt = Δω                          → A[:m, m:] = I
# dΔω/dt = -(ωs/2H)*K*Δδ - (Dm/2H)*Δω → A[m:, :] as below
A = np.zeros((2*m, 2*m))
A[:m, m:]  = np.eye(m)                           # dΔδ/dt = Δω
A[m:, :m]  = -np.diag(WS / (2.0*H_arr)) @ K     # -(ωs/2H)*K
A[m:, m:]  = -np.diag(D_arr / (2.0 * H_arr))    # -(Dm/2H)

eigvals = np.linalg.eigvals(A)
osc     = eigvals[np.abs(eigvals.imag) > 0.1]
pos     = osc[osc.imag > 0]
freq_hz = pos.imag / (2 * np.pi)
zeta    = -pos.real / np.abs(pos)

idx = np.argsort(freq_hz)
freq_hz = freq_hz[idx]; zeta = zeta[idx]; pos = pos[idx]

print(f"\n  {'#':>3}  {'λ':>28}  {'f (Hz)':>8}  {'ζ':>7}  type")
print("  " + "─" * 65)
for k in range(min(len(pos), 20)):
    ev = pos[k]; f = freq_hz[k]; z = zeta[k]
    if f < 0.1:  mtype = "non-osc"
    elif f < 0.8: mtype = "inter-area ★"
    elif f < 2.0: mtype = "local"
    else:         mtype = "plant"
    print(f"  {k+1:>3}  {ev.real:+.4f}{ev.imag:+.4f}j  {f:>8.4f}  {z:>7.4f}  {mtype}")

ia_mask   = (freq_hz > 0.1) & (freq_hz < 0.8)
ia_freqs  = freq_hz[ia_mask]
ia_damps  = zeta[ia_mask]
print(f"\n  Inter-area modes (0.1–0.8 Hz): {ia_mask.sum()} found")
for f, z in zip(ia_freqs, ia_damps):
    print(f"    {f:.4f} Hz,  ζ = {z:.4f}")
print(f"  Expected (Pal&Chaudhuri 2005): 4 modes ≈ 0.38, 0.47, 0.56, 0.65 Hz")

# ── 4. Fault simulation (via simulate_classical) ───────────────────────────
print("\n[4] Fault simulation (classical swing, 3-phase fault at bus 1) ...")

fault_bus_1idx = 1    # Generator bus 1 (NETS area equivalent, H=42 s)
gen_buses_1idx = list(range(1, m+1))

try:
    P_load_pu = BUS[:, 2] / BASEMVA
    Q_load_pu = BUS[:, 3] / BASEMVA

    result = simulate_classical(
        machines      = machine_params,
        Vg_pf         = Vg_gen,
        theta_g_pf    = theta_gen,
        PG            = PG_pu,
        QG            = QG_pu,
        Y_bus_pre     = Ybus_orig,   # original Y-bus (no gen admittances pre-added)
        gen_buses     = gen_buses_1idx,
        fault_bus     = fault_bus_1idx,
        t_fault       = 1.0,
        t_clear       = 1.10,        # 100 ms clearing
        t_end         = 8.0,
        dt            = 0.002,
        omega0        = WS,
        V_pf_all      = V_all,
        P_load_pu     = P_load_pu,
        Q_load_pu     = Q_load_pu,
    )

    t_arr     = result['t']
    delta_arr = result['delta']   # (N, m) degrees
    omega_arr = result['omega']   # (N, m) rad/s

    max_sep   = np.max(delta_arr.max(axis=1) - delta_arr.min(axis=1))
    final_sep = delta_arr[-1].max() - delta_arr[-1].min()
    stable    = final_sep < 180.0

    print(f"  Fault bus: {fault_bus_1idx},  t_fault=1.0s,  t_clear=1.10s")
    print(f"  Peak angle separation:  {max_sep:.2f}°")
    print(f"  Final separation:       {final_sep:.2f}°")
    print(f"  Result: {'STABLE ✓' if stable else 'UNSTABLE ✗'}")

    # Swing curve plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor='white',
                              gridspec_kw={'hspace': 0.38})
    colors = plt.cm.tab20(np.linspace(0, 1, m))
    for i in range(m):
        lbl = f'G{i+1} (H={MD[0,i]:.0f}s)' if i < 6 else None
        axes[0].plot(t_arr, delta_arr[:, i], color=colors[i],
                     lw=0.9, label=lbl, alpha=0.8)
        axes[1].plot(t_arr, (omega_arr[:, i] / WS - 1) * 100,
                     color=colors[i], lw=0.9, alpha=0.8)

    for ax in axes:
        ax.axvspan(1.0, 1.10, color='red', alpha=0.10)
        ax.axvline(1.0,  color='red', lw=0.9, ls='--', label='Fault on/off')
        ax.axvline(1.10, color='red', lw=0.9, ls='--')
        ax.grid(True, lw=0.3, alpha=0.6)

    axes[0].set_ylabel('Rotor angle δ (°)', fontsize=11)
    axes[0].set_title(
        f'IEEE 68-bus NETS/NYPS — 3-phase fault at bus {fault_bus_1idx} '
        f'(t = 1.00–1.10 s, 100 ms)\n'
        f'16 generators  |  Peak separation: {max_sep:.1f}°  |  '
        f'{"STABLE" if stable else "UNSTABLE"}',
        fontsize=11)
    axes[0].legend(fontsize=8, ncol=3, loc='upper right')

    axes[1].set_xlabel('Time (s)', fontsize=11)
    axes[1].set_ylabel('Freq deviation Δf (%)', fontsize=11)
    axes[1].axhline(0, color='k', lw=0.7, ls=':')
    axes[1].set_title('Frequency deviation (all 16 generators)', fontsize=10)

    plt.tight_layout()
    fig.savefig('output/ieee68_fault_sim.png', dpi=150)
    plt.close()
    print("  → output/ieee68_fault_sim.png")

except Exception as e:
    import traceback
    print(f"  Simulation error: {e}")
    traceback.print_exc()

# ── 5. Eigenvalue map ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 8), facecolor='white')

# Damping lines at 5%, 10%, 20%
for zt in [0.05, 0.10, 0.20]:
    ang = np.arccos(zt)
    ax.plot([-4, 0], [0,  4*np.tan(ang)], 'k--', lw=0.5, alpha=0.4)
    ax.plot([-4, 0], [0, -4*np.tan(ang)], 'k--', lw=0.5, alpha=0.4)
    ax.text(-0.05, 4*np.tan(ang)*0.97, f'ζ={int(zt*100)}%',
            fontsize=7, ha='right', color='gray', va='top')

# All modes
ax.scatter(pos.real,  pos.imag, s=55, c='steelblue', zorder=3, label='modes')
ax.scatter(pos.real, -pos.imag, s=55, c='steelblue', zorder=3)

# Highlight inter-area modes
if ia_mask.any():
    ia_ev = pos[ia_mask]
    ax.scatter(ia_ev.real,  ia_ev.imag, s=120, c='red', zorder=4,
               label=f'inter-area ({ia_mask.sum()} modes)')
    ax.scatter(ia_ev.real, -ia_ev.imag, s=120, c='red', zorder=4)
    for ev, fq in zip(ia_ev, ia_freqs):
        ax.annotate(f'{fq:.3f} Hz', xy=(ev.real, ev.imag),
                    xytext=(ev.real + 0.05, ev.imag + 0.3),
                    fontsize=9, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.7))

ax.axhline(0, color='k', lw=0.5)
ax.axvline(0, color='k', lw=0.5)
ax.set_xlabel('Real part  σ  (1/s)', fontsize=11)
ax.set_ylabel('Imaginary part  jω  (rad/s)', fontsize=11)
ax.set_title(
    'IEEE 68-bus (NETS/NYPS) — Eigenvalue Map\n'
    '16-machine classical swing model  |  '
    f'Inter-area modes (★) vs local modes\n'
    f'Expected: ~4 inter-area modes ≈ 0.38, 0.47, 0.56, 0.65 Hz  '
    f'(Pal & Chaudhuri 2005)',
    fontsize=10)
ax.legend(fontsize=9)
ax.set_xlim(-1.5, 0.3)
ax.set_ylim(-15, 15)
ax.grid(True, lw=0.3, alpha=0.5)
plt.tight_layout()
fig.savefig('output/ieee68_eigenvalues.png', dpi=150)
plt.close()
print("\n  → output/ieee68_eigenvalues.png")

print("\n" + "=" * 60)
print("IEEE 68-bus validation complete.")
print("=" * 60)
