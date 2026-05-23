"""Newton-Raphson AC power flow — MATPOWER-compatible.

Build Y-bus from branch data and solve the power flow equations using the
polar-form Jacobian (same algorithm as MATPOWER's ``newtonpf``).

Bus type codes (MATPOWER convention):
    1 — PQ  (load bus)
    2 — PV  (voltage-controlled bus)
    3 — Slack (reference / V-theta fixed)

MATPOWER bus array column indices (0-based):
    0  BUS_I     — bus number
    1  BUS_TYPE  — 1=PQ, 2=PV, 3=slack
    2  PD        — real power demand [MW]
    3  QD        — reactive power demand [MVAr]
    4  GS        — shunt conductance [MW at V=1 pu]
    5  BS        — shunt susceptance [MVAr at V=1 pu]
    6  BUS_AREA  — area number
    7  VM        — voltage magnitude [pu]
    8  VA        — voltage angle [deg]
    9  BASE_KV   — base voltage [kV]
    10 ZONE      — zone number
    11 VMAX      — maximum voltage [pu]
    12 VMIN      — minimum voltage [pu]

MATPOWER branch array column indices (0-based):
    0  F_BUS    — from bus number
    1  T_BUS    — to bus number
    2  BR_R     — resistance [pu]
    3  BR_X     — reactance [pu]
    4  BR_B     — total line charging susceptance [pu]
    5  RATE_A   — MVA rating A (long-term)
    6  RATE_B   — MVA rating B (short-term)
    7  RATE_C   — MVA rating C (emergency)
    8  TAP      — transformer off-nominal turns ratio (0 → 1)
    9  SHIFT    — transformer phase shift angle [deg]
    10 BR_STATUS — initial branch status (1=in service, 0=out)
    11 ANGMIN   — minimum angle difference [deg]
    12 ANGMAX   — maximum angle difference [deg]

MATPOWER generator array column indices (0-based):
    0  GEN_BUS  — bus number
    1  PG       — real power output [MW]
    2  QG       — reactive power output [MVAr]
    3  QMAX     — max reactive power [MVAr]
    4  QMIN     — min reactive power [MVAr]
    5  VG       — voltage magnitude setpoint [pu]
    6  MBASE    — machine base [MVA]
    7  GEN_STATUS — status (1=in service)
    8  PMAX     — max real power [MW]
    9  PMIN     — min real power [MW]

Legacy simple interface (build_ybus with 2 args, run_powerflow with 2 args)
is preserved for backward compatibility.

References:
    R. D. Zimmerman, C. E. Murillo-Sanchez, R. J. Thomas, "MATPOWER: Steady-State
    Operations, Planning and Analysis Tools for Power Systems Research and Education,"
    IEEE Transactions on Power Systems, vol.26, no.1, pp.12-19, Feb 2011.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Y-bus construction  (MATPOWER-compatible)
# ---------------------------------------------------------------------------

def build_ybus(
    buses_or_branch_data,
    branches_or_n_buses,
    baseMVA: float = 100.0,
) -> np.ndarray:
    """Build the complex nodal admittance matrix Y-bus.

    Supports two calling conventions:

    **MATPOWER-style** (3 arguments)::

        Ybus = build_ybus(buses, branches, baseMVA=100.0)

        buses    : ndarray (n, 13+)  — MATPOWER bus array
        branches : ndarray (nl, 13+) — MATPOWER branch array
        baseMVA  : float             — system MVA base

    **Legacy simple style** (2 arguments)::

        Ybus = build_ybus(branch_data, n_buses)

        branch_data : ndarray (nl, 7)
            Columns: [from_bus(1-based), to_bus, R, X, B_total, tap, shift_deg]
        n_buses : int

    Parameters
    ----------
    buses_or_branch_data : ndarray
        Either the MATPOWER bus array or a simplified branch_data array.
    branches_or_n_buses : ndarray or int
        Either the MATPOWER branch array or the number of buses (int).
    baseMVA : float
        System MVA base (only used in MATPOWER-style call).

    Returns
    -------
    Ybus : ndarray, shape (n, n), complex128
        Nodal admittance matrix.

    Notes
    -----
    For a branch from bus *f* to bus *t* with series impedance z = R + jX,
    total line charging b, and off-nominal turns ratio *a = |tap|*exp(j*shift*):

        y_series = 1/z
        Y_ff = (y_series + jb/2) / |a|²
        Y_tt =  y_series + jb/2
        Y_ft = -y_series / conj(a)
        Y_tf = -y_series / a
    """
    # Detect calling convention
    if isinstance(branches_or_n_buses, (int, np.integer)):
        # Legacy: build_ybus(branch_data, n_buses)
        return _build_ybus_simple(buses_or_branch_data, int(branches_or_n_buses))
    else:
        # MATPOWER: build_ybus(buses, branches, baseMVA)
        return _build_ybus_matpower(
            np.asarray(buses_or_branch_data, dtype=float),
            np.asarray(branches_or_n_buses,  dtype=float),
            baseMVA,
        )


def _build_ybus_simple(branch_data: np.ndarray, n_buses: int) -> np.ndarray:
    """Legacy Y-bus builder (branch_data columns: from,to,R,X,B,tap,shift_deg)."""
    n_br = branch_data.shape[0]
    Y = np.zeros((n_buses, n_buses), dtype=complex)
    for k in range(n_br):
        fb  = int(branch_data[k, 0]) - 1
        tb  = int(branch_data[k, 1]) - 1
        R   = branch_data[k, 2]
        X   = branch_data[k, 3]
        B   = branch_data[k, 4]
        tap = branch_data[k, 5]
        shift = np.deg2rad(branch_data[k, 6])
        ys = 1.0 / complex(R, X) if (R != 0 or X != 0) else 0j
        bc = 1j * B / 2.0
        t  = tap * np.exp(1j * shift) if tap != 0 else np.exp(1j * shift)
        Y[fb, fb] += (ys + bc) / (abs(t) ** 2)
        Y[tb, tb] += ys + bc
        Y[fb, tb] -= ys / np.conj(t)
        Y[tb, fb] -= ys / t
    return Y


def _build_ybus_matpower(
    buses: np.ndarray,
    branches: np.ndarray,
    baseMVA: float,
) -> np.ndarray:
    """MATPOWER-format Y-bus builder."""
    n  = buses.shape[0]
    nl = branches.shape[0]

    bus_nums = buses[:, 0].astype(int)
    bus_idx  = {b: i for i, b in enumerate(bus_nums)}

    Ybus = np.zeros((n, n), dtype=complex)

    # Bus shunt admittances (GS in MW, BS in MVAr at V=1 pu → divide by baseMVA)
    for i in range(n):
        GS = buses[i, 4] / baseMVA
        BS = buses[i, 5] / baseMVA
        Ybus[i, i] += GS + 1j * BS

    # Branch contributions
    for k in range(nl):
        br_status = int(branches[k, 10]) if branches.shape[1] > 10 else 1
        if br_status == 0:
            continue

        fi = bus_idx[int(branches[k, 0])]
        ti = bus_idx[int(branches[k, 1])]

        R   = branches[k, 2]
        X   = branches[k, 3]
        B   = branches[k, 4]
        tap = branches[k, 8] if branches[k, 8] != 0.0 else 1.0
        shift_deg = branches[k, 9] if branches.shape[1] > 9 else 0.0

        a   = tap * np.exp(1j * np.deg2rad(shift_deg))
        z   = R + 1j * X
        y_s = 1.0 / z if abs(z) > 1e-15 else (1e10 + 0j)
        y_c = 1j * B / 2.0

        Ybus[fi, fi] += (y_s + y_c) / (tap * tap)
        Ybus[ti, ti] += y_s + y_c
        Ybus[fi, ti] -= y_s / np.conj(a)
        Ybus[ti, fi] -= y_s / a

    return Ybus


# ---------------------------------------------------------------------------
# Sparse Y-bus and Jacobian (for large systems)
# ---------------------------------------------------------------------------

def _build_ybus_sparse(
    buses: np.ndarray,
    branches: np.ndarray,
    baseMVA: float,
) -> sps.csr_matrix:
    """Build Y-bus as a scipy.sparse CSR matrix (efficient for large networks)."""
    n  = buses.shape[0]
    nl = branches.shape[0]
    bus_nums = buses[:, 0].astype(int)
    bus_idx  = {b: i for i, b in enumerate(bus_nums)}

    rows, cols, data = [], [], []

    # Bus shunts
    for i in range(n):
        GS = buses[i, 4] / baseMVA
        BS = buses[i, 5] / baseMVA
        rows.append(i); cols.append(i); data.append(GS + 1j * BS)

    # Branches
    for k in range(nl):
        br_status = int(branches[k, 10]) if branches.shape[1] > 10 else 1
        if br_status == 0:
            continue
        fi = bus_idx[int(branches[k, 0])]
        ti = bus_idx[int(branches[k, 1])]
        R   = branches[k, 2]; X = branches[k, 3]; B = branches[k, 4]
        tap = branches[k, 8] if branches[k, 8] != 0.0 else 1.0
        shift = branches[k, 9] if branches.shape[1] > 9 else 0.0
        a   = tap * np.exp(1j * np.deg2rad(shift))
        z   = R + 1j * X
        y_s = (1.0 / z) if abs(z) > 1e-15 else (1e10 + 0j)
        y_c = 1j * B / 2.0

        rows += [fi, ti, fi, ti]
        cols += [fi, ti, ti, fi]
        data += [(y_s + y_c) / (tap * tap), y_s + y_c,
                 -y_s / np.conj(a), -y_s / a]

    Y = sps.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=complex)
    return Y.tocsr()


def _sparse_jacobian(
    Ybus: sps.csr_matrix,
    V: np.ndarray,
    pvpq: np.ndarray,
    pq: np.ndarray,
) -> "tuple[sps.csr_matrix, np.ndarray, np.ndarray]":
    """Build sparse polar-form Jacobian.

    Uses vectorised COO assembly — O(nnz) instead of O(n²).

    Returns
    -------
    J : csr_matrix   shape (n_pvpq + n_pq) × (n_pvpq + n_pq)
    P : ndarray      real power injection
    Q : ndarray      reactive power injection
    """
    n      = len(V)
    Vmag   = np.abs(V)
    Vang   = np.angle(V)

    I_bus  = Ybus @ V
    S_bus  = V * np.conj(I_bus)
    P      = S_bus.real
    Q      = S_bus.imag

    # Extract COO entries of Y-bus
    Ycoo  = Ybus.tocoo()
    ry, cy = Ycoo.row, Ycoo.col
    G_y   = Ycoo.data.real
    B_y   = Ycoo.data.imag

    th    = Vang[ry] - Vang[cy]
    ViVj  = Vmag[ry] * Vmag[cy]
    sin_t = np.sin(th)
    cos_t = np.cos(th)

    H_all = ViVj * (G_y * sin_t - B_y * cos_t)   # dP/dθ off-diag
    N_all = ViVj * (G_y * cos_t + B_y * sin_t)   # dP/dV·V off-diag

    # Diagonal of Y (scalar arrays, length n)
    Yd    = np.asarray(Ybus.diagonal())
    B_d   = Yd.imag
    G_d   = Yd.real

    # Vectorised index maps: bus index → local pvpq/pq index (-1 if absent)
    pvpq_loc = np.full(n, -1, dtype=int)
    pvpq_loc[pvpq] = np.arange(len(pvpq))
    pq_loc = np.full(n, -1, dtype=int)
    pq_loc[pq] = np.arange(len(pq))
    pq_in_pvpq = pvpq_loc[pq]   # row in pvpq block for each PQ bus

    n_pvpq = len(pvpq);  n_pq = len(pq)
    total  = n_pvpq + n_pq

    od  = ry != cy           # off-diagonal mask
    r_od, c_od = ry[od], cy[od]
    H_od = H_all[od]
    N_od = N_all[od]

    # Diagonal values (only for included buses)
    H_d_vals = -Q[pvpq] - B_d[pvpq] * Vmag[pvpq] ** 2
    N_d_vals =  P[pq]   + G_d[pq]   * Vmag[pq]   ** 2
    J_d_vals =  P[pq]   - G_d[pq]   * Vmag[pq]   ** 2
    L_d_vals =  Q[pq]   - B_d[pq]   * Vmag[pq]   ** 2

    # ── H block (pvpq × pvpq): dP/dθ ────────────────────────────────────
    r_h = pvpq_loc[r_od]; c_h = pvpq_loc[c_od]
    mh  = (r_h >= 0) & (c_h >= 0)
    H_r = np.r_[r_h[mh], np.arange(n_pvpq)]
    H_c = np.r_[c_h[mh], np.arange(n_pvpq)]
    H_v = np.r_[H_od[mh], H_d_vals]

    # ── N block (pvpq × pq): dP/dV·V ────────────────────────────────────
    r_n = pvpq_loc[r_od]; c_n = pq_loc[c_od]
    mn  = (r_n >= 0) & (c_n >= 0)
    N_r = np.r_[r_n[mn], pq_in_pvpq]
    N_c = np.r_[c_n[mn], np.arange(n_pq)]
    N_v = np.r_[N_od[mn], N_d_vals]

    # ── Jm block (pq × pvpq): dQ/dθ ─────────────────────────────────────
    r_j = pq_loc[r_od]; c_j = pvpq_loc[c_od]
    mj  = (r_j >= 0) & (c_j >= 0)
    Jm_r = np.r_[r_j[mj], np.arange(n_pq)]
    Jm_c = np.r_[c_j[mj], pq_in_pvpq]
    Jm_v = np.r_[-N_od[mj], J_d_vals]   # dQ/dθ = -dP/dV·V (off-diag)

    # ── L block (pq × pq): dQ/dV·V ───────────────────────────────────────
    r_l = pq_loc[r_od]; c_l = pq_loc[c_od]
    ml  = (r_l >= 0) & (c_l >= 0)
    L_r = np.r_[r_l[ml], np.arange(n_pq)]
    L_c = np.r_[c_l[ml], np.arange(n_pq)]
    L_v = np.r_[H_od[ml], L_d_vals]   # dQ/dV·V = dP/dθ (off-diag)

    # Stack into full Jacobian using block offsets
    all_r = np.r_[H_r,           N_r,           Jm_r + n_pvpq,   L_r + n_pvpq]
    all_c = np.r_[H_c,           N_c + n_pvpq,  Jm_c,            L_c + n_pvpq]
    all_v = np.r_[H_v,           N_v,           Jm_v,            L_v          ]

    J = sps.coo_matrix((all_v, (all_r, all_c)), shape=(total, total)).tocsr()
    return J, P, Q


def _run_pf_sparse(
    buses:      np.ndarray,
    branches:   np.ndarray,
    generators: np.ndarray,
    baseMVA:    float,
    tol:        float,
    max_iter:   int,
) -> Dict:
    """Sparse Newton-Raphson power flow (for large systems ≥ 300 buses)."""
    n  = buses.shape[0]
    ng = generators.shape[0] if generators is not None else 0

    bus_nums = buses[:, 0].astype(int)
    bus_idx  = {b: i for i, b in enumerate(bus_nums)}
    bus_type = buses[:, 1].astype(int)

    slack_i  = np.where(bus_type == 3)[0]
    pv_i     = np.where(bus_type == 2)[0]
    pq_i     = np.where(bus_type == 1)[0]
    pvpq     = np.concatenate([pv_i, pq_i])

    Vmag = buses[:, 7].copy()
    Vang = np.deg2rad(buses[:, 8].copy())

    P_sched = -buses[:, 2] / baseMVA
    Q_sched = -buses[:, 3] / baseMVA

    if generators is not None:
        for g in range(ng):
            gs = int(generators[g, 7]) if generators.shape[1] > 7 else 1
            if gs == 0:
                continue
            gi = bus_idx[int(generators[g, 0])]
            P_sched[gi] += generators[g, 1] / baseMVA
            if bus_type[gi] in (2, 3):
                Vmag[gi] = generators[g, 5]
            else:
                Q_sched[gi] += generators[g, 2] / baseMVA

    Ybus = _build_ybus_sparse(buses, branches, baseMVA)

    converged  = False
    iterations = 0

    for _it in range(max_iter):
        V = Vmag * np.exp(1j * Vang)
        J, P_calc, Q_calc = _sparse_jacobian(Ybus, V, pvpq, pq_i)

        dP = (P_sched - P_calc)[pvpq]
        dQ = (Q_sched - Q_calc)[pq_i]
        F  = np.r_[dP, dQ]
        mis = float(np.max(np.abs(F)))
        if mis < tol:
            converged  = True
            iterations = _it
            break

        try:
            dx = spla.spsolve(J, F)
        except Exception:
            break

        n_pvpq = len(pvpq)
        # Backtracking Armijo line search
        alpha = 1.0
        for _ in range(10):
            Vang_t = Vang.copy()
            Vmag_t = Vmag.copy()
            Vang_t[pvpq] += np.clip(dx[:n_pvpq] * alpha, -np.pi, np.pi)
            Vmag_t[pq_i] *= np.maximum(1.0 + np.clip(dx[n_pvpq:] * alpha, -0.5, 0.5), 0.01)
            Vmag_t = np.maximum(Vmag_t, 0.01)
            Vt     = Vmag_t * np.exp(1j * Vang_t)
            _, P_t, Q_t = _sparse_jacobian(Ybus, Vt, pvpq, pq_i)
            Ft = np.r_[(P_sched - P_t)[pvpq], (Q_sched - Q_t)[pq_i]]
            if np.max(np.abs(Ft)) < mis:
                break
            alpha *= 0.5
        Vang[pvpq] += np.clip(dx[:n_pvpq] * alpha, -np.pi, np.pi)
        Vmag[pq_i] *= np.maximum(1.0 + np.clip(dx[n_pvpq:] * alpha, -0.5, 0.5), 0.01)
        Vmag = np.maximum(Vmag, 0.01)
    else:
        iterations = max_iter

    V_sol   = Vmag * np.exp(1j * Vang)
    S_final = V_sol * np.conj(Ybus @ V_sol)
    dP_f    = (P_sched - S_final.real)[pvpq]
    dQ_f    = (Q_sched - S_final.imag)[pq_i]
    mismatch = float(np.max(np.abs(np.r_[dP_f, dQ_f]))) if n > 1 else 0.0

    P_gen = np.zeros(ng); Q_gen = np.zeros(ng)
    if generators is not None:
        for g in range(ng):
            gi = bus_idx[int(generators[g, 0])]
            P_gen[g] = generators[g, 1] / baseMVA
            if bus_type[gi] in (2, 3):
                Q_gen[g] = S_final[gi].imag + buses[gi, 3] / baseMVA

    return {
        'V':          V_sol,
        'Vmag':       Vmag,
        'Vang_deg':   np.degrees(Vang),
        'P_gen':      P_gen,
        'Q_gen':      Q_gen,
        'P_inj':      S_final.real,
        'Q_inj':      S_final.imag,
        'converged':  converged,
        'iterations': iterations,
        'mismatch':   mismatch,
        'Ybus':       Ybus,
    }


# ---------------------------------------------------------------------------
# Newton-Raphson power flow
# ---------------------------------------------------------------------------

def run_powerflow(
    buses_or_bus_data,
    branches_or_branch_data,
    generators: Optional[np.ndarray] = None,
    baseMVA: float = 100.0,
    tol: float = 1e-8,
    max_iter: int = 50,
):
    """Solve AC power flow using Newton-Raphson in polar form.

    Supports two calling conventions:

    **MATPOWER-style** (3-6 arguments)::

        result = run_powerflow(buses, branches, generators, baseMVA, tol, max_iter)

        buses      : ndarray (n, 13+)  — MATPOWER bus array (VM/VA as flat start)
        branches   : ndarray (nl, 13+) — MATPOWER branch array
        generators : ndarray (ng, 10+) — MATPOWER generator array
        Returns dict with keys: V, Vmag, Vang_deg, P_gen, Q_gen, P_inj, Q_inj,
                                converged, iterations, mismatch, Ybus

    **Legacy simple style** (2 arguments)::

        V, S_inj, n_iter, converged = run_powerflow(bus_data, branch_data)

        bus_data : ndarray (n, 10)
            Cols: [bus_id, type(1=slack/2=PV/3=PQ), Vmag, Vang_deg,
                   Pgen_MW, Qgen_Mvar, Pload_MW, Qload_Mvar, Gsh, Bsh]
        branch_data : ndarray (nl, 7)  — see build_ybus() legacy form
        Returns (V_complex, S_inj_complex, n_iter, converged)

    Parameters
    ----------
    buses_or_bus_data : ndarray
    branches_or_branch_data : ndarray
    generators : ndarray, optional
        Required for MATPOWER-style call. If None, legacy call assumed.
    baseMVA : float
    tol : float
        Convergence tolerance on infinity-norm of P,Q mismatch [pu].
    max_iter : int

    Notes
    -----
    Polar-form Newton-Raphson (standard textbook algorithm):
        f = [ΔP; ΔQ],  J·[Δθ; ΔV/V] = f,  θ += Δθ,  V *= (1 + ΔV/V)

    PQ buses: both P and Q mismatches.
    PV buses: P mismatch only; |V| fixed.
    Slack bus: no mismatch; θ and |V| fixed.
    """
    if generators is None:
        # Legacy call
        V, S_inj, n_iter, conv = _run_pf_simple(
            np.asarray(buses_or_bus_data,      dtype=float),
            np.asarray(branches_or_branch_data, dtype=float),
            tol=tol, max_iter=max_iter,
        )
        return V, S_inj, n_iter, conv
    else:
        buses_arr = np.asarray(buses_or_bus_data,      dtype=float)
        brchs_arr = np.asarray(branches_or_branch_data, dtype=float)
        gens_arr  = np.asarray(generators,              dtype=float)
        # Use sparse solver for large systems (≥ 300 buses) — O(nnz) Jacobian
        if buses_arr.shape[0] >= 300:
            return _run_pf_sparse(
                buses_arr, brchs_arr, gens_arr,
                baseMVA=baseMVA, tol=tol, max_iter=max_iter,
            )
        return _run_pf_matpower(
            buses_arr, brchs_arr, gens_arr,
            baseMVA=baseMVA, tol=tol, max_iter=max_iter,
        )


def _run_pf_simple(
    bus_data: np.ndarray,
    branch_data: np.ndarray,
    tol: float,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """Legacy NR power flow (bus_data cols: id,type,Vmag,Vang,Pg,Qg,Pl,Ql,Gs,Bs)."""
    n = bus_data.shape[0]
    S_BASE = 100.0

    bus_type = bus_data[:, 1].astype(int)  # 1=slack, 2=PV, 3=PQ
    Vmag  = bus_data[:, 2].copy()
    Vang  = np.deg2rad(bus_data[:, 3].copy())
    P_spec = (bus_data[:, 4] - bus_data[:, 6]) / S_BASE
    Q_spec = (bus_data[:, 5] - bus_data[:, 7]) / S_BASE

    Y = _build_ybus_simple(branch_data, n)
    G, B = Y.real, Y.imag

    slack_idx = np.where(bus_type == 1)[0]
    pv_idx    = np.where(bus_type == 2)[0]
    pq_idx    = np.where(bus_type == 3)[0]
    theta_idx = np.concatenate([pv_idx, pq_idx])
    v_idx     = pq_idx

    V = Vmag * np.exp(1j * Vang)
    n_iter = 0
    converged = False

    for n_iter in range(1, max_iter + 1):
        Vmag  = np.abs(V)
        theta = np.angle(V)
        S_calc = V * np.conj(Y @ V)
        P_calc, Q_calc = S_calc.real, S_calc.imag

        dP = P_spec - P_calc
        dQ = Q_spec - Q_calc
        dP[slack_idx] = 0.0
        dQ[slack_idx] = 0.0
        dQ[pv_idx]    = 0.0
        mismatch = np.concatenate([dP[theta_idx], dQ[v_idx]])
        if np.max(np.abs(mismatch)) < tol:
            converged = True
            break

        n_th = len(theta_idx)
        n_v  = len(v_idx)
        J = np.zeros((n_th + n_v, n_th + n_v))

        for a, i in enumerate(theta_idx):
            for b, j in enumerate(theta_idx):
                if i == j:
                    J[a, b] = -Q_calc[i] - B[i, i] * Vmag[i] ** 2
                else:
                    th = theta[i] - theta[j]
                    J[a, b] = Vmag[i] * Vmag[j] * (G[i,j]*np.sin(th) - B[i,j]*np.cos(th))
        for a, i in enumerate(theta_idx):
            for b, j in enumerate(v_idx):
                if i == j:
                    J[a, n_th + b] = P_calc[i] / Vmag[i] + G[i, i] * Vmag[i]
                else:
                    th = theta[i] - theta[j]
                    J[a, n_th + b] = Vmag[i] * (G[i,j]*np.cos(th) + B[i,j]*np.sin(th))
        for a, i in enumerate(v_idx):
            for b, j in enumerate(theta_idx):
                if i == j:
                    J[n_th + a, b] = P_calc[i] / Vmag[i] - G[i, i] * Vmag[i]
                else:
                    th = theta[i] - theta[j]
                    J[n_th + a, b] = -Vmag[i] * Vmag[j] * (G[i,j]*np.cos(th) + B[i,j]*np.sin(th))
        for a, i in enumerate(v_idx):
            for b, j in enumerate(v_idx):
                if i == j:
                    J[n_th + a, n_th + b] = Q_calc[i] / Vmag[i] - B[i, i] * Vmag[i]
                else:
                    th = theta[i] - theta[j]
                    J[n_th + a, n_th + b] = Vmag[i] * (G[i,j]*np.sin(th) - B[i,j]*np.cos(th))

        dx = np.linalg.solve(J, mismatch)
        for a, i in enumerate(theta_idx):
            theta[i] += dx[a]
        for a, i in enumerate(v_idx):
            Vmag[i] += dx[n_th + a]
        V = Vmag * np.exp(1j * theta)

    S_inj = V * np.conj(Y @ V)
    return V, S_inj, n_iter, converged


def _run_pf_matpower(
    buses: np.ndarray,
    branches: np.ndarray,
    generators: np.ndarray,
    baseMVA: float,
    tol: float,
    max_iter: int,
) -> Dict:
    """MATPOWER-format NR power flow returning result dict."""
    n  = buses.shape[0]
    ng = generators.shape[0]

    bus_nums = buses[:, 0].astype(int)
    bus_idx  = {b: i for i, b in enumerate(bus_nums)}
    bus_type = buses[:, 1].astype(int)  # 1=PQ, 2=PV, 3=slack

    pq_idx   = np.where(bus_type == 1)[0]
    pv_idx   = np.where(bus_type == 2)[0]
    ref_idx  = np.where(bus_type == 3)[0]
    if len(ref_idx) == 0:
        raise ValueError("No slack bus (type 3) found.")
    pvpq_idx = np.concatenate([pv_idx, pq_idx])

    Ybus = _build_ybus_matpower(buses, branches, baseMVA)
    G, B = Ybus.real, Ybus.imag

    # Scheduled net injections [pu]
    P_sched = -buses[:, 2] / baseMVA   # -PD (load demand)
    Q_sched = -buses[:, 3] / baseMVA   # -QD

    # Initial voltage
    Vmag = buses[:, 7].copy()
    Vang = np.deg2rad(buses[:, 8])

    for g in range(ng):
        gs = int(generators[g, 7]) if generators.shape[1] > 7 else 1
        gb = int(generators[g, 0])
        gi = bus_idx[gb]
        if gs == 0:
            continue
        P_sched[gi] += generators[g, 1] / baseMVA
        if bus_type[gi] in (2, 3):
            Vmag[gi] = generators[g, 5]   # VG setpoint
        else:
            Q_sched[gi] += generators[g, 2] / baseMVA

    converged = False
    iterations = 0

    for _it in range(max_iter):
        V      = Vmag * np.exp(1j * Vang)
        S_calc = V * np.conj(Ybus @ V)
        P_calc = S_calc.real
        Q_calc = S_calc.imag

        dP = P_sched - P_calc
        dQ = Q_sched - Q_calc
        F  = np.concatenate([dP[pvpq_idx], dQ[pq_idx]])
        mis = np.max(np.abs(F))
        if mis < tol:
            converged  = True
            iterations = _it
            break

        npvpq = len(pvpq_idx)
        npq   = len(pq_idx)

        J11 = np.zeros((npvpq, npvpq))
        J12 = np.zeros((npvpq, npq))
        J21 = np.zeros((npq,   npvpq))
        J22 = np.zeros((npq,   npq))

        for ii, i in enumerate(pvpq_idx):
            for jj, j in enumerate(pvpq_idx):
                if i == j:
                    J11[ii, jj] = -Q_calc[i] - B[i, i] * Vmag[i] ** 2
                else:
                    th = Vang[i] - Vang[j]
                    J11[ii, jj] = Vmag[i]*Vmag[j]*(G[i,j]*np.sin(th) - B[i,j]*np.cos(th))
            for jj, j in enumerate(pq_idx):
                if i == j:
                    J12[ii, jj] = P_calc[i] + G[i, i] * Vmag[i] ** 2
                else:
                    th = Vang[i] - Vang[j]
                    J12[ii, jj] = Vmag[i]*Vmag[j]*(G[i,j]*np.cos(th) + B[i,j]*np.sin(th))

        for ii, i in enumerate(pq_idx):
            for jj, j in enumerate(pvpq_idx):
                if i == j:
                    J21[ii, jj] = P_calc[i] - G[i, i] * Vmag[i] ** 2
                else:
                    th = Vang[i] - Vang[j]
                    J21[ii, jj] = -Vmag[i]*Vmag[j]*(G[i,j]*np.cos(th) + B[i,j]*np.sin(th))
            for jj, j in enumerate(pq_idx):
                if i == j:
                    J22[ii, jj] = Q_calc[i] - B[i, i] * Vmag[i] ** 2
                else:
                    th = Vang[i] - Vang[j]
                    J22[ii, jj] = Vmag[i]*Vmag[j]*(G[i,j]*np.sin(th) - B[i,j]*np.cos(th))

        J   = np.block([[J11, J12], [J21, J22]])
        try:
            dx = np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            break

        # Backtracking line search: ensure the mismatch decreases
        alpha = 1.0
        mis0  = np.max(np.abs(F))
        for _ in range(10):
            dth = np.clip(dx[:npvpq] * alpha, -np.pi, np.pi)
            dv  = np.clip(dx[npvpq:] * alpha, -0.5, 0.5)
            Vang_t = Vang.copy(); Vang_t[pvpq_idx] += dth
            Vmag_t = Vmag.copy(); Vmag_t[pq_idx]   *= (1.0 + dv)
            Vmag_t = np.maximum(Vmag_t, 0.01)   # voltage floor
            V_t    = Vmag_t * np.exp(1j * Vang_t)
            S_t    = V_t * np.conj(Ybus @ V_t)
            F_t    = np.concatenate([
                (P_sched - S_t.real)[pvpq_idx],
                (Q_sched - S_t.imag)[pq_idx],
            ])
            if np.max(np.abs(F_t)) < mis0:
                break
            alpha *= 0.5
        Vang[pvpq_idx] += np.clip(dx[:npvpq] * alpha, -np.pi, np.pi)
        Vmag[pq_idx]   *= (1.0 + np.clip(dx[npvpq:] * alpha, -0.5, 0.5))
        Vmag = np.maximum(Vmag, 0.01)
    else:
        iterations = max_iter

    V_sol  = Vmag * np.exp(1j * Vang)
    S_final = V_sol * np.conj(Ybus @ V_sol)

    # Final mismatch
    dP_f = P_sched - S_final.real
    dQ_f = Q_sched - S_final.imag
    F_f  = np.concatenate([dP_f[pvpq_idx], dQ_f[pq_idx]])
    mismatch = float(np.max(np.abs(F_f))) if len(F_f) > 0 else 0.0

    # Generator outputs
    P_gen = np.zeros(ng)
    Q_gen = np.zeros(ng)
    for g in range(ng):
        gs = int(generators[g, 7]) if generators.shape[1] > 7 else 1
        gb = int(generators[g, 0])
        gi = bus_idx[gb]
        P_gen[g] = generators[g, 1] / baseMVA
        if gs and bus_type[gi] in (2, 3):
            Q_gen[g] = S_final[gi].imag + buses[gi, 3] / baseMVA

    return {
        'V':          V_sol,
        'Vmag':       Vmag,
        'Vang_deg':   np.rad2deg(Vang),
        'P_gen':      P_gen,
        'Q_gen':      Q_gen,
        'P_inj':      S_final.real,
        'Q_inj':      S_final.imag,
        'converged':  converged,
        'iterations': iterations,
        'mismatch':   mismatch,
        'Ybus':       Ybus,
    }


# ---------------------------------------------------------------------------
# Utility: extract generator-bus voltages from power flow solution
# ---------------------------------------------------------------------------

def get_gen_voltages(
    pf_result: Dict,
    buses: np.ndarray,
    generators: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract terminal voltage magnitude and angle for each generator bus.

    Parameters
    ----------
    pf_result : dict
        Output of :func:`run_powerflow` (MATPOWER-style).
    buses : ndarray
        MATPOWER bus array.
    generators : ndarray
        MATPOWER generator array.

    Returns
    -------
    Vg : ndarray, shape (ng,)
        Terminal voltage magnitudes [pu].
    theta_g : ndarray, shape (ng,)
        Terminal voltage angles [rad].
    """
    bus_nums = buses[:, 0].astype(int)
    bus_idx  = {b: i for i, b in enumerate(bus_nums)}
    ng = generators.shape[0]
    Vg      = np.zeros(ng)
    theta_g = np.zeros(ng)
    V = pf_result['V']
    for g in range(ng):
        gi = bus_idx[int(generators[g, 0])]
        Vg[g]      = np.abs(V[gi])
        theta_g[g] = np.angle(V[gi])
    return Vg, theta_g
