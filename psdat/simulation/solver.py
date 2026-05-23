"""Time-domain DAE solver for power system dynamics.

Implements RK45 integration of machine ODEs with implicit algebraic equation
solution at each time step (partitioned approach).

The ODE for all machines:
    dx/dt = f(x, V_bus)
with algebraic constraint:
    I_norton(x) = Y_total * V_bus

Fault modeling:
    - 3-phase bolted fault: add large shunt admittance at faulted bus
    - Fault cleared: restore pre-fault Y-bus

Also provides ``DAESystem`` / ``FaultEvent`` / ``SimulationResult`` — a
higher-level interface using the full AEs.m algebraic solver (AlgebraicSystem
+ fsolve/LM) with scipy.integrate.solve_ivp (RK45), matching the MATLAB
PSDAT simulation pipeline exactly.

References:
    Sauer & Pai (1998), Ch. 6.
    PSDAT (Abdulrahman 2020), Simulink model / AEs.m.
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from math import pi

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.integrate import solve_ivp
from psdat.models.machine import MachineParams, derivatives, N_STATES_PER_GEN
from psdat.simulation.algebraic import solve_network, get_vdq


def simulate(
    machines: List[MachineParams],
    x0: np.ndarray,
    Y_bus_pre: np.ndarray,
    Y_bus_fault: np.ndarray,
    Y_bus_post: np.ndarray,
    t_fault: float,
    t_clear: float,
    t_end: float,
    dt: float = 0.001,
    S_base: float = 100.0,
    omega0: float = 2 * np.pi * 60.0,
) -> Dict:
    """Simulate power system dynamics with a 3-phase fault.

    Parameters
    ----------
    machines : list of MachineParams
    x0 : ndarray, shape (n_gen * 11,)
        Initial state vector (from init_all_machines).
    Y_bus_pre : ndarray, complex (n_buses, n_buses)
        Pre-fault Y-bus.
    Y_bus_fault : ndarray, complex (n_buses, n_buses)
        Y-bus during fault (large shunt at fault bus added externally).
    Y_bus_post : ndarray, complex (n_buses, n_buses)
        Post-fault Y-bus (usually = pre-fault for bolted fault cleared by
        disconnecting faulted bus; or same as pre-fault for transient).
    t_fault : float
        Time of fault application (s).
    t_clear : float
        Time of fault clearing (s).
    t_end : float
        Simulation end time (s).
    dt : float
        Time step (s) for RK4 integration.
    S_base, omega0 : float

    Returns
    -------
    result : dict with keys:
        't'      : ndarray (N,)   — time vector
        'x'      : ndarray (N, n_states) — state history
        'V_bus'  : ndarray (N, n_buses), complex — bus voltages
        'delta'  : ndarray (N, n_gen) — rotor angles (rad)
        'omega'  : ndarray (N, n_gen) — speed deviation (pu)
        'Vmag'   : ndarray (N, n_gen) — terminal voltage magnitudes
    """
    n_gen   = len(machines)
    n_buses = Y_bus_pre.shape[0]
    n_states = n_gen * N_STATES_PER_GEN

    # Gen buses: either passed explicitly or default to first n_gen buses
    gen_buses_list = list(range(1, n_gen + 1))

    # --- Time vector ---
    t_vec = np.arange(0.0, t_end + dt, dt)
    N = len(t_vec)

    # Storage
    x_hist    = np.zeros((N, n_states))
    V_hist    = np.zeros((N, n_buses), dtype=complex)
    x = x0.copy()

    for k, t in enumerate(t_vec):
        # Select Y-bus for this time step
        if t < t_fault:
            Y_curr = Y_bus_pre
        elif t < t_clear:
            Y_curr = Y_bus_fault
        else:
            Y_curr = Y_bus_post

        # Solve algebraic equations (network)
        V_bus = solve_network(x, machines, Y_curr, gen_buses_list, S_base=S_base)

        # Store
        x_hist[k] = x
        V_hist[k] = V_bus

        # RK4 step
        if k < N - 1:
            dt_k = t_vec[k + 1] - t
            x = _rk4_step(x, machines, Y_curr, gen_buses_list, dt_k, S_base, omega0)

    # Extract variables of interest
    # PSDAT column-major: state layout is [Eqp*m, Si1d*m, Edp*m, Si2q*m, delta*m, omega*m, ...]
    # delta is at indices [4*m .. 5*m), omega at [5*m .. 6*m)
    delta_hist = np.zeros((N, n_gen))
    omega_hist = np.zeros((N, n_gen))
    Vmag_hist  = np.zeros((N, n_gen))
    for g in range(n_gen):
        # Column-major: state g for variable k is at x[k*n_gen + g]
        delta_col = 4  # delta is 5th block (0-indexed)
        omega_col = 5  # omega is 6th block
        delta_hist[:, g] = np.rad2deg(x_hist[:, delta_col * n_gen + g])
        omega_hist[:, g] = x_hist[:, omega_col * n_gen + g]
        bus_idx = gen_buses_list[g] - 1
        Vmag_hist[:, g] = np.abs(V_hist[:, bus_idx])

    return {
        't':      t_vec,
        'x':      x_hist,
        'V_bus':  V_hist,
        'delta':  delta_hist,
        'omega':  omega_hist,
        'Vmag':   Vmag_hist,
    }


def _rk4_step(
    x: np.ndarray,
    machines: List[MachineParams],
    Y_bus: np.ndarray,
    gen_buses: List[int],
    dt: float,
    S_base: float,
    omega0: float,
) -> np.ndarray:
    """Single RK4 step for all machines."""
    k1 = _compute_derivs(x,              machines, Y_bus, gen_buses, S_base, omega0)
    k2 = _compute_derivs(x + 0.5*dt*k1, machines, Y_bus, gen_buses, S_base, omega0)
    k3 = _compute_derivs(x + 0.5*dt*k2, machines, Y_bus, gen_buses, S_base, omega0)
    k4 = _compute_derivs(x +     dt*k3, machines, Y_bus, gen_buses, S_base, omega0)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def _compute_derivs(
    x: np.ndarray,
    machines: List[MachineParams],
    Y_bus: np.ndarray,
    gen_buses: List[int],
    S_base: float,
    omega0: float,
) -> np.ndarray:
    """Evaluate full state derivative vector (PSDAT column-major state layout).

    State layout: [Eqp*m, Si1d*m, Edp*m, Si2q*m, delta*m, omega*m,
                   Efd*m, RF*m, VR*m, TM*m, PSV*m]
    """
    n_gen = len(machines)
    n_states = n_gen * N_STATES_PER_GEN

    # Solve network algebraics
    V_bus = solve_network(x, machines, Y_bus, gen_buses, S_base=S_base)
    vdq   = get_vdq(x, machines, V_bus, gen_buses)

    # Unpack column-major state into per-generator x_gen arrays
    keys_order = ['Eqp', 'Si1d', 'Edp', 'Si2q', 'delta', 'omega',
                  'Efd', 'RF', 'VR', 'TM', 'PSV']
    # Each block of n_gen values corresponds to one state variable
    state_blocks = [x[k * n_gen: (k + 1) * n_gen] for k in range(N_STATES_PER_GEN)]

    dxdt = np.zeros(n_states)
    for g in range(n_gen):
        p  = machines[g]
        # Extract per-generator state vector (row-major order for derivatives)
        xg = np.array([state_blocks[k][g] for k in range(N_STATES_PER_GEN)])
        Vg    = abs(V_bus[gen_buses[g] - 1])
        THg   = np.angle(V_bus[gen_buses[g] - 1])
        Vd, Vq = vdq[g]
        # Compute Id, Iq from subtransient model
        Xdpp = p.Xdpp; Xqpp = p.Xqpp; Rs = p.Rs
        Eqp  = xg[0]; Si1d = xg[1]; Edp = xg[2]; Si2q = xg[3]
        Xdp  = p.Xdp; Xqp = p.Xqp; Xls = p.Xls
        denom_d = max(Xdp - Xls, 1e-6)
        denom_q = max(Xqp - Xls, 1e-6)
        Eqpp_val = (Xdpp-Xls)/denom_d * Eqp + (Xdp-Xdpp)/denom_d * Si1d
        Edpp_val = (Xqpp-Xls)/denom_q * Edp - (Xqp-Xqpp)/denom_q * Si2q
        if abs(Rs) < 1e-6:
            Id_val = (Eqpp_val - Vq) / max(Xdpp, 1e-6)
            Iq_val = (Vd - Edpp_val) / max(Xqpp, 1e-6)
        else:
            A_ = np.array([[Rs, -Xqpp], [Xdpp, Rs]])
            b_ = np.array([Vd - Edpp_val, Vq - Eqpp_val])
            Id_val, Iq_val = np.linalg.solve(A_, b_)
        # Get Vref and PC from machine (set during initialization)
        Vref_val = getattr(p, '_Vref', Vg + xg[8] / p.KA)  # VR / KA fallback
        PC_val   = getattr(p, '_PC',   xg[10])   # PSV as fallback
        dxg = derivatives(xg, Vg, THg, float(Id_val), float(Iq_val), p,
                          Vref_val, PC_val)
        # Place back into column-major output
        for k in range(N_STATES_PER_GEN):
            dxdt[k * n_gen + g] = dxg[k]

    return dxdt


def build_fault_ybus(
    Y_bus: np.ndarray,
    fault_bus: int,
    Y_fault: float = 1e6,
) -> np.ndarray:
    """Return Y-bus with bolted fault at fault_bus.

    Parameters
    ----------
    Y_bus : ndarray, complex (n, n)
        Pre-fault Y-bus.
    fault_bus : int
        Faulted bus number (1-indexed).
    Y_fault : float
        Fault admittance to ground (large number for bolted fault).

    Returns
    -------
    Y_fault_bus : ndarray, complex (n, n)
    """
    Y = Y_bus.copy()
    Y[fault_bus - 1, fault_bus - 1] += Y_fault
    return Y


# ===========================================================================
# High-level DAE solver: DAESystem + FaultEvent + SimulationResult
# Implements the MATLAB PSDAT simulation pipeline (AEs.m + ODE45 equivalent)
# ===========================================================================

@dataclass
class FaultEvent:
    """Describes a single fault or generator-trip event.

    Parameters
    ----------
    t_fault : float
        Time at which the fault/trip occurs (s).
    t_clear : float
        Time at which the fault is cleared (s).
    fault_bus : int or None
        0-based bus index for a bus (three-phase) fault.
    gen_idx : int or None
        0-based generator index for a generator trip (when fault_bus is None).
    """
    t_fault   : float
    t_clear   : float
    fault_bus : int | None = None
    gen_idx   : int | None = None


@dataclass
class SimulationResult:
    """Container for time-domain simulation output.

    Attributes
    ----------
    t : ndarray, shape (K,)
        Time vector (s).
    x : ndarray, shape (K, 11*m)
        State variable trajectories for all generators.
    z : ndarray, shape (K, 2m+2n)
        Algebraic variable trajectories [Id, Iq, V, TH].
    gen_data : dict
        Derived per-generator signals:
            'delta'  : (K, m) rotor angles (rad)
            'omega'  : (K, m) rotor speed deviations (rad/s)
            'Eqp'    : (K, m)
            'Edp'    : (K, m)
            'Pe'     : (K, m) electrical power
            'V_gen'  : (K, m) terminal bus voltage magnitudes
            'TH_gen' : (K, m) terminal bus voltage angles
    """
    t        : np.ndarray
    x        : np.ndarray
    z        : np.ndarray
    gen_data : dict = field(default_factory=dict)


def _machine_derivs_fullmodel(
    x_gen: np.ndarray,
    Id: float,
    Iq: float,
    Vg: float,
    THg: float,
    params: dict,
    ws: float,
) -> np.ndarray:
    """11-state derivative for a single generator (subtransient model + exciter + gov).

    State ordering: [delta, omega, Eqp, Si1d, Edp, Si2q, Efd, VR, RF, Vref, PSV]
    """
    delta, omega, Eqp, Si1d, Edp, Si2q, Efd, VR, RF, Vref, PSV = x_gen

    Rs    = params["Rs"]
    Xd    = params["Xd"]
    Xdp   = params["Xdp"]
    Xdpp  = params["Xdpp"]
    Xq    = params["Xq"]
    Xqp   = params["Xqp"]
    Xqpp  = params["Xqpp"]
    Xls   = params["Xls"]
    H     = params["H"]
    D     = params["D"]
    Td0p  = params["Td0p"]
    Td0pp = params["Td0pp"]
    Tq0p  = params["Tq0p"]
    Tq0pp = params["Tq0pp"]
    KA    = params["KA"]
    TA    = params["TA"]
    KE    = params["KE"]
    TE    = params["TE"]
    KF    = params["KF"]
    TF    = params["TF"]
    Ax    = params["Ax"]
    Bx    = params["Bx"]
    KG    = params["KG"]
    TG    = params["TG"]
    PC    = params["PC"]

    # Air-gap torque (subtransient, 6th-order)
    Te = (
        Eqp * Iq
        + Edp * Id
        + (Xdpp - Xqpp) * Id * Iq
        - (Xdpp - Xls) / (Xdp - Xls) * Si1d * Iq
        + (Xqpp - Xls) / (Xqp - Xls) * Si2q * Id
    )

    d_delta = omega
    d_omega = (ws / (2.0 * H)) * (PSV - Te - D * omega / ws)

    d_Eqp  = (1.0 / Td0p)  * (-(Eqp + (Xd  - Xdp)  * Id) + Efd)
    d_Si1d = (1.0 / Td0pp) * (-Si1d + Eqp - (Xdp  - Xls) * Id)
    d_Edp  = (1.0 / Tq0p)  * (-Edp  + (Xq  - Xqp)  * Iq)
    d_Si2q = (1.0 / Tq0pp) * (-Si2q - Edp  - (Xqp  - Xls) * Iq)

    SE_sat = Ax * np.exp(Bx * Efd)
    d_Efd  = (1.0 / TE) * (VR - (KE + SE_sat) * Efd)
    d_VR   = (1.0 / TA) * (KA * (Vref - Vg - RF) - VR)
    d_RF   = (1.0 / TF) * ((KF / TF) * Efd - RF)
    d_Vref = 0.0
    d_PSV  = (1.0 / TG) * (PC - PSV - (omega / ws) / KG)

    return np.array([d_delta, d_omega, d_Eqp, d_Si1d, d_Edp, d_Si2q,
                     d_Efd, d_VR, d_RF, d_Vref, d_PSV])


class DAESystem:
    """Full DAE power system model for time-domain integration via solve_ivp.

    Uses the AlgebraicSystem (AEs.m / fsolve-LM) approach for algebraic
    equations at each ODE evaluation, driven by scipy RK45.

    Parameters
    ----------
    machine_params_list : list of dict, length m
        Per-generator parameter dicts (keys: Rs, Xd, Xdp, Xdpp, Xq, Xqp,
        Xqpp, Xls, H, D, Td0p, Td0pp, Tq0p, Tq0pp, KA, TA, KE, TE,
        KF, TF, Ax, Bx, KG, TG, PC).
    pf_result : dict
        Power-flow solution.  Required keys: 'V', 'TH', 'Pg', 'Qg',
        'Pload', 'Qload'.
    Y_bus : ndarray, shape (n, n), complex
        Pre-fault admittance matrix.
    ws : float
        Synchronous angular frequency (rad/s). Default 2π·60.
    """

    def __init__(
        self,
        machine_params_list: list,
        pf_result: dict,
        Y_bus: np.ndarray,
        ws: float = 2.0 * pi * 60.0,
    ) -> None:
        from psdat.simulation.algebraic import AlgebraicSystem
        from psdat.simulation.initializer import compute_initial_conditions

        self.params_list = machine_params_list
        self.pf_result   = pf_result
        self.Y_bus       = Y_bus.astype(complex)
        self.ws          = ws
        self.m           = len(machine_params_list)
        self.n           = Y_bus.shape[0]

        load_P = np.asarray(pf_result.get("Pload", np.zeros(self.n)), dtype=float)
        load_Q = np.asarray(pf_result.get("Qload", np.zeros(self.n)), dtype=float)

        x0, z0, aux = compute_initial_conditions(pf_result, machine_params_list, ws)
        self._x0 = x0
        self._z0 = z0
        self._aux = aux

        for k, p in enumerate(self.params_list):
            p.setdefault("PC", float(aux["PC"][k]))

        self._alg = AlgebraicSystem(
            Y_bus=self.Y_bus,
            n_gen=self.m,
            n_bus=self.n,
            params_list=machine_params_list,
            load_P=load_P,
            load_Q=load_Q,
        )
        self._load_P    = load_P
        self._load_Q    = load_Q
        self._z_cache   = z0.copy()

    def _unpack_x(self, x_flat: np.ndarray) -> dict:
        m = self.m
        X = x_flat.reshape(m, 11)
        return {
            "delta":  X[:, 0],  "omega":  X[:, 1],
            "Eqp":    X[:, 2],  "Si1d":   X[:, 3],
            "Edp":    X[:, 4],  "Si2q":   X[:, 5],
            "Efd":    X[:, 6],  "VR":     X[:, 7],
            "RF":     X[:, 8],  "Vref":   X[:, 9],
            "PSV":    X[:, 10],
        }

    def _build_dyn_states(self, states: dict) -> dict:
        return {
            "Eqp":   states["Eqp"],   "Si1d":  states["Si1d"],
            "Edp":   states["Edp"],   "Si2q":  states["Si2q"],
            "delta": states["delta"],
        }

    def _set_fault(
        self,
        fault_bus: int | None = None,
        t_fault: float | None = None,
        t_clear: float | None = None,
    ) -> None:
        self._alg.fault_bus = fault_bus
        self._alg.t_fault   = t_fault
        self._alg.t_clear   = t_clear

    def rhs(self, t: float, x_flat: np.ndarray) -> np.ndarray:
        """Evaluate dx/dt: solve algebraics then compute machine derivatives."""
        states     = self._unpack_x(x_flat)
        dyn_states = self._build_dyn_states(states)
        try:
            z = self._alg.solve(self._z_cache, dyn_states, t)
        except RuntimeError as exc:
            warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
            z = self._z_cache.copy()
        self._z_cache = z.copy()

        m, n = self.m, self.n
        Id  = z[0:m];  Iq = z[m:2*m]
        V   = z[2*m:2*m+n];  TH = z[2*m+n:2*m+2*n]

        dxdt = np.empty(11 * m, dtype=float)
        for k, params in enumerate(self.params_list):
            dx_k = _machine_derivs_fullmodel(
                x_flat[k*11:(k+1)*11],
                Id=float(Id[k]),  Iq=float(Iq[k]),
                Vg=float(V[k]),   THg=float(TH[k]),
                params=params,    ws=self.ws,
            )
            dxdt[k*11:(k+1)*11] = dx_k
        return dxdt

    def simulate(
        self,
        t_span: tuple,
        fault_events: list | None = None,
        dt_store: float = 0.001,
    ) -> SimulationResult:
        """Integrate the DAE over t_span with optional fault events.

        Uses scipy.integrate.solve_ivp (RK45, max_step=dt_store, rtol=1e-6,
        atol=1e-8).  Fault events trigger discontinuity restarts.

        Parameters
        ----------
        t_span : (t0, tf) in seconds.
        fault_events : list of FaultEvent.
        dt_store : float — output time resolution (s).

        Returns
        -------
        SimulationResult
        """
        fault_events = fault_events or []
        t0, tf = float(t_span[0]), float(t_span[1])

        # Build breakpoints at every fault/clear boundary.
        bps = sorted({t0, tf} | {
            v for ev in fault_events
            for v in (ev.t_fault, ev.t_clear)
            if t0 < v < tf
        })

        x_cur = self._x0.copy()
        self._z_cache = self._z0.copy()

        t_all, x_all, z_all = [], [], []

        for seg_idx in range(len(bps) - 1):
            t_a, t_b = bps[seg_idx], bps[seg_idx + 1]

            # Activate fault if any event covers this segment.
            active = next(
                (ev for ev in fault_events if ev.t_fault <= t_a < ev.t_clear),
                None,
            )
            if active is not None and active.fault_bus is not None:
                self._set_fault(active.fault_bus, active.t_fault, active.t_clear)
            else:
                self._set_fault()

            n_pts  = max(2, int(np.ceil((t_b - t_a) / dt_store)) + 1)
            t_eval = np.linspace(t_a, t_b, n_pts)

            res = solve_ivp(
                fun=self.rhs,
                t_span=(t_a, t_b),
                y0=x_cur,
                method="RK45",
                t_eval=t_eval,
                max_step=dt_store,
                rtol=1.0e-6,
                atol=1.0e-8,
                dense_output=False,
            )
            if not res.success:
                warnings.warn(
                    f"solve_ivp failed [{t_a:.4f},{t_b:.4f}]: {res.message}",
                    RuntimeWarning, stacklevel=2,
                )

            t_seg = res.t
            x_seg = res.y.T

            z_seg  = np.empty((len(t_seg), self._z0.size), dtype=float)
            z_warm = self._z_cache.copy()
            for i, (ti, xi) in enumerate(zip(t_seg, x_seg)):
                st_i = self._unpack_x(xi)
                dy_i = self._build_dyn_states(st_i)
                try:
                    z_i = self._alg.solve(z_warm, dy_i, ti)
                except RuntimeError:
                    z_i = z_warm.copy()
                z_seg[i] = z_i
                z_warm   = z_i.copy()

            if seg_idx == 0:
                t_all.append(t_seg);  x_all.append(x_seg);  z_all.append(z_seg)
            else:
                t_all.append(t_seg[1:]);  x_all.append(x_seg[1:]);  z_all.append(z_seg[1:])

            x_cur         = x_seg[-1].copy()
            self._z_cache = z_seg[-1].copy()

        t_out = np.concatenate(t_all)
        x_out = np.vstack(x_all)
        z_out = np.vstack(z_all)
        gen_data = self._build_gen_data(x_out, z_out)
        return SimulationResult(t=t_out, x=x_out, z=z_out, gen_data=gen_data)

    def _build_gen_data(self, x_out: np.ndarray, z_out: np.ndarray) -> dict:
        m, n = self.m, self.n
        K  = x_out.shape[0]
        X  = x_out.reshape(K, m, 11)
        delta  = X[:, :, 0];  omega = X[:, :, 1]
        Eqp    = X[:, :, 2];  Edp   = X[:, :, 4]
        Id  = z_out[:, 0:m];  Iq  = z_out[:, m:2*m]
        V   = z_out[:, 2*m:2*m+n];  TH = z_out[:, 2*m+n:2*m+2*n]
        Vg  = V[:, 0:m];  THg = TH[:, 0:m]
        d_th = delta - THg
        Pe   = Id * Vg * np.sin(d_th) + Iq * Vg * np.cos(d_th)
        return {"delta": delta, "omega": omega, "Eqp": Eqp, "Edp": Edp,
                "Pe": Pe, "V_gen": Vg, "TH_gen": THg}

    @property
    def x0(self) -> np.ndarray:   return self._x0
    @property
    def z0(self) -> np.ndarray:   return self._z0
    @property
    def aux(self) -> dict:        return self._aux


# ---------------------------------------------------------------------------
# NOTE: FaultEvent, DAESystem, and SimulationResult are defined earlier
# in this module (search for @dataclass above). The definitions there are
# the canonical ones matching the MATLAB PSDAT AEs.m pipeline.
# ---------------------------------------------------------------------------
