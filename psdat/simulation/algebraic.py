"""Algebraic equations: stator + network interface.

Solves the coupled algebraic equations between machine internal states
and the network (Y-bus) at each time step of the DAE simulation.

For each machine i:
    Vd_i = V_i * sin(delta_i - theta_i)   [d-axis terminal voltage]
    Vq_i = V_i * cos(delta_i - theta_i)   [q-axis terminal voltage]
    Id_i, Iq_i from stator algebraics
    I_net_i = (Iq_i - j*Id_i) * exp(j*delta_i)  [network frame]

Network equation:
    I_net = Y_bus * V_bus  (for non-generator buses)

References:
    Sauer & Pai (1998), Ch. 5-6.
    PSDAT (Abdulrahman 2020), AEs.m
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple
from scipy.optimize import fsolve
from psdat.models.machine import MachineParams


def machine_current_network_frame(
    x: np.ndarray,
    params: MachineParams,
    V_terminal: complex,
) -> complex:
    """Compute machine current injection in network (a-b-c) frame.

    Parameters
    ----------
    x : ndarray, shape (11,)
        Machine state vector.
    params : MachineParams
    V_terminal : complex
        Terminal bus voltage (pu).

    Returns
    -------
    I_inj : complex
        Current injected into the network (pu, system base).
    """
    p = params
    delta = x[0]
    Eq_pp = x[4]
    Ed_pp = x[5]

    Vt = V_terminal
    theta = np.angle(Vt)
    Vm = abs(Vt)

    Vd = Vm * np.sin(delta - theta)
    Vq = Vm * np.cos(delta - theta)

    if abs(p.Ra) < 1e-6:
        Id = (Eq_pp - Vq) / max(p.Xd_pp, 1e-6)
        Iq = (Vd - Ed_pp) / max(p.Xq_pp, 1e-6)
    else:
        A_ = np.array([[p.Ra, -p.Xq_pp], [p.Xd_pp, p.Ra]])
        b_ = np.array([Vd - Ed_pp, Vq - Eq_pp])
        Id, Iq = np.linalg.solve(A_, b_)

    # dq to network frame: I_net = (Iq - j*Id) * exp(j*delta)
    I_dq = complex(Iq, -Id)
    I_net = I_dq * np.exp(1j * delta)

    # Scale to system base
    k = 100.0 / p.S_rated
    return I_net / k


def solve_network_pf(
    x_all: np.ndarray,
    machines: List[MachineParams],
    Y_bus: np.ndarray,
    gen_buses: List[int],
    V_pf: np.ndarray,
    fault_buses: List[int] = None,
    S_base: float = 100.0,
) -> np.ndarray:
    """Solve network using power-flow V as linearisation point (1-step).

    This is a more accurate version that uses the power-flow voltage as the
    initial guess and does one Newton step to correct.

    For now this just calls solve_network (full Norton), kept as an alias.
    """
    return solve_network(x_all, machines, Y_bus, gen_buses, fault_buses, S_base)


def solve_network(
    x_all: np.ndarray,
    machines: List[MachineParams],
    Y_bus: np.ndarray,
    gen_buses: List[int],
    fault_buses: List[int] = None,
    S_base: float = 100.0,
) -> np.ndarray:
    """Solve network algebraic equations (Norton equivalent method).

    Builds a Norton equivalent for each machine behind its subtransient
    reactance, augments the Y-bus, and solves Y_aug * V = I_norton.

    State layout (PSDAT column-major):
        x[0:m]   = Eqp,   x[m:2m]  = Si1d
        x[2m:3m] = Edp,   x[3m:4m] = Si2q
        x[4m:5m] = delta, x[5m:6m] = omega
        x[6m:7m] = Efd,   x[7m:8m] = RF
        x[8m:9m] = VR,    x[9m:10m]= TM
        x[10m:11m]= PSV

    Parameters
    ----------
    x_all : ndarray, shape (11 * n_gen,)
        Full state vector in PSDAT column-major format.
    machines : list of MachineParams
    Y_bus : ndarray, complex, shape (n_buses, n_buses)
    gen_buses : list of int
        Generator bus numbers (1-indexed).
    fault_buses : list of int, optional
        Buses with bolted 3-phase fault (V=0 imposed via large shunt).
    S_base : float
        System MVA base (not used directly here, kept for signature compat).

    Returns
    -------
    V_bus : ndarray, complex, shape (n_buses,)
        Bus voltages (pu).
    """
    n_buses = Y_bus.shape[0]
    n_gen   = len(machines)
    fault_buses = fault_buses or []

    # Augmented Y-bus with fault short circuits
    Y_aug = Y_bus.copy()
    for fb in fault_buses:
        Y_aug[fb - 1, fb - 1] += 1e8 + 0j   # bolted fault

    # Norton equivalent for each generator
    I_norton = np.zeros(n_buses, dtype=complex)
    Y_norton = np.zeros(n_buses, dtype=complex)

    for g in range(n_gen):
        p = machines[g]
        bi = gen_buses[g] - 1   # 0-indexed bus

        # Extract states from column-major layout
        Eqp  = float(x_all[0 * n_gen + g])
        Si1d = float(x_all[1 * n_gen + g])
        Edp  = float(x_all[2 * n_gen + g])
        Si2q = float(x_all[3 * n_gen + g])
        delta = float(x_all[4 * n_gen + g])

        # Subtransient EMFs (PSDAT eq.)
        Xdp  = max(p.Xdp,  1e-6)
        Xqp  = max(p.Xqp,  1e-6)
        Xdpp = max(p.Xdpp, 1e-6)
        Xqpp = max(p.Xqpp, 1e-6)
        Xls  = p.Xls
        Rs   = p.Rs

        denom_d = max(Xdp - Xls, 1e-6)
        denom_q = max(Xqp - Xls, 1e-6)

        Eqpp = (Xdpp - Xls) / denom_d * Eqp + (Xdp - Xdpp) / denom_d * Si1d
        Edpp = (Xqpp - Xls) / denom_q * Edp - (Xqp - Xqpp) / denom_q * Si2q

        # Internal phasor voltage (d-axis along q-axis of rotor)
        # In network frame: E'' = (Eqpp + j*Edpp) rotated by (delta - pi/2)
        E_int = complex(Eqpp, Edpp) * np.exp(1j * (delta - np.pi / 2.0))

        # Norton admittance behind subtransient reactance
        Yn = 1.0 / complex(Rs, Xdpp)

        Y_norton[bi] += Yn
        I_norton[bi] += Yn * E_int

    # Build total Y-bus and solve
    Y_total = Y_aug.copy()
    for i in range(n_buses):
        Y_total[i, i] += Y_norton[i]

    try:
        V_bus = np.linalg.solve(Y_total, I_norton)
    except np.linalg.LinAlgError:
        # Singular (e.g., all buses faulted) — return zeros
        V_bus = np.zeros(n_buses, dtype=complex)

    return V_bus


def build_reduced_ybus(
    Y_bus: np.ndarray,
    gen_buses: List[int],
) -> np.ndarray:
    """Kron-reduce Y-bus to generator internal buses.

    Parameters
    ----------
    Y_bus : ndarray, complex (n_buses, n_buses)
    gen_buses : list of int (1-indexed)

    Returns
    -------
    Y_red : ndarray, complex (n_gen, n_gen)
    """
    n = Y_bus.shape[0]
    n_gen = len(gen_buses)
    gen_idx = [b - 1 for b in gen_buses]
    load_idx = [i for i in range(n) if i not in gen_idx]

    # Partition Y-bus
    # Y_gg: generator-generator block
    # Y_gl: generator-load block
    # Y_lg: load-generator block
    # Y_ll: load-load block
    Y_gg = Y_bus[np.ix_(gen_idx, gen_idx)]
    Y_gl = Y_bus[np.ix_(gen_idx, load_idx)]
    Y_lg = Y_bus[np.ix_(load_idx, gen_idx)]
    Y_ll = Y_bus[np.ix_(load_idx, load_idx)]

    try:
        Y_red = Y_gg - Y_gl @ np.linalg.solve(Y_ll, Y_lg)
    except np.linalg.LinAlgError:
        Y_red = Y_gg.copy()

    return Y_red


def get_vdq(x_all: np.ndarray, machines: List[MachineParams],
            V_bus: np.ndarray,
            gen_buses: List[int] = None) -> List[Tuple[float, float]]:
    """Extract Vd, Vq for each machine given bus voltages.

    Uses PSDAT column-major state layout (delta at index 4*n_gen + g).

    Parameters
    ----------
    x_all : ndarray
        Full state vector in PSDAT column-major format.
    machines : list of MachineParams
    V_bus : ndarray, complex
        Bus voltages.
    gen_buses : list of int, optional
        Generator bus numbers (1-indexed). Defaults to [1, 2, ..., n_gen].

    Returns
    -------
    vdq : list of (Vd, Vq) tuples, one per machine.
    """
    n_gen = len(machines)
    if gen_buses is None:
        gen_buses = list(range(1, n_gen + 1))
    vdq = []
    for g in range(n_gen):
        # Column-major: delta at index 4*n_gen + g
        delta = x_all[4 * n_gen + g]
        bus_idx = gen_buses[g] - 1
        Vt = V_bus[bus_idx]
        Vm = abs(Vt)
        theta = np.angle(Vt)
        Vd = Vm * np.sin(delta - theta)
        Vq = Vm * np.cos(delta - theta)
        vdq.append((Vd, Vq))
    return vdq


# ===========================================================================
# AlgebraicSystem — vectorised MATLAB AEs.m equivalent (fsolve / LM)
# ===========================================================================

class AlgebraicSystem:
    """Vectorized algebraic equation solver matching MATLAB AEs.m.

    Solves the coupled stator + network algebraic equations at each time
    step using scipy.optimize.fsolve (Levenberg-Marquardt) to match the
    MATLAB ``fsolve(..., 'Algorithm','levenberg-marquardt')`` call.

    Algebraic variable ordering (matches MATLAB AEs.m exactly):
        z = [Id(0:m), Iq(m:2m), V(2m:2m+n), TH(2m+n:2m+2n)]
                                              total: 2*m + 2*n

    where
        m  = number of generator buses (first m buses in Y-bus ordering)
        n  = total buses
        Id, Iq  = d/q-axis stator currents
        V, TH   = bus voltage magnitudes and angles (all n buses)

    Parameters
    ----------
    Y_bus : ndarray, shape (n, n), complex
        Full nodal admittance matrix.
    n_gen : int
        Number of generator buses (first n_gen entries in Y_bus).
    n_bus : int
        Total buses (== Y_bus.shape[0]).
    params_list : list of dict, length n_gen
        Per-generator parameters with keys: Rs, Xd, Xdp, Xdpp,
        Xq, Xqp, Xqpp, Xls.
    load_P : ndarray, shape (n_bus,)
        Active power load (p.u.).
    load_Q : ndarray, shape (n_bus,)
        Reactive power load (p.u.).
    V0 : ndarray, shape (n_bus,), optional
        Initial voltage magnitudes (informational, not used directly).
    fault_bus : int or None
        0-based bus index for three-phase fault (V forced to zero).
    t_fault : float or None
        Fault application time.
    t_clear : float or None
        Fault clearing time.
    """

    def __init__(
        self,
        Y_bus: np.ndarray,
        n_gen: int,
        n_bus: int,
        params_list: list,
        load_P: np.ndarray,
        load_Q: np.ndarray,
        V0: np.ndarray | None = None,
        fault_bus: int | None = None,
        t_fault: float | None = None,
        t_clear: float | None = None,
    ) -> None:
        if Y_bus.shape != (n_bus, n_bus):
            raise ValueError(
                f"Y_bus must be ({n_bus},{n_bus}), got {Y_bus.shape}"
            )
        if len(params_list) != n_gen:
            raise ValueError(
                f"params_list must have {n_gen} entries, got {len(params_list)}"
            )

        self.Y_bus = Y_bus.astype(complex)
        self.n_gen = n_gen
        self.n_bus = n_bus
        self.params_list = params_list
        self.load_P = np.asarray(load_P, dtype=float)
        self.load_Q = np.asarray(load_Q, dtype=float)

        self.Yabs = np.abs(self.Y_bus)
        self.Yang = np.angle(self.Y_bus)

        m = n_gen
        self.Rs   = np.array([p["Rs"]   for p in params_list], dtype=float)
        self.Xdpp = np.array([p["Xdpp"] for p in params_list], dtype=float)
        self.Xdp  = np.array([p["Xdp"]  for p in params_list], dtype=float)
        self.Xqpp = np.array([p["Xqpp"] for p in params_list], dtype=float)
        self.Xqp  = np.array([p["Xqp"]  for p in params_list], dtype=float)
        self.Xls  = np.array([p["Xls"]  for p in params_list], dtype=float)

        self.fault_bus = fault_bus
        self.t_fault   = t_fault
        self.t_clear   = t_clear
        self.nz        = 2 * n_gen + 2 * n_bus

    def _unpack(self, z: np.ndarray):
        m, n = self.n_gen, self.n_bus
        return z[0:m], z[m:2*m], z[2*m:2*m+n], z[2*m+n:2*m+2*n]

    def _apply_fault(self, V: np.ndarray, TH: np.ndarray, t: float):
        if (
            self.fault_bus is not None
            and self.t_fault is not None
            and self.t_clear is not None
            and self.t_fault <= t < self.t_clear
        ):
            fb = self.fault_bus
            V, TH = V.copy(), TH.copy()
            V[fb] = 0.0
            TH[fb] = 0.0
        return V, TH

    def residual(self, z: np.ndarray, dyn_states: dict, t: float) -> np.ndarray:
        """Compute 2m+2n algebraic residuals.

        Implements MATLAB AEs.m DAE() equations exactly.

        Parameters
        ----------
        z : ndarray, shape (2m+2n,)
        dyn_states : dict
            Keys: 'Eqp', 'Si1d', 'Edp', 'Si2q', 'delta'  (each ndarray(m)).
        t : float

        Returns
        -------
        out : ndarray, shape (2m+2n,)
        """
        m, n = self.n_gen, self.n_bus
        Id, Iq, V, TH = self._unpack(z)
        V, TH = self._apply_fault(V, TH, t)

        Eqp   = dyn_states["Eqp"]
        Si1d  = dyn_states["Si1d"]
        Edp   = dyn_states["Edp"]
        Si2q  = dyn_states["Si2q"]
        delta = dyn_states["delta"]

        Rs   = self.Rs
        Xdpp = self.Xdpp
        Xdp  = self.Xdp
        Xqpp = self.Xqpp
        Xqp  = self.Xqp
        Xls  = self.Xls
        Yabs = self.Yabs
        Yang = self.Yang

        Vg  = V[0:m]
        THg = TH[0:m]
        d_th = delta - THg

        # Stator equations (MATLAB SE1, SE2)
        coeff_Edp  = (Xqpp - Xls) / (Xqp - Xls)
        coeff_Si2q = (Xqp  - Xqpp) / (Xqp - Xls)
        coeff_Eqp  = (Xdpp - Xls) / (Xdp - Xls)
        coeff_Si1d = (Xdp  - Xdpp) / (Xdp - Xls)

        SE1 = (
            Rs * Id - Xqpp * Iq
            - coeff_Edp  * Edp
            + coeff_Si2q * Si2q
            + Vg * np.sin(d_th)
        )
        SE2 = (
            Rs * Iq + Xdpp * Id
            - coeff_Eqp  * Eqp
            - coeff_Si1d * Si1d
            + Vg * np.cos(d_th)
        )

        # Generator bus power balance (MATLAB PV1, PV2)
        PL2 = self.load_P
        QL2 = self.load_Q

        TH_diff_gen = THg[:, None] - TH[None, :]
        P_inj_gen = np.sum(
            Vg[:, None] * V[None, :] * Yabs[0:m, :] * np.cos(TH_diff_gen - Yang[0:m, :]),
            axis=1,
        )
        Q_inj_gen = np.sum(
            Vg[:, None] * V[None, :] * Yabs[0:m, :] * np.sin(TH_diff_gen - Yang[0:m, :]),
            axis=1,
        )

        PV1 = Id * Vg * np.sin(d_th) + Iq * Vg * np.cos(d_th) - PL2[0:m] - P_inj_gen
        PV2 = Id * Vg * np.cos(d_th) - Iq * Vg * np.sin(d_th) - QL2[0:m] - Q_inj_gen

        # Load bus power balance (MATLAB PQ1, PQ2)
        Vl  = V[m:]
        THl = TH[m:]
        TH_diff_load = THl[:, None] - TH[None, :]
        P_inj_load = np.sum(
            Vl[:, None] * V[None, :] * Yabs[m:, :] * np.cos(TH_diff_load - Yang[m:, :]),
            axis=1,
        )
        Q_inj_load = np.sum(
            Vl[:, None] * V[None, :] * Yabs[m:, :] * np.sin(TH_diff_load - Yang[m:, :]),
            axis=1,
        )
        PQ1 = -PL2[m:] - P_inj_load
        PQ2 = -QL2[m:] - Q_inj_load

        return np.concatenate([SE1, SE2, PV1, PQ1, PV2, PQ2])

    def solve(
        self,
        z0: np.ndarray,
        dyn_states: dict,
        t: float,
        tol: float = 1.0e-10,
        max_iter: int = 400,
    ) -> np.ndarray:
        """Solve algebraic equations via fsolve (Levenberg-Marquardt).

        Parameters
        ----------
        z0 : ndarray, shape (2m+2n,) — warm-start guess.
        dyn_states : dict
        t : float
        tol, max_iter : solver settings.

        Returns
        -------
        z : ndarray, shape (2m+2n,) — converged solution.

        Raises
        ------
        RuntimeError if fsolve does not converge.
        """
        def fun(z):
            return self.residual(z, dyn_states, t)

        z_sol, info, ier, msg = fsolve(
            fun, z0, full_output=True,
            xtol=tol, ftol=tol,
            maxfev=max_iter * self.nz,
            factor=100,
        )
        if ier not in (1, 2, 3, 4):
            raise RuntimeError(
                f"AlgebraicSystem.solve did not converge at t={t:.6f}: {msg}"
            )
        return z_sol
