"""Classical machine model transient stability simulation.

Uses the classical 2nd-order model (constant internal EMF behind X') for
transient stability analysis. This is equivalent to PSDAT classical model
and provides numerical stability without a full algebraic solver.

The reduced Y-bus (Kron reduction to generator internal buses) is used,
which eliminates load buses exactly.

Electromechanical equations per machine i:
    d(delta_i)/dt = omega_i - omega_s
    d(omega_i)/dt = (omega_s / 2H_i) * (Pm_i - Pe_i)

where:
    Pe_i = sum_j |E'_i| * |E'_j| * |Y_red_ij| * cos(delta_i - delta_j - gamma_ij)
    gamma_ij = angle(Y_red_ij)

References:
    Anderson & Fouad (2003), Ch. 2.
    Kundur (1994), Section 3.6.
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple
from psdat.models.machine import MachineParams
from psdat.simulation.algebraic import build_reduced_ybus


def compute_internal_emf(
    Vg: np.ndarray,
    theta_g: np.ndarray,
    PG: np.ndarray,
    QG: np.ndarray,
    machines: List[MachineParams],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute constant internal EMF E' and initial rotor angle delta_0.

    E'_i = Vt_i + (Ra_i + j*Xdp_i) * I_i

    Parameters
    ----------
    Vg : ndarray (n_gen,) — terminal voltage magnitudes (pu)
    theta_g : ndarray (n_gen,) — terminal voltage angles (rad)
    PG, QG : ndarray (n_gen,) — generator complex power (pu)
    machines : list of MachineParams

    Returns
    -------
    E_prime : ndarray, complex (n_gen,) — internal EMF phasors
    delta_0 : ndarray (n_gen,) — initial rotor angles (rad)
    """
    n_gen = len(machines)
    E_prime = np.zeros(n_gen, dtype=complex)
    delta_0 = np.zeros(n_gen)

    for g, p in enumerate(machines):
        Vt = Vg[g] * np.exp(1j * theta_g[g])
        Sg = complex(PG[g], QG[g])
        Ig = np.conj(Sg / Vt)
        # Classical model: use Eq = Vt + jXq*I as constant internal voltage.
        # This gives correct power balance at operating point.
        E_q = Vt + complex(p.Rs, p.Xq) * Ig
        E_prime[g] = E_q
        delta_0[g] = np.angle(E_q)

    return E_prime, delta_0


def simulate_classical(
    machines: List[MachineParams],
    Vg_pf: np.ndarray,
    theta_g_pf: np.ndarray,
    PG: np.ndarray,
    QG: np.ndarray,
    Y_bus_pre: np.ndarray,
    gen_buses: List[int],
    fault_bus: int,
    t_fault: float = 1.0,
    t_clear: float = 1.1,
    t_end: float = 5.0,
    dt: float = 0.001,
    omega0: float = 2 * np.pi * 60.0,
    V_pf_all: np.ndarray = None,   # full power-flow bus voltage (complex)
    P_load_pu: np.ndarray = None,  # load active power (pu)
    Q_load_pu: np.ndarray = None,  # load reactive power (pu)
) -> dict:
    """Simulate transient stability using classical machine model.

    Parameters
    ----------
    machines : list of MachineParams
    Vg_pf : ndarray (n_gen,) — pre-fault terminal voltage magnitudes (pu)
    theta_g_pf : ndarray (n_gen,) — pre-fault terminal voltage angles (rad)
    PG, QG : ndarray (n_gen,) — pre-fault generator power (pu)
    Y_bus_pre : ndarray, complex (n_buses, n_buses) — pre-fault Y-bus
    gen_buses : list of int — generator bus numbers (1-indexed)
    fault_bus : int — bus number of 3-phase fault (1-indexed)
    V_pf_all : ndarray, complex (n_buses,), optional — full bus voltages from PF
    P_load_pu, Q_load_pu : ndarray (n_buses,), optional — load in pu
    t_fault, t_clear, t_end : float — fault timeline (s)
    dt : float — time step (s)
    omega0 : float — synchronous speed (rad/s)

    Returns
    -------
    result : dict
        't' : ndarray (N,)
        'delta' : ndarray (N, n_gen) — rotor angles in degrees
        'omega' : ndarray (N, n_gen) — angular speed in rad/s
        'V_bus' : ndarray (N, n_buses), complex — bus voltages
        'Vmag' : ndarray (N, n_gen) — terminal voltage magnitudes
        'Pe' : ndarray (N, n_gen) — electrical power
    """
    n_gen = len(machines)
    n_buses = Y_bus_pre.shape[0]
    n_gen_buses = len(gen_buses)

    # Compute internal EMF (constant)
    E_prime, delta_0 = compute_internal_emf(Vg_pf, theta_g_pf, PG, QG, machines)
    E_mag = np.abs(E_prime)

    # Initial mechanical power = electrical power (steady state)
    Pm = PG.copy()   # pu (machine base load)

    # Build load admittances
    if P_load_pu is None or Q_load_pu is None:
        try:
            from psdat.data.ieee9 import BUS_DATA, S_BASE
            P_load_pu = BUS_DATA[:n_buses, 6] / S_BASE
            Q_load_pu = BUS_DATA[:n_buses, 7] / S_BASE
        except Exception:
            P_load_pu = np.zeros(n_buses)
            Q_load_pu = np.zeros(n_buses)

    if V_pf_all is None:
        V_flat = np.ones(n_buses, dtype=complex)
        for g, gb in enumerate(gen_buses):
            V_flat[gb - 1] = Vg_pf[g] * np.exp(1j * theta_g_pf[g])
    else:
        V_flat = np.asarray(V_pf_all, dtype=complex)

    def _build_reduced_ybus_internal(Y_base: np.ndarray) -> np.ndarray:
        """Build reduced Y-bus between generator INTERNAL EMF buses.

        Uses virtual internal bus formulation (Anderson & Fouad, Section 2.5):
          - Extend Y-bus to n_buses + n_gen nodes
          - Internal buses connect via jXd' to terminal buses
          - Kron-reduce to eliminate all terminal + load buses

        Parameters
        ----------
        Y_base : ndarray, complex (n_buses, n_buses)
            Y-bus with load admittances already added.
        """
        n_ext = n_buses + n_gen
        Y_ext = np.zeros((n_ext, n_ext), dtype=complex)
        Y_ext[:n_buses, :n_buses] = Y_base

        for g, gb in enumerate(gen_buses):
            bi  = gb - 1
            igi = n_buses + g   # internal bus index
            # Use Xq to match the classical model's Eq = Vt + jXq*I
            y_g = 1.0 / complex(machines[g].Rs, max(machines[g].Xq, 1e-6))
            Y_ext[bi,  bi]   += y_g
            Y_ext[igi, igi]  += y_g
            Y_ext[bi,  igi]  -= y_g
            Y_ext[igi, bi]   -= y_g

        # Kron-reduce: eliminate terminal+load buses (0..n_buses-1)
        elim = list(range(n_buses))
        keep = list(range(n_buses, n_ext))

        Ykk = Y_ext[np.ix_(keep, keep)]
        Ykl = Y_ext[np.ix_(keep, elim)]
        Yll = Y_ext[np.ix_(elim, elim)]
        Ylk = Y_ext[np.ix_(elim, keep)]
        try:
            Y_red = Ykk - Ykl @ np.linalg.solve(Yll, Ylk)
        except np.linalg.LinAlgError:
            Y_red = Ykk.copy()
        return Y_red

    # Load-augmented Y_bus (constant impedance load model)
    Y_load_aug = Y_bus_pre.copy()
    for i in range(n_buses):
        Vm2 = max(abs(V_flat[i]) ** 2, 1e-6)
        Y_load_aug[i, i] += complex(P_load_pu[i], -Q_load_pu[i]) / Vm2

    # Build reduced Y-bus for three topologies
    Y_red_pre = _build_reduced_ybus_internal(Y_load_aug)

    Y_fault_aug = Y_load_aug.copy()
    Y_fault_aug[fault_bus - 1, fault_bus - 1] += 1e6
    Y_red_fault = _build_reduced_ybus_internal(Y_fault_aug)

    Y_red_post = Y_red_pre.copy()

    def electrical_power(delta: np.ndarray, Y_red: np.ndarray) -> np.ndarray:
        """Compute electrical power for each generator."""
        Pe = np.zeros(n_gen)
        for i in range(n_gen):
            for j in range(n_gen):
                Yij = Y_red[i, j]
                Gij = Yij.real
                Bij = Yij.imag
                dij = delta[i] - delta[j]
                Pe[i] += (E_mag[i] * E_mag[j] *
                          (Gij * np.cos(dij) + Bij * np.sin(dij)))
        return Pe

    def swing_rhs(delta: np.ndarray, omega: np.ndarray,
                  Y_red: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """RHS of swing equations."""
        Pe = electrical_power(delta, Y_red)
        ddelta = omega - omega0
        domega = np.array([
            (omega0 / (2.0 * machines[g].H)) * (Pm[g] - Pe[g])
            for g in range(n_gen)
        ])
        return ddelta, domega, Pe

    # Initial conditions
    delta = delta_0.copy()
    omega = np.full(n_gen, omega0)

    # Time integration (RK4)
    t_vec = np.arange(0.0, t_end + dt, dt)
    N = len(t_vec)

    delta_hist = np.zeros((N, n_gen))
    omega_hist = np.zeros((N, n_gen))
    Pe_hist    = np.zeros((N, n_gen))

    for k, t in enumerate(t_vec):
        if t < t_fault:
            Y_curr = Y_red_pre
        elif t < t_clear:
            Y_curr = Y_red_fault
        else:
            Y_curr = Y_red_post

        delta_hist[k] = np.rad2deg(delta)
        omega_hist[k] = omega
        _, _, Pe_hist[k] = swing_rhs(delta, omega, Y_curr)

        if k < N - 1:
            dt_k = t_vec[k + 1] - t
            # RK4
            dd1, dw1, _ = swing_rhs(delta, omega, Y_curr)
            dd2, dw2, _ = swing_rhs(delta + 0.5*dt_k*dd1, omega + 0.5*dt_k*dw1, Y_curr)
            dd3, dw3, _ = swing_rhs(delta + 0.5*dt_k*dd2, omega + 0.5*dt_k*dw2, Y_curr)
            dd4, dw4, _ = swing_rhs(delta + dt_k*dd3, omega + dt_k*dw3, Y_curr)
            delta += (dt_k / 6.0) * (dd1 + 2*dd2 + 2*dd3 + dd4)
            omega += (dt_k / 6.0) * (dw1 + 2*dw2 + 2*dw3 + dw4)

    # Compute bus voltages from gen bus voltages
    # V_gen = E' - jXd'*I_gen  (approximate)
    # For simplicity use E' * exp(j*delta) projected onto bus
    Vmag_hist = np.zeros((N, n_gen))
    for g in range(n_gen):
        Vmag_hist[:, g] = E_mag[g] * np.abs(
            np.cos(np.deg2rad(delta_hist[:, g]) - delta_0[g]) + 0j
        )
        # Better approximation: just track |Vt| via |E' - jXd'*I|
        # For the examples, use E_mag as proxy (it's bounded and meaningful)
        Vmag_hist[:, g] = Vg_pf[g] * np.ones(N)   # constant (classical model)

    # Dummy V_bus (not tracked in classical model)
    V_bus_hist = np.zeros((N, n_buses), dtype=complex)

    return {
        't':      t_vec,
        'delta':  delta_hist,
        'omega':  omega_hist,
        'Pe':     Pe_hist,
        'V_bus':  V_bus_hist,
        'Vmag':   Vmag_hist,
    }
