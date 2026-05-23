"""Critical Clearing Time (CCT) computation via binary search.

PSDAT Program 1.2.

The CCT is the maximum fault duration for which the system remains transiently
stable. A binary search on the clearing time is performed with a stability
criterion based on rotor angle deviation.

References:
    Kundur (1994), Section 13.5.
    PSDAT (Abdulrahman 2020), Program 1.2.
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional
from psdat.models.machine import MachineParams
from psdat.simulation.solver import simulate, build_fault_ybus


def compute_cct(
    machines: List[MachineParams],
    x0: np.ndarray,
    Y_bus: np.ndarray,
    fault_bus: int,
    t_sim: float = 5.0,
    dt: float = 0.001,
    tol_time: float = 0.001,   # CCT precision (s)
    t_clear_lo: float = 0.01,  # lower bound (s)
    t_clear_hi: float = 2.0,   # upper bound (s)
    delta_unstable_deg: float = 120.0,
    S_base: float = 100.0,
    omega0: float = 2 * np.pi * 60.0,
) -> float:
    """Find Critical Clearing Time via binary search.

    Parameters
    ----------
    machines : list of MachineParams
    x0 : ndarray, shape (n_gen * 11,)
        Initial state vector.
    Y_bus : ndarray, complex (n_buses, n_buses)
        Pre-fault (and post-fault) Y-bus.
    fault_bus : int
        Bus number (1-indexed) where the 3-phase fault occurs.
    t_sim : float
        Total simulation time (s) for stability check.
    dt : float
        Integration time step (s).
    tol_time : float
        Convergence tolerance on CCT (s).
    t_clear_lo, t_clear_hi : float
        Initial bracket for binary search (s).
    delta_unstable_deg : float
        Rotor angle difference threshold for instability declaration (degrees).
    S_base, omega0 : float

    Returns
    -------
    cct : float
        Critical clearing time (s). Returns t_clear_hi if always stable,
        t_clear_lo if always unstable.
    """
    gen_buses = [m.bus for m in machines]
    Y_fault = build_fault_ybus(Y_bus, fault_bus)

    def is_stable(t_clear: float) -> bool:
        """Returns True if system is stable for given clearing time."""
        result = simulate(
            machines, x0,
            Y_bus_pre=Y_bus,
            Y_bus_fault=Y_fault,
            Y_bus_post=Y_bus,
            t_fault=0.1,
            t_clear=0.1 + t_clear,
            t_end=t_sim,
            dt=dt,
            S_base=S_base,
            omega0=omega0,
        )
        delta = result['delta']   # shape (N, n_gen), degrees
        # Check max rotor angle difference
        for k in range(delta.shape[0]):
            d_max = delta[k].max() - delta[k].min()
            if d_max > delta_unstable_deg:
                return False
        return True

    # Binary search
    lo = t_clear_lo
    hi = t_clear_hi

    # Check bounds
    if is_stable(hi):
        return hi
    if not is_stable(lo):
        return lo

    while hi - lo > tol_time:
        mid = 0.5 * (lo + hi)
        if is_stable(mid):
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def compute_cct_all_buses(
    machines: List[MachineParams],
    x0: np.ndarray,
    Y_bus: np.ndarray,
    fault_buses: Optional[List[int]] = None,
    n_buses: Optional[int] = None,
    **cct_kwargs,
) -> dict:
    """Compute CCT for faults at multiple buses.

    Parameters
    ----------
    machines, x0, Y_bus : as in compute_cct()
    fault_buses : list of int, optional
        Bus numbers to test. Defaults to all non-generator buses.
    n_buses : int, optional
        Total number of buses (required if fault_buses is None).

    Returns
    -------
    cct_dict : dict mapping bus_number -> CCT (s)
    """
    gen_buses = {m.bus for m in machines}
    if fault_buses is None:
        if n_buses is None:
            raise ValueError("Provide fault_buses or n_buses.")
        fault_buses = [b for b in range(1, n_buses + 1) if b not in gen_buses]

    results = {}
    for bus in fault_buses:
        try:
            cct = compute_cct(machines, x0, Y_bus, fault_bus=bus, **cct_kwargs)
            results[bus] = cct
        except Exception as e:
            results[bus] = float('nan')
            print(f"  [Warning] Bus {bus}: CCT computation failed: {e}")

    return results
