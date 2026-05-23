"""P-V nose curve via continuation power flow.

Incrementally increases load power factor lambda:
    P_load(lambda) = P0 * (1 + lambda)
    Q_load(lambda) = Q0 * (1 + lambda)
until voltage collapse (nose point).

PSDAT Program 1.2 (voltage stability part).

References:
    Ajjarapu & Christy (1992), "The continuation power flow."
    Kundur (1994), Section 14.2.
    PSDAT (Abdulrahman 2020), Program 1.2.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_pv_curve(
    Y_bus: np.ndarray,
    bus_data: np.ndarray,
    load_buses: Optional[List[int]] = None,
    monitored_bus: int = 1,
    lambda_max: float = 5.0,
    dlambda: float = 0.05,
    tol: float = 1e-8,
    max_iter: int = 20,
    S_base: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute P-V curve by parametric continuation.

    Load power is scaled uniformly: P(lambda) = P0*(1+lambda),
    Q(lambda) = Q0*(1+lambda).

    Parameters
    ----------
    Y_bus : ndarray, complex (n, n)
        Y-bus matrix.
    bus_data : ndarray, shape (n, 10)
        Bus data array (cols: id,type,Vmag,Vang,Pg,Qg,Pl,Ql,Gs,Bs).
    load_buses : list of int, optional
        1-indexed bus numbers where load is increased.
        If None, increases load at all PQ buses.
    monitored_bus : int
        1-indexed bus number to monitor voltage.
    lambda_max : float
        Maximum loading parameter.
    dlambda : float
        Step size for lambda.
    tol : float
        Power flow convergence tolerance.
    max_iter : int
        Newton-Raphson iteration limit.
    S_base : float
        System MVA base.

    Returns
    -------
    lambda_vals : ndarray
        Loading parameter values.
    V_mag_vals : ndarray
        Voltage magnitudes at monitored_bus for each lambda.
    """
    from psdat.models.powerflow import _run_pf_simple, _build_ybus_simple

    n = bus_data.shape[0]
    bus_type = bus_data[:, 1].astype(int)
    pq_idx = np.where(bus_type == 3)[0]

    if load_buses is None:
        load_buses_0 = pq_idx   # 0-indexed PQ buses
    else:
        load_buses_0 = np.array([b - 1 for b in load_buses])

    mon_idx = monitored_bus - 1  # 0-indexed

    # Base load values
    P0_load = bus_data[:, 6].copy()   # MW
    Q0_load = bus_data[:, 7].copy()   # Mvar

    lambda_vals = []
    V_mag_vals  = []

    # Need branch data stored somewhere — use a dummy approach:
    # We rebuild branch data from Y_bus using a helper that assumes
    # the user passes Y_bus directly and we do NR on complex voltages.
    # Here we use a direct NR on the Y_bus without rebuilding branch data.

    # Direct NR using Y_bus:
    Vmag  = bus_data[:, 2].copy()
    Vang  = np.deg2rad(bus_data[:, 3].copy())
    V     = Vmag * np.exp(1j * Vang)

    slack_idx = np.where(bus_type == 1)[0]
    pv_idx    = np.where(bus_type == 2)[0]
    theta_idx = np.concatenate([pv_idx, pq_idx])
    v_idx     = pq_idx

    lam = 0.0
    prev_converged = True

    while lam <= lambda_max:
        # Update load
        bd_lam = bus_data.copy()
        bd_lam[:, 6] = P0_load * (1 + lam)
        bd_lam[:, 7] = Q0_load * (1 + lam)

        P_spec = (bd_lam[:, 4] - bd_lam[:, 6]) / S_BASE
        Q_spec = (bd_lam[:, 5] - bd_lam[:, 7]) / S_BASE

        # NR loop
        V_lam = V.copy()
        converged = False
        for _it in range(max_iter):
            Vm   = np.abs(V_lam)
            Vth  = np.angle(V_lam)
            G    = Y_bus.real
            B    = Y_bus.imag
            S_calc = V_lam * np.conj(Y_bus @ V_lam)
            P_calc = S_calc.real
            Q_calc = S_calc.imag

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
                        J[a, b] = -Q_calc[i] - B[i,i]*Vm[i]**2
                    else:
                        th = Vth[i]-Vth[j]
                        J[a, b] = Vm[i]*Vm[j]*(G[i,j]*np.sin(th)-B[i,j]*np.cos(th))
            for a, i in enumerate(theta_idx):
                for b, j in enumerate(v_idx):
                    if i == j:
                        J[a, n_th+b] = P_calc[i]/Vm[i]+G[i,i]*Vm[i]
                    else:
                        th = Vth[i]-Vth[j]
                        J[a, n_th+b] = Vm[i]*(G[i,j]*np.cos(th)+B[i,j]*np.sin(th))
            for a, i in enumerate(v_idx):
                for b, j in enumerate(theta_idx):
                    if i == j:
                        J[n_th+a, b] = P_calc[i]/Vm[i]-G[i,i]*Vm[i]
                    else:
                        th = Vth[i]-Vth[j]
                        J[n_th+a, b] = -Vm[i]*Vm[j]*(G[i,j]*np.cos(th)+B[i,j]*np.sin(th))
            for a, i in enumerate(v_idx):
                for b, j in enumerate(v_idx):
                    if i == j:
                        J[n_th+a, n_th+b] = Q_calc[i]/Vm[i]-B[i,i]*Vm[i]
                    else:
                        th = Vth[i]-Vth[j]
                        J[n_th+a, n_th+b] = Vm[i]*(G[i,j]*np.sin(th)-B[i,j]*np.cos(th))

            try:
                dx = np.linalg.solve(J, mismatch)
            except np.linalg.LinAlgError:
                break

            for a, i in enumerate(theta_idx):
                Vth[i] += dx[a]
            for a, i in enumerate(v_idx):
                Vm[i] += dx[n_th + a]
            V_lam = Vm * np.exp(1j * Vth)

        if not converged:
            break   # nose point reached

        lambda_vals.append(lam)
        V_mag_vals.append(abs(V_lam[mon_idx]))

        # Update starting point for next step
        V = V_lam.copy()
        lam += dlambda

    return np.array(lambda_vals), np.array(V_mag_vals)


# Module-level for import by compute_pv_curve
S_BASE = 100.0


def find_nose_point(
    pv_curve: Tuple[np.ndarray, np.ndarray],
) -> Tuple[float, float]:
    """Find the nose point (maximum loading parameter).

    Parameters
    ----------
    pv_curve : (lambda_vals, V_mag_vals)
        Output from compute_pv_curve().

    Returns
    -------
    (lambda_nose, V_nose) : (float, float)
        Loading parameter and voltage at the nose point.
    """
    lambda_vals, V_mag_vals = pv_curve
    if len(lambda_vals) == 0:
        return 0.0, 0.0
    idx = np.argmax(lambda_vals)
    return float(lambda_vals[idx]), float(V_mag_vals[idx])


def plot_pv_curves(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ax=None,
    xlabel: str = "Loading parameter lambda",
    ylabel: str = "Voltage magnitude (pu)",
    title: str = "P-V Nose Curves",
) -> None:
    """Plot multiple P-V curves on the same axes.

    Parameters
    ----------
    results_dict : dict mapping label -> (lambda_vals, V_mag_vals)
    ax : matplotlib Axes, optional
    xlabel, ylabel, title : str
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    for label, (lam, V) in results_dict.items():
        ax.plot(lam, V, "-o", markersize=3, label=label)
        if len(lam) > 0:
            ax.plot(lam[-1], V[-1], "xk", markersize=8)  # nose point

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.1)
