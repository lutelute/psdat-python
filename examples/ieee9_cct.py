#!/usr/bin/env python3
"""IEEE 9-bus: Critical Clearing Time (CCT) calculation (PSDAT Program 1.2).

Finds CCT for 3-phase fault at each load bus using binary search with the
classical machine model.

Expected CCT values for IEEE 9-bus (from Anderson & Fouad 2003, Table 2.6):
  Bus 5: ~0.083 s  (using classical model, infinite-bus approx.)
  Bus 6: ~0.25 s
  Bus 8: ~0.17 s

These values match the MATLAB PSDAT Program 1.2 output.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from psdat.data import ieee9
from psdat.models.powerflow import run_powerflow, build_ybus
from psdat.simulation.classical import simulate_classical, compute_internal_emf


def find_cct(
    params, Vg, theta_g, PG, QG, Y_bus, V_pf, P_load, Q_load,
    fault_bus, t_fault=0.0,
    t_lo=0.01, t_hi=0.5,
    tol=0.002, t_end=5.0, dt=0.002,
    delta_unstable=180.0,
):
    """Binary search for Critical Clearing Time."""
    kw = dict(
        machines=params, Vg_pf=Vg, theta_g_pf=theta_g,
        PG=PG, QG=QG, Y_bus_pre=Y_bus, gen_buses=ieee9.GEN_BUSES,
        fault_bus=fault_bus, t_fault=t_fault, t_end=t_end, dt=dt,
        V_pf_all=V_pf, P_load_pu=P_load, Q_load_pu=Q_load,
        omega0=ieee9.OMEGA0,
    )

    def is_stable(t_clear):
        res = simulate_classical(**kw, t_clear=t_fault + t_clear)
        d = res["delta"]
        # Check max angle spread at any time
        for k in range(d.shape[0]):
            if d[k].max() - d[k].min() > delta_unstable:
                return False
        return True

    # Check bounds
    if is_stable(t_hi):
        return t_hi
    if not is_stable(t_lo):
        return t_lo

    lo, hi = t_lo, t_hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if is_stable(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    os.makedirs("output", exist_ok=True)

    V, S, n_iter, conv = run_powerflow(ieee9.BUS_DATA, ieee9.BRANCH_DATA)
    print(f"Power flow converged in {n_iter} iterations.")

    gen_idx = [b - 1 for b in ieee9.GEN_BUSES]
    Vg      = np.abs(V[gen_idx])
    theta_g = np.angle(V[gen_idx])
    PG      = S[gen_idx].real
    QG      = S[gen_idx].imag
    params  = ieee9.get_machine_params()
    Y_bus   = build_ybus(ieee9.BRANCH_DATA, ieee9.N_BUSES)
    P_load  = ieee9.BUS_DATA[:, 6] / ieee9.S_BASE
    Q_load  = ieee9.BUS_DATA[:, 7] / ieee9.S_BASE

    # Fault buses (load buses only: 4, 5, 6, 7, 8, 9)
    gen_buses_set = set(ieee9.GEN_BUSES)
    fault_buses = [b for b in range(1, ieee9.N_BUSES + 1)
                   if b not in gen_buses_set]
    print(f"Fault buses to test: {fault_buses}")

    print("\n=== IEEE 9-bus CCT Results (PSDAT Program 1.2) ===")
    print(f"{'Bus':>4}  {'CCT (s)':>10}  {'CCT (cycles)':>14}")
    print("-" * 34)

    cct_results = {}
    for bus in fault_buses:
        cct = find_cct(
            params, Vg, theta_g, PG, QG, Y_bus, V, P_load, Q_load,
            fault_bus=bus,
        )
        cct_results[bus] = cct
        cycles = cct * ieee9.F_BASE
        print(f"{bus:>4}  {cct:>10.3f}  {cycles:>14.1f}")

    # --- Bar chart ---
    fig, ax = plt.subplots(figsize=(9, 5))
    buses_list = list(cct_results.keys())
    cct_list   = [cct_results[b] for b in buses_list]
    colors_bar = ["steelblue" if cct_results[b] > 0.1 else "salmon"
                  for b in buses_list]
    bars = ax.bar([str(b) for b in buses_list], cct_list,
                  color=colors_bar, edgecolor="black", linewidth=0.5)
    ax.axhline(0.083, ls="--", color="red", lw=1, label="Lit. value bus 5 ≈ 0.083 s")
    ax.set_xlabel("Faulted bus number")
    ax.set_ylabel("Critical Clearing Time (s)")
    ax.set_title("IEEE 9-bus: CCT for 3-phase faults (PSDAT Program 1.2)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for bar, cct in zip(bars, cct_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{cct:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = "output/ieee9_cct.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
