#!/usr/bin/env python3
"""IEEE 9-bus: Generator-side fault simulation (PSDAT Program 1.1).

Simulates a 3-phase bolted fault at load bus 5 at t=1.0 s, cleared at
t=1.1 s (100 ms, above CCT → unstable) and t=1.05 s (50 ms, below CCT
→ marginally stable). Uses the classical machine model (constant internal
voltage behind synchronous reactance Xq).

Plots: rotor angles, angular velocity deviations for all 3 generators.
Matches the expected behaviour described in PSDAT Example 1a.
Saves: output/ieee9_fault_gen.png
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
from psdat.simulation.classical import simulate_classical


def setup():
    V, S, n_iter, conv = run_powerflow(ieee9.BUS_DATA, ieee9.BRANCH_DATA)
    if not conv:
        raise RuntimeError(f"Power flow did not converge (iter={n_iter})")
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

    return params, Vg, theta_g, PG, QG, Y_bus, V, P_load, Q_load


def main():
    os.makedirs("output", exist_ok=True)
    params, Vg, theta_g, PG, QG, Y_bus, V_pf, P_load, Q_load = setup()

    fault_bus  = 5
    t_fault    = 1.0
    t_end      = 5.0
    dt         = 0.001
    kw = dict(
        machines=params, Vg_pf=Vg, theta_g_pf=theta_g,
        PG=PG, QG=QG, Y_bus_pre=Y_bus, gen_buses=ieee9.GEN_BUSES,
        fault_bus=fault_bus, t_fault=t_fault, t_end=t_end, dt=dt,
        V_pf_all=V_pf, P_load_pu=P_load, Q_load_pu=Q_load,
        omega0=ieee9.OMEGA0,
    )

    # Two cases: unstable (t_clear=1.1s) and stable (t_clear=1.05s)
    cases = {
        "t_clear=1.1 s (unstable)":  dict(**kw, t_clear=1.1),
        "t_clear=1.05 s (marginal)": dict(**kw, t_clear=1.05),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
    fig.suptitle(
        f"IEEE 9-bus: 3-phase fault at bus {fault_bus}  (classical model, PSDAT Prog. 1.1)",
        fontsize=12,
    )
    labels  = [f"Gen {i+1} (bus {ieee9.GEN_BUSES[i]})" for i in range(3)]
    colors  = plt.cm.tab10([0, 1, 2])
    ws      = ieee9.OMEGA0   # 377 rad/s

    for col, (title, kwargs) in enumerate(cases.items()):
        t_clear = kwargs["t_clear"]
        res     = simulate_classical(**kwargs)
        t       = res["t"]
        delta   = res["delta"]     # degrees
        omega   = res["omega"]     # rad/s

        # --- Rotor angles ---
        ax = axes[0, col]
        for i in range(3):
            ax.plot(t, delta[:, i], label=labels[i], color=colors[i])
        ax.axvspan(t_fault, t_clear, alpha=0.15, color="red", label="Fault")
        ax.set_ylabel("Rotor angle delta (deg)")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, t_end)

        # --- Angular speed deviation ---
        ax = axes[1, col]
        for i in range(3):
            dev_hz = (omega[:, i] - ws) / (2 * np.pi)   # Hz
            ax.plot(t, dev_hz, label=labels[i], color=colors[i])
        ax.axvspan(t_fault, t_clear, alpha=0.15, color="red")
        ax.set_ylabel("Speed deviation (Hz)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, t_end)

        print(f"\n{title}")
        print(f"  Initial delta (deg): {delta[0, :]}")
        print(f"  delta at t={t_clear+0.1:.1f}s: {delta[int((t_clear+0.1)/dt), :]}")

    plt.tight_layout()
    out_path = "output/ieee9_fault_gen.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
