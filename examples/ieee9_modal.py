#!/usr/bin/env python3
"""IEEE 9-bus: Modal analysis (PSDAT Program 1.4).

Computes the linearized state matrix A, eigenvalues, participation factors,
and identifies electromechanical oscillation modes.

Expected results for IEEE 9-bus (60 Hz):
  - 2 electromechanical modes in 0.8-2.0 Hz range
  - All modes stable (sigma < 0)
  - Dominant participation from delta and omega states

Saves: output/ieee9_modal_eigenvalues.png
       output/ieee9_modal_participation.png
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
from psdat.models.machine import (
    get_state_labels,
    init_from_powerflow,
    pack_state_vector,
    unpack_state_vector,
)
from psdat.analysis.modal import (
    compute_modal_analysis,
    compute_state_matrix,
    print_eigenvalue_table,
    identify_electromechanical_modes,
)
from psdat.simulation.classical import simulate_classical, compute_internal_emf
from psdat.simulation.algebraic import build_reduced_ybus


def build_linearized_model():
    """Build linearized swing-equation state matrix for IEEE 9-bus."""
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

    E_prime, delta_0 = compute_internal_emf(Vg, theta_g, PG, QG, params)
    E_mag = np.abs(E_prime)

    # Build load-augmented Y_bus and reduced Y (internal buses via Xq)
    Y_load = Y_bus.copy()
    for i in range(ieee9.N_BUSES):
        Vm2 = max(abs(V[i]) ** 2, 1e-6)
        Y_load[i, i] += complex(P_load[i], -Q_load[i]) / Vm2

    n_bus = ieee9.N_BUSES
    n_gen = ieee9.N_GEN
    n_ext = n_bus + n_gen
    Y_ext = np.zeros((n_ext, n_ext), dtype=complex)
    Y_ext[:n_bus, :n_bus] = Y_load
    for g in range(n_gen):
        bi  = ieee9.GEN_BUSES[g] - 1
        igi = n_bus + g
        y_g = 1.0 / complex(params[g].Rs, max(params[g].Xq, 1e-6))
        Y_ext[bi,  bi]   += y_g;  Y_ext[igi, igi] += y_g
        Y_ext[bi,  igi]  -= y_g;  Y_ext[igi, bi]  -= y_g

    elim = list(range(n_bus))
    keep = list(range(n_bus, n_ext))
    Y_red = (Y_ext[np.ix_(keep, keep)]
             - Y_ext[np.ix_(keep, elim)]
               @ np.linalg.solve(Y_ext[np.ix_(elim, elim)],
                                 Y_ext[np.ix_(elim, keep)]))

    # State: x = [delta_1, ..., delta_n, omega_1, ..., omega_n]
    ws = ieee9.OMEGA0
    H  = np.array([p.H for p in params])

    def swing_rhs(x_vec: np.ndarray) -> np.ndarray:
        """Right-hand side of swing equations at x_vec."""
        delta = x_vec[:n_gen]
        omega = x_vec[n_gen:]
        Pe = np.zeros(n_gen)
        for i in range(n_gen):
            for j in range(n_gen):
                Yij = Y_red[i, j]
                dij = delta[i] - delta[j]
                Pe[i] += E_mag[i] * E_mag[j] * (
                    Yij.real * np.cos(dij) + Yij.imag * np.sin(dij))
        Pm = PG.copy()
        ddelta = omega - ws
        domega = (ws / (2.0 * H)) * (Pm - Pe)
        return np.concatenate([ddelta, domega])

    x0 = np.concatenate([delta_0, np.full(n_gen, ws)])
    A = compute_state_matrix(swing_rhs, x0, eps=1e-6)

    state_labels = (
        [f"delta_{g+1}" for g in range(n_gen)]
        + [f"omega_{g+1}" for g in range(n_gen)]
    )
    return A, state_labels, delta_0, params


def main():
    os.makedirs("output", exist_ok=True)

    A, state_labels, delta_0, params = build_linearized_model()
    n = A.shape[0]

    # Modal analysis
    modal = compute_modal_analysis(A, state_labels)
    print("\n=== IEEE 9-bus Modal Analysis (PSDAT Program 1.4) ===")
    print_eigenvalue_table(modal, n_show=10)

    # Identify electromechanical modes
    n_gen = ieee9.N_GEN
    delta_idx = list(range(n_gen))   # delta states are first
    em_modes  = identify_electromechanical_modes(modal, delta_idx, (0.1, 3.0))
    print(f"\nElectromechanical modes: {[m+1 for m in em_modes]}")
    for m in em_modes:
        lam = modal.eigenvalues[m]
        f   = modal.frequencies_hz[m]
        z   = modal.damping_ratios[m]
        print(f"  Mode {m+1}: lambda={lam.real:.4f}+j{lam.imag:.4f} "
              f"f={f:.3f} Hz  zeta={z:.4f}")

    # --- Plot 1: Eigenvalue s-plane ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("IEEE 9-bus Modal Analysis (PSDAT Program 1.4)", fontsize=12)

    ax = axes[0]
    lams = modal.eigenvalues
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.scatter(lams.real, lams.imag, c="steelblue", zorder=5)
    for m in em_modes:
        lam = lams[m]
        ax.scatter([lam.real], [lam.imag], c="red", s=80, zorder=10,
                   label=f"Mode {m+1}: f={modal.frequencies_hz[m]:.2f}Hz")
    ax.set_xlabel("Real part (1/s)")
    ax.set_ylabel("Imaginary part (rad/s)")
    ax.set_title("Eigenvalue s-plane")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Participation factors ---
    ax = axes[1]
    # Show only electromechanical modes
    n_em = min(len(em_modes), 4)
    pf_mat = modal.participation_factors[:, em_modes[:n_em]]
    im = ax.imshow(pf_mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n_em))
    ax.set_xticklabels([f"Mode {em_modes[k]+1}\n{modal.frequencies_hz[em_modes[k]]:.2f}Hz"
                        for k in range(n_em)], fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(state_labels, fontsize=8)
    ax.set_title("Participation Factors (electromechanical modes)")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    out1 = "output/ieee9_modal.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out1}")
    plt.close()


if __name__ == "__main__":
    main()
