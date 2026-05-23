"""Tests: Classical transient stability simulation for IEEE 9-bus."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from psdat.data import ieee9
from psdat.models.powerflow import run_powerflow, build_ybus
from psdat.simulation.classical import simulate_classical, compute_internal_emf


@pytest.fixture(scope="module")
def pf_and_classical():
    """Run power flow and classical simulation once."""
    V, S, _, _ = run_powerflow(ieee9.BUS_DATA, ieee9.BRANCH_DATA)
    gen_idx = [b - 1 for b in ieee9.GEN_BUSES]
    Vg = np.abs(V[gen_idx])
    theta_g = np.angle(V[gen_idx])
    PG = S[gen_idx].real
    QG = S[gen_idx].imag
    params = ieee9.get_machine_params()
    Y_bus = build_ybus(ieee9.BRANCH_DATA, ieee9.N_BUSES)
    P_load = ieee9.BUS_DATA[:, 6] / ieee9.S_BASE
    Q_load = ieee9.BUS_DATA[:, 7] / ieee9.S_BASE
    return params, Vg, theta_g, PG, QG, Y_bus, V, P_load, Q_load


def test_no_fault_equilibrium(pf_and_classical):
    """Without fault, angles must remain within 0.1 deg of initial value."""
    params, Vg, theta_g, PG, QG, Y_bus, V, P_load, Q_load = pf_and_classical
    result = simulate_classical(
        params, Vg, theta_g, PG, QG, Y_bus, ieee9.GEN_BUSES,
        fault_bus=5, t_fault=100.0, t_clear=200.0, t_end=3.0, dt=0.01,
        V_pf_all=V, P_load_pu=P_load, Q_load_pu=Q_load,
    )
    delta0 = result["delta"][0, :]
    for k in range(result["delta"].shape[0]):
        drift = np.max(np.abs(result["delta"][k, :] - delta0))
        assert drift < 0.1, f"Angle drifts {drift:.2f} deg at t={result['t'][k]:.2f}s"


def test_fault_causes_oscillation(pf_and_classical):
    """Short fault (30 ms) should cause oscillation that eventually settles."""
    params, Vg, theta_g, PG, QG, Y_bus, V, P_load, Q_load = pf_and_classical
    result = simulate_classical(
        params, Vg, theta_g, PG, QG, Y_bus, ieee9.GEN_BUSES,
        fault_bus=5, t_fault=1.0, t_clear=1.03, t_end=5.0, dt=0.005,
        V_pf_all=V, P_load_pu=P_load, Q_load_pu=Q_load,
    )
    delta = result["delta"]
    # Angles must swing during fault
    delta_at_1_05 = delta[int(1.05 / 0.005), :]
    delta_at_0    = delta[0, :]
    swing = np.abs(delta_at_1_05 - delta_at_0).max()
    assert swing > 1.0, f"Expected angle swing > 1 deg during fault, got {swing:.3f}"

    # System remains bounded
    final_spread = delta[-1, :].max() - delta[-1, :].min()
    assert final_spread < 300, f"Final angle spread = {final_spread:.1f} deg (unstable?)"


def test_internal_emf_initial():
    """Internal EMF angles must match PSDAT initialization values."""
    V, S, _, _ = run_powerflow(ieee9.BUS_DATA, ieee9.BRANCH_DATA)
    gen_idx = [0, 1, 2]
    Vg = np.abs(V[gen_idx]); theta_g = np.angle(V[gen_idx])
    PG = S[gen_idx].real; QG = S[gen_idx].imag
    params = ieee9.get_machine_params()

    E_prime, delta_0 = compute_internal_emf(Vg, theta_g, PG, QG, params)
    delta_deg = np.rad2deg(delta_0)

    # PSDAT reference: delta ~ [3.5, 59.0, 38.6] degrees
    expected = np.array([3.5, 59.0, 38.6])
    for g in range(3):
        assert abs(delta_deg[g] - expected[g]) < 1.0, (
            f"Gen {g+1}: delta={delta_deg[g]:.2f} deg, expected {expected[g]:.1f} deg"
        )
