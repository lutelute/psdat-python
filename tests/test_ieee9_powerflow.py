"""Tests: IEEE 9-bus power flow validation.

Expected results (from MATPOWER/literature, Anderson & Fouad 2003):
  Bus 1 (slack): V=1.040 pu, theta=0 deg
  Bus 2 (PV):    V=1.025 pu, P=163 MW
  Bus 3 (PV):    V=1.025 pu, P=85 MW
  Bus 5 (PQ):    V in [0.98, 1.01] pu, P_load=125 MW
  Bus 6 (PQ):    V in [1.00, 1.03] pu, P_load=90 MW
  Bus 8 (PQ):    V in [1.00, 1.03] pu, P_load=100 MW
"""
import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from psdat.data import ieee9
from psdat.models.powerflow import run_powerflow, build_ybus


@pytest.fixture(scope="module")
def pf_result():
    """Run power flow once and reuse across all tests."""
    V, S, n_iter, converged = run_powerflow(ieee9.BUS_DATA, ieee9.BRANCH_DATA)
    return {"V": V, "S": S, "n_iter": n_iter, "converged": converged}


def test_powerflow_convergence(pf_result):
    """Power flow must converge in fewer than 10 iterations."""
    assert pf_result["converged"], "Power flow did not converge"
    assert pf_result["n_iter"] < 10, (
        f"Power flow took {pf_result['n_iter']} iterations (expected < 10)"
    )


def test_ybus_symmetry():
    """Y-bus matrix must be symmetric."""
    Y = build_ybus(ieee9.BRANCH_DATA, ieee9.N_BUSES)
    diff = np.max(np.abs(Y - Y.T))
    assert diff < 1e-10, f"Y-bus is not symmetric: max diff = {diff:.2e}"


def test_ybus_shape():
    """Y-bus must be 9x9."""
    Y = build_ybus(ieee9.BRANCH_DATA, ieee9.N_BUSES)
    assert Y.shape == (9, 9), f"Expected (9,9), got {Y.shape}"


def test_bus1_slack_voltage(pf_result):
    """Bus 1 (slack): V=1.040 pu, theta=0."""
    V = pf_result["V"]
    assert abs(abs(V[0]) - 1.040) < 1e-4, f"Bus 1 V={abs(V[0]):.4f}, expected 1.040"
    assert abs(np.angle(V[0])) < 1e-6, f"Bus 1 theta={np.rad2deg(np.angle(V[0])):.4f} deg, expected 0"


def test_bus2_pv_voltage(pf_result):
    """Bus 2 (PV): V=1.025 pu."""
    V = pf_result["V"]
    assert abs(abs(V[1]) - 1.025) < 1e-4, f"Bus 2 V={abs(V[1]):.4f}, expected 1.025"


def test_bus3_pv_voltage(pf_result):
    """Bus 3 (PV): V=1.025 pu."""
    V = pf_result["V"]
    assert abs(abs(V[2]) - 1.025) < 1e-4, f"Bus 3 V={abs(V[2]):.4f}, expected 1.025"


def test_bus_voltages_within_limits(pf_result):
    """All bus voltages must be within 0.9-1.1 pu."""
    V = pf_result["V"]
    Vmag = np.abs(V)
    assert Vmag.min() >= 0.90, f"Min voltage = {Vmag.min():.4f} < 0.90"
    assert Vmag.max() <= 1.10, f"Max voltage = {Vmag.max():.4f} > 1.10"


def test_bus5_load_voltage(pf_result):
    """Bus 5 (load, P=125 MW): voltage within 2% of typical values."""
    V = pf_result["V"]
    Vm5 = abs(V[4])
    assert 0.96 <= Vm5 <= 1.04, f"Bus 5 V={Vm5:.4f} outside [0.96, 1.04]"


def test_bus6_load_voltage(pf_result):
    """Bus 6 (load, P=90 MW): voltage within typical range."""
    V = pf_result["V"]
    Vm6 = abs(V[5])
    assert 0.98 <= Vm6 <= 1.05, f"Bus 6 V={Vm6:.4f} outside [0.98, 1.05]"


def test_bus8_load_voltage(pf_result):
    """Bus 8 (load, P=100 MW): voltage within typical range."""
    V = pf_result["V"]
    Vm8 = abs(V[7])
    assert 0.98 <= Vm8 <= 1.05, f"Bus 8 V={Vm8:.4f} outside [0.98, 1.05]"


def test_generator_output_bus1(pf_result):
    """Bus 1 (slack) generation: ~70 MW."""
    S = pf_result["S"]
    PG1 = S[0].real * 100.0   # MW
    assert 60.0 <= PG1 <= 80.0, f"Bus 1 PG={PG1:.1f} MW outside [60, 80]"


def test_generator_output_bus2(pf_result):
    """Bus 2 (PV): P_scheduled=163 MW."""
    S = pf_result["S"]
    PG2 = S[1].real * 100.0   # MW
    assert abs(PG2 - 163.0) < 2.0, f"Bus 2 PG={PG2:.1f} MW, expected 163 MW"


def test_generator_output_bus3(pf_result):
    """Bus 3 (PV): P_scheduled=85 MW."""
    S = pf_result["S"]
    PG3 = S[2].real * 100.0   # MW
    assert abs(PG3 - 85.0) < 2.0, f"Bus 3 PG={PG3:.1f} MW, expected 85 MW"


def test_power_balance(pf_result):
    """Total generation must equal total load (plus losses)."""
    S = pf_result["S"]
    total_P_inj = S.real.sum() * 100.0   # MW
    # Total injection should be ~0 (P_gen - P_load ≈ losses, small)
    total_load = ieee9.BUS_DATA[:, 6].sum()    # 315 MW
    total_gen  = ieee9.BUS_DATA[:, 4].sum()    # 248 MW (approx, Pgen col)
    # Check net injection sum is small (losses only)
    assert abs(total_P_inj) < 10.0, (
        f"Net power injection = {total_P_inj:.1f} MW (expected ~0)"
    )


def test_initial_conditions():
    """Machine initialization must produce physically meaningful values."""
    from psdat.models.machine import init_from_powerflow, pack_state_vector, unpack_state_vector

    V, S, _, _ = run_powerflow(ieee9.BUS_DATA, ieee9.BRANCH_DATA)
    gen_idx = [b - 1 for b in ieee9.GEN_BUSES]
    Vg      = np.abs(V[gen_idx])
    theta_g = np.angle(V[gen_idx])
    PG      = S[gen_idx].real
    QG      = S[gen_idx].imag
    params  = ieee9.get_machine_params()

    ic = init_from_powerflow(Vg, theta_g, PG, QG, params)
    x0 = pack_state_vector(ic)

    assert x0.shape == (11 * 3,), f"x0 shape {x0.shape}, expected (33,)"

    states = unpack_state_vector(x0, 3)
    delta_deg = np.rad2deg(states["delta"])

    # Rotor angles should be between -90 and 150 deg
    assert delta_deg.min() > -90, f"Min delta={delta_deg.min():.1f} deg"
    assert delta_deg.max() < 150, f"Max delta={delta_deg.max():.1f} deg"

    # Omega should be close to synchronous speed
    omega = states["omega"]
    ws    = ieee9.OMEGA0   # ~377 rad/s
    assert np.all(np.abs(omega - ws) < 1.0), (
        f"omega deviates from ws by more than 1 rad/s: {omega}"
    )

    # Efd should be positive
    assert np.all(ic["Efd"] > 0), f"Negative Efd: {ic['Efd']}"
