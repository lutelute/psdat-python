# PSDAT-Python

Pure-Python reimplementation of the **Power System Dynamic Analysis Toolbox (PSDAT)**
originally developed in MATLAB/Simulink by Ismael Abdulrahman (2020).

**Reference**: I. Abdulrahman, "MATLAB-Based Programs for Power System Dynamic Analysis,"
*IEEE Access*, 2020. DOI: [10.1109/ACCESS.2019.2962421](https://doi.org/10.1109/ACCESS.2019.2962421)

---

## What's included

| Module | Description | PSDAT equivalent |
|--------|-------------|-----------------|
| `psdat.data.ieee9` | IEEE 9-bus system data (MATPOWER-compatible) | `IEEE9Bus.m` / `DynData.m` |
| `psdat.data.ieee68` | IEEE 68-bus (NETS/NYPS) system data | `IEEE68Bus.m` |
| `psdat.models.machine` | 6th-order SG + DC1A exciter + governor | Machine model blocks |
| `psdat.models.powerflow` | Newton-Raphson AC power flow (MATPOWER-compatible) | MATPOWER `newtonpf` |
| `psdat.simulation.algebraic` | Norton network solver + AlgebraicSystem (AEs.m) | `AEs.m` |
| `psdat.simulation.solver` | RK45 DAE solver + DAESystem class | Simulink ODE45 |
| `psdat.simulation.classical` | Classical 2nd-order transient stability | Classical model |
| `psdat.analysis.modal` | Eigenvalues + participation factors | Program 1.4 |
| `psdat.analysis.residue` | Transfer function residues + optimal PSS location | Program 1.5 |
| `psdat.analysis.pss` | Phase compensation PSS design | Program 1.6 |
| `psdat.analysis.cct` | Critical clearing time (binary search) | Program 1.2 |
| `psdat.analysis.nose_curve` | P-V nose curve (continuation power flow) | Program 1.2 |

---

## Test systems

- **IEEE 9-bus**: 3 generators, 9 buses, 60 Hz (Anderson & Fouad 2003)
- **IEEE 68-bus**: 16 generators, 68 buses, New England/New York system (Pal & Chaudhuri 2005)

---

## Quick start

```python
from psdat.data import ieee9
from psdat.models.powerflow import run_powerflow, build_ybus
from psdat.models.machine import init_from_powerflow, pack_state_vector
import numpy as np

# 1. Power flow
V, S, n_iter, converged = run_powerflow(ieee9.BUS_DATA, ieee9.BRANCH_DATA)
print(f"Converged in {n_iter} iterations: V_bus 1 = {abs(V[0]):.4f} pu")

# 2. Initialise machine states
gen_idx = [b - 1 for b in ieee9.GEN_BUSES]
params  = ieee9.get_machine_params()
ic = init_from_powerflow(
    np.abs(V[gen_idx]), np.angle(V[gen_idx]),
    S[gen_idx].real, S[gen_idx].imag,
    params,
)
x0 = pack_state_vector(ic)
print(f"Rotor angles (deg): {np.rad2deg(ic['delta'])}")

# 3. Transient stability (classical model)
from psdat.simulation.classical import simulate_classical
Y_bus  = build_ybus(ieee9.BRANCH_DATA, ieee9.N_BUSES)
P_load = ieee9.BUS_DATA[:, 6] / 100.0
Q_load = ieee9.BUS_DATA[:, 7] / 100.0

result = simulate_classical(
    params, np.abs(V[gen_idx]), np.angle(V[gen_idx]),
    S[gen_idx].real, S[gen_idx].imag,
    Y_bus, ieee9.GEN_BUSES,
    fault_bus=5, t_fault=1.0, t_clear=1.07, t_end=5.0, dt=0.001,
    V_pf_all=V, P_load_pu=P_load, Q_load_pu=Q_load,
)
print(f"Max angle spread at t=2s: {result['delta'][2000, :].ptp():.1f} deg")
```

---

## Examples

Run from the repository root:

```bash
# Fault simulation (Program 1.1)
python3 examples/ieee9_fault_gen.py

# Modal analysis (Program 1.4)
python3 examples/ieee9_modal.py

# Critical clearing time (Program 1.2)
python3 examples/ieee9_cct.py
```

Output figures are saved to `examples/output/`.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Machine model state vector

The full machine state vector (PSDAT column-major layout, 11 states per generator):

```
x[0:m]    = Eqp    — d-axis transient EMF (Eq')
x[m:2m]   = Si1d   — d-axis subtransient flux state
x[2m:3m]  = Edp    — q-axis transient EMF (Ed')
x[3m:4m]  = Si2q   — q-axis subtransient flux state
x[4m:5m]  = delta  — rotor angle [rad]
x[5m:6m]  = omega  — rotor angular speed [rad/s]
x[6m:7m]  = Efd    — field voltage
x[7m:8m]  = RF     — exciter rate-feedback signal
x[8m:9m]  = VR     — voltage regulator output
x[9m:10m] = TM     — mechanical torque
x[10m:11m]= PSV    — steam valve position
```

For `m = 3` (IEEE 9-bus), `len(x0) = 33`.

---

## Numerical equivalence

| Quantity | MATLAB PSDAT | PSDAT-Python | Error |
|----------|-------------|--------------|-------|
| Bus 1 voltage (pu) | 1.0400 | 1.0400 | 0.00% |
| Bus 2 voltage (pu) | 1.0250 | 1.0250 | 0.00% |
| Gen 1 rotor angle (deg) | 3.50 | 3.50 | < 0.1% |
| Gen 2 rotor angle (deg) | 59.0 | 58.97 | < 0.1% |
| Gen 3 rotor angle (deg) | 38.6 | 38.64 | < 0.1% |
| Mode 1 frequency (Hz) | ~1.09 | 1.09 | < 1% |
| Mode 2 frequency (Hz) | ~0.72 | 0.72 | < 1% |
| CCT bus 5 (s) | 0.083 | 0.080 | < 4% |

---

## Architecture

```
psdat/
  data/
    ieee9.py        — IEEE 9-bus (MATPOWER bus/branch/gen arrays + MachineParams)
    ieee68.py       — IEEE 68-bus
  models/
    machine.py      — 6th-order SG + DC1A exciter + IEEEG1 governor
    powerflow.py    — Newton-Raphson power flow (MATPOWER-compatible)
  simulation/
    algebraic.py    — Norton network solver + AlgebraicSystem (AEs.m)
    solver.py       — RK4/RK45 DAE solver + DAESystem class
    classical.py    — Classical model transient stability
    initializer.py  — Initialization from power flow
  analysis/
    modal.py        — Eigenvalue analysis + participation factors
    residue.py      — Transfer function residues + optimal PSS location
    pss.py          — Phase compensation PSS design
    cct.py          — Critical clearing time (binary search)
    nose_curve.py   — P-V nose curve (continuation power flow)
examples/
  ieee9_fault_gen.py  — PSDAT Program 1.1
  ieee9_modal.py      — PSDAT Program 1.4
  ieee9_cct.py        — PSDAT Program 1.2
tests/
  test_ieee9_powerflow.py   — Power flow validation (15 tests)
  test_reconstruction.py    — Simulation validation (3 tests)
```

---

## License

MIT License. See LICENSE file.
