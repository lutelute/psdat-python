"""IEEE 9-bus test system data.

Matches MATPOWER case9 and the PSDAT IEEE9Bus.m / DynData.m files exactly.

Network: 9 buses, 3 generators (buses 1, 2, 3), 6 load buses, 9 branches.
Base:    100 MVA, 60 Hz, ws = 2*pi*60 rad/s.

Generator dynamic data notation (PSDAT DynData.m column order per generator):
    Machine data MD (14 rows × 3 cols):
        0  H    [s]
        1  D    (damping, pu)  -> Dm in MachineParams
        2  Xd   [pu]
        3  Xdp  [pu]
        4  Xdpp [pu]
        5  Xq   [pu]
        6  Xqp  [pu]
        7  Xqpp [pu]
        8  Td0p  [s]
        9  Td0pp [s]
        10 Tq0p  [s]
        11 Tq0pp [s]
        12 Rs   [pu]
        13 Xls  [pu]

    Exciter data ED (8 rows × 3 cols):
        0  KA
        1  TA  [s]
        2  KE
        3  TE  [s]
        4  KF
        5  TF  [s]
        6  Ax
        7  Bx

    Turbine data TD (3 rows × 3 cols):
        0  TCH  [s]
        1  TSV  [s]
        2  RD   (droop)

References:
    Anderson & Fouad (2003), Appendix D.
    Pai (1989), Energy Function Analysis for Power System Stability.
    MATPOWER case9 (MATPOWER 7.1).
    Abdulrahman (2020), PSDAT DynData.m.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List

from psdat.models.machine import MachineParams


# ---------------------------------------------------------------------------
# System constants
# ---------------------------------------------------------------------------

BASEMVA = 100.0
WS      = 2.0 * np.pi * 60.0   # synchronous speed [rad/s]

# Legacy aliases
S_BASE = BASEMVA
F_BASE = 60.0
OMEGA0 = WS

N_BUSES   = 9
N_GEN     = 3
GEN_BUSES = [1, 2, 3]   # 1-indexed


# ---------------------------------------------------------------------------
# MATPOWER bus array  (n=9, 13 columns)
# Columns: [BUS_I, TYPE, PD, QD, GS, BS, AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN]
# TYPE: 1=PQ, 2=PV, 3=slack
# Power in MW/MVAr
# ---------------------------------------------------------------------------

BUS = np.array([
    # id  type   PD      QD    GS    BS  area   VM      VA   kV  zone Vmax  Vmin
    [1,    3,    0.0,    0.0,  0.0,  0.0,  1, 1.040,  0.0,   0,    1, 1.1, 0.9],
    [2,    2,    0.0,    0.0,  0.0,  0.0,  1, 1.025,  0.0,   0,    1, 1.1, 0.9],
    [3,    2,    0.0,    0.0,  0.0,  0.0,  1, 1.025,  0.0,   0,    1, 1.1, 0.9],
    [4,    1,    0.0,    0.0,  0.0,  0.0,  1, 1.000,  0.0,   0,    1, 1.1, 0.9],
    [5,    1,  125.0,   50.0,  0.0,  0.0,  1, 1.000,  0.0,   0,    1, 1.1, 0.9],
    [6,    1,   90.0,   30.0,  0.0,  0.0,  1, 1.000,  0.0,   0,    1, 1.1, 0.9],
    [7,    1,    0.0,    0.0,  0.0,  0.0,  1, 1.000,  0.0,   0,    1, 1.1, 0.9],
    [8,    1,  100.0,   35.0,  0.0,  0.0,  1, 1.000,  0.0,   0,    1, 1.1, 0.9],
    [9,    1,    0.0,    0.0,  0.0,  0.0,  1, 1.000,  0.0,   0,    1, 1.1, 0.9],
], dtype=float)

# Legacy alias used by run_powerflow legacy interface
BUS_DATA = np.array([
    # id  type  Vmag  Vang    Pgen    Qgen   Pload  Qload  Gsh  Bsh
    [1,   1,   1.040, 0.0,   71.64,  27.05,   0.0,   0.0,  0.0, 0.0],
    [2,   2,   1.025, 0.0,  163.0,   6.65,    0.0,   0.0,  0.0, 0.0],
    [3,   2,   1.025, 0.0,   85.0,  -10.87,   0.0,   0.0,  0.0, 0.0],
    [4,   3,   1.000, 0.0,    0.0,    0.0,    0.0,   0.0,  0.0, 0.0],
    [5,   3,   1.000, 0.0,    0.0,    0.0,  125.0,  50.0,  0.0, 0.0],
    [6,   3,   1.000, 0.0,    0.0,    0.0,   90.0,  30.0,  0.0, 0.0],
    [7,   3,   1.000, 0.0,    0.0,    0.0,    0.0,   0.0,  0.0, 0.0],
    [8,   3,   1.000, 0.0,    0.0,    0.0,  100.0,  35.0,  0.0, 0.0],
    [9,   3,   1.000, 0.0,    0.0,    0.0,    0.0,   0.0,  0.0, 0.0],
], dtype=float)


# ---------------------------------------------------------------------------
# MATPOWER branch array  (nl=9, 13 columns)
# Columns: [F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C,
#            TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX]
# All in pu on 100-MVA base
# ---------------------------------------------------------------------------

BRANCH = np.array([
    # f   t        R        X        B     rA  rB  rC  tap shift st angmin angmax
    # Anderson & Fouad (2003) Appendix D / PSDAT IEEE9Bus.m exact values
    [1,  4,   0.0000,  0.0576,  0.0000,   0,  0,  0,  0,  0,   1, -360, 360],  # T1
    [4,  5,   0.0100,  0.0850,  0.1760,   0,  0,  0,  0,  0,   1, -360, 360],  # line
    [5,  6,   0.0170,  0.0920,  0.1580,   0,  0,  0,  0,  0,   1, -360, 360],  # line
    [3,  6,   0.0000,  0.0586,  0.0000,   0,  0,  0,  0,  0,   1, -360, 360],  # T3
    [6,  7,   0.0320,  0.1610,  0.3060,   0,  0,  0,  0,  0,   1, -360, 360],  # line
    [7,  8,   0.0085,  0.0720,  0.1490,   0,  0,  0,  0,  0,   1, -360, 360],  # line
    [8,  2,   0.0000,  0.0625,  0.0000,   0,  0,  0,  0,  0,   1, -360, 360],  # T2
    [8,  9,   0.0119,  0.1008,  0.2090,   0,  0,  0,  0,  0,   1, -360, 360],  # line
    [9,  4,   0.0100,  0.0850,  0.1760,   0,  0,  0,  0,  0,   1, -360, 360],  # line
], dtype=float)

# Legacy alias for simple Y-bus builder: [from, to, R, X, B, tap, shift_deg]
BRANCH_DATA = np.array([
    [1,  4,  0.0000, 0.0576, 0.0000, 1.0, 0.0],
    [4,  5,  0.0100, 0.0850, 0.1760, 1.0, 0.0],
    [5,  6,  0.0170, 0.0920, 0.1580, 1.0, 0.0],
    [3,  6,  0.0000, 0.0586, 0.0000, 1.0, 0.0],
    [6,  7,  0.0320, 0.1610, 0.3060, 1.0, 0.0],
    [7,  8,  0.0085, 0.0720, 0.1490, 1.0, 0.0],
    [8,  2,  0.0000, 0.0625, 0.0000, 1.0, 0.0],
    [8,  9,  0.0119, 0.1008, 0.2090, 1.0, 0.0],
    [9,  4,  0.0100, 0.0850, 0.1760, 1.0, 0.0],
], dtype=float)


# ---------------------------------------------------------------------------
# MATPOWER generator array  (ng=3, 10 columns)
# Columns: [GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN]
# Power in MW/MVAr
# ---------------------------------------------------------------------------

GEN = np.array([
    # bus   PG      QG     Qmax   Qmin   VG    Mbase  stat  Pmax   Pmin
    [1,   71.64,  27.05,   300,  -300, 1.040,  100,    1,   250,   10],
    [2,  163.00,   6.65,   300,  -300, 1.025,  100,    1,   300,   10],
    [3,   85.00, -10.87,   300,  -300, 1.025,  100,    1,   270,   10],
], dtype=float)


# ---------------------------------------------------------------------------
# Dynamic machine data (MD) — 14 parameters x 3 generators
# Each column corresponds to one generator (gen 1, gen 2, gen 3).
# Values from Anderson & Fouad (2003) Appendix D / PSDAT DynData.m
# ---------------------------------------------------------------------------

MD = np.array([
    # row   param          gen1     gen2     gen3
    # 0     H [s]
    [                    23.64,    6.40,    3.01],
    # 1     D (damping Dm)
    [                     0.0,     0.0,     0.0],
    # 2     Xd [pu]
    [                     0.1460,  0.8958,  1.3125],
    # 3     Xdp [pu]
    [                     0.0608,  0.1198,  0.1813],
    # 4     Xdpp [pu]
    [                     0.0608,  0.1198,  0.1813],
    # 5     Xq [pu]
    [                     0.0969,  0.8645,  1.2578],
    # 6     Xqp [pu]
    [                     0.0969,  0.1969,  0.2500],
    # 7     Xqpp [pu]
    [                     0.0969,  0.1969,  0.2500],
    # 8     Td0p [s]
    [                     8.96,    6.00,    5.89],
    # 9     Td0pp [s]
    [                     0.031,   0.0350,  0.0600],
    # 10    Tq0p [s]
    [                     0.31,    0.535,   0.60],
    # 11    Tq0pp [s]
    [                     0.050,   0.0500,  0.0600],
    # 12    Rs [pu]
    [                     0.0,     0.0,     0.0],
    # 13    Xls [pu]
    [                     0.019,   0.0521,  0.0742],
])

# ---------------------------------------------------------------------------
# Exciter data (ED) — 8 parameters x 3 generators
# ---------------------------------------------------------------------------

ED = np.array([
    # row   param     gen1    gen2    gen3
    # 0     KA
    [                20.0,   20.0,   20.0],
    # 1     TA [s]
    [                 0.2,    0.2,    0.2],
    # 2     KE
    [                 1.0,    1.0,    1.0],
    # 3     TE [s]
    [                 0.314,  0.314,  0.314],
    # 4     KF
    [                 0.063,  0.063,  0.063],
    # 5     TF [s]
    [                 0.35,   0.35,   0.35],
    # 6     Ax
    [                 0.0039, 0.0039, 0.0039],
    # 7     Bx
    [                 1.555,  1.555,  1.555],
])

# ---------------------------------------------------------------------------
# Turbine data (TD) — 3 parameters x 3 generators
# ---------------------------------------------------------------------------

TD = np.array([
    # row   param     gen1    gen2    gen3
    # 0     TCH [s]
    [                 0.30,   0.30,   0.30],
    # 1     TSV [s]
    [                 0.20,   0.20,   0.20],
    # 2     RD (droop)
    [                 0.05,   0.05,   0.05],
])

# Legacy dict-based format (backward compatibility)
MACHINE_DATA = [
    {   # Generator 1 - bus 1 (slack)
        "bus": 1, "S_rated": 100.0,
        "Ra": 0.0,   "Xd": 0.1460, "Xd_p": 0.0608, "Xd_pp": 0.0608,
        "Xq": 0.0969, "Xq_p": 0.0969, "Xq_pp": 0.0969,
        "Xl": 0.019,
        "Td0_p": 8.96, "Td0_pp": 0.031, "Tq0_p": 0.31, "Tq0_pp": 0.050,
        "H": 23.64, "D": 0.0,
        "Ka": 20.0, "Ta": 0.2, "Ke": 1.0, "Te": 0.314,
        "Kf": 0.063, "Tf": 0.35, "Vr_max": 1.0, "Vr_min": -1.0,
        "Kg": 1.0, "Tg": 0.1, "R_gov": 0.05,
    },
    {   # Generator 2 - bus 2
        "bus": 2, "S_rated": 100.0,
        "Ra": 0.0,   "Xd": 0.8958, "Xd_p": 0.1198, "Xd_pp": 0.1198,
        "Xq": 0.8645, "Xq_p": 0.1969, "Xq_pp": 0.1969,
        "Xl": 0.0521,
        "Td0_p": 6.0, "Td0_pp": 0.0350, "Tq0_p": 0.535, "Tq0_pp": 0.0500,
        "H": 6.40, "D": 0.0,
        "Ka": 20.0, "Ta": 0.2, "Ke": 1.0, "Te": 0.314,
        "Kf": 0.063, "Tf": 0.35, "Vr_max": 1.0, "Vr_min": -1.0,
        "Kg": 1.0, "Tg": 0.1, "R_gov": 0.05,
    },
    {   # Generator 3 - bus 3
        "bus": 3, "S_rated": 100.0,
        "Ra": 0.0,   "Xd": 1.3125, "Xd_p": 0.1813, "Xd_pp": 0.1813,
        "Xq": 1.2578, "Xq_p": 0.2500, "Xq_pp": 0.2500,
        "Xl": 0.0742,
        "Td0_p": 5.89, "Td0_pp": 0.0600, "Tq0_p": 0.60, "Tq0_pp": 0.0600,
        "H": 3.01, "D": 0.0,
        "Ka": 20.0, "Ta": 0.2, "Ke": 1.0, "Te": 0.314,
        "Kf": 0.063, "Tf": 0.35, "Vr_max": 1.0, "Vr_min": -1.0,
        "Kg": 1.0, "Tg": 0.1, "R_gov": 0.05,
    },
]


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def get_system() -> Dict:
    """Return the complete IEEE 9-bus system as a dict of numpy arrays.

    Returns
    -------
    sys : dict
        'baseMVA'   — float, 100 MVA
        'ws'        — float, 2*pi*60 rad/s
        'bus'       — ndarray (9, 13), MATPOWER bus array
        'branch'    — ndarray (9, 13), MATPOWER branch array
        'gen'       — ndarray (3, 10), MATPOWER generator array
        'MD'        — ndarray (14, 3), machine dynamic data
        'ED'        — ndarray (8, 3),  exciter data
        'TD'        — ndarray (3, 3),  turbine data
        'params'    — list of 3 MachineParams objects
    """
    return {
        'baseMVA': BASEMVA,
        'ws':      WS,
        'bus':     BUS.copy(),
        'branch':  BRANCH.copy(),
        'gen':     GEN.copy(),
        'MD':      MD.copy(),
        'ED':      ED.copy(),
        'TD':      TD.copy(),
        'params':  get_machine_params(),
    }


def get_machine_params() -> List[MachineParams]:
    """Return list of 3 MachineParams for the IEEE 9-bus generators.

    Returns
    -------
    params : list of MachineParams, length 3
        One per generator (gen 1 at bus 1, gen 2 at bus 2, gen 3 at bus 3).
    """
    params = []
    for g in range(3):
        p = MachineParams(
            H     = float(MD[0, g]),
            Dm    = float(MD[1, g]),
            Xd    = float(MD[2, g]),
            Xdp   = float(MD[3, g]),
            Xdpp  = float(MD[4, g]),
            Xq    = float(MD[5, g]),
            Xqp   = float(MD[6, g]),
            Xqpp  = float(MD[7, g]),
            Td0p  = float(MD[8, g]),
            Td0pp = float(MD[9, g]),
            Tq0p  = float(MD[10, g]),
            Tq0pp = float(MD[11, g]),
            Rs    = float(MD[12, g]),
            Xls   = float(MD[13, g]),
            KA    = float(ED[0, g]),
            TA    = float(ED[1, g]),
            KE    = float(ED[2, g]),
            TE    = float(ED[3, g]),
            KF    = float(ED[4, g]),
            TF    = float(ED[5, g]),
            Ax    = float(ED[6, g]),
            Bx    = float(ED[7, g]),
            TCH   = float(TD[0, g]),
            TSV   = float(TD[1, g]),
            RD    = float(TD[2, g]),
            ws    = WS,
        )
        params.append(p)
    return params
