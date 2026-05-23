"""6th-order subtransient synchronous generator with DC1A exciter and steam governor.

State vector per generator — 11 states (PSDAT MATLAB column-major layout):
    When stacking m generators the full state vector x of length 11*m is:
        x[0:m]    = Eqp   — transient d-axis voltage (q-axis open-circuit transient EMF)
        x[m:2m]   = Si1d  — subtransient d-axis state (d-axis damper winding flux)
        x[2m:3m]  = Edp   — transient q-axis voltage (d-axis open-circuit transient EMF)
        x[3m:4m]  = Si2q  — subtransient q-axis state (q-axis damper winding flux)
        x[4m:5m]  = delta — rotor angle [rad]
        x[5m:6m]  = omega — rotor angular velocity [rad/s]
        x[6m:7m]  = Efd   — field voltage
        x[7m:8m]  = RF    — rate-feedback signal (exciter inner loop)
        x[8m:9m]  = VR    — voltage regulator output
        x[9m:10m] = TM    — mechanical torque [pu]
        x[10m:11m]= PSV   — steam valve position [pu]

    For a single machine (local index within its own 11-element slice):
        [0] Eqp, [1] Si1d, [2] Edp, [3] Si2q, [4] delta, [5] omega,
        [6] Efd, [7] RF, [8] VR, [9] TM, [10] PSV

This layout matches the PSDAT MATLAB toolbox exactly (Abdulrahman 2020).

References:
    Abdulrahman (2020), PSDAT MATLAB toolbox.
    Kundur (1994), Power System Stability and Control, Ch. 4-5.
    Anderson & Fouad (2003), Power System Control and Stability, 2nd ed.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List


N_STATES_PER_GEN = 11   # states per machine


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class MachineParams:
    """All parameters for a single synchronous generator unit.

    Machine (electrical):
        H      — inertia constant [s]
        Xd     — d-axis synchronous reactance [pu]
        Xdp    — d-axis transient reactance [pu]
        Xdpp   — d-axis subtransient reactance [pu]
        Xq     — q-axis synchronous reactance [pu]
        Xqp    — q-axis transient reactance [pu]
        Xqpp   — q-axis subtransient reactance [pu]
        Td0p   — d-axis open-circuit transient time constant [s]
        Td0pp  — d-axis open-circuit subtransient time constant [s]
        Tq0p   — q-axis open-circuit transient time constant [s]
        Tq0pp  — q-axis open-circuit subtransient time constant [s]
        Rs     — stator resistance [pu]
        Xls    — stator leakage reactance [pu]
        Dm     — mechanical damping coefficient [pu torque / pu speed deviation]

    DC1A Exciter (IEEE type DC1A):
        KA     — voltage regulator gain
        TA     — voltage regulator time constant [s]
        KE     — exciter self-excitation constant
        TE     — exciter time constant [s]
        KF     — rate-feedback gain
        TF     — rate-feedback time constant [s]
        Ax     — saturation coefficient A (SE = Ax*exp(Bx*|Efd|))
        Bx     — saturation coefficient B

    Steam turbine / governor:
        TCH    — chest time constant [s]
        TSV    — steam valve time constant [s]
        RD     — droop (speed regulation) [pu]

    System:
        ws     — synchronous speed [rad/s]  default = 2*pi*60

    Legacy aliases (for backward compatibility with other parts of the codebase):
        Ra  → Rs
        Xl  → Xls
        D   → Dm
        Ka  → KA
        Ta  → TA
        Ke  → KE
        Te  → TE
        Kf  → KF
        Tf  → TF
        R_gov → RD
    """

    # Machine electrical
    H: float = 6.5
    Xd: float = 1.0
    Xdp: float = 0.3
    Xdpp: float = 0.2
    Xq: float = 0.95
    Xqp: float = 0.65
    Xqpp: float = 0.2
    Td0p: float = 8.0
    Td0pp: float = 0.03
    Tq0p: float = 0.4
    Tq0pp: float = 0.05
    Rs: float = 0.0
    Xls: float = 0.15
    Dm: float = 0.0

    # DC1A exciter
    KA: float = 20.0
    TA: float = 0.2
    KE: float = 1.0
    TE: float = 0.314
    KF: float = 0.063
    TF: float = 0.35
    Ax: float = 0.0039
    Bx: float = 1.555

    # Governor / turbine
    TCH: float = 0.3
    TSV: float = 0.2
    RD: float = 0.05

    # System base
    ws: float = 2.0 * np.pi * 60.0  # 377 rad/s for 60-Hz systems

    # Legacy aliases exposed as properties for backward compatibility
    @property
    def Ra(self) -> float:
        return self.Rs

    @Ra.setter
    def Ra(self, v: float) -> None:
        self.Rs = v

    @property
    def Xl(self) -> float:
        return self.Xls

    @Xl.setter
    def Xl(self, v: float) -> None:
        self.Xls = v

    @property
    def D(self) -> float:
        return self.Dm

    @D.setter
    def D(self, v: float) -> None:
        self.Dm = v

    @property
    def Ka(self) -> float:
        return self.KA

    @Ka.setter
    def Ka(self, v: float) -> None:
        self.KA = v

    @property
    def Ta(self) -> float:
        return self.TA

    @Ta.setter
    def Ta(self, v: float) -> None:
        self.TA = v

    @property
    def Ke(self) -> float:
        return self.KE

    @Ke.setter
    def Ke(self, v: float) -> None:
        self.KE = v

    @property
    def Te(self) -> float:
        return self.TE

    @Te.setter
    def Te(self, v: float) -> None:
        self.TE = v

    @property
    def Kf(self) -> float:
        return self.KF

    @Kf.setter
    def Kf(self, v: float) -> None:
        self.KF = v

    @property
    def Tf(self) -> float:
        return self.TF

    @Tf.setter
    def Tf(self, v: float) -> None:
        self.TF = v

    @property
    def R_gov(self) -> float:
        return self.RD

    @R_gov.setter
    def R_gov(self, v: float) -> None:
        self.RD = v

    # Subtransient aliases (Xd_pp, Xq_pp style)
    @property
    def Xd_p(self) -> float:
        return self.Xdp

    @Xd_p.setter
    def Xd_p(self, v: float) -> None:
        self.Xdp = v

    @property
    def Xd_pp(self) -> float:
        return self.Xdpp

    @Xd_pp.setter
    def Xd_pp(self, v: float) -> None:
        self.Xdpp = v

    @property
    def Xq_p(self) -> float:
        return self.Xqp

    @Xq_p.setter
    def Xq_p(self, v: float) -> None:
        self.Xqp = v

    @property
    def Xq_pp(self) -> float:
        return self.Xqpp

    @Xq_pp.setter
    def Xq_pp(self, v: float) -> None:
        self.Xqpp = v

    @property
    def Td0_p(self) -> float:
        return self.Td0p

    @Td0_p.setter
    def Td0_p(self, v: float) -> None:
        self.Td0p = v

    @property
    def Td0_pp(self) -> float:
        return self.Td0pp

    @Td0_pp.setter
    def Td0_pp(self, v: float) -> None:
        self.Td0pp = v

    @property
    def Tq0_p(self) -> float:
        return self.Tq0p

    @Tq0_p.setter
    def Tq0_p(self, v: float) -> None:
        self.Tq0p = v

    @property
    def Tq0_pp(self) -> float:
        return self.Tq0pp

    @Tq0_pp.setter
    def Tq0_pp(self, v: float) -> None:
        self.Tq0pp = v


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def saturation(Efd, Ax: float = 0.0039, Bx: float = 1.555):
    """DC exciter saturation function SE(Efd) = Ax * exp(Bx * |Efd|).

    Parameters
    ----------
    Efd : float or ndarray
        Field voltage [pu].
    Ax, Bx : float
        Saturation coefficients from machine data.

    Returns
    -------
    SE : same type/shape as Efd
        Saturation value (always >= 0).
    """
    return Ax * np.exp(Bx * np.abs(Efd))


# ---------------------------------------------------------------------------
# Initialization  (PSDAT Main_File.m convention)
# ---------------------------------------------------------------------------

def init_from_powerflow(
    Vg,
    theta_g=None,
    PG=None,
    QG=None,
    params=None,
    # Legacy single-machine call: init_from_powerflow(params, V_complex, I_complex, ...)
    _v_terminal=None,
    _i_terminal=None,
    S_base: float = 100.0,
    omega0: float = None,
) -> Dict[str, np.ndarray]:
    """Compute initial state variable values from a power-flow solution.

    **Multi-machine PSDAT call** (matches MATLAB Main_File.m exactly)::

        ic = init_from_powerflow(Vg, theta_g, PG, QG, params)

        Vg      : ndarray (m,) — terminal voltage magnitudes [pu]
        theta_g : ndarray (m,) — terminal voltage angles [rad]
        PG      : ndarray (m,) — active power [pu on system base]
        QG      : ndarray (m,) — reactive power [pu on system base]
        params  : list of MachineParams, length m

    Returns dict with keys:
        'Eqp', 'Si1d', 'Edp', 'Si2q', 'delta', 'omega',
        'Efd', 'RF', 'VR', 'TM', 'PSV', 'Vref', 'PC', 'Id0', 'Iq0'

    Notes
    -----
    The initialization follows PSDAT Main_File.m exactly:

        Vphasor =  VG0.*exp(1i*(THG0));
        Iphasor =  conj((PG+1i*QG)./Vphasor);
        E0      =  Vphasor + (Rs+1i*Xq).*Iphasor;
        D0      =  angle(E0);
        Id0     =  real(Iphasor.*exp(-1i*(D0-pi/2)));
        Iq0     =  imag(Iphasor.*exp(-1i*(D0-pi/2)));
        Edp0    = (Xq-Xqp).*Iq0;
        Si2q0   = (Xls-Xq).*Iq0;
        Eqp0    =  Rs.*Iq0+Xdp.*Id0+V0.*cos(D0-TH0);
        Si1d0   =  Eqp0-(Xdp-Xls).*Id0;
        Efd0    =  Eqp0+(Xd-Xdp).*Id0;
        TM0     = ((Xdpp-Xls)/(Xdp-Xls))*Eqp0*Iq0
                + ((Xdp-Xdpp)/(Xdp-Xls))*Si1d0*Iq0
                + ((Xqpp-Xls)/(Xqp-Xls))*Edp0*Id0
                - ((Xqp-Xqpp)/(Xqp-Xls))*Si2q0*Id0
                + (Xqpp-Xdpp)*Id0*Iq0;
        VR0     = (KE+Ax.*exp(Bx.*Efd0)).*Efd0;
        RF0     = (KF./TF).*Efd0;
        Vref    = V0+VR0./KA;
        PSV0    = TM0;  PC = PSV0;
    """
    # Detect calling convention
    if isinstance(Vg, MachineParams):
        # Legacy single-machine call (not used in PSDAT style but kept for compat)
        p           = Vg
        V_terminal  = theta_g      # positional arg repurposed
        I_terminal  = PG
        S_base_arg  = QG if isinstance(QG, float) else S_base
        ws_i        = p.ws
        Vt          = complex(V_terminal)
        It          = np.conj((p.H,))   # placeholder — not actually used
        # Fall through to legacy path below
        raise TypeError(
            "Legacy single-machine signature not supported. "
            "Use init_from_powerflow(Vg, theta_g, PG, QG, params)."
        )

    # Multi-machine PSDAT path
    Vg_arr      = np.asarray(Vg,      dtype=float)
    theta_g_arr = np.asarray(theta_g, dtype=float)
    PG_arr      = np.asarray(PG,      dtype=float)
    QG_arr      = np.asarray(QG,      dtype=float)
    m = len(params)

    Eqp   = np.zeros(m)
    Si1d  = np.zeros(m)
    Edp   = np.zeros(m)
    Si2q  = np.zeros(m)
    delta = np.zeros(m)
    omega = np.zeros(m)
    Efd   = np.zeros(m)
    RF    = np.zeros(m)
    VR    = np.zeros(m)
    TM    = np.zeros(m)
    PSV   = np.zeros(m)
    Vref  = np.zeros(m)
    PC    = np.zeros(m)
    Id0   = np.zeros(m)
    Iq0   = np.zeros(m)

    for i, p in enumerate(params):
        V  = Vg_arr[i]
        th = theta_g_arr[i]

        # Step 1 — Terminal phasors
        Vphasor = V * np.exp(1j * th)
        Iphasor = np.conj((PG_arr[i] + 1j * QG_arr[i]) / Vphasor)

        # Step 2 — Internal EMF behind Xq to find rotor angle
        E0 = Vphasor + (p.Rs + 1j * p.Xq) * Iphasor
        D0 = np.angle(E0)

        # Step 3 — Park transformation: rotate to dq frame
        rot = np.exp(-1j * (D0 - np.pi / 2.0))
        Id_i = float(np.real(Iphasor * rot))
        Iq_i = float(np.imag(Iphasor * rot))

        # Step 4 — Transient/subtransient initial states
        Edp_i  = (p.Xq  - p.Xqp)  * Iq_i
        Si2q_i = (p.Xls - p.Xq)   * Iq_i   # Xls < Xq → negative

        # Eqp from stator algebraic equation
        Eqp_i  = p.Rs * Iq_i + p.Xdp * Id_i + V * np.cos(D0 - th)
        Si1d_i = Eqp_i - (p.Xdp - p.Xls) * Id_i

        # Step 5 — Field voltage
        Efd_i  = Eqp_i + (p.Xd - p.Xdp) * Id_i

        # Step 6 — Mechanical torque (= electrical torque at steady state)
        TM_i = (
            ((p.Xdpp - p.Xls) / (p.Xdp - p.Xls)) * Eqp_i  * Iq_i
          + ((p.Xdp  - p.Xdpp) / (p.Xdp - p.Xls)) * Si1d_i * Iq_i
          + ((p.Xqpp - p.Xls) / (p.Xqp - p.Xls)) * Edp_i  * Id_i
          - ((p.Xqp  - p.Xqpp) / (p.Xqp - p.Xls)) * Si2q_i * Id_i
          + (p.Xqpp - p.Xdpp) * Id_i * Iq_i
        )

        # Step 7 — Exciter states
        SE_i   = saturation(Efd_i, p.Ax, p.Bx)
        VR_i   = (p.KE + SE_i) * Efd_i
        RF_i   = (p.KF / p.TF) * Efd_i
        Vref_i = V + VR_i / p.KA

        # Step 8 — Governor states
        PSV_i = TM_i
        PC_i  = PSV_i

        omega_i = p.ws   # at steady state omega == ws

        Eqp[i]   = Eqp_i
        Si1d[i]  = Si1d_i
        Edp[i]   = Edp_i
        Si2q[i]  = Si2q_i
        delta[i] = D0
        omega[i] = omega_i
        Efd[i]   = Efd_i
        RF[i]    = RF_i
        VR[i]    = VR_i
        TM[i]    = TM_i
        PSV[i]   = PSV_i
        Vref[i]  = Vref_i
        PC[i]    = PC_i
        Id0[i]   = Id_i
        Iq0[i]   = Iq_i

    return {
        'Eqp':   Eqp,
        'Si1d':  Si1d,
        'Edp':   Edp,
        'Si2q':  Si2q,
        'delta': delta,
        'omega': omega,
        'Efd':   Efd,
        'RF':    RF,
        'VR':    VR,
        'TM':    TM,
        'PSV':   PSV,
        'Vref':  Vref,
        'PC':    PC,
        'Id0':   Id0,
        'Iq0':   Iq0,
    }


# ---------------------------------------------------------------------------
# Differential equations
# ---------------------------------------------------------------------------

def derivatives(
    x_gen: np.ndarray,
    Vg: float,
    theta_g: float,
    Id: float,
    Iq: float,
    params: MachineParams,
    Vref: float,
    PC: float,
) -> np.ndarray:
    """Evaluate the 11 state-variable derivatives for a single generator.

    Parameters
    ----------
    x_gen : ndarray, shape (11,)
        Current state vector for this generator in PSDAT order:
        [Eqp, Si1d, Edp, Si2q, delta, omega, Efd, RF, VR, TM, PSV]
    Vg : float
        Terminal voltage magnitude [pu].
    theta_g : float
        Terminal voltage angle [rad].
    Id : float
        d-axis stator current [pu] (solved from algebraic equations).
    Iq : float
        q-axis stator current [pu] (solved from algebraic equations).
    params : MachineParams
        Machine parameters.
    Vref : float
        Voltage reference setpoint [pu].
    PC : float
        Governor power command [pu].

    Returns
    -------
    dxdt : ndarray, shape (11,)
        [dEqp, dSi1d, dEdp, dSi2q, ddelta, domega, dEfd, dRF, dVR, dTM, dPSV]

    Notes
    -----
    6th-order subtransient model ODEs (Anderson & Fouad, Kundur, PSDAT):

        dEqp/dt  = (Efd - Eqp - (Xd - Xdp)*Id) / Td0p
        dSi1d/dt = (Eqp - (Xdp - Xls)*Id - Si1d) / Td0pp
        dEdp/dt  = (-Edp + (Xq - Xqp)*Iq) / Tq0p
        dSi2q/dt = (-Edp - (Xqp - Xls)*Iq - Si2q) / Tq0pp
        ddelta/dt = omega - ws
        domega/dt = (ws/(2H)) * (TM - Te - Dm*(omega-ws)/ws)

    Electrical torque (subtransient, 6th-order):
        Te = ((Xdpp-Xls)/(Xdp-Xls))*Eqp*Iq  + ((Xdp-Xdpp)/(Xdp-Xls))*Si1d*Iq
           + ((Xqpp-Xls)/(Xqp-Xls))*Edp*Id  - ((Xqp-Xqpp)/(Xqp-Xls))*Si2q*Id
           + (Xqpp - Xdpp)*Id*Iq

    DC1A Exciter:
        dRF/dt  = (-RF + (KF/TF)*Efd) / TF
        dVR/dt  = (-VR + KA*(Vref - Vt - (KF/TF)*Efd + RF)) / TA
        dEfd/dt = (-KE*Efd - SE(Efd)*Efd + VR) / TE
        SE(Efd) = Ax*exp(Bx*|Efd|)

    Steam governor / turbine (two first-order blocks):
        dPSV/dt = (PC - PSV - (omega-ws)/(RD*ws)) / TSV
        dTM/dt  = (PSV - TM) / TCH
    """
    p = params

    # Unpack state (PSDAT order)
    Eqp, Si1d, Edp, Si2q, delta_s, omega_s, Efd_s, RF_s, VR_s, TM_s, PSV_s = x_gen

    # --- Machine flux-decay ODEs ---
    dEqp  = (Efd_s - Eqp - (p.Xd  - p.Xdp)  * Id) / p.Td0p
    dSi1d = (Eqp - (p.Xdp - p.Xls) * Id - Si1d) / p.Td0pp
    dEdp  = (-Edp + (p.Xq  - p.Xqp)  * Iq) / p.Tq0p
    dSi2q = (-Edp - (p.Xqp - p.Xls) * Iq - Si2q) / p.Tq0pp

    # --- Rotor dynamics ---
    ddelta = omega_s - p.ws

    # Electrical torque (subtransient 6th-order expression from PSDAT TM0 formula)
    Te = (
        ((p.Xdpp - p.Xls) / (p.Xdp - p.Xls)) * Eqp  * Iq
      + ((p.Xdp  - p.Xdpp) / (p.Xdp - p.Xls)) * Si1d * Iq
      + ((p.Xqpp - p.Xls) / (p.Xqp - p.Xls)) * Edp  * Id
      - ((p.Xqp  - p.Xqpp) / (p.Xqp - p.Xls)) * Si2q * Id
      + (p.Xqpp - p.Xdpp) * Id * Iq
    )

    domega = (p.ws / (2.0 * p.H)) * (TM_s - Te - p.Dm * (omega_s - p.ws) / p.ws)

    # --- DC1A Exciter ---
    SE   = saturation(Efd_s, p.Ax, p.Bx)
    Vt   = Vg  # terminal voltage magnitude

    dRF  = (-RF_s + (p.KF / p.TF) * Efd_s) / p.TF
    Vin  = Vref - Vt - (p.KF / p.TF) * Efd_s + RF_s
    dVR  = (-VR_s + p.KA * Vin) / p.TA
    dEfd = (-p.KE * Efd_s - SE * Efd_s + VR_s) / p.TE

    # --- Steam governor / turbine ---
    dPSV = (PC - PSV_s - (omega_s - p.ws) / (p.RD * p.ws)) / p.TSV
    dTM  = (PSV_s - TM_s) / p.TCH

    return np.array([
        dEqp, dSi1d, dEdp, dSi2q, ddelta, domega,
        dEfd, dRF, dVR, dTM, dPSV,
    ])


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_state_labels(gen_id: int) -> List[str]:
    """Return list of 11 state variable label strings for generator *gen_id*.

    Parameters
    ----------
    gen_id : int
        1-based generator index (matches MATLAB/PSDAT convention).

    Returns
    -------
    labels : list of str, length 11
    """
    g = str(gen_id)
    return [
        f"Eq'_{g}",
        f"Si1d_{g}",
        f"Ed'_{g}",
        f"Si2q_{g}",
        f"delta_{g}",
        f"omega_{g}",
        f"Efd_{g}",
        f"RF_{g}",
        f"VR_{g}",
        f"TM_{g}",
        f"PSV_{g}",
    ]


def pack_state_vector(ic: Dict[str, np.ndarray]) -> np.ndarray:
    """Pack initial-condition dict into the column-major PSDAT state vector.

    Parameters
    ----------
    ic : dict
        Output of :func:`init_from_powerflow`.

    Returns
    -------
    x0 : ndarray, shape (11*m,)
        Stacked state vector ordered as in PSDAT Main_File.m.
    """
    return np.concatenate([
        ic['Eqp'],
        ic['Si1d'],
        ic['Edp'],
        ic['Si2q'],
        ic['delta'],
        ic['omega'],
        ic['Efd'],
        ic['RF'],
        ic['VR'],
        ic['TM'],
        ic['PSV'],
    ])


def unpack_state_vector(x: np.ndarray, m: int) -> Dict[str, np.ndarray]:
    """Unpack the PSDAT column-major state vector into named arrays.

    Parameters
    ----------
    x : ndarray, shape (11*m,)
        Full state vector.
    m : int
        Number of generators.

    Returns
    -------
    states : dict
        Keys: 'Eqp', 'Si1d', 'Edp', 'Si2q', 'delta', 'omega',
              'Efd', 'RF', 'VR', 'TM', 'PSV'.
    """
    keys = ['Eqp', 'Si1d', 'Edp', 'Si2q', 'delta', 'omega',
            'Efd', 'RF', 'VR', 'TM', 'PSV']
    return {k: x[i * m:(i + 1) * m] for i, k in enumerate(keys)}
