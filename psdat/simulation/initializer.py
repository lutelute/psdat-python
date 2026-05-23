"""Compute all initial conditions from a power-flow solution.

This module implements the MATLAB initialization sequence from PSDAT
(RunMe*.m + IC.m in the Abdulrahman 2020 toolbox) for the 6th-order
synchronous machine model with IEEE Type-1 exciter and steam governor.

State-variable ordering (11 states per generator, matching PSDAT):
    x[k*11 + 0]  = delta   (rotor angle, rad)
    x[k*11 + 1]  = omega   (rotor speed deviation, rad/s)
    x[k*11 + 2]  = Eqp     (q-axis transient EMF)
    x[k*11 + 3]  = Si1d    (d-axis subtransient flux linkage)
    x[k*11 + 4]  = Edp     (d-axis transient EMF)
    x[k*11 + 5]  = Si2q    (q-axis subtransient flux linkage)
    x[k*11 + 6]  = Efd     (field voltage)
    x[k*11 + 7]  = VR      (exciter voltage regulator output)
    x[k*11 + 8]  = RF      (exciter rate feedback signal)
    x[k*11 + 9]  = Vref    (AVR voltage reference — constant, stored as IC)
    x[k*11 + 10] = PSV     (governor steam valve position)

Algebraic variable ordering (2m + 2n):
    z[0:m]        = Id   (d-axis stator current per generator)
    z[m:2m]       = Iq   (q-axis stator current per generator)
    z[2m:2m+n]    = V    (bus voltage magnitudes)
    z[2m+n:2m+2n] = TH   (bus voltage angles)

References
----------
Abdulrahman (2020) PSDAT Toolbox, IC.m and RunMe1.m.
Anderson & Fouad (2003) Power System Control and Stability, Ch. 4.
Kundur (1994) Power System Stability and Control, Ch. 6.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_initial_conditions(
    pf_result: dict,
    machine_params_list: list[dict],
    ws: float = 2.0 * np.pi * 60.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute DAE initial conditions from a power-flow solution.

    Implements the MATLAB initialization sequence from PSDAT's IC.m exactly.

    Parameters
    ----------
    pf_result : dict
        Output of ``run_powerflow``.  Required keys:
            'V'     : np.ndarray (n,) — bus voltage magnitudes (p.u.)
            'TH'    : np.ndarray (n,) — bus voltage angles (rad)
            'Pg'    : np.ndarray (m,) — generator active power output (p.u.)
            'Qg'    : np.ndarray (m,) — generator reactive power output (p.u.)
            'Pload' : np.ndarray (n,) — bus active load (p.u.)
            'Qload' : np.ndarray (n,) — bus reactive load (p.u.)
        The first *m* buses are generator buses (same ordering as
        ``machine_params_list``).

    machine_params_list : list of dict, length m
        One dict per generator.  Required keys:
            Rs, Xd, Xdp, Xdpp, Xq, Xqp, Xqpp, Xls,
            H, D, KA, TA, KE, TE, KF, TF, Ax, Bx, KG, TG.
        Optional (with defaults):
            PC : None — governor load reference (default = TM0).

    ws : float
        Synchronous angular frequency (rad/s).  Default = 2π·60.

    Returns
    -------
    x0 : np.ndarray, shape (11*m,)
        Flattened initial state vector for all generators.
    z0 : np.ndarray, shape (2m + 2n,)
        Initial algebraic variable vector.
    aux : dict
        Per-generator auxiliary quantities:
            'Vref' : np.ndarray (m,) — AVR voltage reference
            'PC'   : np.ndarray (m,) — governor load reference
            'Efd0' : np.ndarray (m,) — initial field voltage
            'TM0'  : np.ndarray (m,) — initial mechanical torque
    """
    V   = np.asarray(pf_result["V"],  dtype=float)
    TH  = np.asarray(pf_result["TH"], dtype=float)
    Pg  = np.asarray(pf_result["Pg"], dtype=float)
    Qg  = np.asarray(pf_result["Qg"], dtype=float)

    n   = len(V)
    m   = len(machine_params_list)

    if len(Pg) != m or len(Qg) != m:
        raise ValueError("Pg and Qg must have length m = len(machine_params_list)")

    # Generator terminal voltages/angles from power flow.
    Vg   = V[0:m]    # (m,) — magnitudes
    THg  = TH[0:m]   # (m,) — angles

    # Generator terminal phasor and current phasor in network reference.
    Vg_phasor = Vg * np.exp(1j * THg)                     # (m,)
    Sg        = Pg + 1j * Qg                               # (m,) apparent power
    Ig_phasor = np.conj(Sg / Vg_phasor)                   # (m,) current phasor

    # ----------------------------------------------------------------
    # Step 1  — Internal EMF and rotor angle
    # ----------------------------------------------------------------
    # E0 = Vphasor + (Rs + j*Xq) * Iphasor
    # delta0 = angle(E0)
    # MATLAB: E0 = Vg + (Rs + j*Xq)*Ig;  delta0 = angle(E0);

    Rs = np.array([p["Rs"] for p in machine_params_list], dtype=float)
    Xq = np.array([p["Xq"] for p in machine_params_list], dtype=float)
    Xd = np.array([p["Xd"] for p in machine_params_list], dtype=float)

    Z_q        = Rs + 1j * Xq              # (m,)
    E0         = Vg_phasor + Z_q * Ig_phasor  # (m,) internal voltage in q-axis
    delta0     = np.angle(E0)              # (m,) rotor angle (rad)

    # ----------------------------------------------------------------
    # Step 2  — d-q axis currents
    # ----------------------------------------------------------------
    # Park transformation: rotate current phasor by -(delta - π/2)
    # Id0 + jIq0 = Ig * exp(-j*(delta - π/2))
    # MATLAB: rot = exp(-j*(delta - pi/2));
    #         Id0 = real(Ig .* rot);   Iq0 = imag(Ig .* rot);

    rot  = np.exp(-1j * (delta0 - np.pi / 2.0))  # (m,)
    Idq0 = Ig_phasor * rot                         # (m,) complex
    Id0  = np.real(Idq0)                           # (m,)
    Iq0  = np.imag(Idq0)                           # (m,)

    # ----------------------------------------------------------------
    # Step 3  — Transient and subtransient flux linkage initial values
    # ----------------------------------------------------------------
    Xqp  = np.array([p["Xqp"]  for p in machine_params_list], dtype=float)
    Xdp  = np.array([p["Xdp"]  for p in machine_params_list], dtype=float)
    Xqpp = np.array([p["Xqpp"] for p in machine_params_list], dtype=float)
    Xdpp = np.array([p["Xdpp"] for p in machine_params_list], dtype=float)
    Xls  = np.array([p["Xls"]  for p in machine_params_list], dtype=float)

    # d-axis transient EMF (q-axis oriented)
    # MATLAB: Edp0 = (Xq - Xqp)*Iq0;
    Edp0  = (Xq - Xqp) * Iq0   # (m,)

    # q-axis subtransient flux linkage
    # MATLAB: Si2q0 = (Xls - Xq)*Iq0;
    Si2q0 = (Xls - Xq) * Iq0   # (m,)

    # q-axis transient EMF (d-axis oriented)
    # From stator SE2 at steady state (omega=1, d/dt=0):
    #   0 = Rs*Iq0 + Xdp*Id0 - ((Xdpp-Xls)/(Xdp-Xls))*Eqp0
    #       - ((Xdp-Xdpp)/(Xdp-Xls))*Si1d0 + Vg*cos(delta-theta)
    # At init Si1d0 = Eqp0 - (Xdp-Xls)*Id0 (from subtransient equation at SS),
    # so the expression simplifies to:
    #   Eqp0 = Rs*Iq0 + Xdp*Id0 + Vg*cos(delta-theta)
    # MATLAB: Eqp0 = Rs*Iq0 + Xdp*Id0 + Vg*cos(delta0-theta_g);
    Eqp0  = Rs * Iq0 + Xdp * Id0 + Vg * np.cos(delta0 - THg)   # (m,)

    # d-axis subtransient flux linkage
    # MATLAB: Si1d0 = Eqp0 - (Xdp-Xls)*Id0;
    Si1d0 = Eqp0 - (Xdp - Xls) * Id0   # (m,)

    # ----------------------------------------------------------------
    # Step 4  — Field voltage
    # ----------------------------------------------------------------
    # MATLAB: Efd0 = Eqp0 + (Xd-Xdp)*Id0;
    Efd0 = Eqp0 + (Xd - Xdp) * Id0   # (m,)

    # ----------------------------------------------------------------
    # Step 5  — Mechanical torque (subtransient air-gap torque)
    # ----------------------------------------------------------------
    # Full formula (6th-order subtransient model):
    #   TM0 = Eqp*Iq + Edp*Id + (Xdpp-Xqpp)*Id*Iq
    #         - (Xdpp-Xls)/(Xdp-Xls)*Si1d*Iq
    #         + (Xqpp-Xls)/(Xqp-Xls)*Si2q*Id
    # MATLAB: TM0 = Eqp0.*Iq0 + Edp0.*Id0 + (Xdpp-Xqpp).*Id0.*Iq0
    #              - (Xdpp-Xls)./(Xdp-Xls).*Si1d0.*Iq0
    #              + (Xqpp-Xls)./(Xqp-Xls).*Si2q0.*Id0;
    TM0 = (
        Eqp0 * Iq0
        + Edp0 * Id0
        + (Xdpp - Xqpp) * Id0 * Iq0
        - (Xdpp - Xls) / (Xdp - Xls) * Si1d0 * Iq0
        + (Xqpp - Xls) / (Xqp - Xls) * Si2q0 * Id0
    )   # (m,)

    # ----------------------------------------------------------------
    # Step 6  — IEEE Type-1 Exciter initial values
    # ----------------------------------------------------------------
    KA = np.array([p["KA"] for p in machine_params_list], dtype=float)
    KE = np.array([p["KE"] for p in machine_params_list], dtype=float)
    KF = np.array([p["KF"] for p in machine_params_list], dtype=float)
    TF = np.array([p["TF"] for p in machine_params_list], dtype=float)
    Ax = np.array([p["Ax"] for p in machine_params_list], dtype=float)
    Bx = np.array([p["Bx"] for p in machine_params_list], dtype=float)

    # Exciter saturation: SE_sat = Ax * exp(Bx * Efd0)
    SE_sat = Ax * np.exp(Bx * Efd0)   # (m,)

    # VR0 = (KE + SE_sat) * Efd0
    # MATLAB: VR0 = (KE + Ax.*exp(Bx.*Efd0)).*Efd0;
    VR0 = (KE + SE_sat) * Efd0   # (m,)

    # Rate feedback: RF0 = (KF/TF)*Efd0
    # MATLAB: RF0 = (KF./TF).*Efd0;
    RF0 = (KF / TF) * Efd0   # (m,)

    # AVR voltage reference: Vref = |Vg| + VR0/KA
    # MATLAB: Vref = abs(Vg) + VR0./KA;
    Vref = Vg + VR0 / KA   # (m,)

    # ----------------------------------------------------------------
    # Step 7  — Governor initial values
    # ----------------------------------------------------------------
    # PSV0 = PC = TM0  (valve at steady state equals mechanical torque)
    # MATLAB: PSV0 = PC = TM0;
    PC0  = TM0.copy()   # governor load reference = initial mech torque
    PSV0 = TM0.copy()   # steam valve position

    # ----------------------------------------------------------------
    # Assemble state vector x0  (11 states × m generators, row-major)
    # ----------------------------------------------------------------
    # omega0 = 0  (speed deviation zero at steady state)
    omega0 = np.zeros(m, dtype=float)

    x0 = np.empty(11 * m, dtype=float)
    for k in range(m):
        base = k * 11
        x0[base + 0]  = delta0[k]
        x0[base + 1]  = omega0[k]
        x0[base + 2]  = Eqp0[k]
        x0[base + 3]  = Si1d0[k]
        x0[base + 4]  = Edp0[k]
        x0[base + 5]  = Si2q0[k]
        x0[base + 6]  = Efd0[k]
        x0[base + 7]  = VR0[k]
        x0[base + 8]  = RF0[k]
        x0[base + 9]  = Vref[k]   # stored as a "state" for convenience
        x0[base + 10] = PSV0[k]

    # ----------------------------------------------------------------
    # Assemble algebraic variable vector z0  (2m + 2n)
    # z = [Id(m), Iq(m), V(n), TH(n)]
    # ----------------------------------------------------------------
    z0 = np.empty(2 * m + 2 * n, dtype=float)
    z0[0:m]           = Id0
    z0[m:2*m]         = Iq0
    z0[2*m:2*m+n]     = V
    z0[2*m+n:2*m+2*n] = TH

    # ----------------------------------------------------------------
    # Auxiliary outputs
    # ----------------------------------------------------------------
    aux = {
        "Vref": Vref,
        "PC":   PC0,
        "Efd0": Efd0,
        "TM0":  TM0,
        "Id0":  Id0,
        "Iq0":  Iq0,
        "delta0": delta0,
    }

    return x0, z0, aux
