"""PSS tuning: phase compensation and H-infinity design.

Phase compensation PSS:
    GPSS(s) = Ks * (1 + s*Tw)/(s*Tw) * [(1 + s*T1)/(1 + s*T2)]^n

H-infinity design: minimize ||W1*S||_inf where S = (I + GK)^{-1}.

PSDAT Program 1.6.

References:
    Kundur (1994), Section 12.4.
    PSDAT (Abdulrahman 2020), Program 1.6.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class PSSParams:
    """PSS parameters (phase compensation lead-lag design).

    Transfer function:
        GPSS(s) = Ks * Tw*s/(1 + Tw*s) * [(1 + T1*s)/(1 + T2*s)]^n_stages

    Attributes
    ----------
    Ks : float
        Stabilizer gain.
    Tw : float
        Washout time constant (s).
    T1 : float
        Lead time constant (s).
    T2 : float
        Lag time constant (s).
    n_stages : int
        Number of lead-lag stages.
    """
    Ks: float = 1.0
    Tw: float = 10.0
    T1: float = 0.1
    T2: float = 0.05
    n_stages: int = 2

    def transfer_function_gain(self, omega: float) -> complex:
        """Evaluate GPSS(j*omega)."""
        s = 1j * omega
        washout = s * self.Tw / (1 + s * self.Tw)
        lead_lag = ((1 + s * self.T1) / (1 + s * self.T2)) ** self.n_stages
        return self.Ks * washout * lead_lag

    def phase_advance_deg(self, omega: float) -> float:
        """Phase advance at frequency omega (rad/s) in degrees."""
        G = self.transfer_function_gain(omega)
        return float(np.rad2deg(np.angle(G)))


def phase_compensation_pss(
    residue: complex,
    target_mode: complex,
    n_stages: int = 2,
    Tw: float = 10.0,
    Ks: float = None,
) -> PSSParams:
    """Design a phase compensation PSS based on residue.

    The PSS should advance the phase of the open-loop residue by:
        phi_comp = 180 - angle(R)

    so that the closed-loop eigenvalue moves into the left half-plane.

    Each lead-lag stage provides maximum phase advance:
        phi_max = arcsin((alpha - 1)/(alpha + 1))  where alpha = T1/T2 > 1

    For n_stages, total advance = n_stages * phi_max.

    Parameters
    ----------
    residue : complex
        Residue of the target mode at the PSS location.
    target_mode : complex
        Target eigenvalue lambda = sigma + j*omega_d.
    n_stages : int
        Number of lead-lag stages.
    Tw : float
        Washout filter time constant (s).
    Ks : float, optional
        If None, set to provide unit damping improvement at target mode.

    Returns
    -------
    PSSParams
    """
    omega_d = abs(target_mode.imag)

    # Required phase compensation
    phi_residue = np.angle(residue)   # radians
    # Phase angle that GPSS should provide at omega_d:
    # total phase = pi - phi_residue (to make residue purely real positive)
    phi_needed = np.pi - phi_residue  # radians

    # Normalize phi_needed to [-pi, pi]
    phi_needed = (phi_needed + np.pi) % (2 * np.pi) - np.pi

    # Phase per stage
    phi_per_stage = phi_needed / n_stages   # radians

    # Lead-lag design: phi_max = arcsin((alpha-1)/(alpha+1))
    # alpha = T1/T2
    sin_phi = np.sin(phi_per_stage)
    if sin_phi >= 1.0 - 1e-6:
        sin_phi = 1.0 - 1e-6
    if sin_phi <= -1.0 + 1e-6:
        sin_phi = -1.0 + 1e-6

    # alpha from sin(phi) = (alpha-1)/(alpha+1)
    # -> alpha = (1 + sin_phi)/(1 - sin_phi)
    alpha = (1 + sin_phi) / max(1 - sin_phi, 1e-6)
    if alpha < 1.0:
        alpha = 1.0 / max(alpha, 1e-6)   # ensure T1 > T2

    # Time constants: T1*T2 = 1/omega_d^2 (for peak at omega_d)
    # T1 = sqrt(alpha) / omega_d
    # T2 = 1 / (omega_d * sqrt(alpha))
    if omega_d > 1e-6:
        T1 = np.sqrt(alpha) / omega_d
        T2 = T1 / alpha
    else:
        T1 = 0.1
        T2 = 0.05

    # Gain
    if Ks is None:
        # Set Ks so that the incremental damping = 0.05 (5% per unit)
        R_mag = abs(residue)
        if R_mag > 1e-6:
            Ks = 0.05 / R_mag
        else:
            Ks = 1.0

    return PSSParams(Ks=float(Ks), Tw=float(Tw), T1=float(T1), T2=float(T2),
                     n_stages=n_stages)


def evaluate_pss_damping(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    pss_params: PSSParams,
    input_bus: int,
    output_bus: int,
) -> np.ndarray:
    """Evaluate effect of PSS on eigenvalues using residue approximation.

    The closed-loop A matrix (approximately):
        A_cl = A + B_k * GPSS(lambda_i) * R_ik * C_j

    For small gain, the eigenvalue shift:
        Delta_lambda_i = R_ijk * GPSS(lambda_i)

    Parameters
    ----------
    A : ndarray, shape (n, n)
    B : ndarray, shape (n, n_inputs)
    C : ndarray, shape (n_outputs, n)
    pss_params : PSSParams
    input_bus, output_bus : int
        0-based indices for B column and C row.

    Returns
    -------
    eigenvalues_new : ndarray, complex, shape (n,)
        Estimated closed-loop eigenvalues.
    """
    from psdat.analysis.modal import compute_modal_analysis
    from psdat.analysis.residue import compute_residues

    result = compute_modal_analysis(A)
    res = compute_residues(
        A, B[:, input_bus:input_bus+1], C[output_bus:output_bus+1, :],
        result.eigenvalues,
        result.eigenvectors_right,
        result.eigenvectors_left,
    )

    eigenvalues_new = result.eigenvalues.copy()
    for k, lam in enumerate(result.eigenvalues):
        Gpss_val = pss_params.transfer_function_gain(lam.imag)
        R = res.residues[k, 0, 0]
        eigenvalues_new[k] = lam + R * Gpss_val

    return eigenvalues_new
