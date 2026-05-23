"""Residue analysis for PSS design and optimal placement.

For each input-output pair (u_k, y_j), the residue of mode lambda_i is:
    R_ij = C_j @ phi_i * psi_i^H @ B_k

where phi_i = right eigenvector, psi_i = left eigenvector.

The optimal PSS location maximizes |R| for the target mode.

References:
    Kundur (1994), Section 12.3.
    PSDAT Program 1.5 (Abdulrahman 2020).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ResidueResult:
    """Results of residue analysis.

    Attributes
    ----------
    residues : ndarray, complex, shape (n_modes, n_outputs, n_inputs)
        Transfer function residues.
    mode_indices : ndarray, int, shape (n_modes,)
        Indices of selected modes (into eigenvalues array).
    eigenvalues : ndarray, complex, shape (n_modes,)
        Eigenvalues for the selected modes.
    optimal_location : int
        Input index (0-based) with maximum |R| for the most lightly damped
        electromechanical mode. This identifies the best PSS location.
    optimal_residue_mag : float
        Magnitude of residue at optimal location.
    """
    residues: np.ndarray
    mode_indices: np.ndarray
    eigenvalues: np.ndarray
    optimal_location: int
    optimal_residue_mag: float


def compute_residues(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors_right: np.ndarray,
    eigenvectors_left: np.ndarray,
) -> ResidueResult:
    """Compute transfer function residues for all modes.

    The residue of mode i for transfer function G_jk(s) = C_j (sI - A)^{-1} B_k:
        R_ijk = (C @ phi_i) * (psi_i^H @ B)_k

    Parameters
    ----------
    A : ndarray, shape (n, n)
        State matrix.
    B : ndarray, shape (n, n_inputs)
        Input matrix.
    C : ndarray, shape (n_outputs, n)
        Output matrix.
    eigenvalues : ndarray, complex, shape (n,)
    eigenvectors_right : ndarray, complex, shape (n, n)
        Columns are right eigenvectors phi_i.
    eigenvectors_left : ndarray, complex, shape (n, n)
        Columns are left eigenvectors psi_i.

    Returns
    -------
    ResidueResult
    """
    n = A.shape[0]
    n_inputs  = B.shape[1] if B.ndim > 1 else 1
    n_outputs = C.shape[0] if C.ndim > 1 else 1

    if B.ndim == 1:
        B = B[:, np.newaxis]
    if C.ndim == 1:
        C = C[np.newaxis, :]

    n_modes = n
    residues = np.zeros((n_modes, n_outputs, n_inputs), dtype=complex)

    for i in range(n_modes):
        phi = eigenvectors_right[:, i]      # right eigenvector, shape (n,)
        psi = eigenvectors_left[:, i]       # left eigenvector,  shape (n,)
        # Normalization: psi^H @ phi = 1
        norm = np.dot(np.conj(psi), phi)
        if abs(norm) > 1e-12:
            psi = psi / norm

        # C @ phi_i  -> shape (n_outputs,)
        C_phi = C @ phi
        # psi_i^H @ B -> shape (n_inputs,)
        psi_B = np.conj(psi) @ B

        # Outer product
        residues[i] = np.outer(C_phi, psi_B)

    # Find optimal location: mode with lowest damping ratio, largest |R|
    sigma  = eigenvalues.real
    omega_d = eigenvalues.imag
    with np.errstate(invalid='ignore', divide='ignore'):
        zeta = -sigma / np.sqrt(sigma**2 + omega_d**2 + 1e-30)

    # Only consider oscillatory modes with positive frequency in [0.1, 2] Hz
    freq_hz = np.abs(omega_d) / (2 * np.pi)
    em_mask = (freq_hz >= 0.1) & (freq_hz <= 2.0) & (omega_d > 0)

    if em_mask.any():
        # Most lightly damped electromechanical mode
        em_zeta = np.where(em_mask, zeta, np.inf)
        target_mode = int(np.argmin(em_zeta))
    else:
        # Fall back to most oscillatory
        target_mode = int(np.argmax(np.abs(omega_d)))

    # For target mode, find input (PSS location) with max |R|
    # Sum over outputs
    R_target = np.abs(residues[target_mode]).sum(axis=0)   # shape (n_inputs,)
    opt_loc = int(np.argmax(R_target))
    opt_mag = float(R_target[opt_loc])

    return ResidueResult(
        residues=residues,
        mode_indices=np.arange(n_modes),
        eigenvalues=eigenvalues,
        optimal_location=opt_loc,
        optimal_residue_mag=opt_mag,
    )


def optimal_pss_location(
    residue_result: ResidueResult,
    target_freq_hz_range: Tuple[float, float] = (0.1, 2.0),
) -> Tuple[int, float]:
    """Find optimal PSS bus from residue analysis.

    Returns the input index with maximum residue magnitude for the most
    lightly damped mode in target_freq_hz_range.

    Parameters
    ----------
    residue_result : ResidueResult
    target_freq_hz_range : (f_min, f_max) Hz

    Returns
    -------
    (bus_idx, residue_magnitude) : (int, float)
        0-based input index of optimal PSS location and the residue magnitude.
    """
    eigenvalues = residue_result.eigenvalues
    omega_d = eigenvalues.imag
    sigma   = eigenvalues.real
    freq_hz = np.abs(omega_d) / (2 * np.pi)
    f_min, f_max = target_freq_hz_range

    em_mask = (freq_hz >= f_min) & (freq_hz <= f_max) & (omega_d > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        zeta = -sigma / np.sqrt(sigma**2 + omega_d**2 + 1e-30)

    if em_mask.any():
        em_zeta = np.where(em_mask, zeta, np.inf)
        target_mode = int(np.argmin(em_zeta))
    else:
        target_mode = int(np.argmax(np.abs(omega_d)))

    R_target = np.abs(residue_result.residues[target_mode]).sum(axis=0)
    opt_loc = int(np.argmax(R_target))
    opt_mag = float(R_target[opt_loc])
    return opt_loc, opt_mag


def plot_residue_compass(
    result: ResidueResult,
    mode_idx: int,
    output_idx: int = 0,
    ax=None,
    labels: Optional[List[str]] = None,
) -> None:
    """Compass plot of residues for a given mode (matches MATLAB compass()).

    Parameters
    ----------
    result : ResidueResult
    mode_idx : int
        Mode to plot.
    output_idx : int
        Output index to plot.
    ax : matplotlib Axes, optional
        Polar axes to plot on.
    labels : list of str, optional
        Input labels.
    """
    import matplotlib.pyplot as plt

    R = result.residues[mode_idx, output_idx, :]  # shape (n_inputs,)
    n_inputs = len(R)

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    lam = result.eigenvalues[mode_idx]
    f_hz = abs(lam.imag) / (2 * np.pi)
    sigma = lam.real
    with np.errstate(invalid='ignore'):
        zeta = -sigma / abs(lam) if abs(lam) > 0 else 0.0

    ax.set_title(
        f"Residues — Mode {mode_idx+1}\n"
        f"λ={lam.real:.3f}+j{lam.imag:.3f}  f={f_hz:.3f}Hz  ζ={zeta:.3f}",
        fontsize=9,
    )

    angles = np.angle(R)
    mags   = np.abs(R)
    colors = plt.cm.tab10(np.linspace(0, 1, n_inputs))

    for k in range(n_inputs):
        ax.annotate(
            "",
            xy=(angles[k], mags[k]),
            xytext=(angles[k], 0),
            arrowprops=dict(arrowstyle="->", color=colors[k], lw=1.5),
        )
        label = labels[k] if labels else f"Input {k+1}"
        ax.text(
            angles[k], mags[k] * 1.1, label,
            ha="center", va="center", fontsize=7, color=colors[k],
        )

    ax.set_rlabel_position(90)
