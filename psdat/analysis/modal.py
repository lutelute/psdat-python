"""Modal analysis: eigenvalues, participation factors, mode shapes.

Computes the state matrix A via numerical perturbation (or analytically)
and then decomposes it using eigenvalue analysis.

PSDAT Program 1.4.

References:
    Kundur (1994), Ch. 12.
    Rogers (2000), "Power System Oscillations".
    PSDAT (Abdulrahman 2020), Program 1.4.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ModalResult:
    """Results of modal analysis."""
    eigenvalues: np.ndarray         # complex, shape (n_states,)
    eigenvectors_right: np.ndarray  # complex, shape (n_states, n_states)
    eigenvectors_left: np.ndarray   # complex, shape (n_states, n_states)
    participation_factors: np.ndarray  # real, shape (n_states, n_states)
    frequencies_hz: np.ndarray      # real, shape (n_states,) — Im(lambda)/(2*pi)
    damping_ratios: np.ndarray      # real, shape (n_states,)
    A: np.ndarray                   # state matrix
    state_labels: Optional[List[str]] = None


def compute_state_matrix(
    f_deriv,
    x0: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute state matrix A = df/dx by central finite differences.

    Parameters
    ----------
    f_deriv : callable
        f(x) -> ndarray — right-hand side of dx/dt = f(x).
    x0 : ndarray, shape (n,)
        Equilibrium point.
    eps : float
        Perturbation size.

    Returns
    -------
    A : ndarray, shape (n, n)
    """
    n = len(x0)
    f0 = f_deriv(x0)
    A = np.zeros((n, n))
    for j in range(n):
        xp = x0.copy(); xp[j] += eps
        xm = x0.copy(); xm[j] -= eps
        A[:, j] = (f_deriv(xp) - f_deriv(xm)) / (2 * eps)
    return A


def compute_modal_analysis(
    A: np.ndarray,
    state_labels: Optional[List[str]] = None,
) -> ModalResult:
    """Eigenvalue decomposition and participation factor analysis.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        State matrix.
    state_labels : list of str, optional
        Names of state variables.

    Returns
    -------
    ModalResult
    """
    n = A.shape[0]
    eigenvalues, phi = np.linalg.eig(A)   # right eigenvectors

    # Sort by imaginary part (oscillatory frequency)
    idx = np.argsort(np.abs(eigenvalues.imag))[::-1]
    eigenvalues = eigenvalues[idx]
    phi = phi[:, idx]

    # Left eigenvectors: psi = inv(phi)^T
    try:
        psi = np.linalg.inv(phi).T   # shape (n, n), each column is left eigvec
    except np.linalg.LinAlgError:
        psi = np.conj(phi)   # fallback: use conjugate (approx for normal matrices)

    # Participation factors: P_ki = phi_ki * psi_ik
    # Shape (n_states, n_modes) — column k is participation factors for mode k
    PF = np.zeros((n, n), dtype=complex)
    for k in range(n):
        for i in range(n):
            PF[i, k] = phi[i, k] * psi[i, k]

    # Normalize each mode so max |PF| = 1
    pf_abs = np.abs(PF)
    col_max = pf_abs.max(axis=0)
    col_max[col_max < 1e-12] = 1.0
    PF_norm = pf_abs / col_max[np.newaxis, :]

    # Frequency (Hz) and damping ratio
    sigma = eigenvalues.real
    omega_d = eigenvalues.imag
    freq_hz = np.abs(omega_d) / (2 * np.pi)
    with np.errstate(invalid='ignore', divide='ignore'):
        zeta = -sigma / np.sqrt(sigma**2 + omega_d**2)
        zeta = np.where(np.isfinite(zeta), zeta, np.sign(-sigma))

    return ModalResult(
        eigenvalues=eigenvalues,
        eigenvectors_right=phi,
        eigenvectors_left=psi,
        participation_factors=PF_norm,
        frequencies_hz=freq_hz,
        damping_ratios=zeta.real,
        A=A,
        state_labels=state_labels,
    )


def print_eigenvalue_table(result: ModalResult, n_show: int = 20) -> None:
    """Print eigenvalue table (matches MATLAB tb table format).

    Parameters
    ----------
    result : ModalResult
    n_show : int
        Maximum number of modes to display.
    """
    print(f"\n{'Mode':>4}  {'Real':>10}  {'Imag':>10}  {'Freq(Hz)':>10}  "
          f"{'Zeta':>8}  {'Dominant state'}")
    print("-" * 70)
    n = min(len(result.eigenvalues), n_show)
    for k in range(n):
        lam = result.eigenvalues[k]
        f   = result.frequencies_hz[k]
        z   = result.damping_ratios[k]
        # Find dominant state
        dom_idx = np.argmax(result.participation_factors[:, k])
        dom_label = (result.state_labels[dom_idx]
                     if result.state_labels else str(dom_idx))
        print(f"{k+1:>4}  {lam.real:>10.4f}  {lam.imag:>10.4f}  "
              f"{f:>10.4f}  {z:>8.4f}  {dom_label}")


def identify_electromechanical_modes(
    result: ModalResult,
    delta_indices: List[int],
    freq_range: Tuple[float, float] = (0.1, 2.5),
) -> List[int]:
    """Return mode indices for electromechanical oscillations.

    Electromechanical modes have:
    - Frequency in freq_range (Hz)
    - Significant participation from rotor angle (delta) states

    Parameters
    ----------
    result : ModalResult
    delta_indices : list of int
        State indices corresponding to rotor angle (delta) states.
    freq_range : (f_min, f_max) in Hz

    Returns
    -------
    em_modes : list of int
        Indices into result.eigenvalues that are electromechanical.
    """
    f_min, f_max = freq_range
    em_modes = []
    n_modes = len(result.eigenvalues)
    for k in range(n_modes):
        f = result.frequencies_hz[k]
        lam = result.eigenvalues[k]
        if not (f_min <= f <= f_max):
            continue
        if lam.imag == 0:
            continue
        # Check participation from delta states
        pf_delta = result.participation_factors[delta_indices, k].sum()
        if pf_delta > 0.1:
            em_modes.append(k)
    return em_modes


# ===========================================================================
# Linearization and full modal analysis (MATLAB RunMe9.m equivalent)
# ===========================================================================

def _jacobian_central(fun, x: np.ndarray, eps: float = 1.0e-7) -> np.ndarray:
    """Central-difference Jacobian of fun at x.  Shape (m, n) where m=len(fun(x))."""
    n  = x.size
    f0 = np.asarray(fun(x), dtype=float)
    m  = f0.size
    J  = np.zeros((m, n), dtype=float)
    ei = np.zeros(n, dtype=float)
    for i in range(n):
        ei[i] = eps
        J[:, i] = (np.asarray(fun(x + ei), float) - np.asarray(fun(x - ei), float)) / (2 * eps)
        ei[i] = 0.0
    return J


def linearize(
    dae_system,
    x0: np.ndarray,
    z0: np.ndarray,
    eps: float = 1.0e-7,
) -> np.ndarray:
    """Compute the reduced-order state matrix A via Schur complement.

    A = df/dx - (df/dz) @ inv(dg/dz) @ (dg/dx)

    where f is the differential RHS and g is the algebraic residual.
    All Jacobians use central finite differences with step eps.

    Parameters
    ----------
    dae_system : DAESystem
        Must expose: .rhs(t, x), ._alg.residual(z, dyn, t),
        ._unpack_x(x), ._build_dyn_states(s), ._z_cache.
    x0 : ndarray, shape (nx,)
    z0 : ndarray, shape (nz,)
    eps : float

    Returns
    -------
    A : ndarray, shape (nx, nx)
    """
    import numpy.linalg as la

    states0     = dae_system._unpack_x(x0)
    dyn_states0 = dae_system._build_dyn_states(states0)

    def f_of_x(x):
        return dae_system.rhs(0.0, x)

    def f_of_z(z):
        old = dae_system._z_cache.copy()
        dae_system._z_cache = z.copy()
        dx = dae_system.rhs(0.0, x0)
        dae_system._z_cache = old
        return dx

    def g_of_x(x):
        st = dae_system._unpack_x(x)
        dy = dae_system._build_dyn_states(st)
        return dae_system._alg.residual(z0, dy, 0.0)

    def g_of_z(z):
        return dae_system._alg.residual(z, dyn_states0, 0.0)

    dfdx = _jacobian_central(f_of_x, x0, eps)
    dfdz = _jacobian_central(f_of_z, z0, eps)
    dgdx = _jacobian_central(g_of_x, x0, eps)
    dgdz = _jacobian_central(g_of_z, z0, eps)

    X, _, _, _ = la.lstsq(dgdz, dgdx, rcond=None)
    return dfdx - dfdz @ X


def modal_analysis(
    dae_system,
    x0: np.ndarray,
    z0: np.ndarray,
    state_labels: Optional[List[str]] = None,
    eps: float = 1.0e-7,
) -> ModalResult:
    """Full DAE-aware modal analysis matching MATLAB RunMe9.m.

    Linearizes the DAE, computes eigenvalues, damping ratios,
    frequencies, and normalised participation factors.

    Parameters
    ----------
    dae_system : DAESystem
    x0 : ndarray, shape (nx,)
    z0 : ndarray, shape (nz,)
    state_labels : list of str, length nx, optional
    eps : float

    Returns
    -------
    ModalResult
        Sorted by |Im(eigenvalue)| descending (same as compute_modal_analysis).
    """
    A = linearize(dae_system, x0, z0, eps=eps)
    if state_labels is None:
        m = dae_system.m
        _names = ["delta", "omega", "Eqp", "Si1d", "Edp", "Si2q",
                  "Efd", "VR", "RF", "Vref", "PSV"]
        state_labels = [f"Gen{k+1}_{s}" for k in range(m) for s in _names]
    return compute_modal_analysis(A, state_labels)


def plot_eigenvalues(
    result: ModalResult,
    ax=None,
    show_damping_lines: bool = True,
    damping_levels: tuple = (0.05, 0.10, 0.20),
):
    """Plot eigenvalues in the s-plane.

    Parameters
    ----------
    result : ModalResult
    ax : matplotlib Axes, optional
    show_damping_lines : bool
    damping_levels : tuple of float

    Returns
    -------
    ax
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    lam   = result.eigenvalues
    sigma = np.real(lam)
    nu    = np.imag(lam)

    ax.plot(sigma, nu, "o", markersize=6, color="steelblue", label="Eigenvalues")

    if show_damping_lines:
        nu_max = max(np.abs(nu).max() * 1.1, 1.0)
        nu_line = np.linspace(0, nu_max, 300)
        for zeta_ref in damping_levels:
            if zeta_ref < 1.0:
                s_line = -zeta_ref / np.sqrt(1.0 - zeta_ref**2) * nu_line
                ax.plot(s_line,  nu_line, "k--", linewidth=0.8, alpha=0.5)
                ax.plot(s_line, -nu_line, "k--", linewidth=0.8, alpha=0.5)
                ax.annotate(f"ζ={zeta_ref:.0%}", xy=(s_line[-1], nu_line[-1]),
                            fontsize=7, color="gray")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Real  σ  (1/s)")
    ax.set_ylabel("Imaginary  jω  (rad/s)")
    ax.set_title("Eigenvalue Plot — s-plane")
    ax.legend(fontsize=8)
    return ax


def plot_participation_heatmap(
    result: ModalResult,
    top_modes: int = 10,
    ax=None,
):
    """Plot normalised participation factor heatmap.

    Parameters
    ----------
    result : ModalResult
    top_modes : int
    ax : matplotlib Axes, optional

    Returns
    -------
    ax
    """
    import matplotlib.pyplot as plt

    PF = result.participation_factors
    n_modes_plot = min(top_modes, PF.shape[1])
    PF_plot = PF[:, :n_modes_plot]

    if ax is None:
        _, ax = plt.subplots(
            figsize=(max(6, n_modes_plot * 0.6),
                     max(4, PF_plot.shape[0] * 0.3))
        )

    im = ax.imshow(PF_plot, aspect="auto", cmap="YlOrRd",
                   vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, label="Normalised PF")

    mode_labels = []
    for k in range(n_modes_plot):
        lam_k = result.eigenvalues[k]
        f_k   = result.frequencies_hz[k]
        z_k   = result.damping_ratios[k]
        if f_k > 1e-3:
            mode_labels.append(f"M{k+1}\n{f_k:.2f}Hz\nζ={z_k:.2f}")
        else:
            mode_labels.append(f"M{k+1}\nreal\nζ={z_k:.2f}")

    ax.set_xticks(range(n_modes_plot))
    ax.set_xticklabels(mode_labels, fontsize=7)

    nl = len(result.state_labels) if result.state_labels else PF_plot.shape[0]
    ax.set_yticks(range(nl))
    if result.state_labels:
        ax.set_yticklabels(result.state_labels, fontsize=7)
    ax.set_title(f"Participation Factors (top {n_modes_plot} modes)")
    ax.set_xlabel("Mode")
    ax.set_ylabel("State")
    return ax
