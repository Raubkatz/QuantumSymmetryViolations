#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analytic_tomography.py
======================

Single-qubit **state** and **process** tomography from first principles,
implemented with NumPy/SciPy only. Gates are built from their true
Lie-algebra generators via :func:`scipy.linalg.expm` and can carry an
**additive generator noise** term (inside the exponent) for every gate.
Optionally, a **post-circuit CPTP channel** can be applied, composed of
amplitude damping (probability p) and/or phase damping (dephasing rate λ).
Both are OFF by default (None), preserving legacy behavior.

What this module provides (big picture)
---------------------------------------
• A compact single-qubit gate/circuit layer that constructs ideal gates from
  the usual generators (σx, σy, σz) and optionally perturbs each gate by an
  additive 2×2 complex matrix Δ_g *inside the exponential*:
      U_g(θ) = exp[-i * (G_g * θ/2  +  Δ_g)]
  where G_g is the Hermitian generator for the gate (e.g., σx for Rx).
  This formulation lets you model coherent control errors and small
  non-Hermiticity to approximate certain dissipative effects in the unitary
  surrogate used to induce a channel.

• Two complementary routes to obtain a process matrix (PTM, 4×4 real):
  (1) **Heisenberg-exact**: a closed-form, sampling-free mapping from a
      2×2 matrix U to a 4×4 real Pauli-Liouville PTM R. This is analytic and
      deterministic; good for optimization when you want noiseless feedback.
  (2) **Affine (shots-based) tomography**: reconstruct R from a minimal
      prepare-and-measure schedule using finite shots to mimic hardware
      statistics. This supports optional constraints like enforcing
      trace-preservation (TP) and optional post-circuit CPTP noise.

• A consistent Pauli basis convention {I, X, Y, Z} with explicit helpers for
  state preparation and Pauli measurements, so every PTM we build is in the
  same ordering and comparable across methods.

How to read the code below
--------------------------
• Section 0 defines Pauli primitives, a Pauli basis list (order I,X,Y,Z),
  and canonical preparation kets used by the tomography schedule.
• Later sections (not shown in this snippet) typically include:
  - Gate generators and a small circuit class that composes gates with
    optional per-gate Δ noise,
  - Exact PTM mapping utilities (Heisenberg picture),
  - An affine tomography routine that runs a minimal schedule with finite
    shots (binomial/multinomial sampling) and can optionally enforce TP,
  - Optional post-circuit amplitude/phase damping channels applied before
    readout in both exact and shot-based pipelines when configured.

Switches/knobs you’ll encounter elsewhere in this project
---------------------------------------------------------
• `ASSUME_TP_AFFINE` (bool): if True, the affine estimator enforces
  trace-preservation when inverting the schedule; if False, it returns the
  unconstrained least-squares solution. Turning it ON reduces variance but
  may bias results if the underlying effective map is non-TP.
• `FIT_WITH_SHOTS` (bool in calling script): whether the *optimizer* scores
  candidates using exact PTMs (False) or shot-estimated PTMs (True).
• `SHOTS_PER_EVAL` (int): shots per setting for the affine estimator.
• Post-circuit damping parameters (if present in the Circuit class):
  amplitude damping probability `p` and phase damping rate `lam`.
  Setting them to None disables the channel (legacy behavior).

Shapes/conventions
------------------
• Single-qubit state-vectors are 2×1 complex; density matrices are 2×2 complex.
• Single-qubit gates/matrices (including noisy surrogates) are 2×2 complex.
• PTMs act on the Pauli basis {I, X, Y, Z} and are 4×4 real arrays:
    R_ij = (1/2) Tr[ P_i  E(P_j) ],  with P_0=I, P_1=X, P_2=Y, P_3=Z.
• Preparation labels map to normalized kets shown below.

Dependencies
------------
NumPy for linear algebra and SciPy for the matrix exponential. No quantum
framework is required; everything is explicit linear algebra to stay
transparent for research/debugging.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# SciPy for matrix exponential (used to build gates from generators)
try:
    import scipy.linalg as la
except Exception as exc:  # pragma: no cover
    raise ImportError("This module requires SciPy. Please `pip install scipy`.") from exc

# ──────────────────────────────────────────────────────────────
# 0) Pauli primitives, basis, and small helpers
#    -----------------------------------------------------------
#    • σx, σy, σz are the standard Pauli matrices.
#    • I2 is the 2×2 identity.
#    • PAULIS list defines the global basis order used for PTMs: {I, X, Y, Z}.
#    • _PREP provides canonical pure states for tomography preparations.
# ──────────────────────────────────────────────────────────────

σx = np.array([[0, 1], [1, 0]], complex)      # Pauli X
σy = np.array([[0, -1j], [1j,  0]], complex)   # Pauli Y
σz = np.array([[1,  0], [0, -1]], complex)     # Pauli Z
I2  = np.eye(2, dtype=complex)                 # Identity on qubit space

PAULIS: List[np.ndarray] = [I2, σx, σy, σz]    # Pauli basis ordering: I, X, Y, Z

# Canonical preparation kets (unit-norm). These are used by the affine schedule:
#  • Z± are computational basis |0>, |1>.
#  • X± are ±1 eigenstates of X.
#  • Y± are ±1 eigenstates of Y.
_PREP: Dict[str, np.ndarray] = {
    "Z+": np.array([1, 0], complex),                  # |0>
    "Z-": np.array([0, 1], complex),                  # |1>
    "X+": 1/np.sqrt(2) * np.array([1,  1], complex),  # |+>  (X eigenvalue +1)
    "X-": 1/np.sqrt(2) * np.array([1, -1], complex),  # |->  (X eigenvalue −1)
    "Y+": 1/np.sqrt(2) * np.array([1,  1j], complex), # |+i> (Y eigenvalue +1)
    "Y-": 1/np.sqrt(2) * np.array([1, -1j], complex), # |-i> (Y eigenvalue −1)
}

# Notes on numerical stability and conventions:
# • We keep arrays explicitly complex to avoid accidental dtype upcasts/downs.
# • All Pauli eigenkets above are normalized; density matrices ρ = |ψ⟩⟨ψ|.
# • When building PTMs elsewhere in this module, the PAULIS order here is the
#   source of truth for row/column indexing so that all R[*,*] align consistently.

def _proj_plus(P: np.ndarray) -> np.ndarray:
    """
    Compute the projector onto the +1 eigenspace of a Pauli observable.

    Context
    -------
    In single-qubit tomography, measuring a Pauli operator P ∈ {σx, σy, σz}
    yields outcomes in {+1, −1}. The probability of getting +1 on state ρ is
        p_plus = Tr[ Π_+ ρ ],  where  Π_+ = (I + P)/2 .
    This helper returns that projector Π_+ for a provided Pauli matrix P.

    Parameters
    ----------
    P : np.ndarray, shape (2,2), complex
        A single-qubit Pauli matrix (σx, σy, σz). The function does not assert
        that P is exactly one of those, but correctness assumes P^2 = I and
        eigenvalues in {+1, −1}.

    Returns
    -------
    np.ndarray, shape (2,2), complex
        The projector Π_+ = (I + P)/2 onto the +1 eigenspace of P.

    Notes
    -----
    • If you pass I (identity), the result is I (since all states are +1 eigenstates).
    • For non-Pauli inputs that still satisfy P^2=I, the same algebra holds.
    • This projector is used by the shots-based estimator to turn theoretical
      +1 probabilities into Binomial draws.
    """
    # Formula: Π_+ = (I + P) / 2. I2 is the global 2×2 identity defined elsewhere.
    return 0.5 * (I2 + P)


def _binomial_expectation(p_plus: float, shots: int, rng: np.random.Generator) -> float:
    """
    Convert a +1 probability into a single empirical Pauli expectation ⟨P⟩ via sampling.

    Purpose
    -------
    Given the theoretical probability p_plus = Pr(outcome=+1) for measuring
    some Pauli observable P on a state ρ, this routine draws Binomial samples
    with 'shots' trials and converts the resulting counts into the empirical
    expectation value:
        ⟨P⟩_empirical = (n_plus - n_minus) / shots = 2 * (n_plus/shots) - 1.

    Parameters
    ----------
    p_plus : float
        The ideal probability of a +1 outcome (will be clamped into [0, 1]).
    shots : int
        Number of measurement repetitions. If <= 0, we *skip sampling* and
        return the noiseless expectation 2*p_plus - 1.
    rng : np.random.Generator
        NumPy PRNG used for the Binomial draw (reproducible when seeded).

    Returns
    -------
    float
        The sampled (or deterministic if shots<=0) expectation value in [-1, +1].

    Edge cases & behavior
    ---------------------
    • p_plus is clipped into [0,1] to avoid numerical issues.
    • shots <= 0 → returns exact mean 2*p_plus - 1 (useful for debugging).
    • shots > 0 → draws n_plus ~ Binomial(shots, p_plus) and returns
      (n_plus - (shots - n_plus))/shots.

    Why this matters
    ----------------
    The affine (shots-based) PTM reconstructor needs actual expectation values,
    not probabilities. This helper bridges the two, respecting finite-sample
    noise consistent with hardware-like counts.
    """
    # Clamp to [0, 1] to avoid invalid probabilities due to floating error.
    p_plus = float(min(1.0, max(0.0, p_plus)))

    # If no shots are requested, return the noiseless expectation (debug path).
    if shots <= 0:
        return 2.0 * p_plus - 1.0

    # Draw +1 counts from a Binomial and convert counts -> expectation in [-1, +1].
    n_plus = rng.binomial(shots, p_plus)
    return (n_plus - (shots - n_plus)) / shots

# ──────────────────────────────────────────────────────────────
# 0.5) CPTP channels (Kraus) – optional post-circuit layer
#     These helpers construct standard single-qubit noise channels in Kraus form
#     and provide a simple sequential composition operator. They are used to
#     model *post-circuit* noise in both the Heisenberg-exact and the
#     shots-based pipelines when enabled by the calling code.
# ──────────────────────────────────────────────────────────────

def kraus_amplitude_damping(p: float) -> List[np.ndarray]:
    """
    Return the Kraus operators for single-qubit amplitude damping with probability p.

    Physical meaning
    ----------------
    Amplitude damping models energy relaxation |1⟩ → |0⟩ with probability p.
    In Kraus form, a CPTP map E(ρ) = Σ_k K_k ρ K_k† with:
        K0 = [[1, 0], [0, sqrt(1-p)]],
        K1 = [[0, sqrt(p)], [0, 0]].

    Parameters
    ----------
    p : float
        Relaxation probability. Values are clamped to [0, 1] for robustness.

    Returns
    -------
    List[np.ndarray]
        [K0, K1] each of shape (2,2), complex dtype.

    Usage notes
    -----------
    • p=0 → identity channel (no relaxation).
    • p=1 → full decay to |0⟩ (all |1⟩ population vanishes after the channel).
    • Compose with a unitary U as: ρ' = Σ_k K_k (U ρ U†) K_k†.
    • In PTM-land, you can either (a) convert Kraus to a superoperator and
      left-multiply, or (b) apply to states within a tomography schedule.
    """
    # Sanitize p into [0, 1] to ensure CPTP parameters.
    p = float(p)
    p = 0.0 if p < 0 else (1.0 if p > 1 else p)

    # Standard two-Kraus representation
    K0 = np.array([[1.0, 0.0],[0.0, np.sqrt(1.0 - p)]], complex)
    K1 = np.array([[0.0, np.sqrt(p)],[0.0, 0.0]], complex)
    return [K0, K1]

def kraus_phase_damping(lam: float) -> List[np.ndarray]:
    """
    Return the Kraus operators for single-qubit phase (pure dephasing) damping.

    Physical meaning
    ----------------
    Phase damping with strength λ (0 ≤ λ ≤ 1) reduces off-diagonal coherences
    while leaving populations unchanged. A simple 2-Kraus realization is:
        K0 = diag(1, sqrt(1-λ)),
        K1 = diag(0, sqrt(λ)).
    This attenuates |0⟩⟨1| and |1⟩⟨0| terms by sqrt(1-λ) and redistributes
    the "lost" coherence probabilistically through K1.

    Parameters
    ----------
    lam : float
        Dephasing strength λ. Values are clamped to [0, 1] for robustness.

    Returns
    -------
    List[np.ndarray]
        [K0, K1] each of shape (2,2), complex dtype.

    Usage notes
    -----------
    • λ=0 → identity channel (no dephasing).
    • λ=1 → full dephasing: coherences are fully suppressed.
    • Combine with amplitude damping by sequential composition using
      :func:`compose_kraus` below (order matters).
    """
    # Sanitize λ into [0, 1] to ensure CPTP parameters.
    lam = float(lam)
    lam = 0.0 if lam < 0 else (1.0 if lam > 1 else lam)

    # Simple diagonal Kraus pair that damps coherences only.
    K0 = np.array([[1.0, 0.0],[0.0, np.sqrt(1.0 - lam)]], complex)
    K1 = np.array([[0.0, 0.0],[0.0, np.sqrt(lam)]], complex)
    return [K0, K1]

def compose_kraus(As: List[np.ndarray], Bs: List[np.ndarray]) -> List[np.ndarray]:
    """
    Sequentially compose two CPTP channels given in Kraus form.

    Math
    ----
    Let A(ρ) = Σ_i A_i ρ A_i† and B(ρ) = Σ_j B_j ρ B_j†.
    Then the composition C = B ∘ A has Kraus operators:
        { C_k } = { B_j A_i } for all i, j,
    i.e., apply A first, then B.

    Parameters
    ----------
    As : List[np.ndarray]
        Kraus operators {A_i} of the first channel to apply (A runs first).
    Bs : List[np.ndarray]
        Kraus operators {B_j} of the second channel to apply (B runs after A).

    Returns
    -------
    List[np.ndarray]
        The composed set {B_j A_i} forming B ∘ A.

    Notes & caveats
    ---------------
    • The result is CPTP if each input set is CPTP (closed under composition).
    • Order matters: compose_kraus(As, Bs) ≠ compose_kraus(Bs, As) in general.
    • If you wish to add *both* amplitude and phase damping after a circuit,
      pick an order that matches your physical model (e.g., amplitude then
      dephasing), and call:
          Ks = compose_kraus(kraus_amplitude_damping(p), kraus_phase_damping(lam))
      which yields channel ρ → PhaseDamp( AmpDamp(ρ) ).
    """
    # Cartesian product over Kraus lists implements sequential composition.
    return [B @ A for B in Bs for A in As]

# ──────────────────────────────────────────────────────────────
# 1) Gates from generators, with per-gate additive Δ inside expm
# ──────────────────────────────────────────────────────────────
class Gate:
    """
    Gate factory for single-qubit matrices (2×2 complex ndarrays).

    Big picture
    -----------
    Every gate here is constructed by exponentiating a (possibly *noisy*)
    generator in su(2). Concretely, for a target axis generator G and a
    rotation angle θ, we synthesize
        U(θ, Δ) = exp(−i * (θ/2) * (G + Δ)),
    where Δ is an *additive generator perturbation* (2×2 complex) that
    captures coherent/non-Hermitian model error at the generator level.
    Setting Δ=None yields the ideal gate.

    Why additive Δ (inside the exponential)?
    ----------------------------------------
    • It models calibration-like errors that *distort the generator itself*
      (e.g., slight axis tilt, over/under-rotation bias, small non-Hermitian
      leakage surrogate). This is more expressive than post-multiplying by an
      error unitary.
    • It keeps a single source of truth for both Heisenberg-exact and
      shots-based tomography in the rest of the pipeline.

    Conventions
    -----------
    • Angles are in radians.
    • σx, σy, σz (Paulis) are defined at module scope.
    • For composite “named” gates (H, S, T, etc.), we map them to rotations
      around fixed axes with fixed angles, then apply the same Δ mechanism.
    • Passing `noise=None` means *no additive generator noise* for that gate.

    Gotchas
    -------
    • If Δ is not Hermitian, expm(−i*(θ/2)*(G+Δ)) need not be unitary. This is
      intentional: the broader framework can convert such effective 2×2
      matrices into (possibly non-unitary) channels and then into PTMs.

    """

    @staticmethod
    def _exp_rot(G: np.ndarray, angle: float, delta: Optional[np.ndarray]) -> np.ndarray:
        """
        Core constructor: exponentiate a (possibly perturbed) generator.

        Parameters
        ----------
        G : np.ndarray
            Ideal su(2) generator (e.g., σx, σy, σz, or a normalized axis).
        angle : float
            Rotation angle in radians (the usual θ).
        delta : Optional[np.ndarray]
            Additive generator noise Δ (2×2 complex). If None, no perturbation.

        Returns
        -------
        np.ndarray
            The 2×2 complex matrix U = exp(−i * (angle/2) * (G + Δ)).

        Notes
        -----
        • We add Δ *inside* the exponential: G_eff = G + Δ.
        • The (angle/2) factor matches standard single-qubit rotation
          conventions R_axis(θ) = exp(−i θ σ_axis / 2).
        """
        Geff = G if delta is None else G + delta
        return la.expm(-1j * (angle / 2.0) * Geff)

    # Pauli-axis rotations
    @staticmethod
    def Rx(theta: float, noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Rotation about X:  R_x(θ) = exp(−i * θ * σx / 2) with optional Δ.
        `noise` is an additive generator perturbation applied to σx.
        """
        return Gate._exp_rot(σx, theta, noise)

    @staticmethod
    def Ry(theta: float, noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Rotation about Y:  R_y(θ) = exp(−i * θ * σy / 2) with optional Δ.
        `noise` perturbs the σy generator inside the exponential.
        """
        return Gate._exp_rot(σy, theta, noise)

    @staticmethod
    def Rz(phi: float, noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Rotation about Z:  R_z(φ) = exp(−i * φ * σz / 2) with optional Δ.
        `noise` perturbs the σz generator inside the exponential.
        """
        return Gate._exp_rot(σz, phi, noise)

    # Derived single-qubit gates
    @staticmethod
    def H(noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Hadamard via a π rotation about the axis (X+Z)/√2.
        Ideal: H ∝ exp(−i * π * (σx+σz)/√2 / 2). We use the exact rotation
        form with the same noise mechanism (Δ added to the axis generator).
        """
        n = (σx + σz) / np.sqrt(2.0)
        return Gate._exp_rot(n, np.pi, noise)

    @staticmethod
    def X(noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Pauli-X as a π rotation about X: X = Rx(π) with optional Δ on σx.
        """
        return Gate.Rx(np.pi, noise)

    @staticmethod
    def Y(noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Pauli-Y as a π rotation about Y: Y = Ry(π) with optional Δ on σy.
        """
        return Gate.Ry(np.pi, noise)

    @staticmethod
    def Z(noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Pauli-Z as a π rotation about Z: Z = Rz(π) with optional Δ on σz.
        """
        return Gate.Rz(np.pi, noise)

    @staticmethod
    def SX(noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        √X gate as a +π/2 rotation about X: SX = Rx(+π/2) with optional Δ.
        """
        return Gate.Rx(+np.pi/2, noise)

    @staticmethod
    def SXdg(noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        (√X)† gate as a −π/2 rotation about X: SX† = Rx(−π/2) with optional Δ.
        """
        return Gate.Rx(-np.pi/2, noise)

    @staticmethod
    def S(noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Phase gate S as a +π/2 rotation about Z: S = Rz(+π/2) with optional Δ.
        """
        return Gate.Rz(+np.pi/2, noise)

    @staticmethod
    def Sdg(noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        S† as a −π/2 rotation about Z: S† = Rz(−π/2) with optional Δ.
        """
        return Gate.Rz(-np.pi/2, noise)

    @staticmethod
    def T(noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        T (π/8) gate as a +π/4 rotation about Z: T = Rz(+π/4) with optional Δ.
        """
        return Gate.Rz(+np.pi/4, noise)

    @staticmethod
    def Tdg(noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        T† as a −π/4 rotation about Z: T† = Rz(−π/4) with optional Δ.
        """
        return Gate.Rz(-np.pi/4, noise)

    @staticmethod
    def U(theta: float, phi: float, lam: float, noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        U(θ, φ, λ) = Rz(φ) · Ry(θ) · Rz(λ) with optional Δ applied to the *last* Rz.

        Rationale & knobs
        -----------------
        • This mirrors a common parameterization used in many frameworks.
        • We attach the additive Δ to the final Rz(λ) factor. If you instead
          want Δ to affect a different factor or *all* factors, adjust the
          call site accordingly (the rest of the pipeline is agnostic).

        Parameters
        ----------
        theta : float
            Ry rotation angle.
        phi : float
            First Rz rotation angle.
        lam : float
            Final Rz rotation angle (this one receives the `noise` Δ).
        noise : Optional[np.ndarray]
            Additive generator perturbation for the final Rz.

        Returns
        -------
        np.ndarray
            The composed 2×2 matrix U = Rz(φ) @ Ry(θ) @ Rz(λ, Δ).
        """
        return Gate.Rz(phi) @ Gate.Ry(theta) @ Gate.Rz(lam, noise)

# ──────────────────────────────────────────────────────────────
# 2) Circuit composer with optional post-circuit damping
# ──────────────────────────────────────────────────────────────
@dataclass
class Circuit:
    r"""
    Minimal single-qubit circuit composer with optional *post-circuit* CPTP
    noise (amplitude / phase damping). The class is intentionally lightweight:
    it stores a sequence of gate *tokens* and a per-gate **additive generator
    noise** map Δ, constructs the overall 2×2 matrix product \(U\), and—if
    requested—exposes a set of Kraus operators that model a channel applied
    *after* the coherent circuit.

    ── How gates are built (recap) ────────────────────────────────────────────
    Each gate matrix is constructed from its su(2) generator with an **additive
    generator perturbation** Δ inside the exponential:
        \( U(\theta, \Delta) = \exp[-i (\theta/2)\, (G + \Delta)] \).
    This Δ is provided per gate name via `noise_map`. If a gate name is absent
    in `noise_map` or if Δ is `None`, the ideal generator is used.

    ── What this class *does* vs *doesn't* do ─────────────────────────────────
    • Does:
      - Combine your chosen gates (with their per-gate Δ) into a single 2×2
        matrix \(U\) using a clear multiplication convention (see below).
      - Optionally append **post-circuit** amplitude- and phase-damping via
        a returned Kraus set (to be used downstream when forming channels/PTMs).

    • Doesn't:
      - Perform measurement sampling or tomography itself (that is handled by
        `ProcessTomography` elsewhere).
      - Inject noise *between* gates automatically—only the per-gate Δ and the
        optional *final* damping layer are considered here.

    Parameters
    ----------
    tokens : Sequence[tuple]
        The circuit as a list/tuple of gate *specifications* in **diagram
        order** (left→right). Each spec is a tuple like:
            ("Rx", theta), ("Rz", phi), ("H",), ("SX",), ...
        The first element is the `Gate` method name; remaining elements are the
        gate parameters (angles, etc.). Example: `[("Rx", np.pi)]`.

    noise_map : dict[str, np.ndarray], optional
        Maps *gate name* → **Δ** (2×2 complex) to add inside the exponential of
        that gate's generator. If a gate name is missing, the ideal generator
        is used for that gate. Typical usage is to provide Δ for a subset of:
        {"Rx","Ry","Rz","H","SX","S","T",...}. Passing `None` disables all Δ.

        • Knob behavior:
          - Change Δ magnitude/structure to model per-gate coherent error.
          - Set to `None` (or leave names absent) to recover ideal gates.

    amp_damp_p : Optional[float]
        Amplitude-damping probability \(p \in [0,1]\) applied **after** the
        coherent unitary product \(U\). `None` ⇒ OFF (identity channel).
        Larger \(p\) increases |1⟩→|0⟩ relaxation.

    phase_damp_lam : Optional[float]
        Phase-damping (pure dephasing) strength \( \lambda \in [0,1] \),
        also **after** the unitary. `None` ⇒ OFF. Larger \( \lambda \)
        damps off-diagonals more strongly (reduces coherences).

    Notes on multiplication order
    -----------------------------
    • This implementation multiplies as:
          U_tot ← U_k @ U_tot
      while iterating tokens left→right. That realizes the *diagram order*
      you see in the token list. Do **not** change this order unless you also
      change how you author `tokens` throughout the code base.

    • If you come from a convention where the first token acts *first* on
      kets (rightmost matrix), this code already accounts for that by the
      chosen accumulation rule; your `tokens` should be written left→right.

    Usage patterns
    --------------
    >>> circ = Circuit([("Rx", np.pi)], noise_map={"Rx": Δx})
    >>> U = circ.unitary()          # coherent part (2×2)
    >>> Ks = circ.kraus()           # post-circuit noise (list of 2×2 Kraus ops)

    Pitfalls / gotchas
    ------------------
    • Supplying non-Hermitian Δ leads to non-unitary U. That's allowed here;
      downstream tomography converts the resulting linear map to a PTM.
    • `kraus()` *does not* insert noise between gates—only after the full U.
    """
    tokens: Sequence[tuple]
    noise_map: Optional[Dict[str, np.ndarray]] = None
    amp_damp_p: Optional[float] = None
    phase_damp_lam: Optional[float] = None

    def unitary(self, verbose: bool = False) -> np.ndarray:
        """
        Construct and return the **coherent** circuit matrix \(U\) (2×2).

        Implementation details
        ----------------------
        • Iterates over `self.tokens` in list order (left→right).
        • For each token:
            - Looks up a `Gate.<name>` constructor.
            - Passes through any per-gate Δ from `self.noise_map`.
            - Multiplies as: `U_tot = U_k @ U_tot` (see class docstring).

        Parameters
        ----------
        verbose : bool
            When True, prints a human-readable assembly log with per-step
            matrices and the running product—useful for debugging angle values,
            Δ usage, and multiplication ordering.

        Returns
        -------
        np.ndarray
            The final 2×2 complex matrix \(U\) representing the coherent
            (noise-in-generator) effect of `tokens` with the supplied Δ-map.

        Tuning & diagnostics
        --------------------
        • If results look “reversed,” revisit your mental model of order and
          the left-multiplication used here; the printed verbose trace helps.
        • If a gate name is misspelled or unsupported, a ValueError is raised.
        """
        U_tot = I2                      # start from identity
        nm = self.noise_map or {}       # treat None as empty mapping

        if verbose:
            print("┌── circuit assembly ─────────────────────────────")
        for i, spec in enumerate(self.tokens, 1):
            # Unpack gate token: ("Rx", theta) or ("H",) etc.
            name, *pars = spec

            # Validate gate name early (prevents silent fall-through)
            if not hasattr(Gate, name):
                raise ValueError(f"Unknown gate name: {name}")

            # Pull the additive generator noise Δ for this gate (if any)
            delta = nm.get(name)

            # Build this step's gate matrix with the Δ inside the exponential
            Uk = getattr(Gate, name)(*pars, noise=delta)

            # Accumulate with our left-multiplication convention
            U_tot = Uk @ U_tot

            # Optional trace for humans
            if verbose:
                pstr = ", ".join(
                    f"{p/np.pi:.3g}·π" if isinstance(p, (int, float)) else str(p)
                    for p in pars
                ) or "∅"
                print(f"│ step {i:>2}:  {name}({pstr})")
                print("│ U_k =\n", np.round(Uk, 6))
                print("│ running product =\n", np.round(U_tot, 6))
                print("├─────────────────────────────────────────────")

        if verbose:
            print("└── end assembly ─────────────────────────────\n")
        return U_tot

    def kraus(self) -> List[np.ndarray]:
        """
        Return the **post-circuit** Kraus operators for the configured
        amplitude and/or phase damping. If both are `None`, returns `[I2]`.

        Channel composition & order
        ---------------------------
        The returned set corresponds to applying (after the unitary):
            1) Amplitude damping with probability `amp_damp_p` (if not None),
            2) then Phase damping with strength `phase_damp_lam` (if not None).
        Composition is implemented by concatenating Kraus lists:
            {B_j A_i} for sequential A then B.

        Returns
        -------
        List[np.ndarray]
            A list of 2×2 Kraus operators \(\{K_\ell\}\). Use them to transform
            a density matrix ρ as ρ ↦ Σ_\ell K_\ell ρ K_\ell†.

        Knobs & scenarios
        -----------------
        • `amp_damp_p=None`, `phase_damp_lam=None`  → deterministic identity
          channel `[I2]`.
        • `amp_damp_p=p`, `phase_damp_lam=None`     → only relaxation noise.
        • `amp_damp_p=None`, `phase_damp_lam=λ`     → only pure dephasing.
        • Both set                                   → relaxation then dephasing.

        Caution
        -------
        This function *does not* apply the Kraus channel to any state; it only
        returns the operators. Downstream tomography utilities decide when and
        how to incorporate them (e.g., when building PTMs).
        """
        Ks: List[np.ndarray] = [I2]  # identity (no post-circuit channel by default)

        # Optional amplitude damping; replaces identity if enabled
        if self.amp_damp_p is not None:
            Ks = kraus_amplitude_damping(self.amp_damp_p)

        # Optional phase damping; composes after amplitude damping (if any)
        if self.phase_damp_lam is not None:
            Ps = kraus_phase_damping(self.phase_damp_lam)
            Ks = compose_kraus(Ks, Ps)

        return Ks

# ──────────────────────────────────────────────────────────────
# 3) PauliPrep and PauliMeas schedules (Qiskit-style)
# ──────────────────────────────────────────────────────────────
def prep_unitary(label: str) -> np.ndarray:
    r"""
    Return the state-preparation unitary \(U_{\text{prep}}\) that maps \(|0\rangle\)
    to the requested Pauli eigenstate.

    Parameters
    ----------
    label : {"Z+", "Z-", "X+", "Y+"} (case-insensitive)
        Shorthand for the target eigenstate to prepare **from** \(|0\rangle\):
        • "Z+" → \(|0\rangle\)             via \(I\)
        • "Z-" → \(|1\rangle\)             via \(X\)
        • "X+" → \(|+\rangle\)             via \(H\)
        • "Y+" → \(|+i\rangle\)            via \(S\,H\)

        Notes:
        - These choices match common minimal tomography schedules.
        - "X−" or "Y−" are not used here for the *prep* side; they appear on
          the *measurement* side in the affine schedule.

    Returns
    -------
    np.ndarray
        A 2×2 complex unitary that, when applied to \(|0\rangle\), prepares the
        requested eigenstate.

    Raises
    ------
    ValueError
        If the label is not one of {"Z+", "Z-", "X+", "Y+"}.

    Tips / scenarios
    ----------------
    • If you want to extend the prep basis, add more labels mapping to
      appropriate single-qubit unitaries (e.g., "X−" → `Gate.H() @ Gate.Z()`).
    • Keep this consistent with the measurement side so that your schedule
      remains informationally complete for process tomography.
    """
    l = label.upper()
    if l == "Z+": return I2
    if l == "Z-": return Gate.X()
    if l == "X+": return Gate.H()
    if l == "Y+": return Gate.S() @ Gate.H()
    raise ValueError("prep ∈ {Z+, Z-, X+, Y+}")

def meas_unitary(label: str) -> np.ndarray:
    r"""
    Return the pre-measurement basis-change unitary \(U_{\text{meas}}\) that
    rotates the desired Pauli observable into the computational Z-basis.

    Parameters
    ----------
    label : {"X", "Y", "Z"} (case-insensitive)
        • "Z" → measure \(Z\): no rotation (identity).
        • "X" → measure \(X\): rotate with \(H\).
        • "Y" → measure \(Y\): rotate with \(H\,S^\dagger\).

    Returns
    -------
    np.ndarray
        A 2×2 complex unitary applied **before** standard Z measurement so that
        counts correspond to the chosen Pauli eigenprojectors.

    Raises
    ------
    ValueError
        If the label is not one of {"X", "Y", "Z"}.

    Notes
    -----
    • This mirrors the standard Qiskit-style measurement compilation where all
      measurements are done in Z after a basis change.
    • Extendable to other single-qubit observables by inserting the right
      rotation that diagonalizes the desired Hermitian operator.
    """
    l = label.upper()
    if l == "Z": return I2
    if l == "X": return Gate.H()
    if l == "Y": return Gate.H() @ Gate.Sdg()
    raise ValueError("meas ∈ {X, Y, Z}")

# ──────────────────────────────────────────────────────────────
# 4) Measurement simulation  (exact or with shots) + CPTP support
# ──────────────────────────────────────────────────────────────
def run_one_schedule(U_under_test: np.ndarray,
                     prep_label: str,
                     meas_label: str,
                     shots: int = 0,
                     rng: Optional[np.random.Generator] = None,
                     kraus: Optional[List[np.ndarray]] = None,
                     ) -> Tuple[float, float, float]:
    r"""
    Simulate a **single** state-preparation / measurement schedule with optional
    *post-unitary* CPTP noise and either exact expectations (shots=0) or finite
    sampling (shots>0).

    Pipeline
    --------
    \(|0\rangle \xrightarrow{U_{\text{prep}}}\) prepared state
    \(\xrightarrow{U_{\text{under test}}}\) transformed state
    \(\xrightarrow{\text{(optional channel)}}\)
    \(\xrightarrow{U_{\text{meas}}}\) rotate-into-Z
    \(\xrightarrow{\text{Z measurement}}\) counts/expectation.

    Parameters
    ----------
    U_under_test : np.ndarray
        2×2 complex matrix of the **coherent** process under test (e.g., the
        product from `Circuit.unitary(...)`). This can be non-unitary if you
        used non-Hermitian generator noise Δ; that's allowed here.

    prep_label : {"Z+", "Z-", "X+", "Y+"}
        Which eigenstate to prepare from \(|0\rangle\); see `prep_unitary`.

    meas_label : {"X", "Y", "Z"}
        Which Pauli to measure; see `meas_unitary`. Internally we rotate into Z
        and then compute Z-outcome probabilities.

    shots : int, default 0
        • `shots == 0`  → **analytic** expectation (no sampling): we return
          \( \langle P \rangle = p_0 - p_1 = 2p_0-1 \) exactly.
        • `shots  > 0`  → **finite-sample** expectation: draw Binomial(\(n=\)shots,
          \(p=p_0\)) and convert to the empirical expectation \( (n_+ - n_-)/n \).

        Trade-offs / scenarios:
        - Larger `shots` reduces variance at the cost of runtime.
        - For GA cost evaluations, you may want modest `shots` but average over
          several repeats (see the outer COST_SHOT_REPEATS knob in your driver).

    rng : np.random.Generator, optional
        Source of randomness for the shot noise. If `None`, a fresh default
        generator is created. To get **common random numbers** across candidates
        (variance reduction), pass the same seeded `rng` through callers.

    kraus : list of np.ndarray, optional
        Optional post-unitary **Kraus operators** modeling a CPTP channel
        applied *after* `U_under_test` and *before* the measurement rotation.
        If `None`, no channel is applied. Typical producer: `Circuit.kraus()`.

        • Example: amplitude damping (p) + phase damping (λ) composed in order.
        • If you want *inter-gate* noise, you must build that into your U or
          modify the scheduling; this hook is strictly post-unitary.

    Returns
    -------
    (exp, p0, p1) : tuple of floats
        • `exp` : the returned Pauli expectation value \(\langle P \rangle\).
          - Exact if `shots==0`, sampled if `shots>0`.
        • `p0`  : probability of outcome \(|0\rangle\) **after** measurement
          rotation (i.e., "plus" outcome for the chosen Pauli).
        • `p1`  : probability of outcome \(|1\rangle\) (== 1 - p0).

    Notes & correctness checks
    --------------------------
    • Probabilities p0/p1 are clipped via the binomial helper when sampling,
      ensuring numerical stability even with tiny negative/overshoot due to
      floating-point roundoff.
    • If your affine PTM ever shows structurally zero entries while the
      Heisenberg PTM does not, investigate:
        (1) your *schedule coverage* (prep/meas pairs),
        (2) the Kraus hook (are you passing it consistently?),
        (3) shot RNG seeding and cost weighting,
        (4) whether `assume_tp` is enforced upstream during reconstruction,
            which can couple entries and constrain rows/columns.
    """
    # Ensure we have an RNG; using a shared one upstream enables CRN variance reduction.
    rng = np.random.default_rng() if rng is None else rng

    # Build the basis-change unitaries for prep and measurement.
    Up = prep_unitary(prep_label)
    Um = meas_unitary(meas_label)

    # Start from |0>, prepare the requested eigenstate, apply the device-under-test.
    psi0 = np.array([1,0], complex)[:,None]   # column vector |0>
    psi  = Up @ psi0
    phi  = U_under_test @ psi

    # Density operator after the coherent process.
    rho  = phi @ phi.conj().T

    # Optional post-unitary CPTP channel via Kraus operators.
    if kraus is not None:
        rho = sum( K @ rho @ K.conj().T for K in kraus )

    # Rotate into Z basis for measurement of the requested Pauli observable.
    rho_m = Um @ rho @ Um.conj().T

    # Computational-basis projectors (measure Z after rotation).
    Pi0 = np.array([[1,0],[0,0]], complex)
    Pi1 = np.array([[0,0],[0,1]], complex)

    # Exact probabilities of outcomes |0>, |1> after the full pipeline.
    p0  = float(np.trace(Pi0 @ rho_m).real)
    p1  = float(np.trace(Pi1 @ rho_m).real)

    # Convert to expectation value <P>. With shots>0, draw Binomial samples.
    exp = _binomial_expectation(p0, shots, rng)

    # Return the expectation and the underlying probabilities for diagnostics.
    return exp, p0, p1

class QPT12Strict_old:
    """
    Strict 12-circuit single-qubit process tomography (IBM basis {Rz, SX}).

    Token order matches left-multiply accumulation:
        U_tot = U_k @ U_tot  (rightmost token acts first on the state)
        → tokens must be: prep + target + meas
          so the state sees PREP → TARGET → MEAS.

    Measurement Y is H · S†.
    With left-multiply, encode it as tokens [Sdg, H] so S† acts first.
    """

    def __init__(self, target_tokens=(("X",),)):
        self.target_tokens = list(target_tokens)

    @staticmethod
    def _H_tokens() -> list:
        # Canonical IBM H = Rz(+π/2) · SX · Rz(+π/2)
        return [("Rz", np.pi/2), ("SX",), ("Rz", np.pi/2)]

    @classmethod
    def _meas_tokens(cls, label: str) -> list:
        L = label.upper()
        if L == "Z":
            return []
        if L == "X":
            return cls._H_tokens()
        if L == "Y":
            # MUST be [Sdg, H] under left-multiply (state sees Sdg then H) ⇒ H · S†
            return [("Rz", -np.pi/2)] + cls._H_tokens()
        raise ValueError("meas ∈ {X, Y, Z}")

    @classmethod
    def _prep_tokens(cls, label: str) -> list:
        L = label.upper()
        if L == "Z+": return []
        if L == "Z-": return [("X",)]
        if L == "X+": return cls._H_tokens()                    # |+> = H|0>
        if L == "Y+":                                           # |+i> = S·H|0>
            # With left-multiply, tokens [H, S] ⇒ net S·H on the state (correct).
            return cls._H_tokens() + [("Rz", +np.pi/2)]
        raise ValueError("prep ∈ {Z+, Z-, X+, Y+}")

    def circuits(self, noise_map=None, amp_damp_p=None, phase_damp_lam=None):
        preps = ["Z+", "Z-", "X+", "Y+"]
        meass = ["Z", "X", "Y"]
        out = []
        for p in preps:
            P = self._prep_tokens(p)
            for m in meass:
                M = self._meas_tokens(m)
                tokens = list(P) + list(self.target_tokens) + list(M)
                out.append(Circuit(tokens=tokens,
                                   noise_map=noise_map,
                                   amp_damp_p=amp_damp_p,
                                   phase_damp_lam=phase_damp_lam))
        return out

    def ptm(self,
            shots=0,
            rng=None,
            assume_tp=False,
            noise_map=None,
            amp_damp_p=None,
            phase_damp_lam=None,
            verbose=False,
            readout_loss: float = 0.0,
            *,
            # NEW (optional) stats controls — default keeps old behavior:
            component_stats: bool = False,
            repeats_for_stats: int = 16,
            return_standard_error: bool = False):
        """
        Run the 12 circuits, collect r_p = [⟨X⟩,⟨Y⟩,⟨Z⟩], assemble:
            t = ½ (r_Z+ + r_Z−)
            T = [ r_X+ − t,  r_Y+ − t,  r_Z+ − t ]  (as columns X, Y, Z)

        First row handling:
          • assume_tp=True  → force R[0,:] = [1,0,0,0]
          • assume_tp=False → reconstruct from trace-after signals s_p
                               (needs s_p ≠ 1 to avoid zeros; enable readout_loss).

        readout_loss:
          • If shots>0 and readout_loss>0, we drop a Binomial(shots, loss) “nulls”
            and only sample ± on the remaining shots. This makes s_p < 1 on average.

        Returns (default, unchanged):
            R, t, T

        If component_stats=True and shots>0, returns:
            R_mean, t_mean, T_mean, R_std
        If additionally return_standard_error=True, returns:
            R_mean, t_mean, T_mean, R_std, R_se
        where R_std is the component-wise standard deviation across 'repeats_for_stats'
        shot realizations, and R_se = R_std / sqrt(repeats_for_stats).
        """

        rng = np.random.default_rng() if rng is None else rng
        circs = self.circuits(noise_map=noise_map,
                              amp_damp_p=amp_damp_p,
                              phase_damp_lam=phase_damp_lam)

        def _binomial_expectation(p0: float, shots: int, rng_local) -> float:
            if shots <= 0:
                return p0 - (1.0 - p0)
            n = shots
            k = rng_local.binomial(n, max(0.0, min(1.0, p0)))
            return (k - (n - k)) / float(n)

        def _run_one_R(rng_local):
            def _run(c: Circuit):
                # Build state after full (prep + target + meas) circuit
                U = c.unitary()
                Ks = c.kraus() or [I2]

                psi0 = np.array([1, 0], complex)[:, None]
                phi = U @ psi0
                rho = phi @ phi.conj().T
                rho = sum(K @ rho @ K.conj().T for K in Ks)

                # True probabilities for Z-basis outcomes after embedded basis change
                Pi0 = np.array([[1, 0], [0, 0]], complex)
                Pi1 = np.array([[0, 0], [0, 1]], complex)
                p0 = float(np.trace(Pi0 @ rho).real)
                p1 = float(np.trace(Pi1 @ rho).real)

                # Analytic / no-loss path → standard binomial expectation
                if shots <= 0 or readout_loss <= 0.0:
                    exp = _binomial_expectation(p0, shots, rng_local)
                    return exp, p0, p1  # note: p0+p1≈1 (CPTP), but we don't force it

                # Shots with loss: introduce nulls so s_p := p0_hat+p1_hat < 1
                loss = max(0.0, min(1.0, float(readout_loss)))
                n_null = rng_local.binomial(shots, loss)
                n_eff = shots - n_null
                if n_eff <= 0:
                    # degenerate, but defined
                    return 0.0, 0.0, 0.0

                # Sample outcomes on remaining trials
                n_plus = rng_local.binomial(n_eff, max(0.0, min(1.0, p0)))
                n_minus = n_eff - n_plus

                exp_hat = (n_plus - n_minus) / float(n_eff)    # expectation on effective shots
                p0_hat  = n_plus / float(shots)                # normalize to total → s_p < 1
                p1_hat  = n_minus / float(shots)
                return exp_hat, p0_hat, p1_hat

            preps = ["Z+", "Z-", "X+", "Y+"]
            exps = {}
            traces = {}  # s_p = p0_hat + p1_hat per prep

            idx = 0
            for p in preps:
                ez, p0z, p1z = _run(circs[idx + 0])  # meas Z
                ex, p0x, p1x = _run(circs[idx + 1])  # meas X
                ey, p0y, p1y = _run(circs[idx + 2])  # meas Y
                idx += 3

                # Affine assembly expects [⟨X⟩,⟨Y⟩,⟨Z⟩]
                exps[p] = np.array([ex, ey, ez], float)

                # "trace-after": identical across meas axes by construction
                traces[p] = float(p0z + p1z)

            # Assemble affine map
            rZp, rZm, rXp, rYp = exps["Z+"], exps["Z-"], exps["X+"], exps["Y+"]
            t = 0.5 * (rZp + rZm)
            T = np.column_stack((rXp - t, rYp - t, rZp - t))

            R = np.zeros((4, 4), float)
            R[1:, 0]  = t
            R[1:, 1:] = T

            if assume_tp:
                R[0, :] = [1.0, 0.0, 0.0, 0.0]
            else:
                # Reconstruct first row from trace-after signals
                sZp, sZm, sXp, sYp = traces["Z+"], traces["Z-"], traces["X+"], traces["Y+"]
                a = 0.5 * (sZp + sZm)  # R[0,0]
                c = 0.5 * (sZp - sZm)  # R[0,3]
                b = sXp - a            # R[0,1]
                d = sYp - a            # R[0,2]
                R[0, :] = [a, b, d, c]

            return R, t, T

        # --- Default single-evaluation path (unchanged API) ---
        if (not component_stats) or (shots <= 0):
            R, t, T = _run_one_R(rng)
            if verbose:
                print("\n[QPT12Strict] t =", np.round(t, 6))
                print("[QPT12Strict] T =\n", np.round(T, 6))
                print("[QPT12Strict] R (I,X,Y,Z) =\n", np.round(R, 6))
            return R, t, T

        # --- Optional component-wise variability over repeated shot draws ---
        reps = int(max(1, repeats_for_stats))
        Rs = []
        ts = []
        Ts = []
        # use independent RNG streams for statistical independence
        for _ in range(reps):
            rloc = np.random.default_rng(rng.integers(0, 2**32 - 1))
            Rk, tk, Tk = _run_one_R(rloc)
            Rs.append(Rk); ts.append(tk); Ts.append(Tk)
        Rs = np.stack(Rs, 0)   # (reps, 4, 4)
        ts = np.stack(ts, 0)   # (reps, 3)
        Ts = np.stack(Ts, 0)   # (reps, 3, 3)

        R_mean = Rs.mean(axis=0)
        t_mean = ts.mean(axis=0)
        T_mean = Ts.mean(axis=0)

        # component-wise standard deviation (ddof=1)
        R_std = Rs.std(axis=0, ddof=1) if reps > 1 else np.zeros_like(R_mean)
        if verbose:
            print("\n[QPT12Strict] component-wise STD computed over", reps, "repeats")

        if return_standard_error and reps > 1:
            R_se = R_std / np.sqrt(reps)
            return R_mean, t_mean, T_mean, R_std, R_se
        else:
            return R_mean, t_mean, T_mean, R_std

    def self_test_signs(self):
        R, _, _ = self.ptm(shots=0, assume_tp=False)
        print("[self-test] ||R - diag(1,1,-1,-1)||_F =",
              np.linalg.norm(R - np.diag([1.0,1.0,-1.0,-1.0]), 'fro'))


import numpy as np

I2 = np.eye(2, dtype=complex)

class QPT12Strict_old_but_working_almost_perfectly:
    """
    Strict 12-circuit single-qubit process tomography in {Rz, SX} (IBM native) with
    explicit, commented construction of every PTM component.

    ─────────────────────────────────────────────────────────────────────────────
    PTM / affine Bloch map (ordering = (I, X, Y, Z)):
      A (possibly non-TP) single-qubit channel Φ acts on Bloch vectors as
          r' = t + T r
      with r, r' ∈ ℝ^3, t ∈ ℝ^3, T ∈ ℝ^{3×3}.
      The Pauli Transfer Matrix R (4×4, real in this idealized setting) is:

          R = [[ R00  R01  R02  R03 ],
               [ t_x  T_xx T_xy T_xz],
               [ t_y  T_yx T_yy T_yz],
               [ t_z  T_zx T_zy T_zz]]

      • Rows 1..3, Col 0  (i.e., R[1:,0])  = t  (affine shift)
      • Rows 1..3, Cols 1..3               = T  (linear 3×3 part)

      The **first row** encodes how the expectation of I transforms. For TP channels
      we have R[0,:] = [1, 0, 0, 0]. For non-TP (e.g., readout loss), the first row
      deviates and must be reconstructed from “trace-after” signals.

    ─────────────────────────────────────────────────────────────────────────────
    What the 12 circuits measure and how we assemble (t, T):
      Preps:  { Z+, Z−, X+, Y+ }
      Meas.:  { Z, X, Y }
      For each prep p we measure Pauli expectations r_p := [⟨X⟩, ⟨Y⟩, ⟨Z⟩].

      We then assemble
          t  = ½ ( r_{Z+} + r_{Z−} )
          T  = [ r_{X+} − t,  r_{Y+} − t,  r_{Z+} − t ]   (columns = X,Y,Z)

      These rules are standard for the affine Bloch embedding of a qubit channel.

    ─────────────────────────────────────────────────────────────────────────────
    FIRST ROW: how we reconstruct it (assume_tp == False)
      Define the “trace-after” scalar s_p for prep p as the *observed* total
      probability mass of detection (0 or 1) after the experiment:

         s_p = (n_plus + n_minus) / n_total       if we simulate loss
         s_p = 1                                   if no loss AND CPTP channel

      In our simulator:
        - With readout_loss == 0, we count all n_total shots → s_p ≡ 1 (for CPTP).
        - With readout_loss > 0, we drop n_null ~ Binomial(n_total, loss);
          then n_eff = n_total − n_null and s_p = n_eff / n_total fluctuates.

      Under the PTM model, the expectation of I for prepared eigenstates is:
          s_p = a + b x_p + d y_p + c z_p,
      where [x_p, y_p, z_p] is the Bloch vector of the prepared eigenstate and
      [a,b,d,c] = first row entries [R00,R01,R02,R03].

      Using the four preps (Z+, Z−, X+, Y+):
          s_{Z+} = a + c
          s_{Z−} = a − c
          s_{X+} = a + b
          s_{Y+} = a + d

      Solve for a,b,c,d:
          a = ½ (s_{Z+} + s_{Z−})
          c = ½ (s_{Z+} − s_{Z−})
          b = s_{X+} − a
          d = s_{Y+} − a
      Then set  R[0,:] = [a, b, d, c].

      Continuity check:
        As readout_loss → 0, s_p → 1 almost surely (n_eff→n), so a→1 and b,d,c→0.
        The variance of s_p shrinks continuously to 0 (Binomial with p≈1). Exactly
        at loss=0, s_p is deterministically 1, hence zero variance in the first row.
        There is no bias discontinuity — only the variance vanishes at the boundary.

    ─────────────────────────────────────────────────────────────────────────────
    SHOT NOISE and independent RNG streams
      • When shots>0 we sample ⟨P⟩ via Binomial draws. That injects variability into
        r_p, hence into t (first column below the top element) and T (3×3 block).
      • If you want independent repeats, we spawn a child RNG for each repeat:
            child = np.random.default_rng(parent.integers(0, 2**32-1))
        Do the same in your outer loops if you call ptm repeatedly.

    Returns (default):
        R, t, T
    If component_stats=True and shots>0:
        R_mean, t_mean, T_mean, R_std   (std over repeats)
    If additionally return_standard_error=True:
        ... plus R_se = R_std / sqrt(repeats_for_stats)
    """

    def __init__(self, target_tokens=(("X",),)):
        self.target_tokens = list(target_tokens)

    # --- gate macros (IBM native) ------------------------------------------------
    @staticmethod
    def _H_tokens():
        # H = Rz(+π/2) · SX · Rz(+π/2) in left-multiply token order
        return [("Rz", np.pi/2), ("SX",), ("Rz", np.pi/2)]

    @classmethod
    def _meas_tokens(cls, label: str):
        L = label.upper()
        if L == "Z":
            return []                   # measure Z
        if L == "X":
            return cls._H_tokens()      # H • Z-meas ⇒ X-meas
        if L == "Y":
            # Y-meas = H · S† on the *state*; with left-multiply encode [Sdg, H]
            return [("Rz", -np.pi/2)] + cls._H_tokens()
        raise ValueError("meas ∈ {X, Y, Z}")

    @classmethod
    def _prep_tokens(cls, label: str):
        L = label.upper()
        if L == "Z+": return []                      # |0⟩
        if L == "Z-": return [("X",)]                # |1⟩ = X|0⟩
        if L == "X+": return cls._H_tokens()         # |+⟩ = H|0⟩
        if L == "Y+": return cls._H_tokens()+[("Rz", +np.pi/2)]  # |+i⟩ = S·H|0⟩
        raise ValueError("prep ∈ {Z+, Z-, X+, Y+}")

    # --- circuits builder --------------------------------------------------------
    def circuits(self, noise_map=None, amp_damp_p=None, phase_damp_lam=None):
        preps = ["Z+", "Z-", "X+", "Y+"]
        meass = ["Z", "X", "Y"]
        out = []
        for p in preps:
            P = self._prep_tokens(p)
            for m in meass:
                M = self._meas_tokens(m)
                tokens = list(P) + list(self.target_tokens) + list(M)
                out.append(Circuit(tokens=tokens,
                                   noise_map=noise_map,
                                   amp_damp_p=amp_damp_p,
                                   phase_damp_lam=phase_damp_lam))
        return out

    # --- tomography core ---------------------------------------------------------
    def ptm(self,
            shots=0,
            rng=None,
            assume_tp=False,
            noise_map=None,
            amp_damp_p=None,
            phase_damp_lam=None,
            verbose=False,
            readout_loss: float = 0.0,
            *,
            component_stats: bool = False,
            repeats_for_stats: int = 16,
            return_standard_error: bool = False):
        """
        Execute all 12 circuits; assemble R, t, T as described above.

        shots:
          0   → analytic expectations (no shot noise)
          >0  → Binomial sampling for expectations; if readout_loss>0 we also
                Binomial-drop “nulls” before sampling outcomes.

        readout_loss:
          probability of a trial producing a 'null' (neither 0 nor 1). This only
          affects the **first row** through s_p; expectations use effective shots.

        component_stats:
          if True and shots>0, repeat the whole 12-circuit set `repeats_for_stats`
          times with independent RNG streams; return means and component-wise std
          (and standard error if requested).
        """
        rng = np.random.default_rng() if rng is None else rng
        circs = self.circuits(noise_map=noise_map,
                              amp_damp_p=amp_damp_p,
                              phase_damp_lam=phase_damp_lam)

        def _exp_from_counts(p0, n, rng_local):
            """
            Convert a Z-basis success probability p0 into ⟨P⟩ via Binomial sampling.
            For shots==0 we return analytic value p0 - (1 - p0).
            """
            if shots <= 0:
                return p0 - (1.0 - p0), None, None  # analytic, no counts
            k = rng_local.binomial(shots, max(0.0, min(1.0, p0)))
            # expectation uses *total* shots when loss==0; with loss>0 we’ll use n_eff
            return (k - (shots - k)) / float(shots), k, shots

        def _run_one_R(rng_local):
            def _run(c: Circuit):
                # PREP→TARGET→MEAS unitary and Kraus map
                U = c.unitary()
                Ks = c.kraus() or [I2]

                psi0 = np.array([1, 0], complex)[:, None]
                phi = U @ psi0
                rho = phi @ phi.conj().T
                rho = sum(K @ rho @ K.conj().T for K in Ks)  # trace≤1 allowed if noise non-TP

                # Z-basis projectors after basis change embedded in U above
                Pi0 = np.array([[1,0],[0,0]], complex)
                Pi1 = np.array([[0,0],[0,1]], complex)
                # Prob. mass for 0/1 clicks w.r.t current (possibly trace-decreased) rho
                p0_true = float(np.trace(Pi0 @ rho).real)
                p1_true = float(np.trace(Pi1 @ rho).real)
                s_true  = p0_true + p1_true   # equals Tr(rho) if we measured all mass

                if shots <= 0:
                    # Analytic expectations; “trace-after” uses the *true* mass s_true.
                    exp = p0_true - (1.0 - p0_true)  # note: if Tr(rho)<1 this still gives the Pauli expectation on the normalized support
                    # For s_p we *do not* force 1: if the channel is trace-decreasing via Kraus,
                    # keep s_true so the first row captures non-TP behavior even analytically.
                    return exp, s_true

                # Shots>0: simulate loss → first decide effective shots
                loss = max(0.0, min(1.0, float(readout_loss)))
                if loss > 0.0:
                    n_null = rng_local.binomial(shots, loss)
                    n_eff  = shots - n_null
                else:
                    n_eff  = shots

                if n_eff <= 0:
                    # everything dropped → zero signal
                    return 0.0, 0.0

                # Sample outcomes on effective shots with success prob proportional to p0_true/s_true
                # If s_true<~1 (trace-decreasing channel), we must *renormalize* outcome probabilities
                # to the detected subspace; otherwise Binomial would undercount probability mass.
                p_det = 0.0 if s_true <= 0.0 else (p0_true / s_true)
                p_det = max(0.0, min(1.0, p_det))
                k_eff = rng_local.binomial(n_eff, p_det)
                n_plus, n_minus = k_eff, (n_eff - k_eff)

                # Expectation is computed on the *effective* shots:
                exp_hat = (n_plus - n_minus) / float(n_eff)

                # “trace-after” s_p is the fraction of effective detections out of total trials:
                s_hat = n_eff / float(shots)

                return exp_hat, s_hat

            # run all 12 and collect:
            preps = ["Z+", "Z-", "X+", "Y+"]
            exps  = {}   # r_p vectors (⟨X⟩,⟨Y⟩,⟨Z⟩)
            svals = {}   # s_p scalars

            idx = 0
            for p in preps:
                ez, sZ = _run(circs[idx + 0])  # meas Z
                ex, sX = _run(circs[idx + 1])  # meas X
                ey, sY = _run(circs[idx + 2])  # meas Y
                idx += 3

                exps[p]  = np.array([ex, ey, ez], float)
                # by construction sZ==sX==sY for a given prep; keep one
                svals[p] = float(sZ)

            # assemble affine blocks
            rZp, rZm, rXp, rYp = exps["Z+"], exps["Z-"], exps["X+"], exps["Y+"]
            t = 0.5 * (rZp + rZm)                                   # → first column (rows 1..3)
            T = np.column_stack((rXp - t, rYp - t, rZp - t))        # → rows 1..3, cols 1..3

            # build PTM
            R = np.zeros((4,4), float)
            R[1:, 0]  = t
            R[1:, 1:] = T

            # FIRST ROW handling
            if assume_tp:
                R[0, :] = [1.0, 0.0, 0.0, 0.0]
            else:
                # s_{Z+} = a + c,  s_{Z−} = a − c,  s_{X+} = a + b,  s_{Y+} = a + d
                sZp = svals["Z+"]
                sZm = svals["Z-"]
                sXp = svals["X+"]
                sYp = svals["Y+"]

                a = 0.5 * (sZp + sZm)     # R[0,0]
                c = 0.5 * (sZp - sZm)     # R[0,3]
                b = sXp - a               # R[0,1]
                d = sYp - a               # R[0,2]
                R[0, :] = [a, b, d, c]

            if verbose:
                print("[ptm] t =", np.round(t, 6))
                print("[ptm] T =\n", np.round(T, 6))
                print("[ptm] R =\n", np.round(R, 6))
            return R, t, T

        # single evaluation or analytic
        if (not component_stats) or (shots <= 0):
            return _run_one_R(rng)

        # repeated stats with independent RNG streams
        reps = int(max(1, repeats_for_stats))
        Rs, ts, Ts = [], [], []
        for _ in range(reps):
            child = np.random.default_rng(rng.integers(0, 2**32 - 1))
            Rk, tk, Tk = _run_one_R(child)
            Rs.append(Rk); ts.append(tk); Ts.append(Tk)

        Rs = np.stack(Rs, 0)
        ts = np.stack(ts, 0)
        Ts = np.stack(Ts, 0)

        R_mean = Rs.mean(axis=0)
        t_mean = ts.mean(axis=0)
        T_mean = Ts.mean(axis=0)

        # component-wise std over repeats (ddof=1 for unbiased sample std)
        R_std  = Rs.std(axis=0, ddof=1) if reps > 1 else np.zeros_like(R_mean)

        if return_standard_error and reps > 1:
            R_se = R_std / np.sqrt(reps)
            return R_mean, t_mean, T_mean, R_std, R_se
        else:
            return R_mean, t_mean, T_mean, R_std

    def self_test_signs(self):
        # quick sanity for an X target under TP reconstruction
        R, _, _ = self.ptm(shots=0, assume_tp=False)
        print("[self-test] ||R - diag(1,1,-1,-1)||_F =",
              np.linalg.norm(R - np.diag([1.0, 1.0, -1.0, -1.0]), ord="fro"))



import numpy as np

I2 = np.eye(2, dtype=complex)
import numpy as np

I2 = np.eye(2, dtype=complex)

class QPT12Strict:
    """
    Strict 12-circuit single-qubit process tomography in {Rz, SX} (IBM native) with
    explicit, commented construction of every PTM component.

    ─────────────────────────────────────────────────────────────────────────────
    PTM / affine Bloch map (ordering = (I, X, Y, Z)):
      A (possibly non-TP) single-qubit channel Φ acts on Bloch vectors as
          r' = t + T r
      with r, r' ∈ ℝ^3, t ∈ ℝ^3, T ∈ ℝ^{3×3}.
      The Pauli Transfer Matrix R (4×4, real in this idealized setting) is:

          R = [[ R00  R01  R02  R03 ],
               [ t_x  T_xx T_xy T_xz],
               [ t_y  T_yx T_yy T_yz],
               [ t_z  T_zx T_zy T_zz]]

      • Rows 1..3, Col 0  (i.e., R[1:,0])  = t  (affine shift)
      • Rows 1..3, Cols 1..3               = T  (linear 3×3 part)

      The **first row** encodes how the expectation of I transforms. For TP channels
      we have R[0,:] = [1, 0, 0, 0]. For non-TP (e.g., trace-decreasing/symmetry-
      violating dynamics), the first row deviates and must be reconstructed from
      “trace-after” signals (survival probabilities) s_p.

    ─────────────────────────────────────────────────────────────────────────────
    What the 12 circuits measure and how we assemble (t, T):
      Preps:  { Z+, Z−, X+, Y+ } ; Meas.: { Z, X, Y }.
      For each prep p we measure r_p := [⟨X⟩, ⟨Y⟩, ⟨Z⟩].

      We assemble
          t  = ½ ( r_{Z+} + r_{Z−} )
          T  = [ r_{X+} − t,  r_{Y+} − t,  r_{Z+} − t ]   (columns = X,Y,Z)

    ─────────────────────────────────────────────────────────────────────────────
    FIRST ROW (assume_tp == False) — where its *randomness* comes from:
      Let s_true := p0_true + p1_true = Tr(ρ_after_target). This is **basis-
      independent** (Z/X/Y give the same sum) and encodes trace preservation/
      loss due to the channel itself. In an experiment with N trials:

        • Surviving (detectable) trials are Binomial(N, q_detect),
          with q_detect = clip(s_true, 0, 1) × (1 − readout_loss).

        • We therefore draw a single n_eff for the whole prep p and reuse it for
          the Z/X/Y measurements of that prep. This produces first-row variability
          even when readout_loss=0, as long as s_true<1 (non-TP).

      From the four preps we reconstruct:
          s_{Z+} = a + c,   s_{Z−} = a − c,   s_{X+} = a + b,   s_{Y+} = a + d,
      hence
          a = ½ (s_{Z+} + s_{Z−}),  c = ½ (s_{Z+} − s_{Z−}),
          b = s_{X+} − a,           d = s_{Y+} − a,
      and set R[0,:] = [a, b, d, c].

      Continuity: as shots→∞, the sampled s_p converges to clip(s_true,0,1)×(1−loss),
      so the shot-based first row converges to the analytic first row that uses s_true.

    ─────────────────────────────────────────────────────────────────────────────
    SHOT NOISE and independent RNG streams
      • When shots>0 we sample ⟨P⟩ via Binomial draws *within the detected subspace*
        (size n_eff). That injects variability into r_p, hence into t and T.
      • First-row variability now comes from the **single Binomial draw of n_eff per
        prep** (survival), matching the physical picture.
      • For independent repeats we spawn child RNGs from the parent.

    Returns (default):
        R, t, T
    If component_stats=True and shots>0:
        R_mean, t_mean, T_mean, R_std   (std over repeats)
    If additionally return_standard_error=True:
        ... plus R_se = R_std / sqrt(repeats_for_stats)
    """

    def __init__(self, target_tokens=(("X",),)):
        self.target_tokens = list(target_tokens)

    # --- gate macros (IBM native) ------------------------------------------------
    @staticmethod
    def _H_tokens():
        # H = Rz(+π/2) · SX · Rz(+π/2) in left-multiply token order
        return [("Rz", np.pi/2), ("SX",), ("Rz", np.pi/2)]

    @classmethod
    def _meas_tokens(cls, label: str):
        L = label.upper()
        if L == "Z":
            return []                   # measure Z
        if L == "X":
            return cls._H_tokens()      # H • Z-meas ⇒ X-meas
        if L == "Y":
            # Y-meas = H · S† on the *state*; with left-multiply encode [Sdg, H]
            return [("Rz", -np.pi/2)] + cls._H_tokens()
        raise ValueError("meas ∈ {X, Y, Z}")

    @classmethod
    def _prep_tokens(cls, label: str):
        L = label.upper()
        if L == "Z+": return []                      # |0⟩
        if L == "Z-": return [("X",)]                # |1⟩ = X|0⟩
        if L == "X+": return cls._H_tokens()         # |+⟩ = H|0⟩
        if L == "Y+": return cls._H_tokens()+[("Rz", +np.pi/2)]  # |+i⟩ = S·H|0⟩
        raise ValueError("prep ∈ {Z+, Z-, X+, Y+}")

    # --- circuits builder --------------------------------------------------------
    def circuits(self, noise_map=None, amp_damp_p=None, phase_damp_lam=None):
        preps = ["Z+", "Z-", "X+", "Y+"]
        meass = ["Z", "X", "Y"]
        out = []
        for p in preps:
            P = self._prep_tokens(p)
            for m in meass:
                M = self._meas_tokens(m)
                tokens = list(P) + list(self.target_tokens) + list(M)
                out.append(Circuit(tokens=tokens,
                                   noise_map=noise_map,
                                   amp_damp_p=amp_damp_p,
                                   phase_damp_lam=phase_damp_lam))
        return out

    # --- tomography core ---------------------------------------------------------
    def ptm(self,
            shots=0,
            rng=None,
            assume_tp=False,
            noise_map=None,
            amp_damp_p=None,
            phase_damp_lam=None,
            verbose=False,
            readout_loss: float = 0.0,
            *,
            component_stats: bool = False,
            repeats_for_stats: int = 16,
            return_standard_error: bool = False):
        """
        Execute all 12 circuits; assemble R, t, T as described above.

        shots:
          0   → analytic expectations (no shot noise)
          >0  → Binomial sampling for expectations; we also Binomial-sample the
                per-prep survival n_eff from q_detect = clip(s_true,0,1)*(1-loss).

        readout_loss:
          probability of a trial producing a 'null' (on top of channel loss). This
          and the channel’s own survival both affect the **first row** through s_p.

        component_stats:
          if True and shots>0, repeat the whole 12-circuit set `repeats_for_stats`
          times with independent RNG streams; return means and component-wise std.
        """
        rng = np.random.default_rng() if rng is None else rng
        circs = self.circuits(noise_map=noise_map,
                              amp_damp_p=amp_damp_p,
                              phase_damp_lam=phase_damp_lam)

        def _run_one_R(rng_local):
            # Helper to evaluate one circuit and return (p0_true, s_true)
            def _true_probs(c: Circuit):
                U = c.unitary()
                Ks = c.kraus() or [I2]

                psi0 = np.array([1, 0], complex)[:, None]
                phi = U @ psi0
                rho = phi @ phi.conj().T
                rho = sum(K @ rho @ K.conj().T for K in Ks)  # may reduce trace if non-TP

                Pi0 = np.array([[1,0],[0,0]], complex)
                Pi1 = np.array([[0,0],[0,1]], complex)
                p0 = float(np.trace(Pi0 @ rho).real)
                p1 = float(np.trace(Pi1 @ rho).real)
                return p0, (p0 + p1)

            preps = ["Z+", "Z-", "X+", "Y+"]
            exps  = {}   # r_p vectors (⟨X⟩,⟨Y⟩,⟨Z⟩)
            svals = {}   # s_p scalars

            idx = 0
            for p in preps:
                cZ = circs[idx + 0]  # meas Z
                cX = circs[idx + 1]  # meas X
                cY = circs[idx + 2]  # meas Y
                idx += 3

                # True probabilities for each measurement axis
                p0Z, sZ_true = _true_probs(cZ)
                p0X, sX_true = _true_probs(cX)
                p0Y, sY_true = _true_probs(cY)

                # Basis-independence of survival: numeric guard via averaging+clip
                s_true = float(np.clip((sZ_true + sX_true + sY_true) / 3.0, 0.0, 1.0))

                if shots <= 0:
                    # Analytic path: expectations from true probs; first-row uses s_true
                    ez = p0Z - (1.0 - p0Z)
                    ex = p0X - (1.0 - p0X)
                    ey = p0Y - (1.0 - p0Y)
                    exps[p] = np.array([ex, ey, ez], float)
                    svals[p] = s_true
                    continue

                # === Shot-based path ===
                # Draw a SINGLE effective count n_eff for this prep (basis-independent survival).
                q_detect = s_true * max(0.0, min(1.0, 1.0 - float(readout_loss)))
                n_eff = rng_local.binomial(shots, max(0.0, min(1.0, q_detect)))

                if n_eff <= 0:
                    exps[p]  = np.array([0.0, 0.0, 0.0], float)
                    svals[p] = 0.0
                    continue

                # Within detected subspace, renormalize ± probability per axis
                def _axis_expect(p0_axis):
                    p_det = 0.0 if s_true <= 0.0 else (p0_axis / s_true)
                    p_det = max(0.0, min(1.0, p_det))
                    k = rng_local.binomial(n_eff, p_det)
                    return (k - (n_eff - k)) / float(n_eff)

                ez = _axis_expect(p0Z)
                ex = _axis_expect(p0X)
                ey = _axis_expect(p0Y)

                exps[p]  = np.array([ex, ey, ez], float)
                svals[p] = n_eff / float(shots)  # empirical survival fraction for this prep

            # Assemble affine blocks
            rZp, rZm, rXp, rYp = exps["Z+"], exps["Z-"], exps["X+"], exps["Y+"]
            t = 0.5 * (rZp + rZm)
            T = np.column_stack((rXp - t, rYp - t, rZp - t))

            # Build PTM
            R = np.zeros((4,4), float)
            R[1:, 0]  = t
            R[1:, 1:] = T

            if assume_tp:
                R[0, :] = [1.0, 0.0, 0.0, 0.0]
            else:
                # Reconstruct first row from s_p (one per prep)
                sZp = svals["Z+"]
                sZm = svals["Z-"]
                sXp = svals["X+"]
                sYp = svals["Y+"]

                a = 0.5 * (sZp + sZm)  # R[0,0]
                c = 0.5 * (sZp - sZm)  # R[0,3]
                b = sXp - a            # R[0,1]
                d = sYp - a            # R[0,2]
                R[0, :] = [a, b, d, c]

            if verbose:
                print("[ptm] t =", np.round(t, 6))
                print("[ptm] T =\n", np.round(T, 6))
                print("[ptm] R =\n", np.round(R, 6))
            return R, t, T

        if (not component_stats) or (shots <= 0):
            return _run_one_R(rng)

        # Stats over repeated shot draws (independent child RNGs)
        reps = int(max(1, repeats_for_stats))
        Rs, ts, Ts = [], [], []
        for _ in range(reps):
            child = np.random.default_rng(rng.integers(0, 2**32 - 1))
            Rk, tk, Tk = _run_one_R(child)
            Rs.append(Rk); ts.append(tk); Ts.append(Tk)

        Rs = np.stack(Rs, 0)
        ts = np.stack(ts, 0)
        Ts = np.stack(Ts, 0)

        R_mean = Rs.mean(axis=0)
        t_mean = ts.mean(axis=0)
        T_mean = Ts.mean(axis=0)
        R_std  = Rs.std(axis=0, ddof=1) if reps > 1 else np.zeros_like(R_mean)

        if return_standard_error and reps > 1:
            R_se = R_std / np.sqrt(reps)
            return R_mean, t_mean, T_mean, R_std, R_se
        else:
            return R_mean, t_mean, T_mean, R_std

    def self_test_signs(self):
        # quick sanity for an X target under TP reconstruction
        R, _, _ = self.ptm(shots=0, assume_tp=False)
        print("[self-test] ||R - diag(1,1,-1,-1)||_F =",
              np.linalg.norm(R - np.diag([1.0, 1.0, -1.0, -1.0]), ord="fro"))



# ──────────────────────────────────────────────────────────────
# 6) Process tomography (both paths support CPTP now)
# ──────────────────────────────────────────────────────────────
class ProcessTomography:
    """
    Single-qubit **process tomography** utilities in the Pauli–Liouville
    (Pauli-transfer) picture. This class provides two complementary
    constructions of the 4×4 PTM `R`, both capable of honoring an optional
    **post-circuit CPTP channel** when the input is a `Circuit` (rather than a
    bare 2×2 matrix):

      • `ptm_from_minimal_schedule(...)`  →  **Affine, shots-based** estimator
        that reconstructs the affine map (t, T) from a minimal set of 4×3
        PauliPrep/PauliMeas experiments (preparations: Z+, Z−, X+, Y+; measurements:
        X, Y, Z). Supports finite-shot sampling and an optional **non-TP** mode
        (when `assume_tp=False`) that reconstructs the full first row R[0,:]
        from “trace-after” statistics. Use this when you want to mimic real
        hardware data acquisition.

      • `ptm_from_heisenberg(...)`        →  **Heisenberg-exact** PTM of the
        same channel using the dual (Heisenberg) action on Pauli operators.
        This is analytic (noise-free) and convenient for validation and
        debugging.

    Inputs & supported modes
    ------------------------
    Both methods accept either:
      • `U_circuit` as a **2×2 ndarray**: treated as the coherent map only.
        No CPTP channel is applied.
      • `U_circuit` as a **Circuit**: we call `unitary()` for the coherent part
        and `kraus()` to fetch the optional post-circuit channel (amplitude and/or
        phase damping). This keeps **shots-based** and **Heisenberg-exact** paths
        consistent with the same physical assumptions.

    Conventions
    -----------
    • Pauli ordering is (I, X, Y, Z) for both rows and columns of R.
    • The affine decomposition uses:
          r_out = T r_in + t,
      where r_in/out are Bloch vectors and `t∈ℝ^3`, `T∈ℝ^{3×3}`. The PTM packs
      these into:
          R = [[1, 0, 0, 0],
               [t_x,   T_xx, T_xy, T_xz],
               [t_y,   T_yx, T_yy, T_yz],
               [t_z,   T_zx, T_zy, T_zz]]
      when `assume_tp=True`. If `assume_tp=False`, the first row `[R[0,:]]` is
      also reconstructed from measured traces (see below).

    Switches / knobs & their impact
    -------------------------------
    • `shots` (int): 0 ⇒ analytic expectations (no sampling). >0 ⇒ binomial
      sampling per setting (closer to hardware).
    • `assume_tp` (bool): If **True** (default), enforce trace preservation:
      fix `R[0,:] = [1,0,0,0]`. If **False**, estimate `R[0,:]` from
      “trace-after” statistics (sum of outcome probabilities) collected during
      the same minimal schedule. This is useful when leakage/loss or SPAM-like
      non-TP effects are present in your effective channel model.
    • `rng` (Generator): supply a seeded RNG to enable common-random-numbers
      variance reduction across candidates in optimization loops.
    • Passing a `Circuit` instead of a 2×2 matrix enables the optional
      post-circuit Kraus channel; passing a matrix disables it.

    Typical workflows
    -----------------
    • Use `ptm_from_minimal_schedule(..., shots>0)` inside optimization when
      you want the **comparator** to reflect finite-shot behavior of the
      tomography protocol you used on hardware.
    • Use `ptm_from_heisenberg(...)` for fast, noise-free diagnostics and to
      sanity-check the shape and scale of the map induced by your Δ-parameters.
    """

    @staticmethod
    def ptm_from_minimal_schedule(U_circuit: Union[np.ndarray, Circuit],
                                  shots: int = 0,
                                  rng: Optional[np.random.Generator] = None,
                                  assume_tp: bool = True,
                                  verbose: bool = False
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Minimal-schedule **affine** PTM reconstruction (t, T) with optional shots.

        Parameters
        ----------
        U_circuit : np.ndarray | Circuit
            • If ndarray (2×2): coherent map only; no CPTP channel applied.
            • If Circuit: use `unitary()` for coherent part and apply the
              post-circuit Kraus set from `kraus()` (if any) during measurement
              simulation.

        shots : int, default 0
            Number of shots per prep/measurement setting.
            0 ⇒ analytic expectations; >0 ⇒ Binomial sampling.

        rng : np.random.Generator, optional
            RNG for shot sampling. Provide a seeded generator for repeatability
            and for common-random-number tricks in optimization.

        assume_tp : bool, default True
            If True, enforce TP: set `R[0,:] = [1,0,0,0]`, reconstruct only
            the affine (t, T). If False, also reconstruct `R[0,:]` from the
            **trace-after** signal s = p0 + p1 measured in Z+, Z−, X+, Y+,
            which detects non-TP behavior (e.g., loss, leakage).

        verbose : bool, default False
            Print `t`, `T`, and the combined `R`.

        Returns
        -------
        R : (4,4) ndarray
            The reconstructed Pauli-transfer matrix (rows/cols in I,X,Y,Z).

        t : (3,) ndarray
            The translation vector.

        T : (3,3) ndarray
            The 3×3 linear part mapping Bloch vectors.

        Notes on estimation
        -------------------
        • Preparations used: Z+, Z−, X+, Y+ (four states sufficient for an
          **affine** single-qubit map).
        • Measurements used: X, Y, Z (three Pauli axes).
        • For each prep p, we record r_p = [⟨X⟩, ⟨Y⟩, ⟨Z⟩]ᵀ after the channel.
          Then:
              t = ½ (r_{Z+} + r_{Z−})
              T = [ r_{X+} − t,  r_{Y+} − t,  r_{Z+} − t ]  (as column stack)
        • In non-TP mode, the first row is reconstructed via:
              a = ½ (s_{Z+} + s_{Z−}) = R[0,0]
              c = ½ (s_{Z+} − s_{Z−}) = R[0,3]
              b = s_{X+} − a         = R[0,1]
              d = s_{Y+} − a         = R[0,2]
          where s_p = p0 + p1 is the total probability of getting a valid
          outcome in the rotated Z-basis for prep p (identical across X/Y/Z
          measurement axes in this construction).
        """
        # Decode input into a coherent matrix U and an optional post-circuit Kraus set Ks.
        if isinstance(U_circuit, np.ndarray):
            U = U_circuit        # bare matrix path: no CPTP channel
            Ks = None
        else:
            U  = U_circuit.unitary()   # coherent part from the Circuit
            Ks = U_circuit.kraus()     # optional post-circuit CPTP channel

        # RNG for shots (analytic if shots==0 inside run_one_schedule)
        rng = np.random.default_rng() if rng is None else rng

        # Minimal set of preparations / measurements for affine tomography
        preps = ["Z+", "Z-", "X+", "Y+"]
        meass = ["X", "Y", "Z"]

        # Collect expectation vectors r_p = [⟨X⟩,⟨Y⟩,⟨Z⟩] for all preps p
        exps: Dict[str, np.ndarray] = {}
        for p in preps:
            exps[p] = np.array(
                [run_one_schedule(U, p, m, shots=shots, rng=rng, kraus=Ks)[0] for m in meass],
                float
            )

        # Unpack the four Bloch vectors
        rZp, rZm, rXp, rYp = exps["Z+"], exps["Z-"], exps["X+"], exps["Y+"]

        # Affine reconstruction (TP form): r_out = T r_in + t
        t = 0.5 * (rZp + rZm)                              # translation
        T = np.column_stack((rXp - t, rYp - t, rZp - t))   # columns: X, Y, Z inputs

        # Assemble PTM in Pauli ordering (I,X,Y,Z).
        R = np.zeros((4,4), float)
        R[1:, 0]  = t
        R[1:, 1:] = T
        if assume_tp:
            # Enforce TP (first row fixed):  Tr[E(ρ)] = Tr[ρ]
            R[0, :] = [1.0, 0.0, 0.0, 0.0]
        else:
            # Non-TP mode: reconstruct the entire first row R[0,:] from trace-after data.
            # We re-sample expectations and also extract s = p0+p1 for each prep (X+,Y+,Z±).
            def _exp_and_trace(prep):
                exps = [run_one_schedule(U, prep, m, shots=shots, rng=rng, kraus=Ks) for m in meass]
                # (exp, p0, p1); trace s = p0 + p1 (identical across axes by construction)
                _, p0, p1 = exps[0]
                s = float(p0 + p1)
                r = np.array([e[0] for e in exps], float)
                return r, s

            rZp, sZp = _exp_and_trace("Z+")
            rZm, sZm = _exp_and_trace("Z-")
            rXp, sXp = _exp_and_trace("X+")
            rYp, sYp = _exp_and_trace("Y+")

            t = 0.5 * (rZp + rZm)
            T = np.column_stack((rXp - t, rYp - t, rZp - t))  # ← NOT 0.5*(rZp - rZm)

            R = np.zeros((4, 4), float)
            R[1:, 0] = t
            R[1:, 1:] = T

            if assume_tp:
                # (Kept for structural symmetry; this branch won't execute because we’re in the else)
                R[0, :] = [1.0, 0.0, 0.0, 0.0]
            else:
                # First row from trace-after signals:
                #   a = R[0,0], b = R[0,1], d = R[0,2], c = R[0,3]
                a = 0.5 * (sZp + sZm)  # average trace on Z± preps ⇒ R[0,0]
                c = 0.5 * (sZp - sZm)  # imbalance on Z±            ⇒ R[0,3]
                b = sXp - a            # trace for X+ minus a       ⇒ R[0,1]
                d = sYp - a            # trace for Y+ minus a       ⇒ R[0,2]
                R[0, :] = [a, b, d, c]

        if verbose:
            print("\n┌─ PROCESS-TOMOGRAPHY (minimal affine) ───────────")
            print("│ t =", np.round(t, 6))
            print("│ T =\n", np.round(T, 6))
            print("│ R (rows/cols I,X,Y,Z) =\n", np.round(R, 6))
            print("└────────────────────────────────────────────────\n")

        return R, t, T

    @staticmethod
    def ptm_from_heisenberg(U_circuit: Union[np.ndarray, Circuit]) -> np.ndarray:
        """
        Analytic Pauli-transfer matrix via **Heisenberg dual** action.

        We compute the real matrix R with entries:
            R[k, j] = ½ Tr[ P_k · E*(P_j) ],
        where {P_j} = {I, X, Y, Z} and E* is the Heisenberg dual of the channel
        consisting of the coherent map U and an optional post-circuit CPTP
        layer with Kraus {K_i}. Specifically:
            E(ρ)   =  (∑_i K_i · (U ρ U†) · K_i†)
            E*(P)  =  U† ( ∑_i K_i† P K_i ) U
        Passing a bare 2×2 matrix disables the CPTP layer (i.e., {K_i} = {I}).

        Returns
        -------
        R : (4,4) ndarray
            Heisenberg-exact PTM in the (I,X,Y,Z) basis. By construction R[0,0]=1.

        When to use
        -----------
        • As a **ground truth** comparator for debugging or for optimization
          runs that should ignore sampling noise.
        • To verify qualitative structure (e.g., off-diagonal couplings) before
          involving the minimal-schedule estimator with shots.
        """
        # Decode input: coherent part U; Kraus set Ks (identity if not provided).
        if isinstance(U_circuit, np.ndarray):
            U = U_circuit
            Ks = [I2]                 # no post-circuit channel in bare-matrix mode
        else:
            U = U_circuit.unitary()
            Ks = U_circuit.kraus() or [I2]

        # Allocate the PTM and apply the Heisenberg dual action on Pauli basis
        R = np.empty((4,4), float)
        Udag = U.conj().T
        for k, Pk in enumerate(PAULIS):           # output axis (rows)
            for j, Pj in enumerate(PAULIS):       # input axis (cols)
                # E*(Pj) = U† ( Σ_i K_i† Pj K_i ) U
                inner = sum(K.conj().T @ Pj @ K for K in Ks)
                EPj = Udag @ inner @ U
                R[k, j] = 0.5 * np.trace(Pk @ EPj).real

        # Enforce the PTM convention R[0,0]=1 (exactly 1 in theory; guard for num. drift)
        R[0,0] = 1.0
        return R
