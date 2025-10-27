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
amplitude damping (probability p) and/or phase damping (lambda λ).
Both are OFF by default (None), preserving legacy behavior.

• Heisenberg-exact PTM and shot-based affine tomography both honor the
  optional damping when provided.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# SciPy for matrix exponential
try:
    import scipy.linalg as la
except Exception as exc:  # pragma: no cover
    raise ImportError("This module requires SciPy. Please `pip install scipy`.") from exc

# ──────────────────────────────────────────────────────────────
# 0) Pauli primitives, basis, and small helpers
# ──────────────────────────────────────────────────────────────
σx = np.array([[0, 1], [1, 0]], complex)
σy = np.array([[0, -1j], [1j,  0]], complex)
σz = np.array([[1,  0], [0, -1]], complex)
I2  = np.eye(2, dtype=complex)

PAULIS: List[np.ndarray] = [I2, σx, σy, σz]   # order I,X,Y,Z

# Preparation kets (eigenstates of Pauli axes)
_PREP: Dict[str, np.ndarray] = {
    "Z+": np.array([1, 0], complex),                  # |0>
    "Z-": np.array([0, 1], complex),                  # |1>
    "X+": 1/np.sqrt(2) * np.array([1,  1], complex),  # |+>
    "X-": 1/np.sqrt(2) * np.array([1, -1], complex),  # |->
    "Y+": 1/np.sqrt(2) * np.array([1,  1j], complex), # |+i>
    "Y-": 1/np.sqrt(2) * np.array([1, -1j], complex), # |-i>
}

def _proj_plus(P: np.ndarray) -> np.ndarray:
    """Projector onto the +1 eigenspace of Pauli P (P ∈ {X,Y,Z})."""
    return 0.5 * (I2 + P)

def _binomial_expectation(p_plus: float, shots: int, rng: np.random.Generator) -> float:
    """Return empirical ⟨P⟩ from Binomial samples (p_plus = Pr[outcome=+1])."""
    p_plus = float(min(1.0, max(0.0, p_plus)))
    if shots <= 0:
        return 2.0 * p_plus - 1.0
    n_plus = rng.binomial(shots, p_plus)
    return (n_plus - (shots - n_plus)) / shots

# ──────────────────────────────────────────────────────────────
# 0.5) CPTP channels (Kraus) – optional post-circuit layer
# ──────────────────────────────────────────────────────────────
def kraus_amplitude_damping(p: float) -> List[np.ndarray]:
    """
    Amplitude damping with probability p (|1>→|0> relaxation).
    K0 = [[1, 0], [0, sqrt(1-p)]],  K1 = [[0, sqrt(p)], [0, 0]]
    """
    p = float(p)
    p = 0.0 if p < 0 else (1.0 if p > 1 else p)
    K0 = np.array([[1.0, 0.0],[0.0, np.sqrt(1.0 - p)]], complex)
    K1 = np.array([[0.0, np.sqrt(p)],[0.0, 0.0]], complex)
    return [K0, K1]

def kraus_phase_damping(lam: float) -> List[np.ndarray]:
    """
    Phase (pure dephasing) damping with strength λ (0≤λ≤1).
    This 2-Kraus form damps coherences while preserving populations:
      K0 = diag(1, sqrt(1-λ)),  K1 = diag(0, sqrt(λ))
    """
    lam = float(lam)
    lam = 0.0 if lam < 0 else (1.0 if lam > 1 else lam)
    K0 = np.array([[1.0, 0.0],[0.0, np.sqrt(1.0 - lam)]], complex)
    K1 = np.array([[0.0, 0.0],[0.0, np.sqrt(lam)]], complex)
    return [K0, K1]

def compose_kraus(As: List[np.ndarray], Bs: List[np.ndarray]) -> List[np.ndarray]:
    """Sequential composition:  ρ → Σ_j B_j ( Σ_i A_i ρ A_i† ) B_j†  ⇒ {B_j A_i}."""
    return [B @ A for B in Bs for A in As]

# ──────────────────────────────────────────────────────────────
# 1) Gates from generators, with per-gate additive Δ inside expm
# ──────────────────────────────────────────────────────────────
class Gate:
    """
    Gate factory – each method returns a 2×2 complex ndarray.

    All gates are generated from their su(2) generators via:
        U(θ, Δ) = exp[ −i * (θ/2) * (G + Δ) ]
    """

    @staticmethod
    def _exp_rot(G: np.ndarray, angle: float, delta: Optional[np.ndarray]) -> np.ndarray:
        Geff = G if delta is None else G + delta
        return la.expm(-1j * (angle / 2.0) * Geff)

    # Pauli-axis rotations
    @staticmethod
    def Rx(theta: float, noise: Optional[np.ndarray] = None) -> np.ndarray:
        return Gate._exp_rot(σx, theta, noise)

    @staticmethod
    def Ry(theta: float, noise: Optional[np.ndarray] = None) -> np.ndarray:
        return Gate._exp_rot(σy, theta, noise)

    @staticmethod
    def Rz(phi: float, noise: Optional[np.ndarray] = None) -> np.ndarray:
        return Gate._exp_rot(σz, phi, noise)

    # Derived single-qubit gates
    @staticmethod
    def H(noise: Optional[np.ndarray] = None) -> np.ndarray:
        n = (σx + σz) / np.sqrt(2.0)
        return Gate._exp_rot(n, np.pi, noise)

    @staticmethod
    def X(noise: Optional[np.ndarray] = None) -> np.ndarray:
        return Gate.Rx(np.pi, noise)

    @staticmethod
    def Y(noise: Optional[np.ndarray] = None) -> np.ndarray:
        return Gate.Ry(np.pi, noise)

    @staticmethod
    def Z(noise: Optional[np.ndarray] = None) -> np.ndarray:
        return Gate.Rz(np.pi, noise)

    @staticmethod
    def SX(noise: Optional[np.ndarray] = None) -> np.ndarray:
        return Gate.Rx(+np.pi/2, noise)

    @staticmethod
    def SXdg(noise: Optional[np.ndarray] = None) -> np.ndarray:
        return Gate.Rx(-np.pi/2, noise)

    @staticmethod
    def S(noise: Optional[np.ndarray] = None) -> np.ndarray:
        return Gate.Rz(+np.pi/2, noise)

    @staticmethod
    def Sdg(noise: Optional[np.ndarray] = None) -> np.ndarray:
        return Gate.Rz(-np.pi/2, noise)

    @staticmethod
    def T(noise: Optional[np.ndarray] = None) -> np.ndarray:
        return Gate.Rz(+np.pi/4, noise)

    @staticmethod
    def Tdg(noise: Optional[np.ndarray] = None) -> np.ndarray:
        return Gate.Rz(-np.pi/4, noise)

    @staticmethod
    def U(theta: float, phi: float, lam: float, noise: Optional[np.ndarray] = None) -> np.ndarray:
        return Gate.Rz(phi) @ Gate.Ry(theta) @ Gate.Rz(lam, noise)

# ──────────────────────────────────────────────────────────────
# 2) Circuit composer with optional post-circuit damping
# ──────────────────────────────────────────────────────────────
@dataclass
class Circuit:
    """
    Minimal single-qubit circuit composer.

    Parameters
    ----------
    tokens : Sequence[tuple]
        Gate sequence in diagram order.
    noise_map : dict[str, np.ndarray], optional
        Map from gate names to additive generator noise matrices Δ.
    amp_damp_p : Optional[float]
        Amplitude-damping probability p (None ⇒ OFF).
    phase_damp_lam : Optional[float]
        Phase-damping strength λ (None ⇒ OFF).
    """
    tokens: Sequence[tuple]
    noise_map: Optional[Dict[str, np.ndarray]] = None
    amp_damp_p: Optional[float] = None
    phase_damp_lam: Optional[float] = None

    def unitary(self, verbose: bool = False) -> np.ndarray:
        """Return the product matrix U = ∏ U_k from left to right."""
        U_tot = I2
        nm = self.noise_map or {}

        if verbose:
            print("┌── circuit assembly ─────────────────────────────")
        for i, spec in enumerate(self.tokens, 1):
            name, *pars = spec
            if not hasattr(Gate, name):
                raise ValueError(f"Unknown gate name: {name}")
            delta = nm.get(name)
            Uk = getattr(Gate, name)(*pars, noise=delta)
            U_tot = Uk @ U_tot

            if verbose:
                pstr = ", ".join(f"{p/np.pi:.3g}·π" if isinstance(p, (int,float)) else str(p) for p in pars) or "∅"
                print(f"│ step {i:>2}:  {name}({pstr})")
                print("│ U_k =\n", np.round(Uk, 6))
                print("│ running product =\n", np.round(U_tot, 6))
                print("├─────────────────────────────────────────────")
        if verbose:
            print("└── end assembly ─────────────────────────────\n")
        return U_tot

    def kraus(self) -> List[np.ndarray]:
        """
        Return the (possibly empty) post-circuit Kraus set according to
        amp_damp_p and phase_damp_lam. None ⇒ OFF.
        Composition order (after the unitary): amplitude → phase.
        """
        Ks: List[np.ndarray] = [I2]  # identity (no channel)
        if self.amp_damp_p is not None:
            Ks = kraus_amplitude_damping(self.amp_damp_p)
        if self.phase_damp_lam is not None:
            Ps = kraus_phase_damping(self.phase_damp_lam)
            Ks = compose_kraus(Ks, Ps)
        return Ks

# ──────────────────────────────────────────────────────────────
# 3) PauliPrep and PauliMeas schedules (Qiskit-style)
# ──────────────────────────────────────────────────────────────
def prep_unitary(label: str) -> np.ndarray:
    l = label.upper()
    if l == "Z+": return I2
    if l == "Z-": return Gate.X()
    if l == "X+": return Gate.H()
    if l == "Y+": return Gate.S() @ Gate.H()
    raise ValueError("prep ∈ {Z+, Z-, X+, Y+}")

def meas_unitary(label: str) -> np.ndarray:
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
    """
    Evaluate a single PauliPrep/PauliMeas schedule with optional post-unitary CPTP.

    Steps: |0> → U_prep → U_under_test → (optional channel) → U_meas → Z-measurement.
    """
    rng = np.random.default_rng() if rng is None else rng

    Up = prep_unitary(prep_label)
    Um = meas_unitary(meas_label)

    psi0 = np.array([1,0], complex)[:,None]
    psi  = Up @ psi0
    phi  = U_under_test @ psi
    rho  = phi @ phi.conj().T

    # Optional post-unitary channel
    if kraus is not None:
        rho = sum( K @ rho @ K.conj().T for K in kraus )

    # rotate to Z and compute outcome probabilities
    rho_m = Um @ rho @ Um.conj().T
    Pi0 = np.array([[1,0],[0,0]], complex)
    Pi1 = np.array([[0,0],[0,1]], complex)
    p0  = float(np.trace(Pi0 @ rho_m).real)
    p1  = float(np.trace(Pi1 @ rho_m).real)

    # expectation of chosen Pauli (p_plus = p0 in the rotated basis)
    exp = _binomial_expectation(p0, shots, rng)
    return exp, p0, p1

# ──────────────────────────────────────────────────────────────
# 5) State tomography (unchanged API; now honors CPTP if Circuit)
# ──────────────────────────────────────────────────────────────
class StateTomography:
    @staticmethod
    def linear_inversion(U_circuit: Union[np.ndarray, Circuit],
                         shots: int = 0,
                         rng: Optional[np.random.Generator] = None,
                         verbose: bool = False
                         ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(U_circuit, np.ndarray):
            U = U_circuit
            Ks = None
        else:
            U  = U_circuit.unitary()
            Ks = U_circuit.kraus()

        rx, _, _ = run_one_schedule(U, "Z+", "X", shots=shots, rng=rng, kraus=Ks)
        ry, _, _ = run_one_schedule(U, "Z+", "Y", shots=shots, rng=rng, kraus=Ks)
        rz, _, _ = run_one_schedule(U, "Z+", "Z", shots=shots, rng=rng, kraus=Ks)
        r = np.array([rx, ry, rz], float)
        rho = 0.5 * (I2 + r[0]*σx + r[1]*σy + r[2]*σz)

        if verbose:
            purity = float(np.trace(rho @ rho).real)
            print("\n┌─ STATE-TOMOGRAPHY ──────────────────────────────")
            print("│ r =", np.round(r, 6))
            print("│ ρ̂ =\n", np.round(rho, 6))
            print("│ purity Tr[ρ̂²] =", f"{purity:.6f}")
            print("└────────────────────────────────────────────────\n")
        return rho, r

# ──────────────────────────────────────────────────────────────
# 6) Process tomography (both paths support CPTP now)
# ──────────────────────────────────────────────────────────────
class ProcessTomography:
    @staticmethod
    def ptm_from_minimal_schedule(U_circuit: Union[np.ndarray, Circuit],
                                  shots: int = 0,
                                  rng: Optional[np.random.Generator] = None,
                                  assume_tp: bool = True,
                                  verbose: bool = False
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(U_circuit, np.ndarray):
            U = U_circuit
            Ks = None
        else:
            U  = U_circuit.unitary()
            Ks = U_circuit.kraus()

        rng = np.random.default_rng() if rng is None else rng
        preps = ["Z+", "Z-", "X+", "Y+"]
        meass = ["X", "Y", "Z"]

        exps: Dict[str, np.ndarray] = {}
        for p in preps:
            exps[p] = np.array([run_one_schedule(U, p, m, shots=shots, rng=rng, kraus=Ks)[0] for m in meass], float)

        rZp, rZm, rXp, rYp = exps["Z+"], exps["Z-"], exps["X+"], exps["Y+"]

        t = 0.5 * (rZp + rZm)
        T = np.column_stack((rXp - t, rYp - t, rZp - t))

        R = np.zeros((4,4), float)
        R[1:, 0]  = t
        R[1:, 1:] = T
        if assume_tp:
            R[0, :] = [1.0, 0.0, 0.0, 0.0]
        else:
            # collect BOTH the expectations and (p0+p1) traces for Z+, Z-, X+, Y+
            def _exp_and_trace(prep):
                exps = [run_one_schedule(U, prep, m, shots=shots, rng=rng, kraus=Ks) for m in meass]
                # each entry is (exp, p0, p1); the “trace after” is p0+p1 (same for any meas-basis)
                _, p0, p1 = exps[0]
                s = float(p0 + p1)
                r = np.array([e[0] for e in exps], float)
                return r, s

            rZp, sZp = _exp_and_trace("Z+")
            rZm, sZm = _exp_and_trace("Z-")
            rXp, sXp = _exp_and_trace("X+")
            rYp, sYp = _exp_and_trace("Y+")

            t = 0.5 * (rZp + rZm)
            T = np.column_stack((rXp - t, rYp - t, rZp - t))

            R = np.zeros((4, 4), float)
            R[1:, 0] = t
            R[1:, 1:] = T

            if assume_tp:
                R[0, :] = [1.0, 0.0, 0.0, 0.0]
            else:
                a = 0.5 * (sZp + sZm)  # R[0,0]
                c = 0.5 * (sZp - sZm)  # R[0,3]
                b = sXp - a  # R[0,1]
                d = sYp - a  # R[0,2]
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
        Exact Pauli-transfer by Heisenberg action for a general CPTP channel:
          E(ρ) = ( post-channel ) ◦ ( ρ ↦ U ρ U† )
        Dual map on observables:
          E*(P) = U† ( Σ_i K_i† P K_i ) U
        where {K_i} is identity if no channel is provided.
        """
        if isinstance(U_circuit, np.ndarray):
            U = U_circuit
            Ks = [I2]
        else:
            U = U_circuit.unitary()
            Ks = U_circuit.kraus() or [I2]

        R = np.empty((4,4), float)
        Udag = U.conj().T
        for k, Pk in enumerate(PAULIS):
            for j, Pj in enumerate(PAULIS):
                # Heisenberg dual: Pj → U† ( Σ K† Pj K ) U
                inner = sum(K.conj().T @ Pj @ K for K in Ks)
                EPj = Udag @ inner @ U
                R[k, j] = 0.5 * np.trace(Pk @ EPj).real
        R[0,0] = 1.0
        return R

# ──────────────────────────────────────────────────────────────
# 7) Small demo when run as a script (unchanged)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("### Analytic tomography demo ###\n")
    tokens = [("Rz", np.pi/2), ("SX",), ("Rz", np.pi/2),
              ("Rz", np.pi/2), ("SX",), ("Rz", np.pi/2)]
    circ   = Circuit(tokens)
    U      = circ.unitary(verbose=True)
    Δz   = 0.02 * σz
    Δsx  = 0.015 * σy
    circ_noisy = Circuit(tokens, noise_map={"Rz": Δz, "SX": Δsx})
    U_noisy    = circ_noisy.unitary(verbose=True)
    from math import pi
    # State tomography (exact and sampled)
    from pprint import pprint
    rho_ex, r_ex = StateTomography.linear_inversion(U, shots=0, verbose=True)
    rho_sh, r_sh = StateTomography.linear_inversion(circ_noisy, shots=8192, verbose=True)
    # Process tomography (affine, TP assumed) + exact Heisenberg check
    R_aff, t, T = ProcessTomography.ptm_from_minimal_schedule(circ_noisy, shots=4096, verbose=True)
    R_hex       = ProcessTomography.ptm_from_heisenberg(circ_noisy)
    print("Heisenberg R (exact):\n", np.round(R_hex, 6))
    print("\nDone.")
