#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
Genetic-Algorithm fitting of per-gate generator noise to a target PTM
================================================================================

WHAT THIS SCRIPT IS FOR
-----------------------
You use this module as the *configuration + constants* block for a GA that fits
per-gate non-Hermitian generator perturbations (Δ_g ∈ ℂ^{2×2}) so that a
simulated process (via your tomography toolkit) matches a *hardware* process
tomography matrix R_target (4×4 Pauli–Liouville / Heisenberg PTM).

The rest of your pipeline (not shown here) will:
  • Build a circuit from these settings,
  • Map a candidate Δ-map to a circuit and evaluate a model PTM R_fit(Δ),
  • Score R_fit(Δ) vs R_target, and
  • Evolve genomes w (latent params for Δ) with a GA until the mismatch is small.

HOW TO READ THE KNOBS
---------------------
• TARGET_DIR / OUT_DIR : where the experimental PTMs are read from and results are written to.
• SEED : reproducibility for RNGs used in GA init/mutations and (optionally) shot-based sims.
• GA hyperparameters (population size, generations, selection, crossover, mutation):
    - Larger populations or more generations improve exploration but increase runtime.
    - ELITISM preserves the best individuals each generation; too large may reduce diversity.
    - SELECTION_TOP_FRAC controls selection pressure; smaller → greedier, larger → more diverse.
    - CROSSOVER_PROB=1.0 means always mix genes uniformly; <1.0 sometimes copies a parent.
    - MUTATION_* knobs shape exploration radius and the probability of full re-seeding.
• SHOTS_PER_EVAL / REEVALS : if you run a shot-based comparator, these set sampling depth and
  how many repeats you average for final summary stats (end-of-run diagnostics).
• USE_WEIGHTED / ERROR_AMPLIFICATION / thresholds : cost shaping. With R_std you can do χ²
  weighting; with amplification you up-weight small-magnitude target entries to match delicate
  components more tightly.
• NOISY_GATES : *ordered* list of gate labels that receive their own Δ_g (8 real dof each:
  Re/Im for each of the 2×2 entries). The order here defines the packing/unpacking layout.
• DELTA_MAX / W_MAX : per-entry bounds for Δ via Δ = DELTA_MAX * tanh(w). Smaller DELTA_MAX
  restricts search to gentler perturbations; W_MAX is a hard clamp in latent w-space.
• SEED_DELTA_MAP / USE_SEED : optionally inject a hand-tuned Δ-map as the first genome.

TYPICAL SCENARIOS
-----------------
• If your affine (shots) estimator looked under-noisy or biased:
    - Increase SHOTS_PER_EVAL to reduce sampling noise.
    - Use ASSUME_TP_AFFINE=True to enforce trace preservation if your target was TP-projected.
• If GA stalls or overfits:
    - Increase MUTATION_SIGMA_W or MUTATION_FRAC_MAX for more exploration.
    - Increase population or generations; or reduce ERROR_AMPLIFICATION.
• If fitted Δ looks too large to be physical:
    - Reduce DELTA_MAX to keep generators in a tighter ball around 0.

NOTES
-----
This file intentionally contains *no executable GA logic*, just the values your
pipeline reads. Additions below are comments/docstrings only; no behavior change.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Standard library & third-party imports
# -----------------------------------------------------------------------------
from pathlib import Path         # filesystem paths for inputs/outputs
import sys                       # only used for early error messages
import json                      # report metadata (e.g., writing settings)
import numpy as np               # numerics, array ops, PTM handling
import matplotlib.pyplot as plt  # plotting utilities used by the pipeline

# -----------------------------------------------------------------------------
# Tomography toolkit imports (your project)
# - Circuit: builds an evaluation circuit from a gate list + per-gate Δ noise.
# - ProcessTomography: returns PTMs via Heisenberg-exact and/or shot-based routes.
# -----------------------------------------------------------------------------
from A2025_class_tomographies_damping_3 import Circuit, ProcessTomography

# =============================================================================
# I/O PATHS
# =============================================================================
TARGET_DIR  = Path("unique_process_tomos")                # folder with R_avg.npy (+ optional R_std.npy)
OUT_DIR     = Path("fit_results_ga_1000shots_10kgens_new_shots_max0p1")  # where artifacts will be saved

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED        = 2025   # set the RNG seed used throughout GA and (optionally) sampling

# =============================================================================
# GENETIC ALGORITHM HYPERPARAMETERS
# =============================================================================
N_POPULATION        = 2000      # number of candidate genomes per generation
N_GENERATIONS_old   = 10000      # legacy placeholder (kept for run comparability)
N_GENERATIONS       = 10000     # number of generations to evolve
ELITISM             = 1          # how many top genomes are copied unchanged each generation
SELECTION_TOP_FRAC  = 0.5        # fraction of population eligible for parent selection
CROSSOVER_PROB      = 1.0        # probability of performing crossover (1.0 = always)
MUTATION_ON         = True       # global on/off switch for mutations
MUTATION_FRAC_MAX   = 0.33333333 # max fraction of non-elite individuals mutated per generation
MUTATION_SIGMA_W    = 0.15       # Gaussian step size in latent w-space for local jitters
MUTATION_REPLACE_PROB = 0.25     # probability to hard-reset a chosen individual (diversity injection)
ASSUME_TP_AFFINE     = False     # if True, enforce trace preservation in affine estimator
COST_SHOT_REPEATS    = 10         # evaluate cost as the mean of this many shot repeats (variance reduction)

# Practical guidance:
# • Larger N_POPULATION or N_GENERATIONS → better search but slower.
# • If convergence is too slow/greedy, tweak SELECTION_TOP_FRAC and MUTATION_*.
# • If your measured R_target was TP-projected in post-processing, set ASSUME_TP_AFFINE=True
#   to compare apples-to-apples during shot-based cost evaluation.

# =============================================================================
# SHOT-BASED EVALUATION SETTINGS (used when you choose the affine comparator)
# =============================================================================
SHOTS_PER_EVAL      = 10000   # number of measurement shots per tomography setting
REEVALS             = 100000  # how many repeats to compute mean±std for final reporting

# Tips:
# • Increase SHOTS_PER_EVAL to reduce sampling noise (cost becomes smoother).
# • Increase REEVALS for tighter end-of-run statistics (longer runtime).

# =============================================================================
# COST SHAPING / WEIGHTING
# =============================================================================
USE_WEIGHTED                 = False  # if True, use χ² weights 1/σ^2 from R_std.npy (when available)
ERROR_AMPLIFICATION          = True   # emphasize entries with small |R_target| (often most revealing)
AMPLIFICATION_THRESHOLD      = 0.05   # magnitude threshold below which to up-weight
AMP_FACTOR                   = 7.0    # multiplicative up-weighting factor

# Behavior:
# • USE_WEIGHTED needs R_std.npy with same shape as R_avg.npy.
# • ERROR_AMPLIFICATION helps fit near-zero matrix elements (e.g., leakage/coherence hints).
# • If costs explode, try turning amplification off or lowering AMP_FACTOR.

# =============================================================================
# GATE INVENTORY (ORDER MATTERS)
# =============================================================================
NOISY_GATES = ("Rx", "Rz", "H", "SX", "S")
# Each gate g receives its own 2×2 complex Δ_g; genomes pack gates in this order with
# 8 real parameters per gate (Re/Im of each matrix entry in row-major order).

# =============================================================================
# REFERENCE CHANNELS / FORMATTING
# =============================================================================
R_X = np.diag([1.0, 1.0, -1.0, -1.0])   # Pauli-Liouville PTM for the ideal X gate (diag(1,1,−1,−1))

np.set_printoptions(precision=5, suppress=True)  # compact, readable matrix printing in logs

# =============================================================================
# Δ PARAMETERIZATION & SEARCH BOX
# =============================================================================
DELTA_MAX_og     = 0.30  # historical bound (kept for comparison across runs)
DELTA_MAX        = 0.1   # active bound: Δ = DELTA_MAX * tanh(w)  (per real/imag component)
W_MAX            = 6.0   # hard clamp limit on latent coordinates w

# Impact:
# • Smaller DELTA_MAX confines Δ near 0 → “gentler” perturbations, safer numerics.
# • Increase DELTA_MAX if fit can’t reach target; decrease if Δ looks unphysical.
# • W_MAX is rarely hit because tanh already bounds Δ, but protects against extreme w values.

# =============================================================================
# OPTIONAL SEEDING OF THE POPULATION WITH A HAND-TUNED Δ-MAP
# =============================================================================
USE_SEED = False  # if True, the first genome is set from SEED_DELTA_MAP (converted to latent w)
SEED_DELTA_MAP = {
    "Rx": np.array([[ 0.29512-0.02885j,  0.00121-0.01125j],
                    [-0.00202+0.00956j,  0.29921-0.03285j]], complex),
    "Rz": np.array([[ 0.1078 -0.1895j ,  0.014  -0.29586j],
                    [-0.27143-0.29796j, -0.13132+0.29537j]], complex),
    "H":  np.array([[-0.29963+0.0554j ,  0.2991 -0.05317j],
                    [-0.29356+0.28569j, -0.26439-0.28756j]], complex),
    "SX": np.array([[ 0.13218-0.15033j,  0.29573-0.29833j],
                    [ 0.27451+0.2982j ,  0.10901-0.04919j]], complex),
    "S":  np.array([[-0.2488 -0.18133j, -0.29469+0.08011j],
                    [-0.27822+0.27014j,  0.28417+0.26363j]], complex),
}
# Notes:
# • The dictionary keys must match NOISY_GATES labels; each value is a 2×2 complex matrix.
# • These Δ values are direct generator perturbations (not yet mapped through tanh); the
#   pipeline will convert them into latent coordinates w via atanh(Δ/DELTA_MAX) (with safety
#   clipping). If any entry exceeds the current DELTA_MAX magnitude, conversion will clip.
# -----------------------------------------------------------------------------
# Comparator switch used by the outer GA cost:
#   • FIT_WITH_SHOTS = False → use analytic, Heisenberg-exact PTM (no sampling)
#   • FIT_WITH_SHOTS = True  → use shot-based affine PTM estimator (finite shots)
#
# Context:
# - This flag does NOT change parameterization or the search space; it only
#   selects which *comparator* produces R_fit for a candidate Δ-map.
# - Typical usage patterns:
#     • Start with Heisenberg-exact for stability/speed.
#     • Switch to shots once you're satisfied and want to mimic hardware stats.
# -----------------------------------------------------------------------------
FIT_WITH_SHOTS = True


def _vec_to_delta_entries(w8: np.ndarray) -> np.ndarray:
    """
    Convert a length-8 real vector in latent w-space into a single gate’s 2×2
    complex generator perturbation Δ.

    Mapping & bounds
    ----------------
    Each real coordinate w_i is squashed via tanh and scaled by DELTA_MAX:
        v_i = DELTA_MAX * tanh(w_i)
    This guarantees every real/imag part of Δ lies in [−DELTA_MAX, +DELTA_MAX].
    The 8 numbers v_i are interpreted (row-major) as:
        [Re(Δ_00), Im(Δ_00), Re(Δ_01), Im(Δ_01),
         Re(Δ_10), Im(Δ_10), Re(Δ_11), Im(Δ_11)]

    Why this matters
    ----------------
    • Keeps the GA search numerically stable and physically “reasonable”.
    • Provides a smooth, saturating map so large |w| values don’t explode Δ.

    Parameters
    ----------
    w8 : np.ndarray
        Shape (8,), the latent parameters for one gate.

    Returns
    -------
    np.ndarray
        Shape (2,2), complex Δ for that gate.
    """
    w8 = np.asarray(w8, dtype=float)
    v = DELTA_MAX * np.tanh(w8)
    rr, ii, r01, i01, r10, i10, r11, i11 = v
    return np.array([[rr + 1j*ii,   r01 + 1j*i01],
                     [r10 + 1j*i10, r11 + 1j*i11]], complex)


def unpack_delta_map(w: np.ndarray) -> dict[str, np.ndarray]:
    """
    Decode the *full* genome vector w into a dictionary {gate → Δ_g}.

    Layout
    ------
    The genome packs gates in NOISY_GATES order, with 8 entries per gate,
    and each 8-block is converted by `_vec_to_delta_entries`.

    Parameters
    ----------
    w : np.ndarray
        Shape (8 * len(NOISY_GATES),), concatenated latent parameters.

    Returns
    -------
    dict[str, np.ndarray]
        Keys are the gate names from NOISY_GATES; values are 2×2 complex Δ_g.
    """
    delta_map: dict[str, np.ndarray] = {}
    k = 0
    for g in NOISY_GATES:
        delta_map[g] = _vec_to_delta_entries(w[k:k+8])
        k += 8
    return delta_map


def _delta_entries_to_w8(D: np.ndarray) -> np.ndarray:
    """
    Inverse map for a single gate: 2×2 complex Δ → 8-vector in w-space.

    Details
    -------
    Given Δ entries y with |y| ≤ DELTA_MAX (componentwise on Re/Im),
      x = clip(y / DELTA_MAX, −1+ε, 1−ε)
      w = arctanh(x)
    The small ε avoids infinities if rounding pushes |x| to 1.

    Use cases
    ---------
    • Seeding: turn a hand-tuned Δ_g into a latent w8 block for initialization.
    • Logging/debugging: round-trip consistency checks.

    Parameters
    ----------
    D : np.ndarray
        Shape (2,2), complex Δ for one gate.

    Returns
    -------
    np.ndarray
        Shape (8,), latent coordinates corresponding to D.
    """
    eps = 1e-6
    w8 = np.empty(8, dtype=float)
    vals = np.array([D.real[0,0], D.imag[0,0],
                     D.real[0,1], D.imag[0,1],
                     D.real[1,0], D.imag[1,0],
                     D.real[1,1], D.imag[1,1]], float)
    x = np.clip(vals / DELTA_MAX, -1.0 + eps, 1.0 - eps)
    w8[:] = np.arctanh(x)
    return w8


def pack_delta_map_to_w(delta_map: dict[str, np.ndarray]) -> np.ndarray:
    """
    Encode a per-gate Δ dictionary back into a single flat genome vector.

    Ordering & clipping
    -------------------
    Gates are packed strictly in NOISY_GATES order, each via `_delta_entries_to_w8`.
    The result is clipped to [−W_MAX, +W_MAX] in latent space as a safety measure.

    Parameters
    ----------
    delta_map : dict[str, np.ndarray]
        {gate → Δ_g}, each Δ_g is 2×2 complex.

    Returns
    -------
    np.ndarray
        Shape (8 * len(NOISY_GATES),), the flattened latent genome.
    """
    chunks = []
    for g in NOISY_GATES:
        chunks.append(_delta_entries_to_w8(delta_map[g]))
    w = np.concatenate(chunks, axis=0)
    return np.clip(w, -W_MAX, W_MAX)


def build_unitary_for_delta_map(delta_map: dict[str, np.ndarray]) -> np.ndarray:
    """
    Build the 2×2 effective matrix U corresponding to the evaluation circuit
    under a specified per-gate Δ noise map.

    Notes
    -----
    • The circuit here is fixed as `[("Rx", π)]` by design (single-qubit demo).
    • Δ entries are injected into the gate generators inside exp(·).
    • U can be non-unitary if Δ is non-Hermitian; it is the linear map the
      tomography routines will lift to a channel/PTM.

    Returns
    -------
    np.ndarray
        Shape (2,2), the effective matrix U(Δ).
    """
    circ = Circuit([("Rx", np.pi)], noise_map=delta_map)
    return circ.unitary(verbose=False)


def build_circuit_for_delta_map(delta_map: dict[str, np.ndarray]) -> Circuit:
    """
    Construct and return the Circuit object itself for a given Δ map.

    Why you might prefer this over U:
    ---------------------------------
    • Shot-based tomography typically needs the *procedural* circuit so it can
      re-prepare/measure basis states per setting (e.g., minimal schedule).
    • Heisenberg-exact can accept U directly since it’s analytic.

    Returns
    -------
    Circuit
        The configured Circuit([("Rx", π)], noise_map=Δ).
    """
    return Circuit([("Rx", np.pi)], noise_map=delta_map)


def ptm_heisenberg_for_U(U: np.ndarray) -> np.ndarray:
    """
    Analytic, sampling-free PTM in the Pauli–Liouville basis for a given U.

    Parameters
    ----------
    U : np.ndarray
        Shape (2,2). Effective matrix from the noisy circuit.

    Returns
    -------
    np.ndarray
        Shape (4,4), real Heisenberg PTM (I,X,Y,Z ordering).
    """
    return ProcessTomography.ptm_from_heisenberg(U)


def ptm_affine_for_U_old(U: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """
    Legacy helper kept for reference: shot-based affine PTM from a minimal
    schedule but *always* with assume_tp=True. Prefer `ptm_affine_for_U` below
    which lets you control trace-preservation.

    Parameters
    ----------
    U : np.ndarray
        Shape (2,2). (Historically accepted, but a Circuit is often better.)
    shots : int
        Measurement shots per setting.
    rng : np.random.Generator
        RNG for sampling.

    Returns
    -------
    np.ndarray
        Shape (4,4), affine PTM estimate with TP enforced.
    """
    R_aff, _, _ = ProcessTomography.ptm_from_minimal_schedule(
        U, shots=shots, rng=rng, assume_tp=True, verbose=False
    )
    return R_aff


def ptm_affine_for_U(U_or_circ, shots: int, rng: np.random.Generator, assume_tp: bool = ASSUME_TP_AFFINE) -> np.ndarray:
    """
    General shot-based affine PTM estimator from the minimal schedule.

    Inputs
    ------
    U_or_circ :
        Either a 2×2 matrix U or a Circuit object. For *best fidelity*, pass
        the Circuit so the routine can actually prepare/measure the proper
        tomography settings. Passing U is supported when the tomography layer
        internally re-embeds U with its own schedule.
    shots : int
        Number of shots per setting (higher → lower variance, slower).
    rng : np.random.Generator
        Random generator for sampling.
    assume_tp : bool
        If True, the estimator enforces trace preservation (TP). This should
        match how your hardware R_target was reconstructed; mismatches can make
        errors appear artificially large or small.

    Returns
    -------
    np.ndarray
        Shape (4,4), the affine (finite-shots) PTM estimate.
    """
    R_aff, _, _ = ProcessTomography.ptm_from_minimal_schedule(
        U_or_circ, shots=shots, rng=rng, assume_tp=assume_tp, verbose=False
    )
    return R_aff
def l2_or_chi2_cost(R_fit: np.ndarray, R_target: np.ndarray, R_std: np.ndarray | None) -> float:
    """
    Compute a least-squares style misfit between a *model* PTM (R_fit) and the
    *hardware/target* PTM (R_target), with optional χ² (per-entry variance)
    weighting and optional small-signal amplification.

    What this cost measures
    -----------------------
    - Base metric is elementwise squared error:  (R_fit - R_target)^2
    - If `R_std` is provided, entries with smaller experimental uncertainty
      (smaller σ_ij) are up-weighted as 1/σ_ij^2, i.e., a χ² objective.
    - If `ERROR_AMPLIFICATION` is True, entries where |R_target_ij| is small
      (below `AMPLIFICATION_THRESHOLD`) get an additional multiplicative weight
      of `AMP_FACTOR`. This forces the fitter to treat near-zero structure as
      important instead of “ignorable”.

    Knobs & switches (global)
    -------------------------
    • USE_WEIGHTED (handled by the caller): whether to pass R_std or None.
    • ERROR_AMPLIFICATION: enable/disable the extra emphasis on small targets.
    • AMPLIFICATION_THRESHOLD: how small |R_target_ij| must be to be emphasized.
    • AMP_FACTOR: how much to up-weight those small entries.

    When to prefer which setting
    ----------------------------
    - If your R_target came from many repeats and you have good per-entry σ,
      setting USE_WEIGHTED=True in the caller gives statistically principled
      weighting.
    - If you care about getting the zero-structure right (e.g., symmetry
      constraints, selection rules), keep ERROR_AMPLIFICATION=True so the GA
      does not “waste” error budget on large entries only.

    Parameters
    ----------
    R_fit : (4,4) ndarray
        Candidate model PTM produced by either Heisenberg-exact or shots-based
        tomography for the current genome.
    R_target : (4,4) ndarray
        Empirical PTM you want to match (e.g., average over hardware runs).
    R_std : (4,4) ndarray | None
        Optional per-entry standard deviations for R_target. If None, the cost
        reduces to plain unweighted L2.

    Returns
    -------
    float
        Non-negative scalar cost. Returns +∞ if something goes numerically bad
        (NaNs/Infs), so such candidates are rejected upstream.
    """
    # Residuals entrywise
    diff = R_fit - R_target

    # Base weights: all-ones if no variance info; else 1/σ^2 where σ>0 and finite
    if R_std is None:
        weights = np.ones_like(diff, dtype=float)
    else:
        sigma = R_std
        weights = np.ones_like(diff, dtype=float)
        mask = np.isfinite(sigma) & (sigma > 0)
        weights[mask] = 1.0 / (sigma[mask] ** 2)

    # Optional emphasis on near-zero targets to preserve delicate structure
    if ERROR_AMPLIFICATION:
        amp_mask = np.abs(R_target) < AMPLIFICATION_THRESHOLD
        weights[amp_mask] *= AMP_FACTOR

    # Weighted sum of squared residuals (Frobenius with weights)
    cost = np.sum(weights * (diff ** 2))

    # Guard against numerical issues
    return float(cost) if np.isfinite(cost) else np.inf


def cost_for_w_old(w: np.ndarray, R_target: np.ndarray, R_std: np.ndarray | None, rng: np.random.Generator) -> float:
    """
    Legacy end-to-end cost for a single GA genome `w`.

    Pipeline used here (older version)
    ----------------------------------
    1) Decode genome → per-gate Δ map via `unpack_delta_map`.
    2) Build effective 2×2 matrix U(Δ) for the evaluation circuit.
    3) Produce R_fit using either:
         • shots-based affine estimator     if FIT_WITH_SHOTS == True,
         • Heisenberg-exact analytic PTM    otherwise.
       (This *older* routine always passes a U to the affine estimator.)
    4) Compute least-squares/χ² cost via `l2_or_chi2_cost`.

    Why keep this around?
    ---------------------
    - For comparison/regression against earlier runs that passed U into the
      shot-based estimator.
    - The newer routine below passes a *Circuit* to the shots estimator, which
      better preserves state preparation & measurement details.

    Parameters
    ----------
    w : (8 * len(NOISY_GATES),) ndarray
        Latent genome (8 real params per gate; Re/Im parts of Δ entries).
    R_target : (4,4) ndarray
        Hardware PTM to match.
    R_std : (4,4) ndarray | None
        Optional per-entry σ for χ² weighting.
    rng : np.random.Generator
        RNG for sampling when FIT_WITH_SHOTS=True.

    Returns
    -------
    float
        Cost value; +∞ on any failure so GA can reject gracefully.
    """
    try:
        # Genome → Δ-map
        delta_map = unpack_delta_map(w)

        # Δ → effective matrix U
        U = build_unitary_for_delta_map(delta_map)

        # Comparator: shots-based vs Heisenberg-exact
        if FIT_WITH_SHOTS:
            R_fit = ptm_affine_for_U(U, shots=SHOTS_PER_EVAL, rng=rng)
        else:
            R_fit = ptm_heisenberg_for_U(U)

        # Reject non-finite results to keep GA stable
        if not np.all(np.isfinite(R_fit)):
            return np.inf

        # Weighted or unweighted cost depending on caller’s USE_WEIGHTED choice
        return l2_or_chi2_cost(R_fit, R_target, R_std if USE_WEIGHTED else None)

    except Exception:
        # Any upstream error → treat as a bad candidate
        return np.inf


def cost_for_w(w: np.ndarray, R_target: np.ndarray, R_std: np.ndarray | None, rng: np.random.Generator) -> float:
    """
    Current end-to-end cost for a single genome `w` (preferred).

    What’s different vs cost_for_w_old?
    -----------------------------------
    - For the shots-based comparator, this *passes a Circuit* to the
      tomography routine rather than a bare U. This preserves the full
      state-preparation/measurement protocol used by the minimal schedule and
      avoids losing information (e.g., entries that would collapse to zero when
      the schedule can’t “see” them from a bare matrix alone).

    Variance control
    ----------------
    - `COST_SHOT_REPEATS` > 1 performs multiple independent shot-based PTM
      estimates and averages them (common-random-numbers style variance
      reduction). This smooths the objective and helps GA convergence.

    Global switches that affect behavior
    ------------------------------------
    • FIT_WITH_SHOTS:
        False → analytic Heisenberg PTM (fast, deterministic).
        True  → affine PTM estimated from minimal schedule with finite shots.
    • ASSUME_TP_AFFINE:
        Whether the affine estimator enforces trace preservation; should match
        how the target PTM was reconstructed to avoid bias.
    • SHOTS_PER_EVAL:
        More shots → lower variance but slower evaluation.
    • USE_WEIGHTED:
        If True, pass R_std into the cost to get χ²-style weighting.

    Parameters
    ----------
    w : (8 * len(NOISY_GATES),) ndarray
        Latent genome (8 real params per gate).
    R_target : (4,4) ndarray
        Hardware PTM to match.
    R_std : (4,4) ndarray | None
        Optional per-entry standard deviations for χ² weighting.
    rng : np.random.Generator
        RNG used when FIT_WITH_SHOTS=True.

    Returns
    -------
    float
        Scalar objective; +∞ on numerical/model failure.
    """
    try:
        # Decode genome → Δ map used to parameterize the circuit/gates
        delta_map = unpack_delta_map(w)

        # Build both representations:
        # - U: used by Heisenberg-exact comparator (analytic)
        # - circ: preferred input for shots-based estimator (preserves schedule)
        U = build_unitary_for_delta_map(delta_map)
        circ = build_circuit_for_delta_map(delta_map)

        # Comparator branch
        if FIT_WITH_SHOTS:
            if COST_SHOT_REPEATS <= 1:
                # Single affine estimate (no averaging)
                R_fit = ptm_affine_for_U(circ, shots=SHOTS_PER_EVAL, rng=rng, assume_tp=ASSUME_TP_AFFINE)
            else:
                # Average multiple shot-based estimates to reduce variance
                Rs = []
                for _ in range(COST_SHOT_REPEATS):
                    Rs.append(ptm_affine_for_U(circ, shots=SHOTS_PER_EVAL, rng=rng, assume_tp=ASSUME_TP_AFFINE))
                R_fit = np.mean(np.stack(Rs, axis=0), axis=0)
        else:
            # Deterministic analytic comparator
            R_fit = ptm_heisenberg_for_U(U)

        # Numerical sanity guard
        if not np.all(np.isfinite(R_fit)):
            return np.inf

        # Return χ² or L2 cost depending on whether the caller supplied R_std
        return l2_or_chi2_cost(R_fit, R_target, R_std if USE_WEIGHTED else None)

    except Exception:
        # On any failure (shape/type/NaN), return +∞ so GA discards this genome
        return np.inf
def save_array(arr: np.ndarray, name: str, outdir: Path) -> None:
    """
    Persist an array to disk in two complementary formats for traceability
    and quick inspection.

    What it saves
    -------------
    • Binary NumPy file:  <outdir>/<name>.npy
      - Exact, lossless serialization; fastest to load back into NumPy.
    • CSV (comma-separated values): <outdir>/<name>.csv
      - Human/glanceable; convenient for spreadsheets and diffs.

    Typical usage in this project
    -----------------------------
    - PTMs (4×4), differences, per-entry std matrices, GA summaries.
    - The dual save strategy makes it easy to both re-use (npy) and eyeball (csv).

    Parameters
    ----------
    arr : np.ndarray
        Array to persist (any shape).
    name : str
        Basename for the artifacts (without extension).
    outdir : Path
        Target directory; created if missing.

    Notes
    -----
    - Directory creation is idempotent (parents=True, exist_ok=True).
    - CSV write uses default float formatting; adjust outside if needed.
    """
    # Ensure the output directory exists (safe if already present).
    outdir.mkdir(parents=True, exist_ok=True)

    # Save the exact binary representation.
    np.save(outdir / f"{name}.npy", arr)

    # Save a human-friendly CSV for quick viewing/diffing.
    np.savetxt(outdir / f"{name}.csv", arr, delimiter=",")


def plot_matrix(mat: np.ndarray, title: str, fname: Path) -> None:
    """
    Render a simple heatmap of a 2D array and save it as an image.

    Intended use here
    -----------------
    - Visualizing 4×4 process matrices (PTMs) in the Pauli ordering {I,X,Y,Z}.
    - Comparing target vs. fit vs. differences at a glance.

    Parameters
    ----------
    mat : np.ndarray
        2D array to visualize (e.g., shape (4,4)).
    title : str
        Figure title (kept succinct for report pages).
    fname : Path
        Output image path (e.g., OUT_DIR / "R_target.png").

    Plot details
    ------------
    - Uses imshow with aspect="auto".
    - Adds colorbar, axis labels, and tight layout.
    """
    # Create a fresh figure and axes.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Show the matrix as an image; values map to color.
    im = ax.imshow(mat, aspect="auto")

    # Lightweight labeling targeting Pauli basis usage.
    ax.set_title(title)
    ax.set_xlabel("columns (I,X,Y,Z)")
    ax.set_ylabel("rows (I,X,Y,Z)")

    # Colorbar + tidy spacing.
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    # Persist the visualization to disk and close the figure to free memory.
    fig.savefig(fname)
    plt.close(fig)


def initialize_population(rng: np.random.Generator, dim: int) -> np.ndarray:
    """
    Create the initial GA population in latent parameter space (w-space).

    Initialization policy
    ---------------------
    • Draw each coordinate ~ N(0, 0.5²) to start near the linear regime of tanh.
    • Clip to [-W_MAX, W_MAX] for numerical safety (tanh will further bound Δ).
    • Optionally inject a hand-crafted seed genome as the very first individual
      when USE_SEED=True; failures in seeding are silently ignored.

    Why this matters
    ----------------
    - Centered, moderate-variance starts help GA explore without immediately
      saturating Δ = DELTA_MAX * tanh(w).
    - Clipping avoids extreme latents that would effectively “flatline” tanh.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for reproducibility.
    dim : int
        Genome length, i.e., 8 × (# noisy gates).

    Returns
    -------
    np.ndarray
        Array of shape (N_POPULATION, dim) — one genome per row.
    """
    # Sample a Gaussian cloud in w-space and clip into a safe box.
    pop = rng.normal(0.0, 0.5, size=(N_POPULATION, dim))
    pop = np.clip(pop, -W_MAX, W_MAX)

    # Optionally place a known good seed at index 0 to guide early search.
    if USE_SEED:
        try:
            w_seed = pack_delta_map_to_w(SEED_DELTA_MAP)
            pop[0, :] = w_seed
        except Exception:
            # If the provided seed is incompatible or malformed, skip it.
            pass

    return pop


def evaluate_population(pop: np.ndarray, R_target: np.ndarray, R_std: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    """
    Compute the scalar objective J(w) for every genome in the current population.

    What happens per individual
    ---------------------------
    1) Decode latent vector → per-gate Δ map.
    2) Build model PTM using the configured comparator:
       - Heisenberg-exact (deterministic), or
       - Shots-based affine estimator (stochastic, uses `rng`).
    3) Compute (weighted) least-squares cost vs R_target.

    Performance notes
    -----------------
    - This is typically the hottest part of the GA loop. If it becomes a
      bottleneck, consider vectorized/parallel variants or lowering SHOTS_PER_EVAL.

    Parameters
    ----------
    pop : np.ndarray
        Shape (N_POPULATION, dim); each row is a genome.
    R_target : np.ndarray
        Hardware PTM (4×4) to match.
    R_std : np.ndarray | None
        Optional per-entry std for χ² weighting; pass None for plain L2.
    rng : np.random.Generator
        RNG used only when FIT_WITH_SHOTS=True.

    Returns
    -------
    np.ndarray
        Costs for each individual, shape (N_POPULATION,).
    """
    # Preallocate the cost buffer for speed and cache locality.
    costs = np.empty(pop.shape[0], dtype=float)

    # Evaluate each candidate genome end-to-end.
    for i in range(pop.shape[0]):
        costs[i] = cost_for_w(pop[i], R_target, R_std, rng)

    return costs


def select_parents(rng: np.random.Generator, pop: np.ndarray, costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick two parent genomes uniformly from the top-performing fraction.

    Selection strategy
    ------------------
    • Rank by ascending cost; keep the best `k = max(2, floor(SELECTION_TOP_FRAC*N))`.
    • Sample two indices uniformly (with replacement) from those top `k`.

    Why uniform within the elite?
    -----------------------------
    - Preserves diversity among good candidates vs. picking only the argmin.
    - Reduces selection pressure compared to tournaments — helpful early on.

    Parameters
    ----------
    rng : np.random.Generator
        RNG for the index draws.
    pop : np.ndarray
        Current population matrix (N×dim).
    costs : np.ndarray
        Vector of fitness values aligned with `pop` rows.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two genomes (views) to be used for crossover.
    """
    # Determine the elite pool size based on configured fraction.
    n = pop.shape[0]
    k = max(2, int(SELECTION_TOP_FRAC * n))

    # Sort indices by fitness and take the top-k.
    idx_sorted = np.argsort(costs)
    top_idx = idx_sorted[:k]

    # Sample two parents uniformly from the elite set.
    pA = rng.choice(top_idx)
    pB = rng.choice(top_idx)
    return pop[pA], pop[pB]


def uniform_crossover(rng: np.random.Generator, parentA: np.ndarray, parentB: np.ndarray) -> np.ndarray:
    """
    Create a child genome by independently choosing each gene from A or B.

    Crossover policy
    ----------------
    • With probability (1 − CROSSOVER_PROB), skip mixing (clone A).
    • Otherwise, for each coordinate j:
        child_j = parentA_j  with prob 0.5
                = parentB_j  with prob 0.5
    • Finally clip to [-W_MAX, W_MAX] to respect the latent search box.

    Why uniform crossover?
    ----------------------
    - Maintains gene-level diversity without positional bias.
    - Simple and effective for independent parameterizations like ours.

    Parameters
    ----------
    rng : np.random.Generator
        RNG for the Bernoulli draws.
    parentA, parentB : np.ndarray
        Vectors of length `dim` (w-space genomes).

    Returns
    -------
    np.ndarray
        New child genome ready for mutation/evaluation.
    """
    # Optional “no crossover” path for exploitation-only steps.
    if CROSSOVER_PROB < 1.0 and rng.random() > CROSSOVER_PROB:
        return parentA.copy()

    # Flip a fair coin per gene to choose A or B.
    mask = rng.random(size=parentA.shape) < 0.5
    child = np.where(mask, parentA, parentB)

    # Keep within the latent safety bounds (prevents runaway tanh saturation).
    return np.clip(child, -W_MAX, W_MAX)


def mutate_population(
    rng: np.random.Generator,
    pop: np.ndarray,
    elite_count: int,
) -> None:
    """
    In-place mutation of a random subset of *non-elite* individuals.

    Mutation scheme
    ---------------
    1) If MUTATION_ON is False or MUTATION_FRAC_MAX ≤ 0, do nothing.
    2) Randomly choose K individuals among indices [elite_count, N) where
       K ∈ {1, ..., floor(MUTATION_FRAC_MAX · N)}.
    3) For each chosen individual:
       • With probability MUTATION_REPLACE_PROB:
           - Reinitialize the entire genome ~ N(0, 0.5²) and clip.
           - Purpose: reintroduce *global* diversity (large jump).
       • Else:
           - Choose a random subset of gates (out of len(NOISY_GATES)).
           - For each selected gate, choose a random subset of its 8 parameters
             (corresponding to Re/Im of the 4 Δ entries).
           - Add Gaussian jitter N(0, MUTATION_SIGMA_W²) to those coordinates.
       • Clip the final genome to [-W_MAX, W_MAX].

    Tuning guidance
    ---------------
    - Increase MUTATION_SIGMA_W if the GA stalls (larger local steps).
    - Increase MUTATION_REPLACE_PROB if the population collapses (diversify).
    - Decrease MUTATION_FRAC_MAX late in training to refine around the minima.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness for all draws.
    pop : np.ndarray
        Population matrix (modified in place).
    elite_count : int
        Number of top individuals to *protect* (no mutations applied).

    Returns
    -------
    None
        Operates by side effect.
    """
    # Global switch and fraction guard.
    if not MUTATION_ON or MUTATION_FRAC_MAX <= 0:
        return

    # Decide how many individuals to mutate this generation.
    n = pop.shape[0]
    max_mut = max(1, int(MUTATION_FRAC_MAX * n))
    K = rng.integers(1, max_mut + 1)

    # Only consider non-elites for mutation.
    cand_idx = np.arange(elite_count, n)
    if cand_idx.size == 0:
        return

    # Pick unique individuals to mutate.
    chosen = rng.choice(cand_idx, size=min(K, cand_idx.size), replace=False)

    # Genome layout helpers.
    dim_per_gate = 8
    n_gates = len(NOISY_GATES)

    # Apply either full reseed or structured local jitters.
    for i in chosen:
        if rng.random() < MUTATION_REPLACE_PROB:
            # Full reseed → large jump for diversity injection.
            pop[i, :] = rng.normal(0.0, 0.5, size=pop.shape[1])
            pop[i, :] = np.clip(pop[i, :], -W_MAX, W_MAX)
            continue

        # Structured mutation: pick a random subset of gates to touch.
        num_gates_mut = rng.integers(1, n_gates + 1)
        gates_idx = rng.choice(np.arange(n_gates), size=num_gates_mut, replace=False)

        for gidx in gates_idx:
            # Determine the slice for this gate in the flattened genome.
            start = gidx * dim_per_gate
            end   = start + dim_per_gate

            # Mutate a random subset of the 8 per-gate parameters.
            num_params = rng.integers(1, dim_per_gate + 1)
            params = rng.choice(np.arange(start, end), size=num_params, replace=False)

            # Add local Gaussian noise to those coordinates.
            pop[i, params] += rng.normal(0.0, MUTATION_SIGMA_W, size=params.shape[0])

        # Keep the mutated genome within latent bounds.
        pop[i, :] = np.clip(pop[i, :], -W_MAX, W_MAX)


def main() -> int:
    """
    Orchestrates the full GA-based fit of per-gate generator noise Δ to a target
    hardware PTM R_target, and produces console diagnostics + saved artifacts.

    High-level flow
    ---------------
    1) Set up RNG & output folder.
    2) Load target PTM (and optional per-entry std for χ² weighting).
    3) Initialize population in latent parameter space (w), one genome per candidate.
    4) Evaluate initial costs; print rich diagnostics comparing:
         • Heisenberg-exact PTM (analytic, noiseless estimator),
         • Shots-based affine PTM (minimal schedule with finite sampling),
         • Noiseless (Δ=0) baselines for both.
    5) Run GA loop for N_GENERATIONS with elitism, selection, uniform crossover,
       and structured mutations in w-space.
    6) Decode the best genome → Δ map → build circuit/unitary → compute
       final Heisenberg + affine PTMs for this best Δ.
    7) Compute additional baselines and shot-averaged statistics, emit plots,
       arrays (.npy/.csv), and human/JSON reports.
    8) Return 0 on success; small non-zero code on early I/O errors.

    Key switches & knobs (see module constants)
    -------------------------------------------
    • FIT_WITH_SHOTS (bool): comparator used during cost evaluation.
        - True  → cost uses shots-based affine PTM (stochastic; uses `rng`).
        - False → cost uses Heisenberg-exact PTM (deterministic).
      Regardless of this setting, the *final report* includes both estimators.
    • SHOTS_PER_EVAL (int): per-setting shots for the affine estimator; larger
      reduces variance but increases runtime.
    • ASSUME_TP_AFFINE (bool): if True, the affine estimator enforces trace
      preservation; if False, it reflects “raw” sampling behavior.
    • COST_SHOT_REPEATS (int): when FIT_WITH_SHOTS=True, average multiple
      affine estimates inside the cost to reduce noise (CRN-style variance reduction).
    • USE_WEIGHTED (bool) + R_std: enable χ² weighting per entry (1/σ²), else
      plain L2. (R_std is loaded if available.)
    • ERROR_AMPLIFICATION (bool) + AMPLIFICATION_THRESHOLD + AMP_FACTOR:
      up-weight residuals where |R_target[i,j]| is small; can stabilize fits
      when near-zero entries carry crucial structure.
    • DELTA_MAX, W_MAX: Δ = DELTA_MAX * tanh(w) bounds the per-entry real/imag
      generator parameters; w itself is further clipped to ±W_MAX for safety.

    Returns
    -------
    int
        0 on success; early small non-zero codes (2/3) on missing/malformed target.
    """
    # -----------------------------
    # 1) RNG + output dir bootstrap
    # -----------------------------
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------
    # 2) Load target PTM (and optional R_std)
    # ----------------------------------------
    R_target_path = TARGET_DIR / "R_avg.npy"
    if not R_target_path.exists():
        # Hard stop: cannot fit without the target matrix to compare against.
        print(f"[ERROR] Missing target PTM: {R_target_path}", file=sys.stderr); return 2
    R_target = np.load(R_target_path)
    if R_target.shape != (4, 4):
        # Defensive guard: PTMs are 4×4 in the Pauli basis {I, X, Y, Z}.
        print(f"[ERROR] R_target shape {R_target.shape} != (4,4)", file=sys.stderr); return 3

    # Optional per-entry uncertainties; unlocks χ² cost if USE_WEIGHTED=True.
    R_std = None
    R_target_std_path = TARGET_DIR / "R_std.npy"
    if R_target_std_path.exists():
        R_tmp = np.load(R_target_std_path)
        if R_tmp.shape == (4, 4):
            R_std = R_tmp
        else:
            # Non-fatal: keep running with unweighted L2.
            print(f"[WARN] R_std shape {R_tmp.shape} != (4,4); ignoring std weighting.")

    # Persist inputs to the run folder for traceability.
    save_array(R_target, "R_target", OUT_DIR)
    if R_std is not None: save_array(R_std, "R_target_std", OUT_DIR)

    # ------------------------------------------------------
    # 3) Initialize GA population in latent w-space (dim = 8
    #    reals per gate: Re/Im for each 2×2 complex entry).
    # ------------------------------------------------------
    dim = 8 * len(NOISY_GATES)
    pop = initialize_population(rng, dim)

    # -----------------------------------------
    # 4) Evaluate initial population and choose
    #    the current global best (min cost).
    # -----------------------------------------
    costs = evaluate_population(pop, R_target, R_std, rng)
    best_idx = int(np.argmin(costs))
    best_w   = pop[best_idx].copy()
    best_cost= float(costs[best_idx])

    # ----------------------------------------------------------------------
    # Console diagnostics: show what we’re fitting and how both comparators
    # (Heisenberg exact vs. shots-based affine) behave for the same Δ, plus
    # the Δ=0 baselines. This is invaluable to catch modeling mismatches.
    # ----------------------------------------------------------------------
    print("\n[Initial Diagnostics]")
    print("We fit a model PTM R_fit(Δ) to the hardware target PTM R_target.")
    print("Two comparators are available and shown below for the SAME current Δ:")
    print("  • Heisenberg-exact PTM (analytic, no sampling).")
    print("  • Affine PTM estimated from a minimal schedule (shots-based).")
    print("We also show the corresponding NOISELESS (Δ=0) baselines for both.\n")
    print("Target PTM (R_target):")
    print(np.array2string(R_target, precision=5, suppress_small=True))

    # Decode the current best genome into per-gate Δ matrices.
    delta_map_init = unpack_delta_map(best_w)

    # Build both a matrix U (for Heisenberg) and a Circuit (for shots-based).
    U_init_noisy   = build_unitary_for_delta_map(delta_map_init)
    U_init_ideal   = build_unitary_for_delta_map({})
    circ_init_noisy = build_circuit_for_delta_map(delta_map_init)
    circ_init_ideal = build_circuit_for_delta_map({})

    # Heisenberg-exact PTMs (deterministic comparator).
    R_hex_noisy  = ptm_heisenberg_for_U(U_init_noisy)
    R_hex_ideal  = ptm_heisenberg_for_U(U_init_ideal)

    # Shots-based affine PTMs (stochastic comparator; may vary across runs).
    # NOTE: Passing Circuit objects ensures the minimal schedule is constructed
    #       using the *same* state prep + measurement conventions as on hardware.
    try:
        R_aff_noisy = ptm_affine_for_U(circ_init_noisy, shots=SHOTS_PER_EVAL, rng=rng)
        R_aff_ideal = ptm_affine_for_U(circ_init_ideal, shots=SHOTS_PER_EVAL, rng=rng)
    except TypeError:
        # Backward-compat fallback for an older signature.
        R_aff_noisy = ptm_affine_for_U(circ_init_noisy, SHOTS_PER_EVAL, rng)
        R_aff_ideal = ptm_affine_for_U(circ_init_ideal, SHOTS_PER_EVAL, rng)

    # Helper: print a labeled matrix and its Frobenius distance to target.
    def _print_model_block(label: str, R: np.ndarray):
        err = np.linalg.norm(R - R_target, ord='fro') if np.all(np.isfinite(R)) else np.inf
        print(f"\n{label}:")
        print(np.array2string(R, precision=5, suppress_small=True))
        print(f"Frobenius error vs target: {err:.6e}")

    # Helper: compute cost under the *current* weighting policy (L2 or χ²).
    def _cost_or_inf(R_fit: np.ndarray) -> float:
        if not np.all(np.isfinite(R_fit)):
            return float('inf')
        return l2_or_chi2_cost(R_fit, R_target, R_std if USE_WEIGHTED else None)

    # Print current-Δ comparators and Δ=0 baselines.
    _print_model_block("Heisenberg-exact PTM with current Δ (noisy)", R_hex_noisy)
    _print_model_block(f"Affine PTM with current Δ (shots={SHOTS_PER_EVAL})", R_aff_noisy)
    _print_model_block("Heisenberg-exact PTM (Δ = 0, noiseless baseline)", R_hex_ideal)
    _print_model_block(f"Affine PTM (Δ = 0, noiseless baseline; shots={SHOTS_PER_EVAL})", R_aff_ideal)

    # Show explicit entrywise differences to the target (noisy & baseline).
    print("\nDifferences vs target (noisy):")
    print("Heisenberg: R_hex_noisy - R_target")
    print(np.array2string(R_hex_noisy - R_target, precision=5, suppress_small=True))
    print("Affine    : R_aff_noisy - R_target")
    print(np.array2string(R_aff_noisy - R_target, precision=5, suppress_small=True))

    print("\nDifferences vs target (noiseless baselines):")
    print("Heisenberg: R_hex_ideal - R_target")
    print(np.array2string(R_hex_ideal - R_target, precision=5, suppress_small=True))
    print("Affine    : R_aff_ideal - R_target")
    print(np.array2string(R_aff_ideal - R_target, precision=5, suppress_small=True))

    # Report initial costs under the configured weighting; useful to verify
    # that both comparators are on comparable numeric scales.
    cost_hex_init = _cost_or_inf(R_hex_noisy)
    cost_aff_init = _cost_or_inf(R_aff_noisy)
    print("\nInitial costs under current Δ (according to your configured weighting):")
    print(f"  Heisenberg-exact comparator cost : {cost_hex_init:.6e}")
    print(f"  Affine (shots) comparator cost   : {cost_aff_init:.6e}")

    # Print the actual Δ matrices per gate to see what the current best encodes.
    print("\nCurrent Δ per gate (2×2 complex matrices):")
    for g in NOISY_GATES:
        D = delta_map_init[g]
        print(f"Δ[{g}] =")
        print("  Re:\n", np.array2string(D.real, precision=5, suppress_small=True))
        print("  Im:\n", np.array2string(D.imag, precision=5, suppress_small=True))

    # Make explicit which comparator drives the GA this run; independent of
    # these prints, the GA always optimizes exactly what you set here.
    comparator_str = "shots-based affine PTM" if FIT_WITH_SHOTS else "Heisenberg-exact PTM"
    print(f"\n[GA] Comparator driving optimization this run: {comparator_str}")
    print(f"[GA] initial best cost (under that comparator) = {best_cost:.6e}\n")

    # Quick GA header for sanity.
    print(f"[GA] pop={N_POPULATION}  gens={N_GENERATIONS}  dim={dim}  elitism={ELITISM}")
    print(f"[GA] initial best cost = {best_cost:.6e}")

    # -------------------------
    # 5) Main GA evolution loop
    # -------------------------
    for gen in range(1, N_GENERATIONS + 1):
        # Rank population by fitness and preserve the elites.
        idx_sorted = np.argsort(costs)
        elites = pop[idx_sorted[:ELITISM], :]

        # Build next population: start with elites, fill remainder with children.
        next_pop = np.empty_like(pop)
        next_pop[:ELITISM, :] = elites

        # Parent selection (uniform within top fraction) + uniform crossover.
        for i in range(ELITISM, N_POPULATION):
            pA, pB = select_parents(rng, pop, costs)
            child = uniform_crossover(rng, pA, pB)
            next_pop[i, :] = child

        # Structured mutation on non-elites; can reseed for diversity.
        mutate_population(rng, next_pop, ELITISM)

        # Replace population and re-evaluate costs under the active comparator.
        pop = next_pop
        costs = evaluate_population(pop, R_target, R_std, rng)

        # Track generation best and update global best if improved.
        gen_best_idx = int(np.argmin(costs))
        gen_best_cost= float(costs[gen_best_idx])
        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_w    = pop[gen_best_idx].copy()

        # Progress log ~10× per run; tune modulo if you want more/less chatter.
        if gen % max(1, N_GENERATIONS // 10) == 0:
            print(f"[GA] gen {gen}/{N_GENERATIONS}  best_gen={gen_best_cost:.6e}  best_all={best_cost:.6e}")

    # ------------------------------------------------------------------
    # 6) Decode the best genome → Δ map; save Δ stack in fixed gate order
    # ------------------------------------------------------------------
    best_delta_map = unpack_delta_map(best_w)
    delta_stack = np.stack([best_delta_map[g] for g in NOISY_GATES], axis=0)
    np.save(OUT_DIR / "best_delta.npy", delta_stack)

    # ---------------------------------------------------------
    # Build best-Δ unitary & circuit; compute final comparators
    # ---------------------------------------------------------
    U_best = build_unitary_for_delta_map(best_delta_map)
    circ_best = build_circuit_for_delta_map(best_delta_map)

    # Heisenberg-exact PTM for best Δ (deterministic).
    R_fit_hex = ptm_heisenberg_for_U(U_best)
    save_array(R_fit_hex, "R_fit_heisenberg", OUT_DIR)

    # Shots-based affine PTM for best Δ: one instance (single run) for the report
    # to show a concrete realization at SHOTS_PER_EVAL.
    R_fit_affine_single = ptm_affine_for_U(circ_best, shots=SHOTS_PER_EVAL, rng=rng, assume_tp=ASSUME_TP_AFFINE)
    save_array(R_fit_affine_single, "R_fit_affine_single", OUT_DIR)
    save_array(R_fit_affine_single - R_target, "R_diff_affine_single", OUT_DIR)

    # ---------------------------------------------
    # 7) No-noise (Δ=0) baselines for both methods
    # ---------------------------------------------
    U_ideal = build_unitary_for_delta_map({})
    circ_ideal = build_circuit_for_delta_map({})
    R_ideal_hex = ptm_heisenberg_for_U(U_ideal)
    save_array(R_ideal_hex, "R_ideal_heisenberg", OUT_DIR)
    save_array(R_ideal_hex - R_target, "R_diff_ideal_heisenberg", OUT_DIR)

    # For the affine baseline, average a capped number of realizations to reduce
    # randomness in the saved “ideal” reference (still finite shots).
    R_ideal_affine_mean_stack = []
    for _ in range(max(1, min(REEVALS, 32))):
        R_ideal_affine_mean_stack.append(
            ptm_affine_for_U(circ_ideal, shots=SHOTS_PER_EVAL, rng=rng, assume_tp=ASSUME_TP_AFFINE))
    R_ideal_affine_mean = np.mean(np.stack(R_ideal_affine_mean_stack, axis=0), axis=0)
    save_array(R_ideal_affine_mean, "R_ideal_affine_mean", OUT_DIR)
    save_array(R_ideal_affine_mean - R_target, "R_diff_ideal_affine_mean", OUT_DIR)

    # ---------------------------------------------
    # Differences & auxiliary comparisons vs. R_X
    # ---------------------------------------------
    R_diff_heisenberg = R_fit_hex - R_target
    save_array(R_diff_heisenberg, "R_diff_heisenberg", OUT_DIR)

    R_err_vs_X_target          = R_target   - R_X
    R_err_vs_X_fit_heisenberg  = R_fit_hex  - R_X
    save_array(R_err_vs_X_target,         "R_err_vs_X_target", OUT_DIR)
    save_array(R_err_vs_X_fit_heisenberg, "R_err_vs_X_fit_heisenberg", OUT_DIR)

    # ---------------------------------------------------------
    # 8) Affine PTM statistics for best Δ: mean ± std over REEVALS
    #    (this can be large; adjust REEVALS/SHOTS_PER_EVAL as needed)
    # ---------------------------------------------------------
    R_stack = []
    for _ in range(REEVALS):
        circ = build_circuit_for_delta_map(best_delta_map)
        R_aff = ptm_affine_for_U(circ, shots=SHOTS_PER_EVAL, rng=rng)
        R_stack.append(R_aff)
    R_stack = np.stack(R_stack)
    R_fit_mean = R_stack.mean(axis=0)
    R_fit_std  = R_stack.std(axis=0, ddof=1)
    save_array(R_fit_mean, "R_fit_mean_affine", OUT_DIR)
    save_array(R_fit_std,  "R_fit_std_affine",  OUT_DIR)

    # ----------------
    # 9) Quick plots
    # ----------------
    plot_matrix(R_target, "Target PTM (hardware avg)", OUT_DIR / "R_target.png")
    plot_matrix(R_fit_hex, "Analytic PTM (Heisenberg exact, GA best)", OUT_DIR / "R_fit_heisenberg.png")
    plot_matrix(R_diff_heisenberg, "Difference (Heisenberg − Target)", OUT_DIR / "R_diff_heisenberg.png")
    plot_matrix(R_fit_mean, "Analytic PTM (Affine mean, shots)", OUT_DIR / "R_fit_mean_affine.png")

    # --------------------------------------------
    # 10) JSON + human report with full run config
    # --------------------------------------------
    comparator_str = "Shots-based affine PTM (minimal schedule)" if FIT_WITH_SHOTS else "Heisenberg exact PTM (no shots)"
    report = {
        "settings": {
            "seed": SEED,
            "n_population": N_POPULATION,
            "n_generations": N_GENERATIONS,
            "elitism": ELITISM,
            "selection_top_frac": SELECTION_TOP_FRAC,
            "crossover_prob": CROSSOVER_PROB,
            "mutation_on": MUTATION_ON,
            "mutation_frac_max": MUTATION_FRAC_MAX,
            "mutation_sigma_w": MUTATION_SIGMA_W,
            "mutation_replace_prob": MUTATION_REPLACE_PROB,
            "shots_per_eval_affine": SHOTS_PER_EVAL,
            "reevals_affine": REEVALS,
            "use_weighted_cost": USE_WEIGHTED,
            "error_amplification": ERROR_AMPLIFICATION,
            "amplification_threshold": AMPLIFICATION_THRESHOLD,
            "amp_factor": AMP_FACTOR,
            "delta_max": DELTA_MAX,
            "w_max": W_MAX,
            "noisy_gates": NOISY_GATES,
            "use_seed": USE_SEED,
            "fit_with_shots": FIT_WITH_SHOTS,
            "assume_tp_affine": ASSUME_TP_AFFINE,
            "cost_shot_repeats": COST_SHOT_REPEATS,
            "comparator": comparator_str,
        },
        # Note: name kept for backward compatibility; it always stores the *best*
        # GA cost under the active comparator (Heisenberg or shots-based).
        "best_cost_heisenberg": float(best_cost),
        "best_delta_gate_order": NOISY_GATES,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Human-readable TXT mirror with embedded matrices for quick inspection.
    with (OUT_DIR / "report.txt").open("w", encoding="utf-8") as f:
        f.write("=== Genetic Algorithm fit of per-gate non-Hermitian generator noise to hardware PTM ===\n")
        f.write(f"Comparator: {comparator_str}\n\n")
        f.write(json.dumps(report, indent=2))
        f.write("\n\nBest Δ per gate (order {}):\n".format(NOISY_GATES))
        for g in NOISY_GATES:
            f.write(f"\nΔ[{g}] =\n{best_delta_map[g]}\n")
        f.write("\n\nR_target (mean):\n")
        f.write(str(R_target))
        if R_std is not None:
            f.write("\n\nR_target (std):\n")
            f.write(str(R_std))
        f.write("\n\nR_fit_heisenberg (exact):\n")
        f.write(str(R_fit_hex))
        f.write("\n\nR_diff_heisenberg = R_fit_heisenberg − R_target:\n")
        f.write(str(R_diff_heisenberg))
        f.write("\n\n[Aux] Affine mean ± std (shots):\n")
        f.write(str(R_fit_mean) + "\n±\n" + str(R_fit_std) + "\n")
        f.write("\n\nR_fit_affine_single (shots, assume_tp={}):\n".format(ASSUME_TP_AFFINE))
        f.write(str(R_fit_affine_single))
        f.write("\n\nR_diff_affine_single = R_fit_affine_single − R_target:\n")
        f.write(str(R_fit_affine_single - R_target))
        f.write("\n\n[Aux] Affine mean ± std (shots, assume_tp={}):\n".format(ASSUME_TP_AFFINE))
        f.write(str(R_fit_mean) + "\n±\n" + str(R_fit_std) + "\n")
        f.write("\n\n[Baseline] R_ideal_heisenberg (Δ=0):\n")
        f.write(str(R_ideal_hex))
        f.write("\n\n[Baseline] R_diff_ideal_heisenberg = R_ideal_heisenberg − R_target:\n")
        f.write(str(R_ideal_hex - R_target))
        f.write("\n\n[Baseline] R_ideal_affine_mean (Δ=0, shots):\n")
        f.write(str(R_ideal_affine_mean))
        f.write("\n\n[Baseline] R_diff_ideal_affine_mean = R_ideal_affine_mean − R_target:\n")
        f.write(str(R_ideal_affine_mean - R_target))

    # Final console note + a few extra plots for convenience.
    print("\n[OK] GA fitting complete. Results in:", OUT_DIR.resolve())
    print(" Best cost:", best_cost)

    plot_matrix(R_fit_affine_single, "Analytic PTM (Affine single, shots)", OUT_DIR / "R_fit_affine_single.png")
    plot_matrix(R_ideal_hex, "Ideal PTM (Heisenberg, Δ=0)", OUT_DIR / "R_ideal_heisenberg.png")
    plot_matrix(R_ideal_affine_mean, "Ideal PTM (Affine mean, Δ=0)", OUT_DIR / "R_ideal_affine_mean.png")

    return 0


if __name__ == "__main__":
    # Keep the standard Python entry point; convert return code to process exit.
    raise SystemExit(main())
