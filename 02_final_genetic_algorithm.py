#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import sys, json
import numpy as np
import matplotlib.pyplot as plt

# QPT12-only import
from class_tomography_final import Circuit
from class_tomography_final import QPT12Strict as QPT12

# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────
TARGET_DIR  = Path("unique_process_tomos")
OUT_DIR     = Path("test_28_11_2025")

# ─────────────────────────────────────────────────────────────────────────────
# RNG
# ─────────────────────────────────────────────────────────────────────────────
SEED = 100

# ─────────────────────────────────────────────────────────────────────────────
# GA HYPERPARAMS
# ─────────────────────────────────────────────────────────────────────────────
N_POPULATION          = 100
N_GENERATIONS_old     = 10000
N_GENERATIONS         = 1000
ELITISM               = 50
SELECTION_TOP_FRAC    = 0.5
CROSSOVER_PROB        = 1.0
MUTATION_ON           = True
MUTATION_FRAC_MAX     = 0.33333333
MUTATION_SIGMA_W      = 0.15
MUTATION_REPLACE_PROB = 0.25
ASSUME_TP_AFFINE      = False   # passed through into QPT12
COST_SHOT_REPEATS     = 1       # how many QPT12 calls per cost evaluation (avg)

# ─────────────────────────────────────────────────────────────────────────────
# SHOTS (used by QPT12 only now)
# ─────────────────────────────────────────────────────────────────────────────
SHOTS_PER_EVAL = 10000
REEVALS        = 10000

# ─────────────────────────────────────────────────────────────────────────────
# COST SHAPING
# ─────────────────────────────────────────────────────────────────────────────
USE_WEIGHTED            = False
ERROR_AMPLIFICATION     = True
AMPLIFICATION_THRESHOLD = 0.05
AMP_FACTOR              = 3.5

# ─────────────────────────────────────────────────────────────────────────────
# GATE INVENTORY (order matters) — STRICT 12-circuit set
# ─────────────────────────────────────────────────────────────────────────────
NOISY_GATES = ("Rz", "SX", "X")  # one Δ per gate label, shared across all 12 circuits

# ─────────────────────────────────────────────────────────────────────────────
# REFS/PRINT
# ─────────────────────────────────────────────────────────────────────────────
R_X = np.diag([1.0, 1.0, -1.0, -1.0])
np.set_printoptions(precision=5, suppress=True)

# ─────────────────────────────────────────────────────────────────────────────
# Δ PARAM BOX
# ─────────────────────────────────────────────────────────────────────────────
DELTA_MAX_og = 0.30
DELTA_MAX    = 0.05   # original 0.1
W_MAX        = 6.0

# ─────────────────────────────────────────────────────────────────────────────
# SEEDING
# ─────────────────────────────────────────────────────────────────────────────
USE_SEED = False
SEED_DELTA_MAP = {
    "Rz": np.array([[ 0.05-0.02j,  0.00-0.01j],
                    [-0.01+0.01j,  0.03-0.02j]], complex),
    "SX": np.array([[ 0.01-0.03j,  0.02-0.02j],
                    [ 0.01+0.02j,  0.02-0.01j]], complex),
    "X":  np.array([[ 0.02+0.00j, -0.01-0.02j],
                    [ 0.01+0.02j,  0.02+0.00j]], complex),
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers: Δ-param ⇄ vector
# ─────────────────────────────────────────────────────────────────────────────
def _vec_to_delta_entries(w8: np.ndarray) -> np.ndarray:
    v = DELTA_MAX * np.tanh(np.asarray(w8, float))
    rr, ii, r01, i01, r10, i10, r11, i11 = v
    return np.array([[rr + 1j*ii,   r01 + 1j*i01],
                     [r10 + 1j*i10, r11 + 1j*i11]], complex)

def unpack_delta_map(w: np.ndarray) -> dict[str, np.ndarray]:
    delta_map = {}
    k = 0
    for g in NOISY_GATES:
        delta_map[g] = _vec_to_delta_entries(w[k:k+8]); k += 8
    return delta_map

def _delta_entries_to_w8(D: np.ndarray) -> np.ndarray:
    eps = 1e-6
    vals = np.array([D.real[0,0], D.imag[0,0], D.real[0,1], D.imag[0,1],
                     D.real[1,0], D.imag[1,0], D.real[1,1], D.imag[1,1]], float)
    x = np.clip(vals/DELTA_MAX, -1+eps, 1-eps)
    return np.arctanh(x)

def pack_delta_map_to_w(delta_map: dict[str, np.ndarray]) -> np.ndarray:
    chunks = [ _delta_entries_to_w8(delta_map[g]) for g in NOISY_GATES ]
    w = np.concatenate(chunks, 0)
    return np.clip(w, -W_MAX, W_MAX)

# ─────────────────────────────────────────────────────────────────────────────
# Basic circuit wrapper (used only for building unitary, if needed elsewhere)
# ─────────────────────────────────────────────────────────────────────────────
def build_unitary_for_delta_map(delta_map: dict[str, np.ndarray]) -> np.ndarray:
    return Circuit([("X",)], noise_map=delta_map).unitary(False)

def build_circuit_for_delta_map(delta_map: dict[str, np.ndarray]) -> Circuit:
    return Circuit([("X",)], noise_map=delta_map)

# ─────────────────────────────────────────────────────────────────────────────
# QPT-12 PTM (ONLY tomography route now)
# ─────────────────────────────────────────────────────────────────────────────
def ptm_qpt12_for_delta_map(delta_map: dict[str, np.ndarray],
                            shots: int,
                            rng: np.random.Generator,
                            assume_tp: bool = ASSUME_TP_AFFINE) -> np.ndarray:
    qpt = QPT12(target_tokens=[("X",)])  # strict 12-circuit schedule
    R12, _, _ = qpt.ptm(shots=shots,
                        rng=rng,
                        assume_tp=assume_tp,
                        noise_map=delta_map,
                        amp_damp_p=None,
                        phase_damp_lam=None,
                        verbose=False)
    return R12

# ─────────────────────────────────────────────────────────────────────────────
# Cost
# ─────────────────────────────────────────────────────────────────────────────
def l2_or_chi2_cost(R_fit: np.ndarray, R_target: np.ndarray, R_std: np.ndarray | None) -> float:
    diff = R_fit - R_target
    if R_std is None:
        weights = np.ones_like(diff)
    else:
        weights = np.ones_like(diff)
        mask = np.isfinite(R_std) & (R_std > 0)
        weights[mask] = 1.0/(R_std[mask]**2)
    if ERROR_AMPLIFICATION:
        weights[np.abs(R_target) < AMPLIFICATION_THRESHOLD] *= AMP_FACTOR
    c = np.sum(weights * (diff**2))
    return float(c) if np.isfinite(c) else np.inf

def cost_for_w(w: np.ndarray,
               R_target: np.ndarray,
               R_std: np.ndarray | None,
               rng: np.random.Generator) -> float:
    """
    Cost = || R_QPT12(Δ(w)) - R_target ||^2 (or χ^2-weighted),
    where R_QPT12 is obtained exclusively via QPT12Strict.
    """
    try:
        delta_map = unpack_delta_map(w)

        # always QPT-12, no other tomography
        if COST_SHOT_REPEATS <= 1:
            R_fit = ptm_qpt12_for_delta_map(delta_map,
                                            SHOTS_PER_EVAL,
                                            rng,
                                            ASSUME_TP_AFFINE)
        else:
            Rs = [ptm_qpt12_for_delta_map(delta_map,
                                          SHOTS_PER_EVAL,
                                          rng,
                                          ASSUME_TP_AFFINE)
                  for _ in range(COST_SHOT_REPEATS)]
            R_fit = np.mean(np.stack(Rs, 0), 0)

        if not np.all(np.isfinite(R_fit)):
            return np.inf
        return l2_or_chi2_cost(R_fit, R_target, R_std if USE_WEIGHTED else None)
    except Exception:
        return np.inf

# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_array(arr: np.ndarray, name: str, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / f"{name}.npy", arr)
    np.savetxt(outdir / f"{name}.csv", arr, delimiter=",")

def plot_matrix(mat: np.ndarray, title: str, fname: Path) -> None:
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    im  = ax.imshow(mat, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("columns (I,X,Y,Z)")
    ax.set_ylabel("rows (I,X,Y,Z)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# GA helpers
# ─────────────────────────────────────────────────────────────────────────────
def initialize_population(rng: np.random.Generator, dim: int) -> np.ndarray:
    pop = np.clip(rng.normal(0.0, 0.5, size=(N_POPULATION, dim)), -W_MAX, W_MAX)
    if USE_SEED:
        try:
            pop[0, :] = pack_delta_map_to_w(SEED_DELTA_MAP)
        except Exception:
            pass
    return pop

def evaluate_population(pop: np.ndarray,
                        R_target: np.ndarray,
                        R_std: np.ndarray | None,
                        rng: np.random.Generator) -> np.ndarray:
    costs = np.empty(pop.shape[0], float)
    for i in range(pop.shape[0]):
        costs[i] = cost_for_w(pop[i], R_target, R_std, rng)
    return costs

def select_parents(rng: np.random.Generator,
                   pop: np.ndarray,
                   costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Elites are allowed to mate: selection draws from the whole top fraction (which includes elites).
    n = pop.shape[0]
    k = max(2, int(SELECTION_TOP_FRAC * n))
    idx_sorted = np.argsort(costs)
    top_idx    = idx_sorted[:k]
    pA = rng.choice(top_idx)
    pB = rng.choice(top_idx)
    return pop[pA], pop[pB]

def uniform_crossover(rng: np.random.Generator,
                      parentA: np.ndarray,
                      parentB: np.ndarray) -> np.ndarray:
    if CROSSOVER_PROB < 1.0 and rng.random() > CROSSOVER_PROB:
        return parentA.copy()
    mask  = rng.random(size=parentA.shape) < 0.5
    child = np.where(mask, parentA, parentB)
    return np.clip(child, -W_MAX, W_MAX)

def mutate_population(rng: np.random.Generator,
                      pop: np.ndarray,
                      elite_count: int) -> None:
    """
    Mutate IN-PLACE but NEVER touch the first `elite_count` rows (frozen elites).
    """
    if not MUTATION_ON or MUTATION_FRAC_MAX <= 0:
        return
    n       = pop.shape[0]
    max_mut = max(1, int(MUTATION_FRAC_MAX * n))

    # Strict guard: only indices >= elite_count are ever considered.
    cand_idx = np.arange(elite_count, n)
    if cand_idx.size == 0:
        return

    K = rng.integers(1, max_mut + 1)
    chosen = rng.choice(cand_idx, size=min(K, cand_idx.size), replace=False)

    dim_per_gate = 8
    n_gates      = len(NOISY_GATES)
    for i in chosen:
        if rng.random() < MUTATION_REPLACE_PROB:
            pop[i, :] = np.clip(rng.normal(0.0, 0.5, size=pop.shape[1]),
                                -W_MAX, W_MAX)
            continue

        num_gates_mut = rng.integers(1, n_gates + 1)
        gates_idx = rng.choice(np.arange(n_gates),
                               size=num_gates_mut,
                               replace=False)
        for gidx in gates_idx:
            start = gidx * dim_per_gate
            end   = start + dim_per_gate
            num_params = rng.integers(1, dim_per_gate + 1)
            params = rng.choice(np.arange(start, end),
                                size=num_params,
                                replace=False)
            pop[i, params] += rng.normal(0.0,
                                         MUTATION_SIGMA_W,
                                         size=params.shape[0])
        pop[i, :] = np.clip(pop[i, :], -W_MAX, W_MAX)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load target PTM
    R_target_path = TARGET_DIR / "R_avg.npy"
    if not R_target_path.exists():
        print(f"[ERROR] Missing target PTM: {R_target_path}", file=sys.stderr)
        return 2
    R_target = np.load(R_target_path)
    if R_target.shape != (4, 4):
        print(f"[ERROR] R_target shape {R_target.shape} != (4,4)", file=sys.stderr)
        return 3

    # Optional std
    R_std = None
    R_target_std_path = TARGET_DIR / "R_std.npy"
    if R_target_std_path.exists():
        R_tmp = np.load(R_target_std_path)
        if R_tmp.shape == (4, 4):
            R_std = R_tmp
        else:
            print(f"[WARN] R_std shape {R_tmp.shape} != (4,4); ignoring.")

    save_array(R_target, "R_target", OUT_DIR)
    if R_std is not None:
        save_array(R_std, "R_target_std", OUT_DIR)

    # GA init
    dim = 8 * len(NOISY_GATES)
    pop   = initialize_population(rng, dim)
    costs = evaluate_population(pop, R_target, R_std, rng)
    best_idx  = int(np.argmin(costs))
    best_w    = pop[best_idx].copy()
    best_cost = float(costs[best_idx])

    print("\n[Initial Diagnostics]")
    print("Target PTM (R_target):")
    print(np.array2string(R_target, precision=7, suppress_small=False))

    # Initial Δ → QPT-12 PTMs (noisy vs ideal)
    delta_map_init  = unpack_delta_map(best_w)

    R_12_noisy = ptm_qpt12_for_delta_map(delta_map_init,
                                         SHOTS_PER_EVAL,
                                         rng,
                                         ASSUME_TP_AFFINE)

    R_12_ideal = ptm_qpt12_for_delta_map({},
                                         SHOTS_PER_EVAL,
                                         rng,
                                         ASSUME_TP_AFFINE)

    def _print_block(label, R):
        err = np.linalg.norm(R - R_target, 'fro') if np.all(np.isfinite(R)) else np.inf
        print(f"\n{label}:")
        print(np.array2string(R, precision=7, suppress_small=False))
        print(f"Frobenius error vs target: {err:.6e}")

    _print_block(f"QPT-12 (shots={SHOTS_PER_EVAL}) (current Δ)", R_12_noisy)
    _print_block(f"QPT-12 (Δ=0, shots={SHOTS_PER_EVAL})", R_12_ideal)

    init_cost = l2_or_chi2_cost(R_12_noisy,
                                R_target,
                                R_std if USE_WEIGHTED else None)

    print("\nInitial costs:")
    print(f"  QPT-12           : {init_cost:.6e}")

    print("\nCurrent Δ per gate:")
    for g in NOISY_GATES:
        D = delta_map_init[g]
        print(f"Δ[{g}] Re:\n{np.array2string(D.real, precision=7, suppress_small=False)}")
        print(f"Δ[{g}] Im:\n{np.array2string(D.imag, precision=7, suppress_small=False)}")

    comparator_str = "QPT-12 PTM (shots)"
    print(f"\n[GA] Comparator: {comparator_str}")
    print(f"[GA] pop={N_POPULATION} gens={N_GENERATIONS} dim={dim} elitism={ELITISM}")
    print(f"[GA] initial best cost = {best_cost:.6e}")

    # GA loop
    for gen in range(1, N_GENERATIONS + 1):
        idx_sorted = np.argsort(costs)
        elites     = pop[idx_sorted[:ELITISM], :].copy()  # frozen
        next_pop   = np.empty_like(pop)

        # 1) keep elites
        next_pop[:ELITISM, :] = elites

        # 2) fill rest with offspring
        for i in range(ELITISM, N_POPULATION):
            pA, pB = select_parents(rng, pop, costs)
            next_pop[i, :] = uniform_crossover(rng, pA, pB)

        # 3) mutations (non-elites only)
        mutate_population(rng, next_pop, ELITISM)

        # 4) evaluate
        pop   = next_pop
        costs = evaluate_population(pop, R_target, R_std, rng)

        # 5) track best
        gen_best_idx  = int(np.argmin(costs))
        gen_best_cost = float(costs[gen_best_idx])
        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_w    = pop[gen_best_idx].copy()
        print(f"[GA] gen {gen}/{N_GENERATIONS}  best_gen={gen_best_cost:.6e}  best_all={best_cost:.6e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Final QPT-12 results for best Δ
    # ─────────────────────────────────────────────────────────────────────────
    best_delta_map = unpack_delta_map(best_w)
    delta_stack = np.stack([best_delta_map[g] for g in NOISY_GATES], 0)
    np.save(OUT_DIR / "best_delta.npy", delta_stack)

    # single QPT-12 run for best Δ
    R_fit_qpt12_single = ptm_qpt12_for_delta_map(best_delta_map,
                                                 SHOTS_PER_EVAL,
                                                 rng,
                                                 ASSUME_TP_AFFINE)
    save_array(R_fit_qpt12_single, "R_fit_qpt12_single", OUT_DIR)
    save_array(R_fit_qpt12_single - R_target, "R_diff_qpt12_single", OUT_DIR)

    # ideal (Δ=0) QPT-12, averaged over limited REEVALS for sanity
    R_ideal_qpt12_stack = []
    for _ in range(max(1, min(REEVALS, 32))):
        R_ideal_qpt12_stack.append(
            ptm_qpt12_for_delta_map({}, SHOTS_PER_EVAL, rng, ASSUME_TP_AFFINE)
        )
    R_ideal_qpt12_mean = np.mean(np.stack(R_ideal_qpt12_stack, 0), 0)
    save_array(R_ideal_qpt12_mean, "R_ideal_qpt12_mean", OUT_DIR)
    save_array(R_ideal_qpt12_mean - R_target, "R_diff_ideal_qpt12_mean", OUT_DIR)

    # reeval best Δ with QPT-12 as well (limited repeats for runtime sanity)
    R_stack = []
    n_reeval = max(1, min(REEVALS, 32))
    for _ in range(n_reeval):
        R_stack.append(
            ptm_qpt12_for_delta_map(best_delta_map,
                                    SHOTS_PER_EVAL,
                                    rng,
                                    ASSUME_TP_AFFINE)
        )
    R_stack     = np.stack(R_stack)
    R_fit_mean  = R_stack.mean(0)
    R_fit_std   = R_stack.std(0, ddof=1) if n_reeval > 1 else np.zeros_like(R_fit_mean)

    save_array(R_fit_mean, "R_fit_mean_qpt12", OUT_DIR)
    save_array(R_fit_std,  "R_fit_std_qpt12",  OUT_DIR)

    # error vs ideal X PTM using QPT-12 mean
    R_err_vs_X_target    = R_target        - R_X
    R_err_vs_X_fit_qpt12 = R_fit_mean      - R_X
    save_array(R_err_vs_X_target,    "R_err_vs_X_target",    OUT_DIR)
    save_array(R_err_vs_X_fit_qpt12, "R_err_vs_X_fit_qpt12", OUT_DIR)

    # ─────────────────────────────────────────────────────────────────────────
    # Plots (QPT-12 only)
    # ─────────────────────────────────────────────────────────────────────────
    plot_matrix(R_target,
                "Target PTM (hardware avg)",
                OUT_DIR / "R_target.png")
    plot_matrix(R_fit_qpt12_single,
                "PTM (QPT-12 best single, shots)",
                OUT_DIR / "R_fit_qpt12_single.png")
    plot_matrix(R_fit_mean,
                "PTM (QPT-12 best mean, shots)",
                OUT_DIR / "R_fit_mean_qpt12.png")
    plot_matrix(R_fit_mean - R_target,
                "Difference (QPT-12 mean − Target)",
                OUT_DIR / "R_diff_qpt12_mean.png")

    # ─────────────────────────────────────────────────────────────────────────
    # Report
    # ─────────────────────────────────────────────────────────────────────────
    comparator_str = "QPT-12 PTM (shots)"
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
            "shots_per_eval_qpt12": SHOTS_PER_EVAL,
            "reevals_qpt12": REEVALS,
            "use_weighted_cost": USE_WEIGHTED,
            "error_amplification": ERROR_AMPLIFICATION,
            "amplification_threshold": AMPLIFICATION_THRESHOLD,
            "amp_factor": AMP_FACTOR,
            "delta_max": DELTA_MAX,
            "w_max": W_MAX,
            "noisy_gates": NOISY_GATES,
            "use_seed": USE_SEED,
            "assume_tp_affine": ASSUME_TP_AFFINE,
            "cost_shot_repeats": COST_SHOT_REPEATS,
            "comparator": comparator_str,
        },
        "best_cost": float(best_cost),
        "best_delta_gate_order": NOISY_GATES,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2),
                                         encoding="utf-8")

    with (OUT_DIR / "report.txt").open("w", encoding="utf-8") as f:
        f.write("=== GA fit to hardware PTM (QPT-12 ONLY) ===\n")
        f.write(f"Comparator: {comparator_str}\n\n")
        f.write(json.dumps(report, indent=2))
        f.write("\n\nBest Δ per gate:\n")
        for g in NOISY_GATES:
            f.write(f"\nΔ[{g}] =\n{best_delta_map[g]}\n")
        f.write("\n\nR_target:\n");           f.write(str(R_target))
        f.write("\n\nR_fit_qpt12_single:\n"); f.write(str(R_fit_qpt12_single))
        f.write("\n\nR_diff_qpt12_single:\n"); f.write(str(R_fit_qpt12_single - R_target))
        f.write("\n\nQPT-12 mean ± std (best Δ):\n")
        f.write(str(R_fit_mean) + "\n±\n" + str(R_fit_std) + "\n")
        f.write("\n\nR_ideal_qpt12_mean (Δ=0):\n"); f.write(str(R_ideal_qpt12_mean))

    print("\n[OK] GA fitting complete (QPT-12 only). Results in:", OUT_DIR.resolve())
    print(" Best cost:", best_cost)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
