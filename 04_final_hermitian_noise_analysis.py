#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_random_noise_sweep_qpt12_minimal.py
=======================================

Goal:
  • Keep BOTH Hermitian and Non-Hermitian random Δ generators.
  • SHOTS-BASED evaluation ONLY (single shots setting).
  • For EACH family (Hermitian / Non-Hermitian):
        - Compute the overall MEAN PTM across N_SAMPLES random Δ maps.
        - Compute the component-wise STD across those samples.
        - Save MEAN/STD arrays and heatmap plots.
        - PRINT the MEAN/STD matrices to console.
  • For the HARDWARE TARGET:
        - Present the target MEAN and component-wise STD the same way
          (load R_avg.npy / R_std.npy if present, otherwise aggregate per-run PTMs).
        - Save MEAN/STD arrays and heatmap plots.
        - PRINT the MEAN/STD matrices to console.
  • Produce exactly TWO z-score matrices (one per family) with ✓ check marks:
        z = |R_mean_family − R_target_mean| / sqrt( STD_target^2 + STD_family^2 ).
        Neat, row-wise printout with ✓ for z ≤ 1, shown in console and appended to report.

No other evaluations are performed.

Dependencies (same as your framework):
    from class_tomographies import Circuit, QPT12Strict
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# ---- Your tomography base ----
from class_tomography_final import Circuit, QPT12Strict

# ------------------------- Config -------------------------
TARGET_DIR   = Path("unique_process_tomos")   # expects R_avg.npy / R_std.npy OR per-run subfolders
OUT_DIR      = Path("fit_hermitian_28112025")

SEED         = 100
SHOTS_PROC   = 10000            # single shots-based setting
ASSUME_TP    = False            # single TP mode for all runs

# ONLY these three generators
GATE_ORDER   = ("Rz", "SX", "X")

# Random noise settings
N_SAMPLES    = 1000             # per family
NOISE_SCALE  = 0.05             # ±range per component

plt.rcParams.update({"figure.dpi": 140})

# ---------------------- I/O helpers ---------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_array(arr: np.ndarray, name: str, outdir: Path) -> None:
    ensure_dir(outdir)
    np.save(outdir / f"{name}.npy", arr)
    try:
        np.savetxt(outdir / f"{name}.csv", arr, delimiter=",")
    except Exception:
        pass

def try_load(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None

# ---------------------- Plot helpers ---------------------
def plot_matrix(M: np.ndarray, title: str, fname: Path) -> None:
    ensure_dir(fname.parent)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(M, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("columns (I,X,Y,Z)")
    ax.set_ylabel("rows (I,X,Y,Z)")
    ax.set_xticks(range(4)); ax.set_xticklabels(["I","X","Y","Z"])
    ax.set_yticks(range(4)); ax.set_yticklabels(["I","X","Y","Z"])
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)

# ---------------------- TXT helpers ---------------------
def mat_block(mat: np.ndarray, title: str) -> str:
    s = np.array2string(mat, precision=6, suppress_small=False, floatmode="maxprec", max_line_width=120)
    return f"{title}\n{s}\n"

def hdr(title: str) -> str:
    bar = "─" * max(10, len(title) + 2)
    return f"\n{bar}\n {title}\n{bar}\n"

# ---------------------- Hardware aggregation (mean/std) ---------------------
def _find_candidate_ptms(run_dir: Path) -> list[Path]:
    out = []
    for fname in ("R.npy","R_hat.npy","R_ptm.npy","R.csv","R_hat.csv","ptm.npy"):
        p = run_dir / fname
        if p.exists():
            out.append(p)
    return out

def _load_ptm_file(p: Path) -> np.ndarray | None:
    try:
        if p.suffix == ".npy":
            arr = np.load(p)
        else:
            arr = np.loadtxt(p, delimiter=",")
        if arr.shape == (4,4):
            return arr.astype(float)
    except Exception:
        return None
    return None

def aggregate_target_mean_std(root: Path) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """
    Return (mean, std, n_used). Strategy:
      1) If R_avg.npy exists, use it as mean.
         If R_std.npy exists, use it as std.
      2) Otherwise, scan per-run subfolders and compute mean/std across runs.
    """
    p_avg = root / "R_avg.npy"
    p_std = root / "R_std.npy"
    R_mean, R_std, n_used = None, None, 0

    if p_avg.exists():
        R_mean = np.load(p_avg).astype(float)
        if p_std.exists():
            R_std = np.load(p_std).astype(float)

    if R_mean is None or R_std is None:
        mats = []
        if root.exists():
            for d in sorted([p for p in root.iterdir() if p.is_dir()]):
                for fp in _find_candidate_ptms(d):
                    arr = _load_ptm_file(fp)
                    if arr is not None:
                        mats.append(arr)
                        break
        if mats:
            stack = np.stack(mats, axis=0)        # (N,4,4)
            mean_ = stack.mean(axis=0)
            std_  = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(mean_)
            if R_mean is None: R_mean = mean_
            if R_std is None:  R_std  = std_
            n_used = stack.shape[0]

    return R_mean, R_std, n_used

# ---------------------- Random Δ samplers ---------------------
def sample_delta_hermitian(rng: np.random.Generator, scale: float) -> np.ndarray:
    # 2x2 Hermitian
    d00 = rng.uniform(-scale, scale)
    d11 = rng.uniform(-scale, scale)
    a01 = rng.uniform(-scale, scale)
    b01 = rng.uniform(-scale, scale)
    return np.array([[d00 + 0j,         a01 + 1j*b01],
                     [a01 - 1j*b01,     d11 + 0j   ]], dtype=complex)

def sample_delta_nonhermitian(rng: np.random.Generator, scale: float) -> np.ndarray:
    # 2x2 fully complex
    Re = rng.uniform(-scale, scale, size=(2,2))
    Im = rng.uniform(-scale, scale, size=(2,2))
    return Re + 1j*Im

def sample_noise_map(rng: np.random.Generator, scale: float, hermitian: bool) -> dict[str, np.ndarray]:
    fn = sample_delta_hermitian if hermitian else sample_delta_nonhermitian
    return {g: fn(rng, scale) for g in GATE_ORDER}

# ---------------------- QPT-12 runner ---------------------
def ptm_qpt12_for_noise_map(delta_map: dict[str, np.ndarray],
                            shots: int,
                            rng: np.random.Generator,
                            assume_tp: bool) -> np.ndarray:
    qpt = QPT12Strict(target_tokens=[("X",)])
    circ = Circuit(tokens=[("X",)], noise_map=delta_map or None)
    R, _, _ = qpt.ptm(shots=shots, rng=rng, assume_tp=assume_tp,
                      noise_map=circ.noise_map,
                      amp_damp_p=None, phase_damp_lam=None,
                      verbose=False, readout_loss=0.0)
    return np.asarray(R, float)

# ---------------------- z-score helpers & neat formatting ---------------------
def z_from_delta_and_sigma(delta: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    # safe division: sigma==0 -> z=0 if delta==0 else inf
    z = np.zeros_like(delta, dtype=float)
    zero = (sigma == 0.0)
    nz   = ~zero
    z[nz] = np.abs(delta[nz]) / sigma[nz]
    z[zero] = np.where(np.abs(delta[zero]) == 0.0, 0.0, np.inf)
    return z

def neat_z_block(Z: np.ndarray, title: str) -> str:
    bar = "─" * max(40, len(title) + 2)
    lines = [bar, f" {title}", bar]
    for i in range(Z.shape[0]):
        row_vals = []
        for j in range(Z.shape[1]):
            val = Z[i, j]
            cell = "∞" if np.isinf(val) else f"{val:.2f}"
            row_vals.append(f"✓ {cell}" if (not np.isinf(val)) and (val <= 1.0) else cell)
        lines.append(f"(row {i}) " + " | ".join(row_vals))
    return "\n".join(lines) + "\n"

# ------------------------- Main ---------------------------
def main() -> int:
    np.set_printoptions(precision=5, suppress=True)
    rng = np.random.default_rng(SEED)
    ensure_dir(OUT_DIR)

    # ===== Load/aggregate hardware target =====
    R_t_mean, R_t_std, n_runs = aggregate_target_mean_std(TARGET_DIR)
    if R_t_mean is None:
        print("[FATAL] Could not load or aggregate hardware target (R_avg or per-run PTMs).")
        return 2
    save_array(R_t_mean, "R_target_mean", OUT_DIR)
    if R_t_std is None:
        R_t_std = np.zeros_like(R_t_mean)
    save_array(R_t_std, "R_target_std", OUT_DIR)

    # Target plots (mean/std)
    plot_matrix(R_t_mean, "Target PTM — mean", OUT_DIR / "R_target_mean.png")
    plot_matrix(R_t_std,  "Target PTM — std",  OUT_DIR / "R_target_std.png")

    summary = {
        "seed": SEED,
        "shots_proc": SHOTS_PROC,
        "assume_tp": ASSUME_TP,
        "n_samples": N_SAMPLES,
        "noise_scale": NOISE_SCALE,
        "gate_order": list(GATE_ORDER),
        "target": {
            "mean_path": str((OUT_DIR / "R_target_mean.npy").resolve()),
            "std_path":  str((OUT_DIR / "R_target_std.npy").resolve()),
            "mean_png":  str((OUT_DIR / "R_target_mean.png").resolve()),
            "std_png":   str((OUT_DIR / "R_target_std.png").resolve()),
            "n_runs_used_for_target": int(n_runs),
        },
        "families": {}
    }

    report_lines = []
    report_lines.append(hdr("CONFIG"))
    report_lines.append(f"Output dir : {OUT_DIR.resolve()}")
    report_lines.append(f"Target dir : {TARGET_DIR.resolve()}")
    report_lines.append(f"Seed       : {SEED}")
    report_lines.append(f"Shots      : {SHOTS_PROC}")
    report_lines.append(f"Assume TP  : {ASSUME_TP}")
    report_lines.append(f"Gates      : {GATE_ORDER}")
    report_lines.append(f"N samples  : {N_SAMPLES} per family")
    report_lines.append(f"Noise scale: ±{NOISE_SCALE}\n")

    report_lines.append(hdr("HARDWARE TARGET (mean / std)"))
    report_lines.append(mat_block(R_t_mean, "R_target_mean:"))
    report_lines.append(mat_block(R_t_std,  "R_target_std:"))

    # Console printouts for TARGET mean/std
    print(hdr("HARDWARE TARGET (mean / std)"), end="")
    print(mat_block(R_t_mean, "R_target_mean:"), end="")
    print(mat_block(R_t_std,  "R_target_std:"), end="")

    # ===== Families to sweep (Hermitian / Non-Hermitian) =====
    families = [
        ("hermitian", True),
        ("nonhermitian", False),
    ]

    for fam_name, is_herm in families:
        fam_dir = OUT_DIR / f"random_{fam_name}"
        ensure_dir(fam_dir)

        # Collect N_SAMPLES PTMs with shots
        stack = []
        for k in range(N_SAMPLES):
            child_seed = (SEED ^ (k * 0x9E3779B1)) & 0xFFFFFFFF
            local_rng = np.random.default_rng(child_seed)
            nm = sample_noise_map(local_rng, NOISE_SCALE, hermitian=is_herm)
            Rk = ptm_qpt12_for_noise_map(nm, shots=SHOTS_PROC, rng=local_rng, assume_tp=ASSUME_TP)
            stack.append(Rk)

        stack = np.stack(stack, axis=0)   # (N,4,4)
        R_mean = stack.mean(axis=0)
        R_std  = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(R_mean)

        # Save mean/std
        save_array(R_mean, "R_mean", fam_dir)
        save_array(R_std,  "R_std",  fam_dir)

        # Plots for family mean/std
        plot_matrix(R_mean, f"{fam_name} — mean (shots-based)", fam_dir / "R_mean.png")
        plot_matrix(R_std,  f"{fam_name} — std (shots-based)",  fam_dir / "R_std.png")

        # Console printouts for FAMILY mean/std
        print(hdr(f"FAMILY: {fam_name.upper()} (mean / std)"), end="")
        print(mat_block(R_mean, f"{fam_name} — R_mean (shots-based):"), end="")
        print(mat_block(R_std,  f"{fam_name} — R_std (shots-based):"), end="")

        # z = |Δ| / sqrt( STD_target^2 + STD_family^2 )
        delta      = R_mean - R_t_mean
        sigma_comb = np.sqrt(np.square(R_t_std) + np.square(R_std))
        Z          = z_from_delta_and_sigma(delta, sigma_comb)

        save_array(delta,      "Delta",      fam_dir)
        save_array(sigma_comb, "Sigma_comb", fam_dir)
        save_array(Z,          "Z",          fam_dir)

        # neat z block (with ✓) for report and console
        neat = neat_z_block(Z, f"Z-SCORES (model vs hardware) — {fam_name}")
        (fam_dir / "Z_neat.txt").write_text(neat, encoding="utf-8")

        # Append to report
        report_lines.append(hdr(f"FAMILY: {fam_name.upper()}"))
        report_lines.append(mat_block(R_mean, f"{fam_name} — R_mean (shots-based):"))
        report_lines.append(mat_block(R_std,  f"{fam_name} — R_std (shots-based):"))
        report_lines.append(neat)

        # JSON paths
        summary["families"][fam_name] = {
            "mean_path":   str((fam_dir / "R_mean.npy").resolve()),
            "std_path":    str((fam_dir / "R_std.npy").resolve()),
            "mean_png":    str((fam_dir / "R_mean.png").resolve()),
            "std_png":     str((fam_dir / "R_std.png").resolve()),
            "delta_path":  str((fam_dir / "Delta.npy").resolve()),
            "sigma_path":  str((fam_dir / "Sigma_comb.npy").resolve()),
            "z_path":      str((fam_dir / "Z.npy").resolve()),
            "z_neat_path": str((fam_dir / "Z_neat.txt").resolve()),
        }

        # Console echo of the neat z section
        print(neat, end="")

    # ===== Write report + JSON =====
    (OUT_DIR / "report.txt").write_text("".join(report_lines), encoding="utf-8")
    (OUT_DIR / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(hdr("DONE"), end="")
    print("  • JSON:", (OUT_DIR / "report.json").resolve())
    print("  • TXT :", (OUT_DIR / "report.txt").resolve())
    print("  • Arrays and plots in:", OUT_DIR.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
