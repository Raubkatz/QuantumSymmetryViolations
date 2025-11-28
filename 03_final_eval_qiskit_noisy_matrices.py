#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QPT-12 (Qiskit basis) — process-only evaluation (means + one std per mean, no differences).

What this script does:
  1) Ideal case (Δ=0) QPT-12:
       - shots=0 (analytic) → R_ideal_analytic, STD=0
       - shots>0 (configurable) → mean/std over N_EVAL_SHOTS re-evaluations (each with 'shots')
  2) (Optional) Hardware recap:
       - Loads R_target (mean) from unique_process_tomos/ (std is ignored/unused)
  3) Noisy matrices:
       - Auto-discovers variants under NOISY_ROOT (stacked best_delta.npy or per-gate {Rz.npy,SX.npy,X.npy})
       - For each variant: QPT-12 with shots=0 (analytic) and shots>0 (mean/std over N_EVAL_SHOTS)

Files saved (no Δ, no stderr):
  • ideal/: R_ideal_shots0_mean.npy, and for sampled shots: R_ideal_shotsK_mean.npy, R_ideal_shotsK_std.npy
  • variants/<name>/shotsK/: R_qpt12_mean.npy, R_qpt12_std.npy (+ mean±std TXT)
  • PNG heatmaps for stored matrices
  • report.txt and report.json (minimal)
"""

from __future__ import annotations

from pathlib import Path
import sys, json, datetime
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Your tomography codebase --------------------
from class_tomography_final import Circuit, QPT12Strict  # keep imports minimal & stable

# -------------------- CONFIG --------------------
PROCESS_TARGET_DIR = Path("unique_process_tomos")
#NOISY_ROOT         = Path("fit_results_qiskit_base_04112025_12k_1500Elites")  # adjust as needed
NOISY_ROOT         = Path("test_28_11_2025")
GATE_ORDER         = ("Rz", "SX", "X")

STAMP    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_ROOT = Path(f"eval_qpt12_process_only_{STAMP}")

# shots list: first is analytic (0), others are sampled
SHOTS_LIST    = [0, 10000]
N_EVAL_SHOTS  = 1000   # number of independent re-evaluations per shots-based estimate

ASSUME_TP    = False     # keep as your GA/eval convention
READOUT_LOSS = 0.0       # off

SEED = 2025
plt.rcParams.update({"figure.dpi": 120})

# -------------------- NUMPY PRINT --------------------
np.set_printoptions(precision=5, suppress=True)

# -------------------- PRETTY PRINT HELPERS --------------------
def _hdr(title: str) -> None:
    bar = "─" * max(10, len(title) + 2)
    print(f"\n{bar}\n {title}\n{bar}")

def _print_mat(label: str, M: np.ndarray) -> None:
    print(f"\n{label}:")
    print(np.array2string(M, precision=5, suppress_small=True))

# -------------------- I/O --------------------
def save_array(arr: np.ndarray, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / f"{name}.npy", arr)
    try:
        np.savetxt(outdir / f"{name}.csv", arr, delimiter=",")
    except Exception:
        # complex arrays or non-2D -> skip CSV
        pass

def plot_matrix(M: np.ndarray, title: str, outpng: Path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(M, aspect="auto")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("cols (I,X,Y,Z)")
    ax.set_ylabel("rows (I,X,Y,Z)")
    fig.tight_layout()
    fig.savefig(outpng)
    plt.close(fig)

def _save_mean_pm_std_text(mean: np.ndarray, std: np.ndarray, path: Path, fmt: str = "{: .6f} ± {: .6f}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if mean.ndim == 1:
            row = [fmt.format(float(m), float(s)) for m, s in zip(mean.tolist(), std.tolist())]
            f.write("\t".join(row) + "\n")
        elif mean.ndim == 2:
            for i in range(mean.shape[0]):
                row = [fmt.format(float(mean[i, j]), float(std[i, j])) for j in range(mean.shape[1])]
                f.write("\t".join(row) + "\n")
        else:
            f.write("Mean:\n")
            f.write(str(mean) + "\n\nStd:\n")
            f.write(str(std) + "\n")

# -------------------- HARDWARE TARGET LOADERS --------------------
def _find_candidate_ptms(run_dir: Path) -> list[Path]:
    candidates = []
    for fname in ["R.npy", "R_hat.npy", "R_ptm.npy", "R.csv", "R_hat.csv", "ptm.npy"]:
        p = run_dir / fname
        if p.exists():
            candidates.append(p)
    return candidates

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

def _aggregate_runs(root: Path) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """
    Scan PROCESS_TARGET_DIR/* subfolders for per-run PTMs and aggregate.
    Returns (mean, std, n_used). We only use the mean; std is ignored by design.
    """
    if not root.exists():
        return None, None, 0
    mats = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        for fp in _find_candidate_ptms(d):
            arr = _load_ptm_file(fp)
            if arr is not None:
                mats.append(arr)
                break  # only one per run-dir
    if not mats:
        return None, None, 0
    stack = np.stack(mats, 0)  # (N,4,4)
    mean = np.mean(stack, 0)
    std  = np.std(stack, 0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(mean)
    return mean, std, stack.shape[0]

def load_process_targets():
    """
    Original behavior:
      - Load R_avg.npy as R
      - Optionally load R_std.npy as Rstd (ignored here).
    Augmented:
      - If R_avg.npy is missing but per-run PTMs exist, fall back to their mean.
    """
    p_avg = PROCESS_TARGET_DIR / "R_avg.npy"

    R = None
    if p_avg.exists():
        R = np.load(p_avg)

    if R is None:
        agg_mean, _, n_used = _aggregate_runs(PROCESS_TARGET_DIR)
        if agg_mean is not None:
            R = agg_mean
            print(f"[AGG] Used {n_used} hardware run(s) to build R_target mean.")

    if R is None:
        print(f"[FATAL] Missing process target PTM: {p_avg} and no usable per-run PTMs found under {PROCESS_TARGET_DIR}/", file=sys.stderr)
        sys.exit(2)

    return R.astype(float)

# -------------------- NOISY MAP DISCOVERY --------------------
def discover_noise_maps(root: Path,
                        gate_order: tuple[str, ...] = ("Rz","SX","X"),
                        debug: bool = True):
    def load_stack(p: Path):
        try:
            data = np.load(p)
        except Exception as e:
            if debug: print(f"[DISCOVER] skip {p}: np.load error → {e}")
            return None
        if data.shape != (len(gate_order), 2, 2):
            if debug: print(f"[DISCOVER] skip {p}: shape {data.shape} != {(len(gate_order),2,2)}")
            return None
        return {g: data[i] for i, g in enumerate(gate_order)}

    def load_per_gate(d: Path):
        maps = {}
        for g in gate_order:
            fp = d / f"{g}.npy"
            if not fp.exists():
                if debug: print(f"[DISCOVER] {d}: missing {fp.name}")
                return None
            try:
                M = np.load(fp)
            except Exception as e:
                if debug: print(f"[DISCOVER] {fp}: np.load error → {e}")
                return None
            if M.shape != (2,2):
                if debug: print(f"[DISCOVER] {fp}: shape {M.shape} != (2,2)")
                return None
            maps[g] = M
        return maps

    variants: list[tuple[str, dict]] = [("ideal", {})]

    if not root.exists():
        print(f"[WARN] NOISY_ROOT does not exist: {root} — only 'ideal' will run.")
        return variants

    # root-level
    root_variant = None
    if (root / "best_delta.npy").exists():
        nm = load_stack(root / "best_delta.npy")
        if nm: root_variant = (root.stem or "root", nm)
    else:
        nm = load_per_gate(root)
        if nm: root_variant = (root.stem or "root", nm)
    if root_variant:
        if debug: print(f"[DISCOVER] root-level variant: {root_variant[0]}")
        variants.append(root_variant)

    # one or two levels below
    candidates = []
    for d in [p for p in root.iterdir() if p.is_dir()]:
        candidates.append(d)
        for d2 in [p2 for p2 in d.iterdir() if p2.is_dir()]:
            candidates.append(d2)

    for d in sorted(candidates):
        name = d.name
        nm = None
        if (d / "best_delta.npy").exists():
            nm = load_stack(d / "best_delta.npy")
            if nm and debug: print(f"[DISCOVER] variant via stack: {name}")
        else:
            nm = load_per_gate(d)
            if nm and debug: print(f"[DISCOVER] variant via per-gate: {name}")
        if nm:
            variants.append((name, nm))

    # dedup on name, keep last
    dedup = {}
    for k, v in variants:
        dedup[k] = v
    out = [("ideal", {})] + [(k, dedup[k]) for k in dedup if k != "ideal"]
    if debug:
        print("\n[DISCOVER] Summary:")
        for k, nm in out:
            print(f"  - {k}: {'ideal' if not nm else 'Δ loaded for ' + ','.join(nm.keys())}")
    return out

# -------------------- QPT-12 RUNNER --------------------
def build_x_target_circuit(noise_map: dict[str, np.ndarray]):
    return Circuit(tokens=[("X",)], noise_map=noise_map or None)

def run_qpt_qiskit(circ,
                   rng: np.random.Generator,
                   shots: int,
                   assume_tp: bool,
                   *,
                   n_eval_shots: int) -> dict:
    """
    QPT-12 (strict).
      • shots=0  → single analytic evaluation (STD = 0)
      • shots>0  → run the tomography n_eval_shots times, each with 'shots' and the
                   given noise map; return component-wise mean and std over those runs
    Returns dict: {'R_mean', 'R_std'}; STD=0 for analytic. (No stderr, no deltas.)
    """
    qpt = QPT12Strict(target_tokens=[("X",)])
    if shots == 0:
        R, _, _ = qpt.ptm(shots=0, rng=rng, assume_tp=assume_tp,
                          noise_map=circ.noise_map,
                          amp_damp_p=None, phase_damp_lam=None,
                          readout_loss=READOUT_LOSS,
                          verbose=False)
        R = np.asarray(R, float)
        return {"R_mean": R, "R_std": np.zeros_like(R)}

    # shots > 0 → repeated re-evaluation
    Rs = []
    for i in range(int(max(1, n_eval_shots))):
        # NEW: independent deterministic RNG per iteration
        child_seed = (int(SEED) ^ (int(shots) * 0x9E3779B1) ^ (i * 0x85EBCA6B)) & 0xFFFFFFFF
        local_rng = np.random.default_rng(child_seed)

        R, _, _ = qpt.ptm(shots=shots, rng=local_rng, assume_tp=assume_tp,
                          noise_map=circ.noise_map,
                          amp_damp_p=None, phase_damp_lam=None,
                          readout_loss=READOUT_LOSS,
                          verbose=False)
        Rs.append(np.asarray(R, float))
    stack = np.stack(Rs, 0)
    Rm = np.mean(stack, 0)
    Rstd = np.std(stack, 0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(Rm)
    return {"R_mean": Rm, "R_std": Rstd}

# -------------------- Z-SCORE UTILS (NEW) --------------------
def _sigma_comb(std_target: np.ndarray, std_model: np.ndarray | None) -> np.ndarray:
    if std_model is None:
        return np.array(std_target, float)
    return np.sqrt(np.asarray(std_target, float)**2 + np.asarray(std_model, float)**2)

def _zscores(R_model: np.ndarray, R_target: np.ndarray,
             std_target: np.ndarray, std_model: np.ndarray | None) -> np.ndarray:
    d = np.abs(np.asarray(R_model, float) - np.asarray(R_target, float))
    sc = _sigma_comb(std_target, std_model)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sc > 0, d / sc, np.inf)
    return z

def _format_row_with_checks(zrow: np.ndarray, decimals: int = 2) -> list[str]:
    out = []
    for val in zrow.tolist():
        s = f"{val:.{decimals}f}"
        if val <= 1.0:
            s = f"✓ {s}"
        out.append(s)
    return out

def _write_latex_z_table(z: np.ndarray, out_tex: Path, caption: str, label: str):
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    with out_tex.open("w", encoding="utf-8") as f:
        f.write(r"\begin{table}[t]"+"\n")
        f.write(r"\centering"+"\n")
        f.write(rf"\caption{{{caption}}}"+"\n")
        f.write(rf"\label{{{label}}}"+"\n")
        f.write(r"\setlength{\tabcolsep}{6pt}"+"\n")
        f.write(r"\renewcommand{\arraystretch}{1.15}"+"\n")
        f.write(r"\begin{tabular}{c|rrrr}"+"\n")
        f.write(r"\toprule"+"\n")
        f.write(r" & \multicolumn{4}{c}{$z$ (\,$\checkmark$ if $z\le 1$\,)}\\"+"\n")
        f.write(r"\midrule"+"\n")
        for i in range(4):
            cells = _format_row_with_checks(z[i], decimals=2)
            row = " & ".join(cells)
            f.write(rf"$({i},\cdot)$ & {row}\\")
            f.write("\n")
        f.write(r"\bottomrule"+"\n")
        f.write(r"\end{tabular}"+"\n")
        f.write(r"\end{table}"+"\n")

# ------------------------------ MAIN ------------------------------
class _Tee:
    """Duplicate stdout to both console and a report file."""
    def __init__(self, stream_a, stream_b):
        self.a = stream_a
        self.b = stream_b
    def write(self, data):
        self.a.write(data)
        self.b.write(data)
    def flush(self):
        try:
            self.a.flush()
        finally:
            # Guard against closed file handles
            try:
                self.b.flush()
            except Exception:
                pass

def main():
    rng = np.random.default_rng(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ---- tee all prints into report.txt as well ----
    report_path = OUT_ROOT / "report.txt"
    report_fh = open(report_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, report_fh)

    # ===== Load / Aggregate hardware target (process) =====
    R_target = load_process_targets()
    # NEW: also compute std across all hardware runs (if runs are available)
    agg_mean, agg_std, n_used = _aggregate_runs(PROCESS_TARGET_DIR)

    _hdr("HARDWARE TARGET (process PTM)")
    _print_mat("R_target (mean)", R_target)
    if agg_std is not None:
        print(f"\n[HARDWARE RUNS USED] n_runs={n_used}")
        _print_mat("R_target_std (over runs)", agg_std)
        save_array(agg_std, OUT_ROOT, "hardware_R_target_std")
        plot_matrix(agg_std, "hardware_R_target_std", OUT_ROOT / "hardware_R_target_std.png")

    # convenience copy (non-breaking)
    save_array(R_target, OUT_ROOT, "hardware_R_target_mean")
    plot_matrix(R_target, "hardware_R_target_mean", OUT_ROOT / "hardware_R_target_mean.png")

    # ===== Ideal (Δ=0) =====
    _hdr("IDEAL (Δ=0) — Process Tomography via QPT-12")
    ideal_dir = OUT_ROOT / "ideal"
    ideal_dir.mkdir(parents=True, exist_ok=True)

    for shots in SHOTS_LIST:
        circ_ideal = build_x_target_circuit({})
        print(f"[RUN] ideal | shots={shots} | TP={ASSUME_TP}")
        res = run_qpt_qiskit(circ_ideal, rng, shots, assume_tp=ASSUME_TP, n_eval_shots=N_EVAL_SHOTS)
        save_array(res["R_mean"], ideal_dir, f"R_ideal_shots{shots}_mean")
        plot_matrix(res["R_mean"], f"R_ideal_shots{shots}_mean", ideal_dir / f"R_ideal_shots{shots}_mean.png")
        _print_mat(f"R_ideal_shots{shots}_mean", res["R_mean"])
        if shots > 0:
            save_array(res["R_std"], ideal_dir, f"R_ideal_shots{shots}_std")
            plot_matrix(res["R_std"], f"R_ideal_shots{shots}_std", ideal_dir / f"R_ideal_shots{shots}_std.png")
            _save_mean_pm_std_text(res["R_mean"], res["R_std"], ideal_dir / f"R_ideal_shots{shots}_mean_pm_std.txt")
            _print_mat(f"R_ideal_shots{shots}_std", res["R_std"])

    # ===== Noisy variants (symmetry-violation candidates) =====
    _hdr("DISCOVER VARIANTS")
    variants = discover_noise_maps(NOISY_ROOT, gate_order=GATE_ORDER)
    for name, nm in variants:
        print("  -", f"{name} ({'with Δ' if nm else 'ideal'})")

    for name, nm in variants:
        vdir = OUT_ROOT / "variants" / name
        vdir.mkdir(parents=True, exist_ok=True)

        if nm:
            stack = np.stack([nm[g] for g in GATE_ORDER], 0)
            save_array(stack, vdir, "delta_stack")
            _hdr(f"VARIANT {name}: Δ generators")
            for g in GATE_ORDER:
                _print_mat(f"Δ[{g}] Re", nm[g].real)
                _print_mat(f"Δ[{g}] Im", nm[g].imag)
        else:
            print(f"\n[VARIANT {name}] Ideal (no Δ)")

        for shots in SHOTS_LIST:
            print(f"[RUN] variant={name} | shots={shots} | TP={ASSUME_TP}")
            circ_hat = build_x_target_circuit(nm)

            # Mean/std over N_EVAL_SHOTS for the noisy estimate
            res_hat = run_qpt_qiskit(circ_hat, rng, shots, assume_tp=ASSUME_TP, n_eval_shots=N_EVAL_SHOTS)

            rdir = vdir / f"shots{shots}"
            rdir.mkdir(parents=True, exist_ok=True)

            # Save estimate mean (+ plot + print)
            save_array(res_hat["R_mean"], rdir, "R_qpt12_mean")
            plot_matrix(res_hat["R_mean"], "R_qpt12_mean", rdir / "R_qpt12_mean.png")
            _print_mat("R_qpt12_mean", res_hat["R_mean"])

            # Save estimate std for sampled shots (+ plot + print) and a mean±std txt table
            if shots > 0:
                save_array(res_hat["R_std"], rdir, "R_qpt12_std")
                plot_matrix(res_hat["R_std"], "R_qpt12_std", rdir / "R_qpt12_std.png")
                _print_mat("R_qpt12_std", res_hat["R_std"])
                _save_mean_pm_std_text(res_hat["R_mean"], res_hat["R_std"], rdir / "R_qpt12_mean_pm_std.txt")

            # --------- Z-SCORES vs hardware (NEW) ----------
            if agg_std is not None:
                if shots == 0:
                    z = _zscores(res_hat["R_mean"], R_target, agg_std, None)
                    save_array(z, rdir, "R_zscores_shots0")
                    np.savetxt(rdir / "R_zscores_shots0.csv", z, delimiter=",")
                    _hdr("Z-SCORES (model vs hardware) — shots=0")
                    # print compact row-wise table
                    for i in range(4):
                        row_str = " | ".join(_format_row_with_checks(z[i], decimals=2))
                        print(f"(row {i}) {row_str}")
                else:
                    z = _zscores(res_hat["R_mean"], R_target, agg_std, res_hat["R_std"])
                    save_array(z, rdir, "R_zscores_shots_sampled")
                    np.savetxt(rdir / "R_zscores_shots_sampled.csv", z, delimiter=",")
                    _hdr("Z-SCORES (model vs hardware) — sampled")
                    for i in range(4):
                        row_str = " | ".join(_format_row_with_checks(z[i], decimals=2))
                        print(f"(row {i}) {row_str}")
                    # also emit a LaTeX table mirroring the manuscript style
                    _write_latex_z_table(
                        z,
                        rdir / "zscores_table.tex",
                        caption="Entry-wise comparison for the sampled model ($\\text{shots}{=}10^4$): $z=|\\Delta R|/\\sigma_{\\text{comb}}$ and $1\\sigma$ consistency.",
                        label="tab:zscores"
                    )

    # ===== Minimal report (no Δ, no stderr) =====
    txt = []
    txt.append("=== QPT-12 Process-Only Evaluation (means + one std per mean; no differences) ===")
    txt.append(f"Output root: {OUT_ROOT.resolve()}")
    txt.append(f"Assume TP: {ASSUME_TP}")
    txt.append(f"Shots list: {SHOTS_LIST} (analytic + sampled)")
    txt.append(f"Re-evaluations per sampled: N_EVAL_SHOTS={N_EVAL_SHOTS}")
    if agg_std is not None:
        txt.append(f"Hardware runs aggregated: n_runs={n_used}")
        # (keep matrices printed above; summary keeps meta)
    _hdr("SUMMARY")
    print("\n".join(txt))
    # APPEND summary to the same report (since all prints already streamed there)
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(txt) + "\n")

    # ===== Structured JSON summary (no Δ, no stderr) =====
    summary = {
        "config": {
            "seed": SEED,
            "assume_tp": ASSUME_TP,
            "shots_list": SHOTS_LIST,
            "n_eval_shots": N_EVAL_SHOTS,
            "readout_loss": READOUT_LOSS,
            "gate_order": list(GATE_ORDER),
            "noisy_root": str(NOISY_ROOT.resolve()),
        },
        "paths": {
            "hardware_R_target_mean": str((OUT_ROOT / "hardware_R_target_mean.npy").resolve()),
            "ideal_dir": str(ideal_dir.resolve()),
            "variants_dir": str((OUT_ROOT / "variants").resolve()),
        }
    }
    # include std path if available
    if agg_std is not None:
        summary["paths"]["hardware_R_target_std"] = str((OUT_ROOT / "hardware_R_target_std.npy").resolve())
        summary["hardware_runs"] = {"n_runs": int(n_used)}

    (OUT_ROOT / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _hdr("DONE")
    print(f"[OK] Results in: {OUT_ROOT.resolve()}")
    print("  • report.txt")
    print("  • report.json")
    print("  • ideal/ and variants/ subfolders with arrays & plots (means + std only)")
    print("  • Per-variant z-scores (NPY/CSV) and LaTeX table for sampled case")

    # close tee file handle
    try:
        report_fh.flush()
    finally:
        try:
            report_fh.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
