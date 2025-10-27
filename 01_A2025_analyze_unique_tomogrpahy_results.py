#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
aggregate_unique_tomos.py
=========================

Purpose
-------
Re-aggregate results **only** from curated folders:
  - unique_process_tomos/
  - unique_state_tomos/

It scans each folder's immediate subdirectories (assumed to be RUN_TAGs),
loads available artifacts, and recomputes:
  • Process: PTM aggregates (R_avg, R_std, R_err=R_avg−R_X)
             + R_mean_std.npy and human-readable R_mean_pm_std.txt
  • State  : rho, bloch aggregates (mean/std if present)
             + rho_mean_std.npz, rho_real/imag_mean_pm_std.txt
             + bloch_mean_std.npy, bloch_mean_pm_std.txt

No folder copying or manifest re-writing beyond a fresh per-type
manifest.json/csv for what was actually loaded this time.

Usage
-----
python aggregate_unique_tomos.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import List, Dict, Optional

import numpy as np

# ---------- Inputs ----------
OUT_PROC = Path("unique_process_tomos")
OUT_STATE = Path("unique_state_tomos")

# Files expected within each run folder (if present)
PTM_FILE = "ptm.npy"      # (4,4)
RHO_FILE = "rho.npy"      # (2,2) complex
BLOCH_FILE = "bloch.npy"  # (3,)   real

# Ideal X PTM in {I, X, Y, Z}
R_X = np.diag([1.0, 1.0, -1.0, -1.0])

np.set_printoptions(precision=5, suppress=True)


def save_array(arr: np.ndarray, name: str, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / f"{name}.npy", arr)
    np.savetxt(outdir / f"{name}.csv", arr, delimiter=",")


def save_mean_pm_std_text(mean: np.ndarray, std: np.ndarray, path: Path, fmt: str = "{: .6f} ± {: .6f}") -> None:
    """
    Save a human-readable table where each entry is formatted as 'mean ± std'.
    Supports 1D or 2D arrays.
    """
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


def aggregate_process(out_dir: Path) -> None:
    """
    Load all ptm.npy from out_dir/<RUN_TAG>/ and aggregate.
    Write summary + aggregates back into out_dir.
    """
    if not out_dir.exists():
        print(f"[INFO] {out_dir} does not exist; skipping process aggregation.")
        return

    # discover run folders (immediate subdirectories)
    run_dirs = sorted([p for p in out_dir.iterdir() if p.is_dir()])
    if not run_dirs:
        print(f"[INFO] No subfolders in {out_dir}; skipping process aggregation.")
        return

    manifest: List[Dict[str, object]] = []
    ptms: List[np.ndarray] = []

    for rd in run_dirs:
        ptm_path = rd / PTM_FILE
        has_ptm = False
        if ptm_path.exists():
            try:
                arr = np.load(ptm_path)
                if arr.shape == (4, 4):
                    ptms.append(arr.astype(float))
                    has_ptm = True
                else:
                    print(f"[WARN] {ptm_path} has shape {arr.shape}, expected (4,4). Skipping.")
            except Exception as exc:
                print(f"[WARN] Failed to load {ptm_path}: {exc}")

        manifest.append(
            {
                "run_folder": rd.name,
                "ptm_present": has_ptm,
                "ptm_relpath": str(PTM_FILE) if has_ptm else None,
            }
        )

    # write manifest for what we used
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run_folder", "ptm_present", "ptm_relpath"])
            for row in manifest:
                w.writerow([row["run_folder"], row["ptm_present"], row["ptm_relpath"]])
    except Exception as exc:
        print(f"[WARN] Failed to write {out_dir/'manifest.csv'}: {exc}")

    summary_lines = []
    summary_lines.append("=== PROCESS TOMOGRAPHY (from curated unique folders) ===")
    summary_lines.append(f"Run folders seen: {len(run_dirs)}")
    summary_lines.append(f"PTMs loaded     : {len(ptms)}")

    if ptms:
        R_all = np.stack(ptms)  # (n,4,4)
        R_avg = R_all.mean(axis=0)
        R_std = R_all.std(axis=0, ddof=1) if R_all.shape[0] >= 2 else np.zeros_like(R_avg)
        R_err = R_avg - R_X

        save_array(R_avg, "R_avg", out_dir)
        save_array(R_std, "R_std", out_dir)
        save_array(R_err, "R_err", out_dir)

        # Also bundle mean & std and human-readable ±
        R_mean_std = np.stack([R_avg, R_std], axis=-1)  # (4,4,2) [mean,std]
        np.save(out_dir / "R_mean_std.npy", R_mean_std)
        save_mean_pm_std_text(R_avg, R_std, out_dir / "R_mean_pm_std.txt")

        summary_lines.append("")
        summary_lines.append("Averaged PTM (R_avg):")
        summary_lines.append(str(R_avg))
        summary_lines.append("\nStandard deviation (R_std):")
        summary_lines.append(str(R_std))
        summary_lines.append("\nError matrix (R_err = R_avg − R_X):")
        summary_lines.append(str(R_err))
    else:
        summary_lines.append("No PTMs found; no aggregate statistics computed.")

    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"[OK] Process outputs written to: {out_dir.resolve()}")


def aggregate_state(out_dir: Path) -> None:
    """
    Load rho.npy and/or bloch.npy from out_dir/<RUN_TAG>/ and aggregate.
    Write summary + aggregates back into out_dir.
    """
    if not out_dir.exists():
        print(f"[INFO] {out_dir} does not exist; skipping state aggregation.")
        return

    run_dirs = sorted([p for p in out_dir.iterdir() if p.is_dir()])
    if not run_dirs:
        print(f"[INFO] No subfolders in {out_dir}; skipping state aggregation.")
        return

    manifest: List[Dict[str, object]] = []
    rhos: List[np.ndarray] = []
    blochs: List[np.ndarray] = []

    for rd in run_dirs:
        rho_path = rd / RHO_FILE
        bloch_path = rd / BLOCH_FILE

        has_rho = False
        has_bloch = False

        if rho_path.exists():
            try:
                rho = np.load(rho_path)
                if rho.shape == (2, 2):
                    rhos.append(rho.astype(complex))
                    has_rho = True
                else:
                    print(f"[WARN] {rho_path} has shape {rho.shape}, expected (2,2). Skipping rho.")
            except Exception as exc:
                print(f"[WARN] Failed to load {rho_path}: {exc}")

        if bloch_path.exists():
            try:
                rvec = np.load(bloch_path)
                if rvec.shape == (3,):
                    blochs.append(rvec.astype(float))
                    has_bloch = True
                else:
                    print(f"[WARN] {bloch_path} has shape {rvec.shape}, expected (3,). Skipping bloch.")
            except Exception as exc:
                print(f"[WARN] Failed to load {bloch_path}: {exc}")

        manifest.append(
            {
                "run_folder": rd.name,
                "rho_present": has_rho,
                "rho_relpath": str(RHO_FILE) if has_rho else None,
                "bloch_present": has_bloch,
                "bloch_relpath": str(BLOCH_FILE) if has_bloch else None,
            }
        )

    # write manifest for what we used
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run_folder", "rho_present", "rho_relpath", "bloch_present", "bloch_relpath"])
            for row in manifest:
                w.writerow([row["run_folder"], row["rho_present"], row["rho_relpath"],
                            row["bloch_present"], row["bloch_relpath"]])
    except Exception as exc:
        print(f"[WARN] Failed to write {out_dir/'manifest.csv'}: {exc}")

    summary_lines = []
    summary_lines.append("=== STATE TOMOGRAPHY (from curated unique folders) ===")
    summary_lines.append(f"Run folders seen : {len(run_dirs)}")
    summary_lines.append(f"rho.npy loaded   : {len(rhos)}")
    summary_lines.append(f"bloch.npy loaded : {len(blochs)}")

    # rho aggregation
    if rhos:
        rho_all = np.stack(rhos)  # (n,2,2)
        rho_avg = rho_all.mean(axis=0)
        rho_std = rho_all.std(axis=0, ddof=1) if rho_all.shape[0] >= 2 else np.zeros_like(rho_avg)

        # Save complex arrays
        np.save(out_dir / "rho_avg.npy", rho_avg)
        np.save(out_dir / "rho_std.npy", rho_std)
        # Bundle both
        np.savez(out_dir / "rho_mean_std.npz", rho_avg=rho_avg, rho_std=rho_std)

        # Human-readable ± for real/imag parts
        save_mean_pm_std_text(rho_avg.real, rho_std.real, out_dir / "rho_real_mean_pm_std.txt")
        save_mean_pm_std_text(rho_avg.imag, rho_std.imag, out_dir / "rho_imag_mean_pm_std.txt")

        summary_lines.append("")
        summary_lines.append("rho_avg (complex):")
        summary_lines.append(str(rho_avg))
        summary_lines.append("\nrho_std (complex):")
        summary_lines.append(str(rho_std))

    # bloch aggregation
    if blochs:
        r_all = np.stack(blochs)  # (n,3)
        r_avg = r_all.mean(axis=0)
        r_std = r_all.std(axis=0, ddof=1) if r_all.shape[0] >= 2 else np.zeros_like(r_avg)

        # Save separate and combined
        save_array(r_avg, "bloch_avg", out_dir)
        save_array(r_std, "bloch_std", out_dir)
        bloch_mean_std = np.stack([r_avg, r_std], axis=0)  # (2,3): [mean; std]
        np.save(out_dir / "bloch_mean_std.npy", bloch_mean_std)
        save_mean_pm_std_text(r_avg, r_std, out_dir / "bloch_mean_pm_std.txt")

        summary_lines.append("")
        summary_lines.append("Bloch vector average (bloch_avg):")
        summary_lines.append(str(r_avg))
        summary_lines.append("\nBloch vector std (bloch_std):")
        summary_lines.append(str(r_std))

    if not rhos and not blochs:
        summary_lines.append("No state artifacts found; no aggregate statistics computed.")

    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"[OK] State outputs written to: {out_dir.resolve()}")


def main() -> int:
    # Process aggregation
    aggregate_process(OUT_PROC)

    # State aggregation
    aggregate_state(OUT_STATE)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
