#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
aggregate_unique_tomos.py
=========================

High-level description
----------------------
This script aggregates **process tomography** results from curated folders
produced by an external tomography / experiment pipeline.

The pipeline is assumed to have already:
  - Performed quantum process tomography for a set of runs.
  - Curated those runs such that only one representative instance per
    configuration (e.g. per gate, per parameter setting, per device, etc.)
    is kept in the directory ``unique_process_tomos/``.
  - Written, for each run, a file ``ptm.npy`` containing the reconstructed
    process matrix (also called Pauli transfer matrix, PTM) in the
    Pauli basis ordered as {I, X, Y, Z} with shape (4, 4).

This script does **not** perform tomography itself. It only:
  1. Scans the curated process-tomography output directory
     ``unique_process_tomos/``.
  2. Treats each immediate subdirectory as a single "run folder"
     (a single experimental or numerical run with its own tomography result).
  3. For every run folder that contains a valid ``ptm.npy`` of shape (4, 4),
     loads that PTM.
  4. Aggregates all loaded PTMs by computing:
       - The element-wise mean over runs (R_avg).
       - The element-wise sample standard deviation over runs (R_std).
       - The deviation of the mean PTM from a fixed reference PTM, here the
         ideal X gate PTM R_X (R_err = R_avg − R_X).
  5. Writes out:
       - Numpy arrays of R_avg, R_std, and R_err as ``.npy`` and ``.csv``.
       - A compact file ``R_mean_std.npy`` containing a stacked
         (4, 4, 2) array [mean, std] for downstream analysis.
       - A human-readable text file ``R_mean_pm_std.txt`` in which each entry
         is printed as "mean ± std".
       - A per-run manifest as ``manifest.json`` and ``manifest.csv``
         that documents which runs were seen and whether a usable PTM was
         found in each of them.
       - A plain-text ``summary.txt`` summarizing the aggregation results
         and printing R_avg, R_std, and R_err.

Intended use in the workflow
----------------------------
In the context of a tomography / characterization study, this script is
intended to be run **after**:
  - All individual tomography reconstructions have completed.
  - A separate curation step has selected a set of "unique" runs
    (e.g. one run per configuration) and placed them into
    ``unique_process_tomos/``.

The aggregated statistics produced here can then be:
  - Used for plotting average process matrices with uncertainty bars.
  - Compared to an ideal reference PTM (here X) to quantify systematic
    deviations.
  - Exported to other analysis scripts, notebooks, or plotting tools.

"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import List, Dict, Optional

import numpy as np

# ---------- Inputs ----------
# Directory that contains curated process-tomography results. Each immediate
# subdirectory is interpreted as a single run folder (RUN_TAG).
OUT_PROC = Path("unique_process_tomos")

# Files expected within each run folder (if present).
# Here only PTM_FILE is used, since this script aggregates process tomographies.
PTM_FILE = "ptm.npy"      # Expected shape: (4, 4), real-valued Pauli transfer matrix

# Ideal X PTM in the {I, X, Y, Z} Pauli basis.
# This is used as a fixed reference to compute a simple error matrix
# R_err = R_avg − R_X for the aggregated PTM.
R_X = np.diag([1.0, 1.0, -1.0, -1.0])

# Global numpy print options for cleaner matrices in summary.txt and stdout.
np.set_printoptions(precision=5, suppress=True)


def save_array(arr: np.ndarray, name: str, outdir: Path) -> None:
    """
    Save a numpy array both as .npy and .csv into a given output directory.

    Parameters
    ----------
    arr : np.ndarray
        Array to be stored. No shape or dtype assumptions are enforced here.
    name : str
        Base name (without extension) used for the output files.
        Files will be written as "<name>.npy" and "<name>.csv".
    outdir : pathlib.Path
        Directory where the files are written. Created if it does not exist.

    Notes
    -----
    - The .npy file preserves the exact dtype and shape.
    - The .csv file is written using comma-separated values, primarily to
      allow quick inspection or import in other tools that handle CSV.
    """
    # Ensure the output directory exists before writing.
    outdir.mkdir(parents=True, exist_ok=True)

    # Save in NumPy's binary format.
    np.save(outdir / f"{name}.npy", arr)

    # Save a CSV representation for quick inspection.
    np.savetxt(outdir / f"{name}.csv", arr, delimiter=",")


def save_mean_pm_std_text(
    mean: np.ndarray,
    std: np.ndarray,
    path: Path,
    fmt: str = "{: .6f} ± {: .6f}",
) -> None:
    """
    Save a human-readable table where each entry is formatted as 'mean ± std'.

    Parameters
    ----------
    mean : np.ndarray
        Array containing mean values. Accepted shapes are 1D or 2D.
    std : np.ndarray
        Array of standard deviations with the same shape as `mean`.
    path : pathlib.Path
        Target text file path.
    fmt : str, optional
        Format string used for each pair (mean, std). The default is a
        fixed-width floating point format with six decimals.

    Behavior
    --------
    - If `mean` is 1D, a single line is written with tab-separated entries,
      each of the form "mean ± std".
    - If `mean` is 2D, one line per row is written, with tab-separated
      entries "mean ± std".
    - For higher-dimensional inputs, the function falls back to printing
      raw `mean` and `std` arrays.

    Purpose
    -------
    This function produces a compact and human-readable summary that can be
    directly included in the supplementary material, logs, or used as input
    to manual inspection without requiring a Python environment.
    """
    # Ensure the parent directory exists.
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        if mean.ndim == 1:
            # 1D case: write all entries into one line, separated by tabs.
            row = [fmt.format(float(m), float(s)) for m, s in zip(mean.tolist(), std.tolist())]
            f.write("\t".join(row) + "\n")
        elif mean.ndim == 2:
            # 2D case: write one line per row of the array.
            for i in range(mean.shape[0]):
                row = [fmt.format(float(mean[i, j]), float(std[i, j])) for j in range(mean.shape[1])]
                f.write("\t".join(row) + "\n")
        else:
            # Fallback: in case someone passes an array with ndim > 2, just dump
            # the raw mean and std, which at least preserves all information.
            f.write("Mean:\n")
            f.write(str(mean) + "\n\nStd:\n")
            f.write(str(std) + "\n")


def aggregate_process(out_dir: Path) -> None:
    """
    Aggregate process-tomography PTMs stored in a curated directory.

    Parameters
    ----------
    out_dir : pathlib.Path
        Directory that contains curated run subdirectories. Each immediate
        subdirectory is expected to represent one run and may contain a
        file named `PTM_FILE` (typically ``ptm.npy``) with shape (4, 4).

    Behavior
    --------
    1. Enumerates all immediate subdirectories of `out_dir` and treats them
       as run folders.
    2. For each run folder:
         - Attempts to load `PTM_FILE`.
         - Checks the shape; if it is (4, 4), the PTM is accepted and stored.
         - Records in a manifest whether a valid PTM was found.
    3. Writes the manifest as both ``manifest.json`` and ``manifest.csv``.
    4. If at least one valid PTM was loaded:
         - Stacks all PTMs into a single array with shape (n, 4, 4).
         - Computes the element-wise mean (R_avg) across the run dimension.
         - Computes the sample standard deviation (R_std) with ddof=1
           (unbiased estimator) if n >= 2; otherwise zeros.
         - Computes R_err = R_avg − R_X, where R_X is the ideal X gate PTM.
         - Writes R_avg, R_std, R_err as numpy arrays and CSV files.
         - Writes a helper file ``R_mean_std.npy`` that stacks the mean and
           standard deviation into shape (4, 4, 2).
         - Writes ``R_mean_pm_std.txt`` as a human-readable "mean ± std"
           table of size 4×4.
         - Writes ``summary.txt`` summarizing the aggregation and printing
           the resulting matrices.
       Otherwise:
         - Writes ``summary.txt`` indicating that no PTMs were found
           and no statistics were computed.

    Notes
    -----
    - Any PTM file that does not have shape (4, 4) is ignored and a warning
      is printed.
    - Any load error (e.g. corrupt file) is reported via a warning and that
      run is recorded as not having a valid PTM.
    """
    # If the curated directory does not exist, there is nothing to aggregate.
    if not out_dir.exists():
        print(f"[INFO] {out_dir} does not exist; skipping process aggregation.")
        return

    # Discover run folders (immediate subdirectories only).
    run_dirs = sorted([p for p in out_dir.iterdir() if p.is_dir()])
    if not run_dirs:
        print(f"[INFO] No subfolders in {out_dir}; skipping process aggregation.")
        return

    # Manifest will capture, per run folder, whether we found a PTM and
    # where it was located relative to the run directory.
    manifest: List[Dict[str, object]] = []
    # List that will collect all successfully loaded PTMs.
    ptms: List[np.ndarray] = []

    # Iterate over all run directories in a deterministic (sorted) order.
    for rd in run_dirs:
        ptm_path = rd / PTM_FILE
        has_ptm = False

        # Try to load the PTM matrix if the expected file exists.
        if ptm_path.exists():
            try:
                arr = np.load(ptm_path)
                if arr.shape == (4, 4):
                    # Store a float version of the PTM. Casting to float ensures
                    # that we drop any complex part if present, which should not
                    # occur for a PTM in the Pauli basis but is enforced here.
                    ptms.append(arr.astype(float))
                    has_ptm = True
                else:
                    # Shape mismatch, ignore this file but inform the user.
                    print(f"[WARN] {ptm_path} has shape {arr.shape}, expected (4,4). Skipping.")
            except Exception as exc:
                # Any load error (I/O, parsing, etc.) is reported, but does not
                # stop the aggregation of other runs.
                print(f"[WARN] Failed to load {ptm_path}: {exc}")

        # Add an entry to the manifest for this run folder.
        manifest.append(
            {
                "run_folder": rd.name,
                "ptm_present": has_ptm,
                "ptm_relpath": str(PTM_FILE) if has_ptm else None,
            }
        )

    # Write manifest as JSON for machine-readable inspection and reproducibility.
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Write manifest as CSV for quick inspection in spreadsheets or text editors.
    try:
        with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run_folder", "ptm_present", "ptm_relpath"])
            for row in manifest:
                w.writerow([row["run_folder"], row["ptm_present"], row["ptm_relpath"]])
    except Exception as exc:
        # Failure to write the CSV is non-critical, but worth reporting.
        print(f"[WARN] Failed to write {out_dir/'manifest.csv'}: {exc}")

    # Prepare a list of human-readable lines that will be written to summary.txt
    # and also printed to stdout for immediate feedback.
    summary_lines = []
    summary_lines.append("=== PROCESS TOMOGRAPHY (from curated unique folders) ===")
    summary_lines.append(f"Run folders seen: {len(run_dirs)}")
    summary_lines.append(f"PTMs loaded     : {len(ptms)}")

    if ptms:
        # Stack all PTMs into a single array of shape (n, 4, 4), where
        # n = number of successfully loaded runs.
        R_all = np.stack(ptms)  # (n, 4, 4)

        # Element-wise mean over the first dimension (runs).
        R_avg = R_all.mean(axis=0)

        # Sample standard deviation with ddof=1 (unbiased estimator)
        # if n >= 2. If there is only a single PTM, define the std to be zero.
        R_std = R_all.std(axis=0, ddof=1) if R_all.shape[0] >= 2 else np.zeros_like(R_avg)

        # Deviation of the mean PTM from the ideal X PTM.
        R_err = R_avg - R_X

        # Save R_avg, R_std, and R_err in both .npy and .csv formats.
        save_array(R_avg, "R_avg", out_dir)
        save_array(R_std, "R_std", out_dir)
        save_array(R_err, "R_err", out_dir)

        # Also bundle mean & std into a single file for downstream use.
        # Shape (4, 4, 2) with last axis indexing [mean, std].
        R_mean_std = np.stack([R_avg, R_std], axis=-1)  # (4, 4, 2) [mean, std]
        np.save(out_dir / "R_mean_std.npy", R_mean_std)

        # And create a human-readable "mean ± std" table.
        save_mean_pm_std_text(R_avg, R_std, out_dir / "R_mean_pm_std.txt")

        # Extend the summary with the actual arrays.
        summary_lines.append("")
        summary_lines.append("Averaged PTM (R_avg):")
        summary_lines.append(str(R_avg))
        summary_lines.append("\nStandard deviation (R_std):")
        summary_lines.append(str(R_std))
        summary_lines.append("\nError matrix (R_err = R_avg − R_X):")
        summary_lines.append(str(R_err))
    else:
        # No PTMs found; record that no statistics were computed.
        summary_lines.append("No PTMs found; no aggregate statistics computed.")

    # Write the summary to disk and echo to stdout.
    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"[OK] Process outputs written to: {out_dir.resolve()}")


def main() -> int:
    """
    Entry point for the script.

    Behavior
    --------
    - Calls :func:`aggregate_process` for the directory defined in OUT_PROC.
    - Returns 0 as an exit code on normal completion.

    Notes
    -----
    All parameters are hard-coded via module-level constants. This function is
    primarily provided to make the script easy to invoke both as a module and
    from the command line (via the ``if __name__ == "__main__"`` guard).
    """
    # Aggregate process-tomography PTMs from the curated directory.
    aggregate_process(OUT_PROC)

    return 0


if __name__ == "__main__":
    # Use SystemExit to ensure that the integer return value from main()
    # becomes the process exit code when run from the command line.
    raise SystemExit(main())
