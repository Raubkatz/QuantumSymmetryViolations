#!/usr/bin/env python
"""
run_qiskit_process_tomography_X_ibm.py
-------------------------------------

Process‑tomography for a **single‑qubit X gate** with full run logging.
Runs on either
  • a real IBM Quantum backend, or
  • the local `AerSimulator` (state‑vector) if `SIMULATOR=True`.

The structure is identical to the state‑tomography version: same
configuration block, same per‑run directory with all artefacts, same
trace log – only the tomography task differs.

Edit the *CONFIGURATION* block, then run:

    python run_qiskit_process_tomography_X_ibm.py
"""

from __future__ import annotations

import datetime as _dt
import json
import warnings
from pathlib import Path
from typing import Dict, Any

import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module="qiskit_experiments")

# ───────────────────────── CONFIGURATION ────────────────────────────
SIMULATOR   = False            # True ⇒ local state‑vector simulator
IBM_TOKEN   = "keys"                # IBM Cloud API key (ignored if SIMULATOR)
INSTANCE    = "CRNs"      # CRN for Quantum_Experiments_1
BACKEND     = "ibm_brisbane"             # Ignored if SIMULATOR
CHANNEL     = "ibm_quantum_platform"      # "ibm_cloud" also possible
SHOTS       = 10_000                       # Positive integer
RUN_TAG = "run_brisbane_xprocesstomo_18"
# ────────────────────────────────────────────────────────────────────

# ------------------------------------------------------------------ #
# RUN TAG and directory                                              #
# ------------------------------------------------------------------ #
RUN_ID  = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path(f"{RUN_TAG}")
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"### RUN TAG: {RUN_ID}\n")

# ------------------------------------------------------------------ #
# helper utilities                                                    #
# ------------------------------------------------------------------ #
class _NpEncoder(json.JSONEncoder):
    """JSON encoder that converts NumPy arrays and complex numbers."""
    def default(self, obj: Any):  # noqa: D401 – short description style
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        return super().default(obj)

def _save_array(arr: np.ndarray, fname: str) -> None:
    np.save(RUN_DIR / f"{fname}.npy", arr)
    np.savetxt(RUN_DIR / f"{fname}.csv", arr, delimiter=",")
    print(f"Saved {fname} as .npy and .csv")

def _save_json(obj: Dict[str, Any], fname: str) -> None:
    with open(RUN_DIR / f"{fname}.json", "w") as f:
        json.dump(obj, f, cls=_NpEncoder, indent=2)
    print(f"Saved {fname}.json")

to_np  = lambda obj: obj.data if hasattr(obj, "data") else np.asarray(obj)
trace: list[str] = [f"RUN TAG: {RUN_ID}"]

# ------------------------------------------------------------------ #
# 0.  backend & target circuit                                        #
# ------------------------------------------------------------------ #
from qiskit import QuantumCircuit

qc = QuantumCircuit(1, name="X_gate")
qc.x(0)
print("### X‑gate circuit")
print(qc.draw("text"), "\n")

# ------------------------------------------------------------------ #
# 1.  backend selection                                               #
# ------------------------------------------------------------------ #
if SIMULATOR:
    trace.append("backend: AerSimulator (statevector)")
    from qiskit_aer import AerSimulator
    BACKEND_OBJ = AerSimulator(method="statevector")
else:
    trace.append("backend: IBM Quantum hardware")
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise RuntimeError("qiskit‑ibm‑runtime not installed – install with 'pip install qiskit‑ibm‑runtime' or set SIMULATOR=True") from exc

    service = (
        QiskitRuntimeService(channel=CHANNEL, token=IBM_TOKEN, instance=INSTANCE)
        if IBM_TOKEN else
        QiskitRuntimeService(instance=INSTANCE)
    )
    BACKEND_OBJ = service.backend(BACKEND)
    trace.append(f"IBM backend name: {BACKEND_OBJ.name}")

print("### Backend confirmed\n")

# ------------------------------------------------------------------ #
# 2.  process tomography                                              #
# ------------------------------------------------------------------ #
from qiskit_experiments.library.tomography import ProcessTomography
trace.append("import ProcessTomography       ✅")

pt_exp = ProcessTomography(qc)
pt_exp.set_transpile_options(optimization_level=0)

# Raw shot memory useful only on real hardware
use_memory = not SIMULATOR

print("### Running process tomography – this may queue on real hardware…\n")
pt_run = pt_exp.run(
    backend=BACKEND_OBJ,
    shots=SHOTS,
    memory=use_memory,
).block_for_results()

# ------------------------------------------------------------------ #
# 2a.  extract Pauli‑transfer matrix                                  #
# ------------------------------------------------------------------ #
resmap = {r.name: r.value for r in pt_run.analysis_results()}

if "pauli_transfer_matrix" in resmap:            # qiskit‑experiments ≤ 0.8
    R = to_np(resmap["pauli_transfer_matrix"])
    trace.append("PTM found under 'pauli_transfer_matrix'")
elif "ptm" in resmap:                            # qiskit‑experiments 0.5
    R = to_np(resmap["ptm"])
    trace.append("PTM found under 'ptm'")
else:                                            # ≥ 0.9 – Choi only
    key = "choi" if "choi" in resmap else "state"
    trace.append(f"No PTM – converting Choi from '{key}'")
    Λ = to_np(resmap[key])                        # 4×4 Choi
    P = [
        np.eye(2),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.array([[1, 0], [0, -1]])
    ]
    R = np.array([[0.5 * np.trace(np.kron(Pk, Pj.T) @ Λ).real for Pj in P] for Pk in P])

print("R  (Pauli‑transfer, rows I,X,Y,Z ; cols I,X,Y,Z)\n", np.round(R, 3), "\n")

# ------------------------------------------------------------------ #
# 3.  save artefacts                                                  #
# ------------------------------------------------------------------ #
_save_array(R, "ptm")

# Raw experiment data
_save_json(pt_run.data(), "process_tomography_raw")

# Additional job‑level metadata (only meaningful for IBM backends)
#try:
#    job_dict = pt_run.job.to_dict()              # type: ignore[attr‑defined]
#    _save_json(job_dict, "job_metadata")
#    trace.append("job metadata saved          ✅")
#except Exception as exc:  # noqa: BLE001 – broad except ok for metadata
#    trace.append(f"job metadata unavailable    ❌  ({exc})")

# ------------------------------------------------------------------ #
# 4.  trace log                                                       #
# ------------------------------------------------------------------ #
print("### TRACE LOG")
for entry in trace:
    print(" •", entry)

(RUN_DIR / "trace_log.txt").write_text("\n".join(trace))
print(f"\nAll artefacts saved to {RUN_DIR}")
