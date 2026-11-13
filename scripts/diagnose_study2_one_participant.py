"""Single-participant Study 2 ROI ERPAC preflight (read-only).

Run from the motor_pac_aging repository root:
    python scripts/diagnose_study2_one_participant.py

Choose a particular subject by setting SUBJECT below; otherwise the first
alphabetical subject with an FTT planning M1 file is selected.
"""
from pathlib import Path
import pickle
import sys
import warnings

import numpy as np
from tensorpac import EventRelatedPac

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from config.paths import PCA_ROI_TC_DIR, ERPAC_DIR  # noqa: E402
from config.config import COUPLINGS  # noqa: E402

# ------------------ PILOT SETTINGS ------------------
GROUP = "Y"
SUBJECT = None  # e.g. "s1_pac_sub01"; None selects first matching subject
ROI = "M1"
CONDITION = "FTT_plan"
COUPLING = "theta_gamma"
METHOD = "circular"
MCP = "bonferroni"
N_JOBS = 1
# ----------------------------------------------------


def scalar(npz, name, default=None):
    if name not in npz.files:
        return default
    value = npz[name]
    return value.item() if value.ndim == 0 else (value.reshape(-1)[0].item() if value.size == 1 else value)


def main():
    candidates = sorted(Path(PCA_ROI_TC_DIR).glob(
        f"{GROUP}/*/{CONDITION}_{ROI}_pca_roi_tc.npz"
    ))
    if SUBJECT is not None:
        candidates = [p for p in candidates if p.parent.name == SUBJECT]
    if not candidates:
        raise FileNotFoundError(
            f"No {GROUP}/{SUBJECT or '*'}/{CONDITION}_{ROI}_pca_roi_tc.npz "
            f"in {PCA_ROI_TC_DIR}"
        )
    filepath = candidates[0]
    print(f"Selected file: {filepath}")
    print(f"Available matches: {len(candidates)}")

    with np.load(filepath, allow_pickle=True) as npz:
        x3 = np.asarray(npz["data"], dtype=float)
        times = np.asarray(npz["times"], dtype=float)
        sub = str(scalar(npz, "subject"))
        group = str(scalar(npz, "group"))
        roi = str(scalar(npz, "roi"))
        task = str(scalar(npz, "task"))
        stage = str(scalar(npz, "task_stage"))
        pc_n = scalar(npz, "n_components_retained")
        variance = scalar(npz, "actual_variance_retained")
        original_indices = np.asarray(npz["original_epoch_indices"]) if "original_epoch_indices" in npz.files else None

    assert group == GROUP and roi == ROI and f"{task}_{stage}" == CONDITION
    assert x3.ndim == 3 and x3.shape[1] == 1
    assert x3.shape[-1] == len(times)
    assert len(times) >= 3 and np.all(np.isfinite(x3)) and np.all(np.isfinite(times))
    assert np.all(np.diff(times) > 0)
    dt = np.diff(times)
    assert np.allclose(dt, np.median(dt), rtol=1e-5, atol=1e-10)
    sfreq = 1.0 / np.median(dt)
    x = x3[:, 0, :]
    n_epochs, n_times = x.shape
    assert n_epochs >= 3, "Too few epochs for a stable ERPAC pilot."

    print("\n--- ROI signal diagnostics ---")
    print(f"Subject/group: {sub}/{group}; ROI: {roi}; condition: {CONDITION}")
    print(f"Signal shape: {x3.shape} (epochs, singleton ROI, times)")
    print(f"Epochs: {n_epochs}; time range: {times[0]:.3f} .. {times[-1]:.3f} s")
    print(f"Sampling frequency: {sfreq:.3f} Hz; duration: {times[-1]-times[0]:.3f} s")
    print(f"PCA components retained: {pc_n}; retained variance: {variance}")
    print(f"Data range: {x.min():.6g} .. {x.max():.6g}; std: {x.std():.6g}")
    trial_std = x.std(axis=1)
    print(f"Trial-wise std (min/median/max): {trial_std.min():.6g} / {np.median(trial_std):.6g} / {trial_std.max():.6g}")
    if original_indices is not None:
        print(f"Original trial indices: {len(original_indices)}; unique: {len(np.unique(original_indices))}")
        assert len(original_indices) == n_epochs
        assert len(np.unique(original_indices)) == n_epochs

    if times[0] >= 0:
        warnings.warn("Saved epochs start at/after event time: inspect filtering edge effects. "
                      "A pre-event buffer may be required for reliable phase estimates.")
    phase_bounds, amp_bounds = COUPLINGS[COUPLING]
    low_f = float(phase_bounds[0])
    low_period = 1.0 / low_f
    print(f"Lowest phase frequency: {low_f:g} Hz; period: {low_period:.3f} s")
    print(f"Epoch duration / lowest-band period: {(times[-1]-times[0])/low_period:.2f} cycles")
    print("Note: duration in cycles is a diagnostic, NOT a full filter-support validation.")

    print("\n--- Reproducing existing EventRelatedPac.filterfit ---")
    p = EventRelatedPac(f_pha=phase_bounds, f_amp=amp_bounds)
    erpac = np.asarray(p.filterfit(sfreq, x, method=METHOD, mcp=MCP, n_jobs=N_JOBS)).squeeze()
    if erpac.ndim == 1:
        erpac = erpac[np.newaxis, :]
    assert erpac.ndim == 2 and erpac.shape[-1] == n_times
    assert np.all(np.isfinite(erpac))
    print(f"ERPAC shape (amplitude-frequency bins, times): {erpac.shape}")
    print(f"ERPAC min/mean/max: {erpac.min():.6g} / {erpac.mean():.6g} / {erpac.max():.6g}")
    print(f"ERPAC gamma frequency vector: {np.asarray(p.yvec)}")
    print("This reproduces the published script's calculation, not an independent estimator validation.")

    existing = Path(ERPAC_DIR) / "PCA" / "erpac_roi_tc" / group / sub / f"{sub}_{CONDITION}_pca_roi_tc_erpac_results.pkl"
    print("\n--- Compare to previously saved output (if available) ---")
    if existing.is_file():
        with existing.open("rb") as fh:
            saved = pickle.load(fh)
        reference = np.asarray(saved[COUPLING][ROI]["erpac"])
        reference = np.squeeze(reference)
        if reference.ndim == 1:
            reference = reference[np.newaxis, :]
        if reference.shape != erpac.shape:
            print(f"SHAPE MISMATCH: saved={reference.shape}; reproduced={erpac.shape}")
        else:
            absdiff = np.abs(reference - erpac)
            print(f"Max absolute difference: {absdiff.max():.12g}")
            print(f"Allclose (atol=1e-10, rtol=1e-7): {np.allclose(reference, erpac, atol=1e-10, rtol=1e-7)}")
    else:
        print(f"No previous result found: {existing}")

    print("\nPreflight complete. No project files modified.")
    print("Next: inspect the printed results, then test trial-shuffled surrogates.")


if __name__ == "__main__":
    main()
