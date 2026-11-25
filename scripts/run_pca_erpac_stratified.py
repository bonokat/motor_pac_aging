"""Study 2: stratified, surrogate-normalized ERPAC on PCA-derived ROI time series.

This file is being developed and reviewed one stage at a time.
CURRENT IMPLEMENTATION: Sections 1–7 (through LMM-ready export; no LMM fitting).

Run from repository root:
    python scripts/run_pca_erpac_stratified.py --group Y --subject s1_pac_sub01

Without --stage3, the command prints input and selection summaries only.
Use --stage3 --group Y --subject s1_pac_sub01 to compute observed ERPAC.
Use --stage4 --group Y --subject s1_pac_sub01 --task FTT --task-stage plan
    to compute surrogate ERPAC for one condition-stage (200 permutations by default).
Use --full to compute all participants and export surrogate-normalized ERPAC.
Use --rebuild-master to regenerate master CSVs from completed conditions.
The older run_pca_erpac_source.py and phase/amplitude scripts are not used.
"""

from __future__ import annotations

# ============================================================
# 0. PROJECT CONFIGURATION AND PROTOCOL
# ============================================================
import argparse
import hashlib
import csv
import os
import sys
import json
import time
import traceback
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import PCA_ROI_TC_DIR, PCA_ROOT  # noqa: E402

# Use metadata instead of importing config.config: its SUBJECTS variable
# lists source folders at import time and is unnecessary for this stage.
EXPECTED_ROIS = ("M1", "S1", "PMC", "SMA")
EXPECTED_TASKS = ("FTT", "DeCRAT")
EXPECTED_STAGES = ("plan", "go")
EXPECTED_GROUPS = ("Y", "O")
MATCHED_TRIALS = 37
N_STRATA = 5
N_SUBSAMPLES = 10
N_SURROGATES = 200
SEED = 7026
TIME_WINDOWS = {
    "early": (0.0, 0.150),
    "middle": (0.150, 0.275),
    "late": (0.275, 0.400),
}
# Source configuration: updated by user to a broadband gamma filter.
# Step 3 imports and verifies COUPLINGS at runtime (user-local config).


# ============================================================
# 1. LOAD AND VALIDATE PCA-DERIVED ROI TIME SERIES
# ============================================================
# One .npz file per participant × task × stage × block/split × ROI.
# Required: data (trials, 1, time), times, subject, group, roi, task,
# task_stage. Optional: task_block, adaptation_split, original_epoch_indices.
# Stage 1 only indexes and validates files; it does not change the EEG data.


def scalar(npz, key: str, default=None):
    if key not in npz.files:
        return default
    value = np.asarray(npz[key])
    if value.size != 1:
        raise ValueError(f"Expected scalar metadata for {key!r}; got {value.shape}")
    return value.reshape(-1)[0].item()


def optional_text(value):
    if value is None:
        return None
    value = str(value).strip()
    return None if value.lower() in {"", "none", "na", "nan"} else value


def standard_condition(task, block, split):
    """Normalize condition metadata without merging distinct valid conditions."""
    block = optional_text(block)
    split = optional_text(split)
    if task == "FTT":
        if block is not None or split is not None:
            raise ValueError(f"Unexpected FTT block/split: {block}, {split}")
        return None, None
    if task != "DeCRAT":
        raise ValueError(f"Unknown task: {task}")
    if block == "baseline" and split is None:
        return block, split
    if block == "adaptation" and split in {"early", "late"}:
        return block, split
    raise ValueError(f"Unexpected DeCRAT condition: block={block}, split={split}")


def inspect_npz(path: Path):
    with np.load(path, allow_pickle=True) as npz:
        required = {"data", "times", "subject", "group", "roi", "task", "task_stage"}
        missing = required.difference(npz.files)
        if missing:
            raise ValueError(f"{path.name}: missing keys: {sorted(missing)}")
        group = str(scalar(npz, "group"))
        subject = str(scalar(npz, "subject"))
        task = str(scalar(npz, "task"))
        stage = str(scalar(npz, "task_stage"))
        roi = str(scalar(npz, "roi"))
        block, split = standard_condition(
            task, scalar(npz, "task_block"), scalar(npz, "adaptation_split")
        )
        data = np.asarray(npz["data"])
        times = np.asarray(npz["times"], dtype=float)
        epoch_indices = (
            np.asarray(npz["original_epoch_indices"])
            if "original_epoch_indices" in npz.files else None
        )
        if data.ndim != 3 or data.shape[1] != 1:
            raise ValueError(f"{path.name}: expected (trials, 1, times), got {data.shape}")
        if times.ndim != 1 or data.shape[2] != len(times):
            raise ValueError(f"{path.name}: inconsistent time dimension")
        if data.shape[0] < MATCHED_TRIALS:
            raise ValueError(f"{path.name}: {data.shape[0]} trials (< {MATCHED_TRIALS})")
        if not np.isfinite(data).all() or not np.isfinite(times).all():
            raise ValueError(f"{path.name}: nonfinite data or times")
        dt = np.diff(times)
        if len(dt) == 0 or not np.all(dt > 0) or not np.allclose(dt, np.median(dt), rtol=1e-5, atol=1e-10):
            raise ValueError(f"{path.name}: invalid/irregular time vector")
        if times[0] > 0 or times[-1] < 0.4:
            raise ValueError(f"{path.name}: epoch does not cover all three windows")
        if epoch_indices is not None:
            if epoch_indices.ndim != 1 or len(epoch_indices) != data.shape[0]:
                raise ValueError(f"{path.name}: invalid original_epoch_indices shape")
            if len(np.unique(epoch_indices)) != len(epoch_indices):
                raise ValueError(f"{path.name}: duplicate original epoch indices")
            if not np.all(np.diff(epoch_indices.astype(float)) > 0):
                raise ValueError(f"{path.name}: original epoch indices not in chronological order")
        if group not in EXPECTED_GROUPS or task not in EXPECTED_TASKS or stage not in EXPECTED_STAGES or roi not in EXPECTED_ROIS:
            raise ValueError(f"{path.name}: unexpected group/task/stage/ROI metadata")
        return {
            "key": (group, subject, task, stage, block, split),
            "roi": roi,
            "n_trials": int(data.shape[0]),
            "n_times": int(len(times)),
            "sfreq": float(1 / np.median(dt)),
            "times": times,
            "epoch_indices": epoch_indices,
            "path": path,
        }


def index_inputs(root: Path, group=None, subject=None):
    if not root.is_dir():
        raise FileNotFoundError(f"PCA ROI time-course directory not found: {root}")
    paths = sorted(root.rglob("*_pca_roi_tc.npz"))
    if not paths:
        raise FileNotFoundError(f"No matching PCA .npz files under {root}")
    indexed = defaultdict(dict)
    for path in paths:
        # Read metadata from disk, not filenames: paths may vary between projects.
        info = inspect_npz(path)
        key = info["key"]
        if group is not None and key[0] != group:
            continue
        if subject is not None and key[1] != subject:
            continue
        if info["roi"] in indexed[key]:
            raise ValueError(f"Duplicate ROI {info['roi']} in condition {key}")
        indexed[key][info["roi"]] = info
    if not indexed:
        raise ValueError("No conditions match --group/--subject filters")
    for key, by_roi in indexed.items():
        missing = set(EXPECTED_ROIS) - set(by_roi)
        if missing:
            raise ValueError(f"Missing ROIs {sorted(missing)} for {key}")
        ref = next(iter(by_roi.values()))
        for roi, info in by_roi.items():
            if info["n_trials"] != ref["n_trials"]:
                raise ValueError(f"Trial-count mismatch for {key} / {roi}")
            if not np.array_equal(info["times"], ref["times"]):
                raise ValueError(f"Time-vector mismatch for {key} / {roi}")
            a, b = ref["epoch_indices"], info["epoch_indices"]
            if (a is None) != (b is None):
                raise ValueError(f"Inconsistent original_epoch_indices presence for {key}")
            if a is not None and not np.array_equal(a, b):
                raise ValueError(f"Original trial-ID mismatch across ROIs for {key} / {roi}")
        if ref["epoch_indices"] is None:
            print(f"WARNING: original_epoch_indices missing for {key}; assuming stored trial order is chronological")
    return dict(indexed)


# ============================================================
# 2. CHRONOLOGICALLY STRATIFIED TRIAL SUBSAMPLING [IMPLEMENTED]
# ============================================================
# Sampling takes place AFTER preprocessing and PCA reconstruction.
# Each participant × condition × stage is sampled independently.
# The selected trial *row positions* are shared between ROIs, coupling
# types and all time windows; stages are NOT matched to one another.
#
# We split the valid trials (already stored chronologically) into five
# consecutive strata. 37 draws are allocated in proportion to stratum
# size using the largest-remainder rule, then sampled without replacement
# within each stratum. Each repetition draws anew; different repetitions
# may overlap, as intended. Repetitions are not independent observations.
#
# If exactly 37 trials remain, only one distinct subset is possible.
# We produce one subset instead of recomputing an identical subset 10×.


def seed_for_condition(key, base_seed=SEED):
    """Stable per-condition seed, independent of Python's salted hash()."""
    label = "|".join("<none>" if x is None else str(x) for x in key)
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") ^ int(base_seed)) % (2**63)


def proportional_counts(sizes, n_select):
    """Largest-remainder allocation with exactly n_select total draws."""
    sizes = np.asarray(sizes, dtype=int)
    if np.any(sizes <= 0) or n_select > sizes.sum():
        raise ValueError("Invalid stratum sizes or requested trial count")
    targets = n_select * sizes / sizes.sum()
    counts = np.floor(targets).astype(int)
    remainder = int(n_select - counts.sum())
    # Stable tie-breaking by chronological stratum index.
    rank = np.argsort(- (targets - counts), kind="stable")
    counts[rank[:remainder]] += 1
    if counts.sum() != n_select or np.any(counts > sizes):
        raise RuntimeError("Invalid proportional allocation")
    return counts


def draw_stratified_subsets(key, n_trials, original_epoch_indices=None):
    """
    Returns selection positions (n_subsets, 37) and stratum metadata.

    n_trials: number of retained, preprocessed trials for one condition/stage.
    original_epoch_indices: optional acquisition-order IDs. These are used
        only to confirm chronological ordering, not to match stages.
    """
    if n_trials < MATCHED_TRIALS:
        raise ValueError(f"{key}: {n_trials} trials; need {MATCHED_TRIALS}")
    if original_epoch_indices is not None:
        ids = np.asarray(original_epoch_indices)
        if len(ids) != n_trials or not np.all(np.diff(ids.astype(float)) > 0):
            raise ValueError(f"{key}: trial identifiers not chronological")

    strata = np.array_split(np.arange(n_trials, dtype=int), N_STRATA)
    sizes = np.array([len(x) for x in strata], dtype=int)
    quotas = proportional_counts(sizes, MATCHED_TRIALS)
    n_repeats = 1 if n_trials == MATCHED_TRIALS else N_SUBSAMPLES
    rng = np.random.default_rng(seed_for_condition(key))

    selections = []
    for _ in range(n_repeats):
        chosen = [rng.choice(block, size=int(q), replace=False)
                  for block, q in zip(strata, quotas)]
        row_positions = np.sort(np.concatenate(chosen))
        if row_positions.size != MATCHED_TRIALS or np.unique(row_positions).size != MATCHED_TRIALS:
            raise RuntimeError(f"{key}: invalid sampled subset")
        selections.append(row_positions)

    subsets = np.stack(selections)
    # Explicitly check chronological representation in every repetition.
    for row in subsets:
        actual = np.array([np.isin(row, block).sum() for block in strata])
        if not np.array_equal(actual, quotas):
            raise RuntimeError(f"{key}: stratification mismatch")

    return {
        "subsets": subsets,
        "stratum_sizes": sizes,
        "stratum_quotas": quotas,
        "seed": seed_for_condition(key),
        "original_trial_ids": (None if original_epoch_indices is None else
                               np.asarray(original_epoch_indices)[subsets]),
    }


# ============================================================
# 3. FREQUENCY FILTERING AND OBSERVED ERPAC [IMPLEMENTED]
# ============================================================
# This is a pilot of observed ERPAC only. It does not generate surrogates.
# Each ROI is filtered ONCE using the entire available epoch; gamma
# amplitude (30–80 Hz) is reused for theta/alpha/beta coupling.
# The circular ERPAC correlation is evaluated for the retained [0, .4)
# time points. The already-filtered phase and amplitude arrays, not the
# raw data, are restricted to that interval. One observed estimate is
# saved per subset × coupling × ROI × analysis window.
#
# IMPORTANT: results here are NOT surrogate-normalized and should NOT
# enter the final LMM. The Stage 3 output is a diagnostic/pilot only.


def checked_couplings():
    # Read your LOCAL config: do not silently run the outdated narrowband
    # (30, 80, 5, 1) configuration from the earlier ERPAC pipeline.
    from config.config import COUPLINGS

    expected = {
        "theta_gamma": (4, 8),
        "alpha_gamma": (8, 12),
        "beta_gamma": (13, 30),
    }
    if set(COUPLINGS) != set(expected):
        raise ValueError(f"Expected couplings {sorted(expected)}, got {sorted(COUPLINGS)}")
    for name, (phase_band, amp_band) in COUPLINGS.items():
        if not np.array_equal(np.asarray(phase_band), np.asarray(expected[name])):
            raise ValueError(f"Unexpected phase band for {name}: {phase_band}")
        if not np.array_equal(np.asarray(amp_band), np.asarray([30, 80])):
            raise ValueError(
                f"{name}: expected single broadband gamma filter [30, 80]; "
                f"config has {amp_band!r}"
            )
    return COUPLINGS


def load_roi_signal(info):
    """Return retained-trial data (trials, time), verifying stage-1 metadata."""
    with np.load(info["path"], allow_pickle=True) as npz:
        data = np.asarray(npz["data"], dtype=float)
        times = np.asarray(npz["times"], dtype=float)
    if data.shape != (info["n_trials"], 1, info["n_times"]):
        raise ValueError(f"Data shape changed since indexing: {info['path']}")
    if not np.array_equal(times, info["times"]):
        raise ValueError(f"Time vector changed since indexing: {info['path']}")
    if not np.isfinite(data).all():
        raise ValueError(f"Nonfinite data in {info['path']}")
    return data[:, 0, :]


def filter_components(x, sfreq, couplings):
    """Filter full epochs; return phase arrays by coupling and one gamma amp."""
    from tensorpac import EventRelatedPac

    phases = {}
    amplitude = None
    for name, (phase_band, amp_band) in couplings.items():
        pac = EventRelatedPac(
            f_pha=phase_band,
            f_amp=amp_band,
            dcomplex="hilbert",
            cycle=(3, 6),
            verbose=False,
        )
        phases[name] = pac.filter(sfreq, x, ftype="phase", n_jobs=1)
        if amplitude is None:
            amplitude = pac.filter(sfreq, x, ftype="amplitude", n_jobs=1)
        if phases[name].shape != (1, x.shape[0], x.shape[1]):
            raise ValueError(f"Unexpected {name} phase shape {phases[name].shape}")
    if amplitude.shape != (1, x.shape[0], x.shape[1]):
        raise ValueError(f"Unexpected gamma amplitude shape {amplitude.shape}")
    return phases, amplitude


def observed_erpac_windows(phase, amplitude, indices, time_mask, masks):
    """Compute one circular ERPAC curve from 37 trials and window means."""
    from tensorpac.methods import erpac as circular_erpac

    pha = phase[:, indices, :][:, :, time_mask]
    amp = amplitude[:, indices, :][:, :, time_mask]
    rho, _ = circular_erpac(
        np.moveaxis(pha, 1, -1),
        np.moveaxis(amp, 1, -1),
    )
    # Expected: gamma frequencies × phase frequencies × retained time points.
    if rho.shape != (1, 1, int(time_mask.sum())):
        raise ValueError(f"Unexpected ERPAC shape: {rho.shape}")
    if not np.isfinite(rho).all():
        raise ValueError("Nonfinite observed ERPAC")
    return {window: float(rho[..., mask].mean()) for window, mask in masks.items()}


def run_stage3(indexed, output_root):
    """Compute observed ERPAC for the validated conditions and save pilot CSV."""
    couplings = checked_couplings()
    rows = []
    for key, by_roi in sorted(indexed.items(), key=lambda item: str(item[0])):
        group, subject, task, stage, block, split = key
        ref = next(iter(by_roi.values()))
        selection = draw_stratified_subsets(
            key, ref["n_trials"], ref["epoch_indices"]
        )
        subs = selection["subsets"]
        times = ref["times"]
        time_mask = (times >= 0.0) & (times < 0.400)
        selected_times = times[time_mask]
        masks = {
            name: (selected_times >= a) & (selected_times < b)
            for name, (a, b) in TIME_WINDOWS.items()
        }
        if int(time_mask.sum()) != 200 or any(not mask.any() for mask in masks.values()):
            raise ValueError(f"Unexpected 0–400 ms samples for {key}")
        print(f"Stage 3: {key} | {len(subs)} subsets")
        for roi, info in sorted(by_roi.items()):
            x = load_roi_signal(info)
            phases, amp = filter_components(x, ref["sfreq"], couplings)
            for coupling, pha in phases.items():
                for i, subset in enumerate(subs, start=1):
                    values = observed_erpac_windows(pha, amp, subset, time_mask, masks)
                    for window, value in values.items():
                        rows.append({
                            "group": group, "sub": subject, "task": task,
                            "task_stage": stage, "task_block": block or "NA",
                            "adaptation_split": split or "NA", "roi": roi,
                            "coupling": coupling, "subsample": i,
                            "time_win": window, "observed_erpac": value,
                            "n_available": ref["n_trials"],
                            "n_matched": MATCHED_TRIALS,
                            "n_subsamples": len(subs),
                            "selection_seed": selection["seed"],
                        })
    if not rows:
        raise RuntimeError("No observed ERPAC values generated")
    output_root.mkdir(parents=True, exist_ok=True)
    # Pilot is run with one subject at a time; do not overwrite other subjects.
    subject = next(iter(indexed))[1]
    path = output_root / f"{subject}_stage3_observed_erpac_PILOT.csv"
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nStage 3 complete: {len(rows)} raw observed ERPAC rows saved to {path}")
    print("No surrogates, z-normalization or final statistical results generated.")
    return path


# ============================================================
# 4. TRIAL-SHUFFLED SURROGATE ERPAC [IMPLEMENTED: ONE-CONDITION PILOT]
# ============================================================
# For each of the 37-trial subsets, generate 200 independently drawn
# permutations of the gamma-amplitude trial axis relative to phase.
# Each permutation is a bijection (no duplicated trials); FIXED POINTS
# are allowed, as in standard random permutation sampling.
#
# Stratification controls the 37-trial selections. Surrogate permutation
# is over ALL 37 selected trials, matching the previously validated
# trial-shuffle method. A within-stratum shuffle would test a different
# null hypothesis and is NOT silently introduced here.
#
# IMPORTANT: same selections AND same permutation arrays are reused
# across all ROIs and frequency couplings for a condition-stage.
# This stage saves each surrogate's three window averages; it does NOT
# compute surrogate means, SDs, z-scores or aggregate subsamples.
#
# PILOT safety: exactly one participant × condition × stage per run;
# --stage4 requires --group --subject --task --task-stage.


def seed_for_surrogates(key):
    """Stable deterministic stream distinct from trial-selection RNG."""
    label = "surrogate-permutations-v1|" + "|".join(
        "<none>" if value is None else str(value) for value in key
    )
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") ^ int(SEED)) % (2**63)


def generate_surrogate_permutations(key, n_subsets, n_surrogates):
    """Array (subsets, surrogates, 37), shared across ROI/coupling."""
    if n_surrogates < 2:
        raise ValueError("At least 2 surrogates are required")
    rng = np.random.default_rng(seed_for_surrogates(key))
    permutations = np.empty(
        (n_subsets, n_surrogates, MATCHED_TRIALS), dtype=np.int16
    )
    for subset_idx in range(n_subsets):
        for surrogate_idx in range(n_surrogates):
            permutations[subset_idx, surrogate_idx] = rng.permutation(MATCHED_TRIALS)
    expected = np.arange(MATCHED_TRIALS)
    if not np.all(np.sort(permutations, axis=-1) == expected):
        raise RuntimeError("Invalid surrogate trial permutation")
    return permutations


def window_averages_from_circular_erpac(pha, amp, masks):
    """ERPAC over already-filtered, time-cropped (freq=1, trial=37, time) arrays."""
    from tensorpac.methods import erpac as circular_erpac

    rho, _ = circular_erpac(
        np.moveaxis(pha, 1, -1),
        np.moveaxis(amp, 1, -1),
    )
    if rho.shape != (1, 1, pha.shape[-1]):
        raise ValueError(f"Unexpected ERPAC shape: {rho.shape}")
    if not np.isfinite(rho).all():
        raise ValueError("Nonfinite surrogate ERPAC")
    values = {win: float(rho[..., mask].mean()) for win, mask in masks.items()}
    if any(value < -1e-12 or value > 1 + 1e-12 for value in values.values()):
        raise ValueError(f"Surrogate ERPAC outside [0,1]: {values}")
    return values


def run_stage4(indexed, output_root, n_surrogates=N_SURROGATES):
    """One condition-stage pilot; save individual surrogate ERPAC window means."""
    if len(indexed) != 1:
        raise ValueError(
            f"Stage 4 pilot expects one participant-condition-stage; got {len(indexed)}"
        )
    couplings = checked_couplings()
    key, by_roi = next(iter(indexed.items()))
    group, subject, task, stage, block, split = key
    ref = next(iter(by_roi.values()))
    selection = draw_stratified_subsets(key, ref["n_trials"], ref["epoch_indices"])
    subsets = selection["subsets"]
    permutations = generate_surrogate_permutations(key, len(subsets), n_surrogates)

    times = ref["times"]
    time_mask = (times >= 0.0) & (times < 0.400)
    selected_times = times[time_mask]
    masks = {
        name: (selected_times >= a) & (selected_times < b)
        for name, (a, b) in TIME_WINDOWS.items()
    }
    if int(time_mask.sum()) != 200 or any(not m.any() for m in masks.values()):
        raise ValueError(f"Unexpected 0–400 ms sampling/windows for {key}")

    condition_label = "_".join(
        str(item) for item in (task, stage, block, split) if item is not None
    )
    output_dir = output_root / group / subject
    output_dir.mkdir(parents=True, exist_ok=True)
    base = f"{subject}_{condition_label}_stage4_{n_surrogates}sur_PILOT"
    csv_path = output_dir / f"{base}.csv"
    provenance_path = output_dir / f"{base}_selection_permutations.npz"
    # Stage 4 output is separate from stage 3; write atomically to avoid
    # leaving a partial CSV if the pilot is interrupted.
    tmp_path = csv_path.with_suffix(".csv.part")

    fieldnames = [
        "group", "sub", "task", "task_stage", "task_block", "adaptation_split",
        "roi", "coupling", "subsample", "surrogate", "time_win",
        "surrogate_erpac", "n_available", "n_matched", "n_subsamples",
        "n_surrogates", "selection_seed", "permutation_seed",
    ]
    rows_written = 0
    print(f"\nStage 4: {key} | {len(subsets)} subsets × {n_surrogates} surrogates")
    print("Full-epoch filtering; ERPAC restricted to 0–400 ms after filtering.")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for roi, info in sorted(by_roi.items()):
                x = load_roi_signal(info)
                phases, amplitude = filter_components(x, ref["sfreq"], couplings)
                amp_toi = amplitude[..., time_mask]
                for coupling, pha in phases.items():
                    pha_toi = pha[..., time_mask]
                    for k, trial_positions in enumerate(subsets):
                        pha_sub = pha_toi[:, trial_positions, :]
                        amp_sub = amp_toi[:, trial_positions, :]
                        for s, perm in enumerate(permutations[k], start=1):
                            values = window_averages_from_circular_erpac(
                                pha_sub, amp_sub[:, perm, :], masks
                            )
                            for window, value in values.items():
                                writer.writerow({
                                    "group": group, "sub": subject, "task": task,
                                    "task_stage": stage, "task_block": block or "NA",
                                    "adaptation_split": split or "NA", "roi": roi,
                                    "coupling": coupling, "subsample": k + 1,
                                    "surrogate": s, "time_win": window,
                                    "surrogate_erpac": value,
                                    "n_available": ref["n_trials"],
                                    "n_matched": MATCHED_TRIALS,
                                    "n_subsamples": len(subsets),
                                    "n_surrogates": n_surrogates,
                                    "selection_seed": selection["seed"],
                                    "permutation_seed": seed_for_surrogates(key),
                                })
                                rows_written += 1
                        print(f"  {roi} / {coupling}: subset {k + 1}/{len(subsets)} complete", flush=True)
        expected_rows = len(subsets) * len(by_roi) * len(couplings) * n_surrogates * len(masks)
        if rows_written != expected_rows:
            raise RuntimeError(f"Expected {expected_rows} rows, generated {rows_written}")
        os.replace(tmp_path, csv_path)
        # Save exact indices and permutations for independent reproducibility
        np.savez_compressed(
            provenance_path,
            subsets=subsets,
            permutations=permutations,
            selection_seed=np.int64(selection["seed"]),
            permutation_seed=np.int64(seed_for_surrogates(key)),
            stratum_sizes=selection["stratum_sizes"],
            stratum_quotas=selection["stratum_quotas"],
            original_trial_ids=(selection["original_trial_ids"] if selection["original_trial_ids"] is not None
                                else np.empty((0,), dtype=np.int64)),
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    print(f"\nStage 4 complete: {rows_written} surrogate ERPAC window estimates")
    print(f"Surrogate CSV: {csv_path}")
    print(f"Trial/permutation provenance: {provenance_path}")
    print("No z-normalization or subsample aggregation performed.")
    return csv_path



# ============================================================
# 5. TEMPORAL AVERAGING [FULL PIPELINE]
# ============================================================
# Filtering is performed over the full epoch, BEFORE selection of 0–400 ms.
# After circular ERPAC is computed at each retained time point, average
# each observed and surrogate ERPAC curve within the same three windows.
# This averaging is performed by window_averages_from_circular_erpac().
# Never average phase or amplitude before the ERPAC correlation.


# ============================================================
# 6. WITHIN-SUBSET Z-NORMALIZATION AND AVERAGING [FULL PIPELINE]
# ============================================================
# For each subset, ROI, coupling and window:
#   z_k = (observed_k - mean(surrogate_k)) / sd(surrogate_k, ddof=1)
# The final dependent variable is mean(z_k) across the K selected subsets.
# The same 37 trials are used across ROIs, couplings and windows WITHIN
# each participant × condition × stage. Stages are sampled independently.
# When exactly 37 valid trials exist, K=1 (no fake repetitions).


def full_protocol_signature(couplings, n_surrogates):
    """Settings that must match when resuming a saved full run."""
    return {
        "schema": "study2-erpac-stratified-full-v1",
        "matched_trials": MATCHED_TRIALS,
        "strata": N_STRATA,
        "subsamples": N_SUBSAMPLES,
        "surrogates": int(n_surrogates),
        "seed": SEED,
        "time_windows": {k: list(v) for k, v in TIME_WINDOWS.items()},
        "couplings": {k: [list(map(float, phase)), list(map(float, amp))]
                      for k, (phase, amp) in sorted(couplings.items())},
        "filter": {"dcomplex": "hilbert", "cycle": [3, 6]},
        "estimator": "tensorpac.methods.erpac circular",
        "surrogate": "trial-shuffle gamma amplitude; unrestricted across 37 selected trials",
        "normalization": "within-subset z then mean across subsets",
        "numpy_version": np.__version__,
    }


def input_signature(by_roi):
    """Detect file modification before reuse of an existing completed condition."""
    return [
        {"roi": roi, "path": str(info["path"].resolve()),
         "size": info["path"].stat().st_size,
         "mtime_ns": info["path"].stat().st_mtime_ns}
        for roi, info in sorted(by_roi.items())
    ]


def stable_label(key):
    _, _, task, stage, block, split = key
    return "_".join(str(v) for v in (task, stage, block, split) if v is not None)


def atomic_csv(path, rows, columns):
    """Write whole CSV before replacing destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_npz(path, arrays):
    """Prevent corrupt final NPZ if process is interrupted during save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def condition_is_complete(folder, signature, source_signature):
    meta_path = folder / "completed.json"
    expected = [folder / "aggregated.csv", folder / "subsamples.csv",
                folder / "full_erpac.npz"]
    if not meta_path.exists() or any(not p.exists() or p.stat().st_size == 0 for p in expected):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return (meta.get("protocol") == signature
                and meta.get("sources") == source_signature
                and meta.get("status") == "complete")
    except (OSError, ValueError):
        return False


def evaluate_full_condition(key, by_roi, n_surrogates, couplings):
    """Compute observed and surrogate ERPAC for one participant-condition-stage.

    Returns LMM-ready aggregated rows, diagnostic subset rows, and raw
    arrays. Arrays are indexed [ROI, coupling, subset, (surrogate), window].
    """
    group, subject, task, stage, block, split = key
    ref = next(iter(by_roi.values()))
    selection = draw_stratified_subsets(key, ref["n_trials"], ref["epoch_indices"])
    subsets = selection["subsets"]
    k_actual = len(subsets)
    permutations = generate_surrogate_permutations(key, k_actual, n_surrogates)
    window_names = list(TIME_WINDOWS)
    roi_names = sorted(by_roi)
    coupling_names = list(couplings)
    n_roi, n_coupling, n_windows = len(roi_names), len(coupling_names), len(window_names)
    obs = np.empty((n_roi, n_coupling, k_actual, n_windows), dtype=np.float64)
    sur = np.empty((n_roi, n_coupling, k_actual, n_surrogates, n_windows), dtype=np.float64)

    times = ref["times"]
    time_mask = (times >= 0.0) & (times < 0.4)
    cropped_times = times[time_mask]
    masks = {name: (cropped_times >= a) & (cropped_times < b)
             for name, (a, b) in TIME_WINDOWS.items()}
    if len(cropped_times) != 200 or any(not m.any() for m in masks.values()):
        raise ValueError(f"Invalid 0–400 ms sample/window counts: {key}")

    for r, roi in enumerate(roi_names):
        roi_start = time.perf_counter()
        info = by_roi[roi]
        x = load_roi_signal(info)
        phases, amplitude = filter_components(x, ref["sfreq"], couplings)
        amp_toi = amplitude[..., time_mask]
        for c, coupling in enumerate(coupling_names):
            pha_toi = phases[coupling][..., time_mask]
            for k, trial_positions in enumerate(subsets):
                pha_sub = pha_toi[:, trial_positions, :]
                amp_sub = amp_toi[:, trial_positions, :]
                observed = window_averages_from_circular_erpac(pha_sub, amp_sub, masks)
                obs[r, c, k] = [observed[name] for name in window_names]
                for j, perm in enumerate(permutations[k]):
                    shuffled = window_averages_from_circular_erpac(
                        pha_sub, amp_sub[:, perm, :], masks)
                    sur[r, c, k, j] = [shuffled[name] for name in window_names]
        print(f"    ROI {roi}: completed in {time.perf_counter()-roi_start:.1f} s", flush=True)

    surrogate_mean = sur.mean(axis=3)
    surrogate_sd = sur.std(axis=3, ddof=1)
    if not np.isfinite(surrogate_sd).all() or np.any(surrogate_sd <= 0):
        raise RuntimeError(f"Invalid surrogate SD for {key}")
    z_subset = (obs - surrogate_mean) / surrogate_sd
    if not np.isfinite(z_subset).all():
        raise RuntimeError(f"Nonfinite subset z-scores for {key}")

    common = {
        "group": group, "sub": subject, "task": task, "task_stage": stage,
        "task_block": block or "NA", "adaptation_split": split or "NA",
        "condition": ("FTT" if task == "FTT" else
                      "DeCRAT_baseline" if block == "baseline" else
                      f"DeCRAT_adaptation_{split}"),
        "n_available": int(ref["n_trials"]), "n_matched": MATCHED_TRIALS,
        "n_subsamples": k_actual, "n_surrogates": n_surrogates,
        "selection_seed": selection["seed"],
        "permutation_seed": seed_for_surrogates(key),
    }
    agg_rows, subset_rows = [], []
    for r, roi in enumerate(roi_names):
        for c, coupling in enumerate(coupling_names):
            for w, window in enumerate(window_names):
                each_z = z_subset[r, c, :, w]
                agg_rows.append({
                    **common, "roi": roi, "coupling": coupling, "time_win": window,
                    "erpac_z": float(each_z.mean()),
                    "z_subsample_sd": float(each_z.std(ddof=1)) if k_actual > 1 else float("nan"),
                    "observed_mean": float(obs[r, c, :, w].mean()),
                    "surrogate_mean": float(surrogate_mean[r, c, :, w].mean()),
                    "surrogate_sd_mean": float(surrogate_sd[r, c, :, w].mean()),
                })
                for k in range(k_actual):
                    subset_rows.append({
                        **common, "roi": roi, "coupling": coupling,
                        "time_win": window, "subsample": k + 1,
                        "observed_erpac": float(obs[r, c, k, w]),
                        "surrogate_mean": float(surrogate_mean[r, c, k, w]),
                        "surrogate_sd": float(surrogate_sd[r, c, k, w]),
                        "erpac_z": float(z_subset[r, c, k, w]),
                    })

    arrays = {
        "observed": obs, "surrogates": sur, "subsets": subsets,
        "permutations": permutations,
        "roi_names": np.array(roi_names),
        "coupling_names": np.array(coupling_names),
        "window_names": np.array(window_names),
        "selection_seed": np.int64(selection["seed"]),
        "permutation_seed": np.int64(seed_for_surrogates(key)),
        "stratum_sizes": selection["stratum_sizes"],
        "stratum_quotas": selection["stratum_quotas"],
        "original_trial_ids": (selection["original_trial_ids"]
                               if selection["original_trial_ids"] is not None
                               else np.empty((0,), dtype=np.int64)),
        "times_analyzed": cropped_times,
    }
    return agg_rows, subset_rows, arrays


# ============================================================
# 7. CONDITION CHECKPOINTS AND EXPORT FOR R [FULL PIPELINE]
# ============================================================
# Each condition-stage is a separately checkpointed unit, so re-running
# --full skips completed conditions with unchanged inputs and settings.
# The last file written is completed.json; any interrupted condition is
# considered incomplete and recomputed on restart. Final CSV aggregates
# have ONE row per group × subject × condition × stage × ROI × coupling
# × window, without pseudo-replication by overlapping subsamples.
# LMM is NOT fitted here; retain the agreed 5-way fixed-effect model in R.


def condition_folder(output_root, key):
    group, subject, *_ = key
    return output_root / "conditions" / group / subject / stable_label(key)


def rebuild_master(output_root, signature):
    """Assemble completed condition checkpoints into two master CSV files."""
    folders = sorted((output_root / "conditions").glob("*/*/*/completed.json"))
    aggregates, subsets = [], []
    completed = 0
    for meta_path in folders:
        folder = meta_path.parent
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if metadata.get("status") != "complete" or metadata.get("protocol") != signature:
            continue
        with (folder / "aggregated.csv").open(newline="", encoding="utf-8") as f:
            aggregates.extend(csv.DictReader(f))
        with (folder / "subsamples.csv").open(newline="", encoding="utf-8") as f:
            subsets.extend(csv.DictReader(f))
        completed += 1
    if aggregates:
        atomic_csv(output_root / "erpac_z_LMM_ready.csv", aggregates, list(aggregates[0]))
        atomic_csv(output_root / "erpac_z_subsample_diagnostics.csv", subsets, list(subsets[0]))
    print(f"Master rebuild: {completed} completed conditions | "
          f"{len(aggregates)} LMM rows | {len(subsets)} subsample rows", flush=True)
    print(f"  Folder: {output_root}", flush=True)
    return completed


def run_full(indexed, output_root, n_surrogates, overwrite=False):
    couplings = checked_couplings()
    signature = full_protocol_signature(couplings, n_surrogates)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if old != signature:
            raise RuntimeError(
                "Existing full-run folder uses different settings. Choose a new "
                "--output-dir to prevent mixing incompatible results."
            )
    else:
        atomic_json(protocol_path, signature)

    keys = sorted(indexed, key=lambda k: (k[0], k[1], str(k)))
    overall_start = time.perf_counter()
    current_subject = None
    subject_start = None
    subject_processed = subject_skipped = 0
    processed = skipped = 0
    print(f"\nFULL RUN: {len(keys)} condition-stage combinations, "
          f"{n_surrogates} surrogates/subset", flush=True)

    for number, key in enumerate(keys, 1):
        group, subject, *_ = key
        subject_key = (group, subject)
        if subject_key != current_subject:
            if current_subject is not None:
                print(f"SUBJECT COMPLETE: {current_subject} | "
                      f"computed={subject_processed}, skipped={subject_skipped} | "
                      f"elapsed={time.perf_counter()-subject_start:.1f} s", flush=True)
            current_subject = subject_key
            subject_start = time.perf_counter()
            subject_processed = subject_skipped = 0
            print(f"\n===== SUBJECT {group}/{subject} =====", flush=True)

        cond_start = time.perf_counter()
        folder = condition_folder(output_root, key)
        sources = input_signature(indexed[key])
        if not overwrite and condition_is_complete(folder, signature, sources):
            skipped += 1
            subject_skipped += 1
            print(f"[{number}/{len(keys)}] SKIP condition {key} | "
                  f"elapsed={time.perf_counter()-cond_start:.1f} s", flush=True)
            continue
        print(f"[{number}/{len(keys)}] START condition {key}", flush=True)
        # Invalidate prior completion marker before recalculating outputs.
        # This prevents partial overwrites being mistaken for a valid checkpoint.
        marker = folder / "completed.json"
        if marker.exists():
            marker.unlink()
        try:
            agg, subs, arrays = evaluate_full_condition(key, indexed[key], n_surrogates, couplings)
            folder.mkdir(parents=True, exist_ok=True)
            atomic_npz(folder / "full_erpac.npz", arrays)
            atomic_csv(folder / "aggregated.csv", agg, list(agg[0]))
            atomic_csv(folder / "subsamples.csv", subs, list(subs[0]))
            meta = {
                "status": "complete", "protocol": signature, "sources": sources,
                "condition": list(key), "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "n_aggregated": len(agg), "n_subsample_rows": len(subs),
            }
            atomic_json(folder / "completed.json", meta)
            processed += 1
            subject_processed += 1
            print(f"[{number}/{len(keys)}] DONE condition {key} | "
                  f"elapsed={time.perf_counter()-cond_start:.1f} s | "
                  f"LMM rows={len(agg)}", flush=True)
        except Exception:
            # Leave prior valid outputs untouched on errors, but never claim
            # this condition completed successfully in the current run.
            print(f"ERROR in condition {key} after "
                  f"{time.perf_counter()-cond_start:.1f} s", flush=True)
            traceback.print_exc()
            print("Resume with the same command; only incomplete conditions rerun.", flush=True)
            raise

    if current_subject is not None:
        print(f"SUBJECT COMPLETE: {current_subject} | "
              f"computed={subject_processed}, skipped={subject_skipped} | "
              f"elapsed={time.perf_counter()-subject_start:.1f} s", flush=True)
    rebuild_master(output_root, signature)
    print(f"FULL RUN COMPLETE | computed={processed}, skipped={skipped} | "
          f"total_elapsed={time.perf_counter()-overall_start:.1f} s", flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=EXPECTED_GROUPS)
    parser.add_argument("--subject", help="e.g. s1_pac_sub01")
    parser.add_argument("--stage3", action="store_true",
                        help="Run observed ERPAC pilot after Stages 1–2 validation")
    parser.add_argument("--stage4", action="store_true",
                        help="Run trial-shuffled surrogate ERPAC for ONE condition-stage")
    parser.add_argument("--task", choices=EXPECTED_TASKS,
                        help="Condition for stage 4 pilot, e.g. FTT")
    parser.add_argument("--task-stage", choices=EXPECTED_STAGES,
                        help="Movement stage for stage 4 pilot, e.g. plan")
    parser.add_argument("--block", choices=("baseline", "adaptation"),
                        help="DeCRAT block for stage 4 pilot")
    parser.add_argument("--split", choices=("early", "late"),
                        help="DeCRAT adaptation split for stage 4 pilot")
    parser.add_argument("--n-surrogates", type=int, default=N_SURROGATES,
                        help="Surrogate count (default: 200; smaller for smoke test)")
    parser.add_argument("--full", action="store_true",
                        help="Run complete surrogate-normalized ERPAC pipeline through R-ready CSV")
    parser.add_argument("--rebuild-master", action="store_true",
                        help="Rebuild master CSVs from checkpointed conditions (no ERPAC)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(PCA_ROOT) / "erpac_stratified" / "full_run_v1",
                        help="Root for resumable full-run outputs")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recalculate even valid checkpointed conditions")
    args = parser.parse_args()
    if sum([args.stage3, args.stage4, args.full, args.rebuild_master]) > 1:
        parser.error("Choose only one of --stage3, --stage4, --full, --rebuild-master")
    if args.n_surrogates < 2:
        parser.error("--n-surrogates must be >= 2")
    if args.rebuild_master:
        signature = full_protocol_signature(checked_couplings(), args.n_surrogates)
        rebuild_master(args.output_dir, signature)
        return
    if args.stage3 and args.stage4:
        parser.error("Run --stage3 or --stage4, not both")
    if args.stage3 and (args.group is None or args.subject is None):
        parser.error("--stage3 requires both --group and --subject (pilot safety)")
    if args.stage4:
        if any(v is None for v in (args.group, args.subject, args.task, args.task_stage)):
            parser.error("--stage4 requires --group --subject --task --task-stage")
        if args.task == "FTT" and (args.block is not None or args.split is not None):
            parser.error("FTT cannot have --block or --split")
        if args.task == "DeCRAT" and (args.block is None or
            (args.block == "adaptation" and args.split is None) or
            (args.block == "baseline" and args.split is not None)):
            parser.error("DeCRAT requires --block; adaptation also requires --split")
        if args.n_surrogates < 2:
            parser.error("--n-surrogates must be >= 2")
    if args.full:
        indexed = index_inputs(Path(PCA_ROI_TC_DIR), group=args.group, subject=args.subject)
        run_full(indexed, args.output_dir, args.n_surrogates, overwrite=args.overwrite)
        return
    print("Input validation and stratified selections; --stage3 for observed or --stage4 for surrogate ERPAC.\n")
    indexed = index_inputs(Path(PCA_ROI_TC_DIR), group=args.group, subject=args.subject)
    print(f"Validated {len(indexed)} participant-condition-stage combinations:\n")
    for key, roi_files in sorted(indexed.items(), key=lambda item: str(item[0])):
        reference = next(iter(roi_files.values()))
        selection = draw_stratified_subsets(
            key, reference["n_trials"], reference["epoch_indices"]
        )
        selected = selection["subsets"]
        used = len(np.unique(selected))
        print(f"{key}: {reference['n_trials']} trials; {reference['n_times']} samples; "
              f"{reference['sfreq']:.1f} Hz; ROIs={','.join(sorted(roi_files))}")
        print(f"  Strata sizes: {selection['stratum_sizes'].tolist()} | "
              f"selected per stratum: {selection['stratum_quotas'].tolist()}")
        print(f"  Draws: {len(selected)} x {selected.shape[1]} trials | "
              f"unique trials represented: {used}/{reference['n_trials']} | "
              f"seed={selection['seed']}")
        print(f"  First draw (0-based row positions): {selected[0].tolist()}")
        if len(selected) == 1:
            print("  NOTE: exactly 37 valid trials; only one distinct draw is possible.")
    print("\nStage 2 complete.")
    if args.stage3:
        run_stage3(indexed, Path(PCA_ROOT) / "erpac_stratified" / "step3_observed_pilot")
    elif args.stage4:
        requested = standard_condition(args.task, args.block, args.split)
        chosen_key = (
            args.group, args.subject, args.task, args.task_stage, *requested
        )
        if chosen_key not in indexed:
            parser.error(f"Requested condition not found: {chosen_key}")
        run_stage4(
            {chosen_key: indexed[chosen_key]},
            Path(PCA_ROOT) / "erpac_stratified" / "step4_surrogates_pilot",
            n_surrogates=args.n_surrogates,
        )
    else:
        print("No data or output files changed. Use --stage3 or --stage4 for pilot ERPAC.")


if __name__ == "__main__":
    main()
