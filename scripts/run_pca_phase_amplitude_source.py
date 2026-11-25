"""Study 2: cache trial-level phase and amplitude, before trial matching.
Place in motor_pac_aging/scripts/. Run from the project's Python environment.
Only load trusted local NPZ files. No surrogate generation or statistical tests.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorpac
from tensorpac import EventRelatedPac
from tensorpac.methods import erpac

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from config.paths import PCA_ROI_TC_DIR, PCA_ROOT
from config.config import COUPLINGS, ROI, GROUPS, TASKS

INPUT_DIR = Path(PCA_ROI_TC_DIR)
OUTPUT_DIR = Path(PCA_ROOT) / 'phase_amplitude_broadband'
PHASE_BANDS = {'theta_gamma': [4, 8], 'alpha_gamma': [8, 12],
               'beta_gamma': [13, 30]}
GAMMA_BAND = [30, 80]
CYCLE = (3, 6)
N_JOBS = 1
SAVE_RAW_ERPAC = False  # optional exploratory all-trial ERPAC
SCHEMA_VERSION = 1


def scalar(z, key, default=None):
    if key not in z.files:
        return default
    a = z[key]
    if a.size != 1:
        raise ValueError(f'Expected scalar metadata for {key}')
    return a.item()


def optional(v):
    return '' if v is None or str(v).lower() in ('', 'none', 'na', 'nan') else str(v)


def read_key(z):
    group, sub, task, stage, roi = [str(scalar(z, k)) for k in
        ('group', 'subject', 'task', 'task_stage', 'roi')]
    block = optional(scalar(z, 'task_block'))
    split = optional(scalar(z, 'adaptation_split'))
    if task == 'FTT':
        block = split = ''
    elif task == 'DeCRAT' and block != 'adaptation':
        split = ''
    return group, sub, task, stage, block, split, roi


def included(key):
    group, sub, task, stage, block, split, roi = key
    return (group in GROUPS and task in TASKS and roi in ROI
            and stage in ('plan', 'go') and
            (task == 'FTT' or (task == 'DeCRAT' and
             (block == 'baseline' or (block == 'adaptation' and split in ('early', 'late'))))))


def signature(path):
    stat = path.stat()
    settings = dict(schema=SCHEMA_VERSION, input=str(path.resolve()),
                    size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    phase_bands=PHASE_BANDS, gamma=GAMMA_BAND, cycle=CYCLE,
                    method='hilbert', raw_erpac=SAVE_RAW_ERPAC,
                    tensorpac=getattr(tensorpac, '__version__', 'unknown'),
                    numpy=np.__version__)
    return json.dumps(settings, sort_keys=True)


def output_path(key):
    group, sub, task, stage, block, split, roi = key
    condition = '_'.join(v for v in (task, stage, block, split) if v)
    return OUTPUT_DIR / group / sub / f'{condition}_{roi}_phase_amp.npz'


def cache_valid(path, sig):
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            required = {'phase_theta', 'phase_alpha', 'phase_beta', 'amp_gamma',
                        'times', 'sfreq', 'n_epochs', 'original_epoch_indices'}
            if SAVE_RAW_ERPAC:
                required |= {f'raw_erpac_{c}' for c in PHASE_BANDS}
            return (required.issubset(z.files) and scalar(z, 'signature') == sig)
    except (OSError, ValueError, KeyError):
        return False


def load_input(path):
    with np.load(path, allow_pickle=True) as z:
        key = read_key(z)
        data = np.asarray(z['data'], dtype=np.float64)
        times = np.asarray(z['times'], dtype=np.float64)
        if data.ndim != 3 or data.shape[1] != 1 or times.ndim != 1 or data.shape[-1] != times.size:
            raise ValueError(f'Unexpected data/time shape: {data.shape}, {times.shape}')
        if data.shape[0] < 2 or times.size < 2 or not np.isfinite(data).all() or not np.isfinite(times).all():
            raise ValueError('Insufficient or nonfinite data')
        dt = np.diff(times)
        if not np.all(dt > 0) or not np.allclose(dt, np.median(dt), rtol=1e-5, atol=1e-10):
            raise ValueError('Invalid sampling times')
        sfreq = float(1 / np.median(dt))
        if not np.isclose(sfreq, 500, rtol=1e-5):
            raise ValueError(f'Unexpected sampling frequency: {sfreq}')
        has_indices = 'original_epoch_indices' in z.files
        indices = np.asarray(z['original_epoch_indices']) if has_indices else np.arange(data.shape[0])
        if indices.ndim != 1 or indices.size != data.shape[0] or indices.dtype.kind not in 'iu' or np.unique(indices).size != indices.size:
            raise ValueError('Invalid original_epoch_indices')
        provenance = {k: float(scalar(z, k, np.nan)) for k in
                      ('variance_threshold', 'n_components_retained', 'actual_variance_retained')}
    return key, data[:, 0, :], times, sfreq, indices, has_indices, provenance


def extract(x, sfreq):
    # Tensorpac filters each trial independently along time.
    # Preserve shape: (1 frequency band, trials, times).
    phases = {}
    amp = None
    for coupling, band in PHASE_BANDS.items():
        pac = EventRelatedPac(f_pha=band, f_amp=GAMMA_BAND,
                              dcomplex='hilbert', cycle=CYCLE, verbose=False)
        phase = pac.filter(sfreq, x, ftype='phase', n_jobs=N_JOBS)
        phases['phase_' + coupling.split('_')[0]] = phase
        if amp is None:
            amp = pac.filter(sfreq, x, ftype='amplitude', n_jobs=N_JOBS)
    arrays = dict(phases, amp_gamma=amp)
    for name, a in arrays.items():
        if a.shape != (1, *x.shape) or not np.isfinite(a).all():
            raise ValueError(f'Unexpected or nonfinite {name}: {a.shape}')
    if SAVE_RAW_ERPAC:
        for coupling in PHASE_BANDS:
            phase = phases['phase_' + coupling.split('_')[0]]
            rho, _ = erpac(np.moveaxis(phase, 1, -1), np.moveaxis(amp, 1, -1))
            if rho.shape != (1, 1, x.shape[-1]) or not np.isfinite(rho).all():
                raise ValueError(f'Invalid raw ERPAC: {coupling}, {rho.shape}')
            arrays['raw_erpac_' + coupling] = rho
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', help='Restrict to a subject ID for a pilot')
    parser.add_argument('--group', choices=GROUPS)
    parser.add_argument('--force', action='store_true', help='Recompute cached files')
    args = parser.parse_args()
    if set(COUPLINGS) != set(PHASE_BANDS) or any(
        list(COUPLINGS[c][0]) != band or list(COUPLINGS[c][1]) != GAMMA_BAND
        for c, band in PHASE_BANDS.items()):
        raise ValueError('COUPLINGS must match theta/alpha/beta and broadband GAMMA=[30,80]. Check config.py.')
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(INPUT_DIR)
    indexed = {}
    for path in sorted(INPUT_DIR.rglob('*_pca_roi_tc.npz')):
        with np.load(path, allow_pickle=True) as z:
            key = read_key(z)
        if not included(key) or (args.subject and key[1] != args.subject) or (args.group and key[0] != args.group):
            continue
        if key in indexed:
            raise ValueError(f'Duplicate input for {key}: {indexed[key]} and {path}')
        indexed[key] = path
    if not indexed:
        raise RuntimeError('No matching input files found')
    conditions = {}
    for key, path in indexed.items():
        conditions.setdefault(key[:-1], {})[key[-1]] = path
    print(f'Found {len(indexed)} ROI files in {len(conditions)} condition-stage combinations.', flush=True)
    print(f'Output: {OUTPUT_DIR}', flush=True)
    start = perf_counter()
    completed = skipped = 0
    for i, (condition, files) in enumerate(sorted(conditions.items()), 1):
        if set(files) != set(ROI):
            raise ValueError(f'Missing ROI(s) for {condition}: {set(ROI)-set(files)}')
        loaded = [load_input(path) for _, path in sorted(files.items())]
        ref = loaded[0]
        for item in loaded[1:]:
            if item[1].shape != ref[1].shape or not np.array_equal(item[2], ref[2]) or not np.array_equal(item[4], ref[4]):
                raise ValueError(f'ROI trial/time mismatch for {condition}')
        if not all(item[5] for item in loaded):
            print(f'WARNING: original epoch IDs absent in some ROI files for {condition}; row alignment cannot be fully verified.', flush=True)
        for key, x, times, sfreq, indices, has_indices, provenance in loaded:
            source = indexed[key]
            dest = output_path(key)
            sig = signature(source)
            if not args.force and cache_valid(dest, sig):
                skipped += 1
                continue
            arrays = extract(x, sfreq)
            group, sub, task, stage, block, split, roi = key
            arrays.update(times=times, sfreq=sfreq, n_epochs=x.shape[0],
                          original_epoch_indices=indices, has_original_epoch_indices=has_indices,
                          group=group, subject=sub, task=task, task_stage=stage,
                          task_block=block, adaptation_split=split, roi=roi,
                          phase_bands_hz=np.asarray(list(PHASE_BANDS.values())),
                          gamma_band_hz=np.asarray(GAMMA_BAND), signature=sig, **provenance)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, dest)
            completed += 1
        print(f'[{i}/{len(conditions)}] {condition} | saved {completed}, skipped {skipped} | elapsed {(perf_counter()-start)/60:.1f} min', flush=True)
    print(f'Done. Saved {completed}, skipped {skipped}.', flush=True)


if __name__ == '__main__':
    main()
