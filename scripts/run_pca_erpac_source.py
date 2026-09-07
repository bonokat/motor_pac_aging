# ============================================================
# ERPAC ON PCA-DENOISED ROI REPRESENTATIVE TIME COURSES
# ============================================================
#
# INPUT:
#   F:\# study 2\eeg_data\erpac\PCA\pca_stcs\stcs_ave_vtx
#       group/
#           sub/
#               <condition>_<ROI>_pca_roi_tc.npz
#
# Each .npz contains:
#   data  -> (epochs, 1, times)
#   times -> (times,)
#   subject, group, roi, task, task_stage,
#   task_block, adaptation_split, etc.
#
# PIPELINE:
#
# PCA ROI time course
# (epochs × 1 × time)
#          ↓
# remove singleton vertex dimension
#          ↓
# (epochs × time)
#          ↓
# EventRelatedPac.filterfit()
#          ↓
# ERPAC (amp_freq × time)
#
# NOTE:
# There is NO additional vertex/label/ROI averaging here.
# The PCA pipeline has already produced one representative
# signal per functional ROI.
# ============================================================

import mne
from mne.viz import Brain
import os
import numpy as np
import pickle
import pyarrow

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from config import check_paths
from config.config import (HEMI, ROI, FS_SUB, FS_SRC_PATH, COUPLINGS,
                           SUBJECTS, GROUPS, TASKS, TASK_STAGES, TASK_BLOCKS)
from config.paths import FS_FOLDER, SOURCE_DATA_DIR, ERPAC_DIR, ROI_STCS_DIR, ERPAC_FIGS_DIR, PCA_ROOT, PCA_ROI_TC_DIR
from utils.helpers import iterate_dataset
from utils.plotting import plot_group_erpac, plot_group_erpac_timecourse

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorpac import EventRelatedPac


# Keep PCA-based ERPAC outputs separate from the original ERPAC.
ERPAC_PCA_DIR = (
    Path(ERPAC_DIR)
    / "PCA"
    / "erpac_roi_tc"
)

ERPAC_PCA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ERPAC SETTINGS
# ============================================================

ERPAC_METHOD = "circular"
MCP = "bonferroni"
N_JOBS = -1


# ============================================================
# HELPERS
# ============================================================

def npz_scalar(npz, key, default=None):
    """
    Read a scalar/string safely from an np.savez file.
    """

    if key not in npz.files:
        return default

    value = npz[key]

    if np.ndim(value) == 0:
        return value.item()

    if value.size == 1:
        return value.reshape(-1)[0].item()

    return value


def clean_optional(value):
    """
    Convert saved 'NA', 'None', empty values, etc. back to None.
    """

    if value is None:
        return None

    value = str(value)

    if value.lower() in {
        "na",
        "none",
        "nan",
        "",
    }:
        return None

    return value


def condition_key_from_npz(npz):
    """
    Return a condition key that uniquely identifies an ERPAC condition.
    """

    group = str(
        npz_scalar(npz, "group")
    )

    sub = str(
        npz_scalar(npz, "subject")
    )

    task = str(
        npz_scalar(npz, "task")
    )

    stage = str(
        npz_scalar(npz, "task_stage")
    )

    block = clean_optional(
        npz_scalar(
            npz,
            "task_block",
            None,
        )
    )

    adaptation_split = clean_optional(
        npz_scalar(
            npz,
            "adaptation_split",
            None,
        )
    )

    return (
        group,
        sub,
        task,
        stage,
        block,
        adaptation_split,
    )


def normalize_task_condition(task, block, adaptation_split):
    """
    Enforce task-specific condition structure.

    FTT:
        block = None
        adaptation_split = None

    DeCRAT:
        block may be baseline/adaptation
        adaptation_split only applies to adaptation
    """

    if task == "FTT":
        return None, None

    if task == "DeCRAT":
        if block != "adaptation":
            adaptation_split = None
        return block, adaptation_split

    return block, adaptation_split


def condition_name(
    task,
    stage,
    block=None,
    adaptation_split=None,
):
    """Stable human-readable condition name."""

    parts = [
        str(task),
        str(stage),
    ]

    if block is not None:
        parts.append(
            str(block)
        )

    if adaptation_split is not None:
        parts.append(
            str(adaptation_split)
        )

    return "_".join(parts)


def infer_sfreq(times):
    """
    Infer sampling frequency from the saved time vector.
    """

    times = np.asarray(
        times,
        dtype=float,
    )

    if times.ndim != 1:
        raise ValueError(
            f"'times' must be 1-D; got shape {times.shape}"
        )

    if len(times) < 2:
        raise ValueError(
            "Need at least two time samples to infer sfreq."
        )

    dt = np.diff(times)

    if not np.all(
        np.isfinite(dt)
    ):
        raise ValueError(
            "Non-finite values found in time differences."
        )

    median_dt = float(
        np.median(dt)
    )

    if median_dt <= 0:
        raise ValueError(
            "Time vector must be strictly increasing."
        )

    # Verify that the sampling interval is effectively regular.
    if not np.allclose(
        dt,
        median_dt,
        rtol=1e-5,
        atol=1e-10,
    ):
        raise ValueError(
            "Time vector is not regularly sampled."
        )

    return 1.0 / median_dt


# ============================================================
# DISCOVER PCA ROI TIME-COURSE FILES
# ============================================================

if not PCA_ROI_TC_DIR.exists():
    raise FileNotFoundError(
        f"PCA ROI time-course directory does not exist:\n"
        f"{PCA_ROI_TC_DIR}"
    )


roi_tc_files = sorted(
    PCA_ROI_TC_DIR.rglob(
        "*_pca_roi_tc.npz"
    )
)


if len(roi_tc_files) == 0:
    raise FileNotFoundError(
        f"No PCA ROI time-course files found under:\n"
        f"{PCA_ROI_TC_DIR}"
    )


print(
    f"Found {len(roi_tc_files)} PCA ROI time-course files."
)


# ============================================================
# INDEX FILES BY CONDITION
# ============================================================
#
# Each condition should have one file per ROI.
#
# Example key:
# (
#   "Y",
#   "s1_pac_sub01",
#   "DeCRAT",
#   "plan",
#   "adaptation",
#   "early"
# )
#
# ============================================================

condition_files = {}


for file_path in roi_tc_files:

    with np.load(
        file_path,
        allow_pickle=True,
    ) as npz:

        task = str(
            npz_scalar(
                npz,
                "task",
            )
        )

        # Restrict analysis to requested task(s).
        if task not in TASKS:
            continue

        key = condition_key_from_npz(
            npz
        )

        roi = str(
            npz_scalar(
                npz,
                "roi",
            )
        )

    condition_files.setdefault(
        key,
        {}
    )

    if roi in condition_files[key]:
        raise RuntimeError(
            f"Duplicate ROI file for condition {key}, ROI {roi}:\n"
            f"{file_path}"
        )

    condition_files[key][roi] = file_path


print(
    f"Indexed {len(condition_files)} participant-condition combinations "
    f"for TASKS={TASKS}."
)


# ============================================================
# MASTER ERPAC RESULTS
# ============================================================

erpac_results_rows = []


# ============================================================
# ITERATE CONDITIONS
# ============================================================

for (
    group,
    sub,
    task,
    stage,
    block,
    adaptation_split,
), roi_files in sorted(
    condition_files.items(),
    key=lambda item: tuple(
        "" if x is None else str(x)
        for x in item[0]
    ),
):

    block, adaptation_split = normalize_task_condition(
        task=task,
        block=block,
        adaptation_split=adaptation_split,
    )

    cond_name = condition_name(
        task=task,
        stage=stage,
        block=block,
        adaptation_split=adaptation_split,
    )

    print("\n")
    print("=" * 70)

    print(
        f"Processing: "
        f"{group} / {sub} / "
        f"{cond_name}"
    )

    print(
        f"ROIs found: "
        f"{sorted(roi_files.keys())}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Participant-condition results saved as:
    #
    # erpac_results[coupling][roi] = {...}
    # --------------------------------------------------------

    erpac_results = {}


    # ========================================================
    # LOAD ALL ROI SIGNALS FOR THIS CONDITION
    # ========================================================

    roi_data = {}


    for roi, file_path in sorted(
        roi_files.items()
    ):

        with np.load(
            file_path,
            allow_pickle=True,
        ) as npz:

            data = np.asarray(
                npz["data"],
                dtype=float,
            )

            times = np.asarray(
                npz["times"],
                dtype=float,
            )

            # Expected:
            # epochs × 1 × time
            if data.ndim != 3:
                raise ValueError(
                    f"{file_path.name}: expected 3-D data "
                    f"(epochs, 1, times), got {data.shape}"
                )

            if data.shape[1] != 1:
                raise ValueError(
                    f"{file_path.name}: expected singleton vertex "
                    f"dimension, got {data.shape}"
                )

            if data.shape[2] != len(
                times
            ):
                raise ValueError(
                    f"{file_path.name}: data/time mismatch: "
                    f"{data.shape[2]} samples vs {len(times)} times"
                )

            # There should be no NaNs after the ROI np.nanmean
            # in the PCA-reconstruction pipeline.
            n_nonfinite = int(
                np.sum(
                    ~np.isfinite(data)
                )
            )

            if n_nonfinite > 0:
                raise ValueError(
                    f"{file_path.name}: "
                    f"{n_nonfinite} non-finite values found in roi_tc."
                )

            sf = infer_sfreq(
                times
            )

            roi_data[roi] = {
                "data": data,
                "times": times,
                "sfreq": sf,

                # PCA provenance
                "variance_threshold":
                    npz_scalar(
                        npz,
                        "variance_threshold",
                        np.nan,
                    ),

                "n_components_retained":
                    npz_scalar(
                        npz,
                        "n_components_retained",
                        np.nan,
                    ),

                "actual_variance_retained":
                    npz_scalar(
                        npz,
                        "actual_variance_retained",
                        np.nan,
                    ),

                "original_epoch_indices":
                    (
                        np.asarray(
                            npz["original_epoch_indices"]
                        )
                        if "original_epoch_indices" in npz.files
                        else None
                    ),
            }

        print(
            f"  {roi}: "
            f"{data.shape} | "
            f"sfreq={sf:.3f} Hz"
        )


    # ========================================================
    # VERIFY ROI FILES REFER TO THE SAME EPOCH CONDITION
    # ========================================================

    n_epochs_by_roi = {
        roi: values["data"].shape[0]
        for roi, values in roi_data.items()
    }

    if len(
        set(n_epochs_by_roi.values())
    ) != 1:

        raise RuntimeError(
            f"Different number of epochs across ROIs for "
            f"{sub} / {cond_name}:\n"
            f"{n_epochs_by_roi}"
        )


    # It is useful, though not mathematically required for separate
    # ROI ERPAC computation, to ensure the same selected trials were
    # used across ROIs.
    epoch_index_arrays = [
        values["original_epoch_indices"]
        for values in roi_data.values()
        if values["original_epoch_indices"] is not None
    ]

    if len(epoch_index_arrays) > 1:

        ref_indices = epoch_index_arrays[0]

        for arr in epoch_index_arrays[1:]:

            if not np.array_equal(
                arr,
                ref_indices,
            ):
                raise RuntimeError(
                    f"Original epoch indices differ across ROIs for "
                    f"{sub} / {cond_name}."
                )


    # ========================================================
    # ERPAC BY COUPLING
    # ========================================================

    for coupling_name, (
        phase_freq,
        amp_freq,
    ) in COUPLINGS.items():

        print(
            f"\nComputing {coupling_name}: "
            f"{cond_name}"
        )

        erpac_results[
            coupling_name
        ] = {}


        # ====================================================
        # ERPAC BY FUNCTIONAL ROI
        # ====================================================

        for roi, roi_info in roi_data.items():

            data = roi_info[
                "data"
            ]

            times = roi_info[
                "times"
            ]

            sf = roi_info[
                "sfreq"
            ]

            # ------------------------------------------------
            # Remove only the singleton vertex dimension:
            #
            # epochs × 1 × time
            #       ↓
            # epochs × time
            # ------------------------------------------------

            x = data[
                :,
                0,
                :
            ]

            n_epochs, n_times = (
                x.shape
            )

            if n_epochs < 2:
                raise RuntimeError(
                    f"ERPAC requires multiple epochs; "
                    f"{sub} / {cond_name} / {roi} has "
                    f"{n_epochs} epoch(s)."
                )


            # ------------------------------------------------
            # IMPORTANT:
            #
            # Use a fresh EventRelatedPac object for each ROI.
            # This avoids accidentally carrying p-values/state
            # from the previous ROI.
            # ------------------------------------------------

            p = EventRelatedPac(
                f_pha=phase_freq,
                f_amp=amp_freq,
            )

            amp_freqs = np.asarray(
                p.yvec
            )


            # ------------------------------------------------
            # ERPAC
            # ------------------------------------------------

            erpac = p.filterfit(
                sf,
                x,
                method=ERPAC_METHOD,
                mcp=MCP,
                n_jobs=N_JOBS,
            ).squeeze()


            # Expected:
            # amp_freq × time
            if erpac.ndim == 1:
                erpac = erpac[
                    np.newaxis,
                    :
                ]

            pvalues = np.asarray(
                p.pvalues
            ).squeeze()

            if pvalues.ndim == 1:
                pvalues = pvalues[
                    np.newaxis,
                    :
                ]


            if erpac.shape[-1] != n_times:
                raise RuntimeError(
                    f"Unexpected ERPAC time dimension for "
                    f"{sub} / {cond_name} / {roi}: "
                    f"{erpac.shape} vs n_times={n_times}"
                )


            # ------------------------------------------------
            # STORE PARTICIPANT-CONDITION RESULTS
            # ------------------------------------------------

            erpac_results[
                coupling_name
            ][
                roi
            ] = {

                "erpac":
                    erpac,

                "p-values":
                    pvalues,

                "times":
                    times,

                "amp_freqs":
                    amp_freqs,

                "sfreq":
                    sf,

                "n_epochs":
                    n_epochs,

                "pca_variance_threshold":
                    roi_info[
                        "variance_threshold"
                    ],

                "pca_n_components_retained":
                    roi_info[
                        "n_components_retained"
                    ],

                "pca_actual_variance_retained":
                    roi_info[
                        "actual_variance_retained"
                    ],
            }


            # ------------------------------------------------
            # LONG-FORMAT MASTER DATA
            # ------------------------------------------------

            for f_idx, amp_frequency in enumerate(
                amp_freqs
            ):

                for t_idx, time in enumerate(
                    times
                ):

                    erpac_results_rows.append(
                        {
                            "sub":
                                sub,

                            "group":
                                group,

                            "task":
                                task,

                            "task_stage":
                                stage,

                            "task_block":
                                (
                                    block
                                    if block is not None
                                    else "NA"
                                ),

                            "adaptation_split":
                                (
                                    adaptation_split
                                    if adaptation_split is not None
                                    else "NA"
                                ),

                            "coupling":
                                coupling_name,

                            "roi":
                                roi,

                            "amp_freq":
                                float(
                                    amp_frequency
                                ),

                            "time":
                                float(
                                    time
                                ),

                            "erpac_value":
                                float(
                                    erpac[
                                        f_idx,
                                        t_idx,
                                    ]
                                ),

                            "n_epochs":
                                int(
                                    n_epochs
                                ),

                            "pca_variance_threshold":
                                float(
                                    roi_info[
                                        "variance_threshold"
                                    ]
                                ),

                            "pca_n_components_retained":
                                int(
                                    roi_info[
                                        "n_components_retained"
                                    ]
                                ),

                            "pca_actual_variance_retained":
                                float(
                                    roi_info[
                                        "actual_variance_retained"
                                    ]
                                ),
                        }
                    )


    # ========================================================
    # SAVE PARTICIPANT-CONDITION ERPAC RESULTS
    # ========================================================

    condition_save_dir = (
        ERPAC_PCA_DIR
        / str(group)
        / str(sub)
    )

    condition_save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    erpac_filename = (
        f"{sub}_"
        f"{cond_name}_"
        f"pca_roi_tc_erpac_results.pkl"
    )


    with open(
        condition_save_dir
        / erpac_filename,
        "wb",
    ) as f:

        pickle.dump(
            erpac_results,
            f,
        )


# ============================================================
# CREATE MASTER ERPAC DATAFRAME
# ============================================================

erpac_df = pd.DataFrame(
    erpac_results_rows
)


if erpac_df.empty:
    raise RuntimeError(
        "No ERPAC results were generated."
    )


# ============================================================
# FREQUENCY-AVERAGED ERPAC DATAFRAME
# ============================================================

erpac_df_mean_freq = (
    erpac_df
    .groupby(
        [
            "sub",
            "group",
            "task",
            "task_stage",
            "task_block",
            "adaptation_split",
            "coupling",
            "roi",
            "time",
            "n_epochs",
            "pca_variance_threshold",
            "pca_n_components_retained",
            "pca_actual_variance_retained",
        ],
        as_index=False,
        observed=True,
    )[
        "erpac_value"
    ]
    .mean()
)


# ============================================================
# OPTIMISE DATA TYPES
# ============================================================

categorical_cols = [
    "sub",
    "group",
    "task",
    "task_stage",
    "task_block",
    "adaptation_split",
    "coupling",
    "roi",
]


for col in categorical_cols:

    erpac_df[col] = (
        erpac_df[col]
        .astype("category")
    )


float_cols = [
    "amp_freq",
    "time",
    "erpac_value",
    "pca_variance_threshold",
    "pca_actual_variance_retained",
]


for col in float_cols:

    erpac_df[col] = (
        erpac_df[col]
        .astype("float32")
    )


erpac_df[
    "pca_n_components_retained"
] = erpac_df[
    "pca_n_components_retained"
].astype(
    "int16"
)

erpac_df[
    "n_epochs"
] = erpac_df[
    "n_epochs"
].astype(
    "int16"
)


# ============================================================
# SAVE MASTER DATASETS
# ============================================================

task_tag = "FTT_DeCRAT" if set(TASKS) == {"FTT", "DeCRAT"} else "_".join(TASKS)


full_parquet_path = (
    ERPAC_PCA_DIR
    / f"{task_tag}_pca_roi_tc_erpac_results.parquet"
)


erpac_df.to_parquet(
    full_parquet_path,
    engine="pyarrow",
    compression="zstd",
    index=False,
)


mean_freq_path = (
    ERPAC_PCA_DIR
    / f"{task_tag}_pca_roi_tc_erpac_results_mean_freq.csv"
)


erpac_df_mean_freq.to_csv(
    mean_freq_path,
    index=False,
)


print("\n")
print("=" * 70)
print(
    "PCA ROI-TIME-COURSE ERPAC ANALYSIS COMPLETE"
)
print("=" * 70)

print(
    f"\nFull ERPAC dataset:\n"
    f"{full_parquet_path}"
)

print(
    f"\nFrequency-averaged dataset:\n"
    f"{mean_freq_path}"
)

print(
    f"\nRows in full dataframe: "
    f"{len(erpac_df):,}"
)