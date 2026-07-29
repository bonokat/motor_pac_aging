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
                           SUBJECTS, GROUPS, TASKS, TASK_STAGES)
from config.paths import FS_FOLDER, SOURCE_DATA_DIR, ERPAC_DIR, ROI_STCS_DIR, ERPAC_FIGS_DIR
from utils.helpers import iterate_dataset
from utils.plotting import plot_group_erpac, plot_group_erpac_timecourse

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorpac import EventRelatedPac

# %matplotlib qt

# ============================================================
# ERPAC RESULTS CONTAINER
# ============================================================
erpac_results_rows = []

# ============================================================
# ERPAC ANALYSIS PARAMETERS
# ============================================================

alpha_threshold = 0.05  # significance threshold for ERPAC surrogate statistics

# ============================================================
# LOAD SOURCE SPACE ATLAS AND DEFINE ROIs
# ============================================================
# Load anatomical labels from FreeSurfer aparc.a2009s atlas.
# These labels are used to extract source-space activity from
# predefined motor-related cortical regions.

labels = mne.read_labels_from_annot(
    subject=FS_SUB,
    parc="aparc.a2009s",
    hemi=HEMI[1:],
    subjects_dir=FS_FOLDER
)


# Create lookup dictionary:
# label name -> mne.Label object
# This allows fast extraction of labels specified in ROI configuration.

label_dict = {
    label.name: label 
    for label in labels
}

# Convert ROI configuration (names) into actual MNE Label objects.
# This will later be used to extract vertices belonging to each ROI.

ROI_labels = {
    roi: {
        label_name: label_dict[label_name]
        for label_name in label_names
    }
    for roi, label_names in ROI.items()
}

# ============================================================
# LOAD SOURCE SPACE INFORMATION
# ============================================================
# Source space contains the cortical mesh information
# (vertices, hemispheres, etc.) required for source estimates.

src = mne.read_source_spaces(
    FS_SRC_PATH
)

# ============================================================
# ITERATE THROUGH PARTICIPANTS AND EXPERIMENTAL CONDITIONS
# ============================================================
# Loop structure:
#
# group
#   -> subject
#       -> task
#           -> task stage (plan/go)
#
# Each iteration performs a complete ROI extraction and ERPAC
# analysis for one participant and condition.

for group, sub, task, stage in iterate_dataset(
        GROUPS,
        SUBJECTS,
        TASKS,
        TASK_STAGES
    ):

    print("Processing:", group, sub, task, stage)

    # Containers for this participant-condition analysis

    roi_data = {}
    erpac_results = {}

    # ========================================================
    # LOAD SOURCE ESTIMATES AND EXTRACT ROI SIGNALS
    # ========================================================
    # Load epoch-wise source estimates.
    # Each STC corresponds to one trial/epoch.
    #
    # Result:
    # epo_stcs = list of SourceEstimate objects
    #
    # shape per STC:
    # (vertices x time)

    source_path = os.path.join(
        SOURCE_DATA_DIR,
        group,
        sub,
        task,
        stage
    )

    epo_stcs = [
        mne.read_source_estimate(
            os.path.join(source_path, fname))
        for fname in os.listdir(source_path)
            if fname.endswith("-lh.stc")
    ]


    # --------------------------------------------------------
    # Extract cortical vertices belonging to each ROI label
    # --------------------------------------------------------

    for roi, labels in ROI_labels.items():

        roi_data[roi] = {}

        for label_name, label in labels.items():

            # Extract only vertices inside this anatomical label
            # for every source-space epoch

            stcs_label = [stc.in_label(label) for stc in epo_stcs]

            # Save extracted source activity
            # data shape:
            # epochs x vertices x time

            roi_data[roi][label_name] = {
                "data": np.stack([stc.data for stc in stcs_label]),
                "vertices": stcs_label[0].vertices[0],
                "hemisphere": HEMI[1:],
                "times": stcs_label[0].times,
                "tstep": stcs_label[0].tstep,
                "sfreq": stcs_label[0].sfreq,
            }

    # ========================================================
    # SAVE ROI SOURCE DATA
    # ========================================================
    # Saving extracted ROI data avoids repeating the expensive
    # source-space extraction step when re-running ERPAC with
    # different parameters.

    roi_save_dir = os.path.join(ROI_STCS_DIR, group, task, stage)
    check_paths(roi_save_dir)

    with open(os.path.join(roi_save_dir, f"{sub}_{task}_{stage}_roi_stcs.pkl"), "wb") as f:
        pickle.dump(roi_data, f)

    # ========================================================
    # ERPAC COMPUTATION
    # ========================================================
    # ERPAC is computed independently for:
    #
    # coupling type -> ROI -> anatomical label -> cortical vertex
    #
    # The final output is later averaged to ROI level.

    sf = roi_data["M1"]["G_precentral-lh"]["sfreq"]

    for coupling_name, (phase_freq, amp_freq) in COUPLINGS.items():

        print(f"\nComputing {coupling_name}")

        p = EventRelatedPac(
            f_pha=phase_freq,
            f_amp=amp_freq
        )
        print(f"Phase frequency: {p.xvec}")

        amp_freqs = p.yvec
        erpac_results[coupling_name] = {}

        for roi, labels in roi_data.items():

            erpac_results[coupling_name][roi] = {}

            for label_name, label_data in labels.items():
                # Source data:
                data = label_data["data"] # (epochs, vertices, times)
                n_epochs, n_vertices, n_times = data.shape
                n_freqs = len(p.yvec)

                # Store ERPAC for every cortical vertex
                vertex_erpac = np.zeros((n_vertices, n_freqs, n_times))
                vertex_pvalues = np.zeros(
                    (n_vertices, n_freqs, n_times)
                )

                # ------------------------------------------------
                # Compute ERPAC independently for each vertex
                # ------------------------------------------------

                for v in range(n_vertices):

                    x = data[:, v, :]

                    erpac = p.filterfit(
                        sf,
                        x,
                        method="circular",
                        mcp="bonferroni",
                        n_jobs=-1
                    ).squeeze()

                    vertex_pvalues[v] = p.pvalues.squeeze()

                    vertex_erpac[v] = erpac # (n_amp, n_times)

                erpac_results[coupling_name][roi][label_name] = {
                    "erpac": vertex_erpac,
                    "p-values": vertex_pvalues,
                    "times": label_data["times"],
                    "vertices": label_data["vertices"],
                }


        # ====================================================
        # ROI-LEVEL AVERAGING
        # ====================================================
        # Average hierarchy:
        #
        # vertices
        #       ↓
        # anatomical label
        #       ↓
        # functional ROI
        #
        # Final matrix:
        # amplitude frequency x time

        for roi, labels in erpac_results[coupling_name].items():

            label_means = [
                res["erpac"].mean(axis=0)
                for res in labels.values()
            ]

            roi_mean = np.mean(
                label_means,
                axis=0
            )   # (amp_freq, time)


            first_label = next(iter(labels.values()))
            times = first_label["times"]

            # Store long-format dataframe rows
            for f_idx, amp_freq in enumerate(amp_freqs):

                for t_idx, time in enumerate(times):

                    erpac_results_rows.append({

                        "sub": sub,
                        "group": group,
                        "task": task,
                        "task_stage": stage,
                        "coupling": coupling_name,
                        "roi": roi,
                        "amp_freq": amp_freq,
                        "time": time,
                        "erpac_value": roi_mean[f_idx, t_idx]

                    })

    # ========================================================
    # SAVE PARTICIPANT ERPAC RESULTS
    # ========================================================

    with open(os.path.join(roi_save_dir, f"{sub}_{task}_{stage}_erpac_results.pkl"), "wb") as f:
        pickle.dump(erpac_results, f)

# ============================================================
# CREATE MASTER ERPAC DATAFRAME
# ============================================================

# Convert accumulated ERPAC results into a long-format dataframe.
# Each row corresponds to one:
# subject × task × stage × coupling × ROI × gamma frequency × time point.

erpac_df = pd.DataFrame(erpac_results_rows)

# ============================================================
# CREATE FREQUENCY-AVERAGED ERPAC DATAFRAME
# ============================================================

# Average ERPAC across all gamma amplitude frequencies while
# preserving the subject, experimental condition, ROI and time.
# This dataframe is useful for statistical analysis and plotting
# temporal ERPAC dynamics without the frequency dimension.
erpac_df_mean_freq = (
    erpac_df
    .groupby(
        [
            "sub",
            "group",
            "task",
            "task_stage",
            "coupling",
            "roi",
            "time"
        ],
        as_index=False
    )
    ["erpac_value"]
    .mean()
)


# Save the frequency-averaged ERPAC dataframe as CSV for
# compatibility with Excel and statistical software.
erpac_df_mean_freq.to_csv(
    os.path.join(ERPAC_DIR, "erpac_results_mean_freq.csv"),
    index=False
)

# ============================================================
# OPTIMISE DATA TYPES
# ============================================================

# Convert repeated string columns to categorical variables to
# substantially reduce memory usage while working in Python.
categorical_cols = [
    "sub",
    "group",
    "task",
    "task_stage",
    "coupling",
    "roi"
]

for col in categorical_cols:
    erpac_df[col] = erpac_df[col].astype("category")

# Store numerical values as float32 instead of float64.
# This halves memory usage with negligible loss of precision
# for ERPAC values, frequencies and time stamps.
float_cols = [
    "amp_freq",
    "time",
    "erpac_value"
]

for col in float_cols:
    erpac_df[col] = erpac_df[col].astype("float32")

# ============================================================
# SAVE FULL ERPAC DATASET
# ============================================================

# Save the complete ERPAC dataframe in compressed Parquet format.
# This preserves the full frequency × time resolution while
# minimising storage requirements and enabling fast loading.
erpac_df.to_parquet(
    os.path.join(
        ERPAC_DIR,
        "erpac_results.parquet"
    ),
    engine="pyarrow",
    compression="zstd",
    index=False
)
# Outputs:
#   erpac_results.parquet      -> Full ERPAC dataset (frequency × time)
#   erpac_results_mean_freq.csv -> Frequency-averaged ERPAC dataset
# 


# ============================================================
# GROUP-LEVEL ERPAC VISUALISATION
# ============================================================

for group in GROUPS:
    for task_stage in TASK_STAGES:
        for coupling in COUPLINGS:
            plot_group_erpac(
                erpac_df,
                task_stage=task_stage,
                coupling=coupling,
                group=group,
                save_dir=ERPAC_FIGS_DIR
            )

for task_stage in TASK_STAGES:
    for coupling in COUPLINGS:
        plot_group_erpac_timecourse(
            erpac_df,
            task_stage=task_stage,
            coupling=coupling,
            save_dir=ERPAC_FIGS_DIR,
            mode="peak_freq" 
        )