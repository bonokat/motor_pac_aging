import mne
from mne.viz import Brain
import os
import numpy as np
import pickle

import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT))
from config import check_paths
from config.config import (HEMI, ROI, FS_SUB, FS_SRC_PATH, GAMMA, COUPLINGS,
                           SUBJECTS, GROUPS, TASKS, TASK_STAGES)
from config.paths import FS_FOLDER, SOURCE_DATA_DIR, ROI_STCS_DIR
from utils.helpers import iterate_dataset

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorpac import EventRelatedPac

# %matplotlib qt

alpha_threshold = 0.05 # for erpac surrogates

labels = mne.read_labels_from_annot(
    subject=FS_SUB,
    parc="aparc.a2009s", # "aparc.a2009s", "Yeo2011_7Networks_N1000", "Yeo2011_17Networks_N1000"
    hemi=HEMI[1:],
    subjects_dir=FS_FOLDER
)

label_dict = {label.name: label for label in labels}

ROI_labels = {
    roi: {
        label_name: label_dict[label_name]
        for label_name in label_names
    }
    for roi, label_names in ROI.items()
}

for label in labels:
    print(label.name)

src = mne.read_source_spaces(FS_SRC_PATH)


for item in iterate_dataset(
    GROUPS,
    SUBJECTS,
    TASKS,
    TASK_STAGES
):

    group = item["group"]
    sub = item["subject"]
    task = item["task"]
    stage = item["stage"]

    roi_data = {}
    erpac_results = {}

    # !!! STOPPED HERE: ADD DATA ACCUMULATION AND AVERAGING FOR GROUP DATASET !!!
    erpac_groups = {}

    # ROI DATA EXTRACTION
    source_path = os.path.join(SOURCE_DATA_DIR, group, sub, task, stage)
    epo_stcs = [mne.read_source_estimate(os.path.join(source_path, fname_stc)) for fname_stc in os.listdir(source_path) if fname_stc.endswith('-lh.stc')]

    for roi, labels in ROI_labels.items():

        roi_data[roi] = {}

        for label_name, label in labels.items():

            stcs_label = [stc.in_label(label) for stc in epo_stcs]

            roi_data[roi][label_name] = {
                "data": np.stack([stc.data for stc in stcs_label]),
                "vertices": stcs_label[0].vertices[0],
                "hemisphere": HEMI[1:],
                "times": stcs_label[0].times,
                "tstep": stcs_label[0].tstep,
                "sfreq": stcs_label[0].sfreq,
            }
    
    roi_save_dir = os.path.join(ROI_STCS_DIR, group, task, stage)
    check_paths(roi_save_dir)

    with open(os.path.join(roi_save_dir, f"{sub}_{task}_{stage}_roi_stcs.pkl"), "wb") as f:
        pickle.dump(roi_data, f)
    

    # ROI ERPAC COMPUTATION
    sf = roi_data["M1"]["G_precentral-lh"]["sfreq"] # extract from any ROI and label, since they all have the same sampling frequency

    for coupling_name, (phase_freq, amp_freq) in COUPLINGS.items():

        print(f"\nComputing {coupling_name}")

        p = EventRelatedPac(
            f_pha=phase_freq,
            f_amp=amp_freq
        )
        print(f"Phase freq is {p.xvec}")

        erpac_results[coupling_name] = {}

        for roi, labels in roi_data.items():

            erpac_results[coupling_name][roi] = {}

            for label_name, label_data in labels.items():

                data = label_data["data"]      # (epochs, vertices, times)

                n_epochs, n_vertices, n_times = data.shape
                n_freqs = len(p.yvec)

                vertex_erpac = np.zeros((n_vertices, n_freqs, n_times))
                vertex_sig = np.zeros((n_vertices, n_times))

                for v in range(n_vertices):

                    x = data[:, v, :]

                    erpac = p.filterfit(
                        sf,
                        x,
                        method="circular",
                        mcp="bonferroni",
                        n_jobs=-1
                    ).squeeze()

                    pvalues = p.pvalues.squeeze()

                    vertex_erpac[v] = erpac # (n_amp, n_times)

                erpac_results[coupling_name][roi][label_name] = {
                    "erpac": vertex_erpac,
                    "p-values": pvalues,
                    "times": label_data["times"],
                    "vertices": label_data["vertices"],
                }
            
    with open(os.path.join(roi_save_dir, f"{sub}_{task}_{stage}_erpac_results.pkl"), "wb") as f:
        pickle.dump(erpac_results, f)