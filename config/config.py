from config.paths import SOURCE_DATA_DIR, FS_FOLDER
import os

# =========================
# EXPERIMENT STRUCTURE
# =========================

GROUPS = ["Y", "O"]   # younger, older
SUBJECTS = {"Y": os.listdir(os.path.join(SOURCE_DATA_DIR, "Y")), "O": os.listdir(os.path.join(SOURCE_DATA_DIR, "O"))}
TASKS = ["FTT"]
TASK_STAGES = ["plan", "go"]

TASK_MAP = {"_BL": "FTT"}  # rename task for file transfer

# =========================
# ERPAC ANALYSIS
# =========================
GAMMA = (30, 80, 5, 1)

COUPLINGS = {
    "theta_gamma": ([4, 8], GAMMA),
    "alpha_gamma": ([8, 12], GAMMA),
    "beta_gamma": ([13, 30], GAMMA),
}

# =========================
# ROI DEFINITIONS
# =========================

HEMI = "-lh"  # or "rh" for right hemisphere
ROI = {
    "M1": [
        f"G_precentral{HEMI}",
        # f"S_central{HEMI}", # divides the precentral (motor) and postcentral (sensory) gyrus
    ],
    "S1": [
        f"G_postcentral{HEMI}",
        f"S_postcentral{HEMI}",
        ],
    "PMC": [
        f"S_precentral-sup-part{HEMI}",
        f"S_precentral-inf-part{HEMI}",
        # f"G_front_middle{HEMI}",
        f"G_front_inf-Opercular{HEMI}",
        ],
    "SMA": [
        f"G_and_S_paracentral{HEMI}",
        f"G_front_sup{HEMI}", # extends a bit too far front: only posterior parts belong to SMA and PMC
        ],
}

# =========================
# FREESURFER SUB PATH
# =========================

FS_SUB = "fsaverage_bem"
FS_SRC_PATH = os.path.join(FS_FOLDER, FS_SUB, "bem", f"{FS_SUB}-ico4-src.fif")