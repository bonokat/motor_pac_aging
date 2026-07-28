from pathlib import Path
import os

# DATA TRANSFER FROM S1
S1_EEG_DATA_DIR = Path("D:\\BonoKat\\research project\\# study 1\\eeg_data\\set")

# FREESURFER FILES LOCATION
FS_FOLDER = Path("D:\\BonoKat\\research project\\# PD project\\fs")

# S2 PROJECT PATHS
S2_DIR = Path("F:\\# study 2")
S2_EEG_DIR = os.path.join(S2_DIR, "eeg_data")
S2_EPOCHS_DIR = os.path.join(S2_EEG_DIR, "epochs")
SOURCE_DATA_DIR = os.path.join(S2_EPOCHS_DIR, "source")
ERPAC_DIR = os.path.join(S2_EEG_DIR, "erpac")
ROI_STCS_DIR = os.path.join(ERPAC_DIR, "roi_stcs")

FIGS_DIR = Path("F:\\# study 2\\eeg_data\\figs")
ERPAC_FIGS_DIR = os.path.join(FIGS_DIR, "erpac")