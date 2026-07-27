from pathlib import Path
import os

# DATA TRANSFER FROM S1
S1_EEG_DATA_DIR = Path("D:\\BonoKat\\research project\\# study 1\\eeg_data\\set")

# FREESURFER FILES LOCATION
FS_FOLDER = "D:\\BonoKat\\research project\\# PD project\\fs"

# S2 PROJECT PATHS
S2_DIR = Path("F:\\# study 2")
S2_EEG_DIR = os.path.join(S2_DIR, "eeg_data")
S2_EPOCHS_DIR = os.path.join(S2_EEG_DIR, "epochs")
SOURCE_DATA_DIR = os.path.join(S2_EPOCHS_DIR, "source")
ROI_STCS_DIR = os.path.join(S2_EPOCHS_DIR, "roi_stcs")

