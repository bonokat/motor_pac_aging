# =========================
# EXPERIMENT STRUCTURE
# =========================

GROUPS = ["Y", "O"]   # younger, older
CONDITIONS = ["plan", "go"]
TASK = "_BL"
TASK_MAP = {"_BL": "FTT"}  # rename task for file transfer

# =========================
# TIME WINDOWS (seconds)
# =========================

# TIME_WINDOWS = {
#     "planning": (-1.0, 0.0),     # adjust to your prep cue
#     "execution": (-0.5, 0.5)     # around key press
# }

# =========================
# FREQUENCY BANDS (Hz)
# =========================

FREQ_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (13, 30),
    "gamma": (30, 80)
}

# =========================
# PAC COMBINATIONS
# =========================

PAC_PAIRS = [
    ("theta", "gamma"),
    ("alpha", "gamma"),
    ("beta", "gamma"),
]

# =========================
# STATS
# =========================

# N_PERMUTATIONS = 5000
# ALPHA = 0.05
# CLUSTER_ALPHA = 0.05
# TAIL = 0

# =========================
# PLOTTING
# =========================

# FIGSIZE = (10, 6)
# DPI = 300
# CMAP = "RdBu_r"
