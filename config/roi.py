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