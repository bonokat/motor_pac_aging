"""
ERPAC figure: coupling type (rows) × age group (columns).

Inputs
------
1. S2_ERPAC_LMM_FINAL_UPD_emmeans_for_plotting.csv
2. S2_ERPAC_LMM_FINAL_UPD_posthocs.csv
3. S2_ERPAC_LMM_FINAL_UPD_posthocs_group_contrasts.csv

Plot layers
-----------
- Pale participant-level observations
- Violin distributions of participant ERPAC values
- Estimated marginal means (EMMs) with 95% confidence intervals
- Optional significance brackets from the contrasts CSV

IMPORTANT ABOUT SIGNIFICANCE
----------------------------
Contrasts with group = NA are pooled across age groups. By default, pooled
contrasts are NOT repeated in the younger and older panels, because that could
misrepresent them as group-specific results. Set SHOW_POOLED_CONDITION_TESTS
to True only if you intentionally want pooled condition comparisons displayed
in both age-group panels (and label that choice in the figure caption).

The script supports common contrast_type names, including:
- condition_within_stage: condition pairs, separately by task stage when
  stage is provided; if group is NA, the contrast is pooled across age groups.
- stage_within_condition / task_stage_within_condition: plan-vs-go contrasts
  within condition, separately by group when group is provided.

If your contrasts CSV uses different contrast_type labels, inspect the printed
summary and update the matching sets below.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------
# 1. FILE PATHS — EDIT THESE
# ---------------------------------------------------------------------
BASE_DIR = Path(r"F:\# study 2\eeg_data\erpac\PCA\erpac_roi_tc\LMM")

EMMEANS_FILE = BASE_DIR / "S2_ERPAC_LMM_FINAL_UPD_emmeans_for_plotting.csv"
PARTICIPANTS_FILE = BASE_DIR / "S2_ERPAC_LMM_FINAL_UPD_posthocs.csv"
CONTRASTS_FILE = BASE_DIR / "S2_ERPAC_LMM_FINAL_UPD_posthocs_group_contrasts.csv"

OUTPUT_DIR = BASE_DIR / "figures_erpac_emmeans"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / "ERPAC_EMMs_by_coupling_age_condition_stage"
OUTPUT_PDF = OUTPUT_DIR / "ERPAC_EMMs_by_coupling_age_condition_stage"

# Figure output resolution
DPI = 300

# Significance threshold; all significant results use a single star.
ALPHA = 0.05

# Do not present pooled contrasts as though they were age-group-specific.
SHOW_POOLED_CONDITION_TESTS = True

# Prevent a large number of significance brackets from making each panel
# unreadable. None means no cap. When capped, the smallest p-values are kept.
MAX_BRACKETS_PER_PANEL = 100

# ---------------------------------------------------------------------
# 2. DISPLAY SETTINGS
# ---------------------------------------------------------------------
COUPLING_ORDER = ["theta_gamma", "alpha_gamma", "beta_gamma"]
COUPLING_LABELS = {
    "theta_gamma": "Theta–gamma",
    "alpha_gamma": "Alpha–gamma",
    "beta_gamma": "Beta–gamma",
}

GROUP_ORDER = ["Y", "O"]
GROUP_LABELS = {"Y": "Younger", "O": "Older"}

STAGE_ORDER = ["plan", "go"]
STAGE_LABELS = {"plan": "Planning", "go": "Execution"}

CONDITION_ORDER = [
    "FTT",
    "DeCRAT_baseline",
    "DeCRAT_adaptation_early",
    "DeCRAT_adaptation_late",
]
CONDITION_LABELS = {
    "FTT": "FTT",
    "DeCRAT_baseline": "De-CRAT\nbaseline",
    "DeCRAT_adaptation_early": "Early\nadaptation",
    "DeCRAT_adaptation_late": "Late\nadaptation",
}

# Colour encodes task stage, not age group.
STAGE_PALETTE = {
    "plan": "#356F75",  # Deep teal — planning
    "go": "#C47F83",    # Dusty rose — execution
    }

POINT_PALETTE = {
    "plan": "#A9CDCA",  # Pale teal — individual points
    "go": "#E9C1C0",    # Pale rose — individual points
    }

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)

Y_LIM = (0.13, 0.295)


# ---------------------------------------------------------------------
# 3. LOAD AND VALIDATE DATA
# ---------------------------------------------------------------------
for path in [EMMEANS_FILE, PARTICIPANTS_FILE, CONTRASTS_FILE]:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find: {path}\n"
            "Please edit BASE_DIR or the individual file paths in section 1."
        )

emm = pd.read_csv(EMMEANS_FILE)
participants = pd.read_csv(PARTICIPANTS_FILE)
contrasts = pd.read_csv(CONTRASTS_FILE)

required_emm = {
    "coupling", "group", "task_stage", "condition",
    "emmean", "lower_ci", "upper_ci",
}
required_participants = {
    "sub", "group", "task_stage", "condition", "coupling", "mean_erpac",
}
required_contrasts = {
    "coupling", "contrast_type", "group", "task_stage",
    "condition", "contrast", "p.value",
}

missing = required_emm - set(emm.columns)
if missing:
    raise ValueError(f"Missing columns in EMMs CSV: {sorted(missing)}")
missing = required_participants - set(participants.columns)
if missing:
    raise ValueError(f"Missing columns in participant CSV: {sorted(missing)}")
missing = required_contrasts - set(contrasts.columns)
if missing:
    raise ValueError(f"Missing columns in contrasts CSV: {sorted(missing)}")


def clean_string_column(series):
    """Convert CSV text values to clean strings while preserving missing values."""
    out = series.astype("string").str.strip()
    return out.replace({"": pd.NA, "NA": pd.NA, "nan": pd.NA, "None": pd.NA})


for frame in [emm, participants, contrasts]:
    for col in ["coupling", "group", "task_stage", "condition"]:
        if col in frame.columns:
            frame[col] = clean_string_column(frame[col])

for frame in [emm, participants]:
    for col in ["emmean", "lower_ci", "upper_ci", "mean_erpac"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

contrasts["p.value"] = pd.to_numeric(contrasts["p.value"], errors="coerce")
if "estimate" in contrasts.columns:
    contrasts["estimate"] = pd.to_numeric(contrasts["estimate"], errors="coerce")

# Retain expected levels and make sure the plotting order is stable.
emm = emm[
    emm["coupling"].isin(COUPLING_ORDER)
    & emm["group"].isin(GROUP_ORDER)
    & emm["task_stage"].isin(STAGE_ORDER)
    & emm["condition"].isin(CONDITION_ORDER)
].copy()

participants = participants[
    participants["coupling"].isin(COUPLING_ORDER)
    & participants["group"].isin(GROUP_ORDER)
    & participants["task_stage"].isin(STAGE_ORDER)
    & participants["condition"].isin(CONDITION_ORDER)
].copy()

if emm.empty:
    raise ValueError("No EMM rows remain after filtering. Check CSV labels.")
if participants.empty:
    raise ValueError("No participant rows remain after filtering. Check CSV labels.")

# A participant-level violin should have at most one observation per subject
# in each group × stage × condition × coupling cell. Average duplicates if any.
participants = (
    participants.groupby(
        ["sub", "group", "task_stage", "condition", "coupling"],
        as_index=False,
        observed=True,
    )["mean_erpac"]
    .mean()
)

print("\nParticipant counts per plotted cell:")
print(
    participants.groupby(
        ["coupling", "group", "task_stage", "condition"],
        observed=True,
    )["sub"].nunique().to_string()
)

print("\nContrast types in the supplied CSV:")
print(contrasts["contrast_type"].value_counts(dropna=False).to_string())
print("\nGroup labels in the contrasts CSV:")
print(contrasts["group"].value_counts(dropna=False).to_string())


# ---------------------------------------------------------------------
# 4. SIGNIFICANCE HELPERS
# ---------------------------------------------------------------------
def norm_label(value):
    """Normalize a condition/stage label for matching contrast strings."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    aliases = {
        "De-CRAT baseline": "DeCRAT_baseline",
        "DeCRAT baseline": "DeCRAT_baseline",
        "De-CRAT adaptation early": "DeCRAT_adaptation_early",
        "DeCRAT adaptation early": "DeCRAT_adaptation_early",
        "De-CRAT adaptation late": "DeCRAT_adaptation_late",
        "DeCRAT adaptation late": "DeCRAT_adaptation_late",
        "planning": "plan",
        "execution": "go",
        "Planning": "plan",
        "Execution": "go",
    }
    return aliases.get(s, s)


def split_contrast(contrast_text):
    """
    Split a pairwise contrast such as 'FTT - DeCRAT_baseline' or
    'plan - go'. Returns (left, right), or (None, None).
    """
    if pd.isna(contrast_text):
        return None, None
    parts = re.split(r"\s+-\s+", str(contrast_text).strip(), maxsplit=1)
    if len(parts) != 2:
        return None, None
    return norm_label(parts[0]), norm_label(parts[1])


CONDITION_CONTRAST_TYPES = {
    "condition_within_stage",
    "condition_within_task_stage",
    "condition",
}
STAGE_CONTRAST_TYPES = {
    "stage_within_condition",
    "task_stage_within_condition",
    "task_stage",
    "stage",
}


def normalized_type(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def contrast_rows_for_panel(coupling, group):
    """Select contrast rows relevant to a coupling × age-group panel."""
    rows = contrasts[contrasts["coupling"] == coupling].copy()
    if rows.empty:
        return rows

    rows["_type"] = rows["contrast_type"].map(normalized_type)

    # Keep contrasts explicitly estimated for this age group.
    group_rows = rows[rows["group"] == group].copy()

    # Pooled condition contrasts are optional and never silently treated as
    # age-group-specific tests.
    pooled_condition_rows = rows[
        rows["group"].isna()
        & rows["_type"].isin(CONDITION_CONTRAST_TYPES)
    ].copy()

    if SHOW_POOLED_CONDITION_TESTS:
        group_rows = pd.concat(
            [group_rows, pooled_condition_rows], ignore_index=True
        )

    return group_rows


def get_condition_position(condition, stage):
    """Dodge positions: planning on the left, execution on the right."""
    x = CONDITION_ORDER.index(condition)
    offset = -0.18 if stage == "plan" else 0.18
    return x + offset


def add_bracket(ax, x1, x2, y, height, stage=None, label="*"):
    """Draw a bracket and its significance label."""
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + height, y + height, y],
        color=STAGE_PALETTE[stage],
        linewidth=0.8,
        clip_on=False,
        zorder=8,
    )
    ax.text(
        (x1 + x2) / 2,
        y + height,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
        color=STAGE_PALETTE[stage],
        zorder=9,
    )


def add_significance(ax, coupling, group, panel_data, panel_emm):
    """
    Add supported significant pairwise contrasts.

    Condition contrasts:
      - require a stage in the contrast row;
      - are drawn between condition positions within that stage.
      - pooled contrasts (group NA) are skipped by default.

    Stage contrasts:
      - require a condition in the contrast row;
      - are drawn between planning and execution within that condition.

    Contrast strings must be pairwise and use labels corresponding to the
    plotted condition or stage names.
    """
    rows = contrast_rows_for_panel(coupling, group)
    if rows.empty:
        return

    rows = rows[rows["p.value"].notna() & (rows["p.value"] < ALPHA)].copy()
    if rows.empty:
        return

    rows["_type"] = rows["contrast_type"].map(normalized_type)
    rows["_pair"] = rows["contrast"].apply(split_contrast)
    rows["_left"] = rows["_pair"].apply(lambda x: x[0])
    rows["_right"] = rows["_pair"].apply(lambda x: x[1])

    # Avoid drawing duplicate pooled comparisons twice if the CSV includes
    # them alongside group-specific rows.
    rows = rows.drop_duplicates(
        subset=["_type", "task_stage", "condition", "contrast", "group"]
    )

    # Keep the strongest evidence first if a bracket cap is requested.
    rows = rows.sort_values("p.value", ascending=True)
    if MAX_BRACKETS_PER_PANEL is not None:
        rows = rows.head(MAX_BRACKETS_PER_PANEL)

    if panel_data.empty:
        return

    data_max = panel_data["mean_erpac"].max()
    emm_max = panel_emm["upper_ci"].max() if not panel_emm.empty else data_max
    y_top = np.nanmax([data_max, emm_max])
    y_bottom = panel_data["mean_erpac"].min()
    y_range = max(y_top - y_bottom, abs(y_top) * 0.08, 0.01)
    base_y = y_top + 0.06 * y_range
    bracket_step = 0.095 * y_range
    bracket_height = 0.025 * y_range
    drawn = 0

    for _, row in rows.iterrows():
        ctype = row["_type"]
        left, right = row["_left"], row["_right"]
        stage = norm_label(row.get("task_stage"))
        condition = norm_label(row.get("condition"))

        if ctype in CONDITION_CONTRAST_TYPES:
            # The contrast should be a pair of plotted conditions.
            if left not in CONDITION_ORDER or right not in CONDITION_ORDER:
                continue
            if stage not in STAGE_ORDER:
                # A condition contrast without a specific stage cannot be
                # positioned on a stage-dodged plot without ambiguity.
                continue
            x1 = get_condition_position(left, stage)
            x2 = get_condition_position(right, stage)

        elif ctype in STAGE_CONTRAST_TYPES:
            if condition not in CONDITION_ORDER:
                continue
            # Expect e.g. "plan - go" or "Planning - Execution".
            stage_left, stage_right = left, right
            stage_aliases = {"planning": "plan", "execution": "go"}
            stage_left = stage_aliases.get(str(stage_left).lower(), stage_left)
            stage_right = stage_aliases.get(str(stage_right).lower(), stage_right)
            if {stage_left, stage_right} != {"plan", "go"}:
                continue
            x1 = get_condition_position(condition, "plan")
            x2 = get_condition_position(condition, "go")
        else:
            # Unknown contrast type; do not guess what it means.
            continue

        add_bracket(
            ax,
            x1,
            x2,
            base_y + drawn * bracket_step,
            bracket_height,
            stage=stage,
            label="*",
        )
        drawn += 1


# ---------------------------------------------------------------------
# 5. PLOT
# ---------------------------------------------------------------------
for coupling in COUPLING_ORDER:
    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 9),
        sharey=True
    )

    rng = np.random.default_rng(20261013)
    dodge_offset = {"plan": -0.18, "go": 0.18}
    violin_width = 0.32

    for c, group in enumerate(GROUP_ORDER):
        ax = axes[c]
        pdat = participants[
            (participants["coupling"] == coupling)
            & (participants["group"] == group)
        ].copy()
        edat = emm[
            (emm["coupling"] == coupling)
            & (emm["group"] == group)
        ].copy()

        # Draw violins and pale individual participant observations, separately
        # for each movement stage so their positions match the EMM markers.
        for condition_idx, condition in enumerate(CONDITION_ORDER):
            for stage in STAGE_ORDER:
                cell = pdat[
                    (pdat["condition"] == condition)
                    & (pdat["task_stage"] == stage)
                ]["mean_erpac"].dropna().to_numpy()

                if len(cell) == 0:
                    continue

                center = condition_idx + dodge_offset[stage]

                # Matplotlib violinplot keeps control over the exact dodge
                # position and works without relying on seaborn API versions.
                violin = ax.violinplot(
                    [cell],
                    positions=[center],
                    widths=violin_width,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                    points=100,
                    bw_method="scott",
                )
                for body in violin["bodies"]:
                    body.set_facecolor(STAGE_PALETTE[stage])
                    body.set_edgecolor(STAGE_PALETTE[stage])
                    body.set_alpha(0.22)
                    body.set_linewidth(0.8)
                    body.set_zorder(1)

                # Pale jittered participant-level values.
                jitter = rng.uniform(-0.055, 0.055, size=len(cell))
                ax.scatter(
                    np.full(len(cell), center) + jitter,
                    cell,
                    s=15,
                    color=POINT_PALETTE[stage],
                    edgecolors="white",
                    linewidths=0.35,
                    alpha=0.9,
                    zorder=3,
                )

        # Overlay EMMs and 95% confidence intervals.
        for stage in STAGE_ORDER:
            stage_emm = edat[edat["task_stage"] == stage].copy()
            if stage_emm.empty:
                continue

            xs = [
                CONDITION_ORDER.index(cond) + dodge_offset[stage]
                for cond in stage_emm["condition"]
            ]
            y = stage_emm["emmean"].to_numpy(dtype=float)
            lower = stage_emm["lower_ci"].to_numpy(dtype=float)
            upper = stage_emm["upper_ci"].to_numpy(dtype=float)
            yerr = np.vstack([y - lower, upper - y])

            ax.errorbar(
                xs,
                y,
                yerr=yerr,
                fmt="D",
                markersize=5.8,
                markerfacecolor=STAGE_PALETTE[stage],
                markeredgecolor="black",
                markeredgewidth=0.7,
                ecolor=STAGE_PALETTE[stage],
                elinewidth=1.25,
                capsize=3,
                zorder=6,
                label=STAGE_LABELS[stage] if (c == 0) else None,
            )

        # Significance brackets from the post-hoc contrast table.
        add_significance(ax, coupling, group, pdat, edat)

        ax.set_title(GROUP_LABELS[group], fontsize=12, fontweight="bold", pad=9)
        ax.set_xlim(-0.55, len(CONDITION_ORDER) - 0.45)
        ax.set_xticks(range(len(CONDITION_ORDER)))
        ax.set_xticklabels(
            [CONDITION_LABELS[x] for x in CONDITION_ORDER],
            fontsize=9,
        )
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if c == 0:
            ax.set_ylabel(
                f"{COUPLING_LABELS[coupling]}\nEstimated marginal mean ERPAC",
                fontsize=10,
                labelpad=10,
            )
        ax.set_xlabel("Condition", fontsize=10)

    # Single shared legend.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Movement stage",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=2,
            frameon=False,
        )

    fig.suptitle(
        "ERPAC by coupling type, age group, condition and movement stage",
        fontsize=14,
        fontweight="bold",
        y=0.998,
    )

    fig.text(
        0.5,
        0.012,
        "Diamonds = estimated marginal means; error bars = 95% confidence intervals; "
        "p < .05 shown as *.",
        ha="center",
        fontsize=9,
    )

    # Allow headroom for significance brackets.
    ax.set_ylim(*Y_LIM)

    plt.tight_layout(rect=[0.04, 0.04, 1, 0.96])
    fig.savefig(f'{OUTPUT_PNG}_{coupling}.png', dpi=DPI, bbox_inches="tight")
    fig.savefig(f'{OUTPUT_PNG}_{coupling}.pdf', bbox_inches="tight")
    plt.show()

    print(f"\nSaved PNG: {f'{OUTPUT_PNG}_{coupling}.png'}")
    print(f"Saved PDF: {f'{OUTPUT_PNG}_{coupling}.pdf'}")

    if not SHOW_POOLED_CONDITION_TESTS:
        pooled_sig = contrasts[
            contrasts["group"].isna()
            & contrasts["p.value"].notna()
            & (contrasts["p.value"] < ALPHA)
        ]
        if not pooled_sig.empty:
            print(
                "\nNOTE: Significant pooled contrasts (group = NA) were not drawn "
                "in age-specific panels. Review these separately or set "
                "SHOW_POOLED_CONDITION_TESTS = True if you explicitly want them "
                "shown in both panels with an appropriate figure-caption caveat."
            )
