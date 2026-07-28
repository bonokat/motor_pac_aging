from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# =========================
# SOURCE ROI PLOTTING CONFIG
# =========================

def plot_group_erpac(
    erpac_df,
    task_stage,
    coupling,
    group=None,
    save_dir=None,
    dpi=300
):

    # Filter data
    df = erpac_df[
        (erpac_df["task_stage"] == task_stage) &
        (erpac_df["coupling"] == coupling)
    ]

    if group is not None:
        df = df[df["group"] == group]


    rois = df["roi"].unique()


    fig, axes = plt.subplots(
        2, 2,
        figsize=(10, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    axes = axes.ravel()


    for ax, roi in zip(axes, rois):

        roi_df = df[df["roi"] == roi]


        # Average across subjects
        roi_mean = (
            roi_df
            .groupby(
                ["amp_freq", "time"]
            )["erpac_value"]
            .mean()
            .reset_index()
        )


        # Convert long format → matrix
        erpac_matrix = (
            roi_mean
            .pivot(
                index="amp_freq",
                columns="time",
                values="erpac_value"
            )
            .values
        )


        amp_freqs = roi_mean["amp_freq"].unique()
        times = roi_mean["time"].unique()


        im = ax.imshow(
            erpac_matrix,
            aspect="auto",
            origin="lower",
            extent=[
                times[0],
                times[-1],
                amp_freqs[0],
                amp_freqs[-1]
            ],
            vmin=0,
            vmax=0.2
        )


        ax.set_title(roi)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Gamma (Hz)")


    # Remove empty axes
    for ax in axes[len(rois):]:
        ax.remove()


    title = (
        f"{group + ' - ' if group else ''}"
        f"{task_stage} - "
        f"{coupling.replace('_', ' ').title()}"
    )

    fig.suptitle(title)

    fig.colorbar(
        im,
        ax=axes.tolist(),
        shrink=0.8,
        label="ERPAC",
        ticks=np.arange(0, 0.201, 0.025)
    )

    # Save figure
    if save_dir is not None:

        save_dir = Path(save_dir)
        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        group_name = group if group else "all"

        fname = (
            f"{group_name}_"
            f"{task_stage}_"
            f"{coupling}_"
            "ERPAC.png"
        )

        fig.savefig(
            save_dir / fname,
            dpi=dpi,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()