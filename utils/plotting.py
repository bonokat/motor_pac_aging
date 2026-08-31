from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# GROUP ERPAC HEATMAP
# ============================================================

def plot_group_erpac(
    erpac_df,
    task,
    task_stage,
    coupling,
    task_block=None,
    group=None,
    save_dir=None,
    dpi=300,
    vmin=0.05,
    vmax=0.175
):

    # --------------------------------------------------------
    # Filter task / stage / coupling
    # --------------------------------------------------------

    df = erpac_df[
        (erpac_df["task"] == task) &
        (erpac_df["task_stage"] == task_stage) &
        (erpac_df["coupling"] == coupling)
    ]


    # --------------------------------------------------------
    # DeCRAT additionally requires block
    # --------------------------------------------------------

    if task == "DeCRAT":

        if task_block is None:
            raise ValueError(
                "task_block must be provided for DeCRAT "
                "('baseline' or 'adaptation')."
            )

        df = df[
            df["task_block"] == task_block
        ]


    # --------------------------------------------------------
    # Optional group filter
    # --------------------------------------------------------

    if group is not None:

        df = df[
            df["group"] == group
        ]


    if df.empty:

        print(
            f"No data for: "
            f"task={task}, "
            f"stage={task_stage}, "
            f"block={task_block}, "
            f"group={group}, "
            f"coupling={coupling}"
        )

        return


    rois = sorted(
        df["roi"].unique()
    )


    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    axes = axes.ravel()


    # --------------------------------------------------------
    # Plot each ROI
    # --------------------------------------------------------

    for ax, roi in zip(
        axes,
        rois
    ):

        roi_df = df[
            df["roi"] == roi
        ]


        # Group-average ERPAC
        roi_mean = (
            roi_df
            .groupby(
                ["amp_freq", "time"],
                observed=True
            )["erpac_value"]
            .mean()
            .reset_index()
        )


        # Long → matrix
        erpac_matrix = (
            roi_mean
            .pivot(
                index="amp_freq",
                columns="time",
                values="erpac_value"
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )


        amp_freqs = (
            erpac_matrix.index.to_numpy()
        )

        times = (
            erpac_matrix.columns.to_numpy()
        )


        im = ax.imshow(
            erpac_matrix.values,
            aspect="auto",
            origin="lower",
            extent=[
                times[0],
                times[-1],
                amp_freqs[0],
                amp_freqs[-1]
            ],
            vmin=vmin,
            vmax=vmax
        )


        ax.set_title(roi)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Gamma (Hz)")

        ax.axvline(
            0,
            color="k",
            linestyle="--",
            linewidth=1
        )


    # Remove unused axes
    for ax in axes[len(rois):]:
        ax.remove()


    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    title_parts = []

    title_parts.append(group)

    title_parts.append(task)
    title_parts.append(task_stage.title())

    if task == "DeCRAT":
        title_parts.append(task_block.title())

    title_parts.append(
        coupling.replace("_", " ").title()
    )


    fig.suptitle(
        " | ".join(title_parts)
    )


    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    fig.colorbar(
        im,
        ax=[
            ax
            for ax in axes[:len(rois)]
        ],
        shrink=0.8,
        label="ERPAC",
        ticks=np.arange(
            vmin,
            vmax,
            0.025
        )
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if save_dir is not None:

        save_dir = Path(save_dir)

        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        group_name = (group)

        filename_parts = [
            group_name,
            task,
            task_stage
        ]

        if task == "DeCRAT":
            filename_parts.append(
                task_block
            )

        filename_parts.extend([
            coupling,
            "ERPAC"
        ])


        fname = (
            "_".join(filename_parts)
            + ".png"
        )


        fig.savefig(
            save_dir / fname,
            dpi=dpi,
            bbox_inches="tight"
        )


    plt.show()
    # plt.close()



def plot_group_erpac_timecourse(
    erpac_df,
    task,
    task_stage,
    coupling,
    task_block=None,
    mode="mean",
    save_dir=None,
    dpi=300,
    y_min=0.115,
    y_max=0.15
):

    # --------------------------------------------------------
    # Filter task / stage / coupling
    # --------------------------------------------------------

    df = erpac_df[
        (erpac_df["task"] == task) &
        (erpac_df["task_stage"] == task_stage) &
        (erpac_df["coupling"] == coupling)
    ]


    # --------------------------------------------------------
    # DeCRAT block filtering
    # --------------------------------------------------------

    if task == "DeCRAT":

        if task_block is None:

            raise ValueError(
                "task_block must be provided for DeCRAT "
                "('baseline' or 'adaptation')."
            )

        df = df[
            df["task_block"] == task_block
        ]


    if df.empty:

        print(
            f"No data for: "
            f"task={task}, "
            f"stage={task_stage}, "
            f"block={task_block}, "
            f"coupling={coupling}"
        )

        return


    rois = sorted(
        df["roi"].unique()
    )

    groups = sorted(
        df["group"].unique()
    )


    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    axes = axes.ravel()


    # --------------------------------------------------------
    # Plot each ROI
    # --------------------------------------------------------

    for ax, roi in zip(
        axes,
        rois
    ):

        for group in groups:

            roi_df = df[
                (df["roi"] == roi) &
                (df["group"] == group)
            ]


            if roi_df.empty:
                continue


            # ------------------------------------------------
            # Group-average frequency × time surface
            # ------------------------------------------------

            roi_mean = (
                roi_df
                .groupby(
                    ["amp_freq", "time"],
                    observed=True
                )["erpac_value"]
                .mean()
                .reset_index()
            )


            erpac_matrix = (
                roi_mean
                .pivot(
                    index="amp_freq",
                    columns="time",
                    values="erpac_value"
                )
                .sort_index(axis=0)
                .sort_index(axis=1)
            )


            times = (
                erpac_matrix.columns.to_numpy()
            )

            matrix = (
                erpac_matrix.to_numpy()
            )


            # ------------------------------------------------
            # Collapse frequency dimension
            # ------------------------------------------------

            if mode == "max":

                y = np.nanmax(
                    matrix,
                    axis=0
                )

                ylabel = "Maximum ERPAC"


            elif mode == "mean":

                y = np.nanmean(
                    matrix,
                    axis=0
                )

                ylabel = "Mean ERPAC"


            else:

                raise ValueError(
                    "mode must be 'max' or 'mean'"
                )


            # ------------------------------------------------
            # Plot
            # ------------------------------------------------

            ax.plot(
                times,
                y,
                linewidth=2,
                label=group
            )


        ax.set_title(roi)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)

        ax.axvline(
            0,
            color="k",
            linestyle="--",
            linewidth=1
        )

        ax.set_xlim(
            df["time"].min(),
            df["time"].max()
        )

        ax.set_ylim(
            y_min,
            y_max
        )

        ax.legend()


    # Remove unused axes
    for ax in axes[len(rois):]:
        ax.remove()


    # --------------------------------------------------------
    # Figure title
    # --------------------------------------------------------

    title_parts = [
        task,
        task_stage.title()
    ]

    if task == "DeCRAT":
        title_parts.append(
            task_block.title()
        )

    title_parts.extend([
        coupling.replace("_", " ").title(),
        mode.title()
    ])


    fig.suptitle(
        " | ".join(title_parts)
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if save_dir is not None:

        save_dir = Path(save_dir)

        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        filename_parts = [
            task,
            task_stage
        ]

        if task == "DeCRAT":
            filename_parts.append(
                task_block
            )

        filename_parts.extend([
            coupling,
            mode,
            "timecourse"
        ])


        fname = (
            "_".join(filename_parts)
            + ".png"
        )


        fig.savefig(
            save_dir / fname,
            dpi=dpi,
            bbox_inches="tight"
        )


    plt.show()
    # plt.close()


# OLE VERSION: 1.0
# from pathlib import Path
# import matplotlib.pyplot as plt
# import numpy as np

# # =========================
# # SOURCE ROI PLOTTING CONFIG
# # =========================

# def plot_group_erpac(
#     erpac_df,
#     task_stage,
#     coupling,
#     group=None,
#     save_dir=None,
#     dpi=300
# ):

#     # Filter data
#     df = erpac_df[
#         (erpac_df["task_stage"] == task_stage) &
#         (erpac_df["coupling"] == coupling)
#     ]

#     if group is not None:
#         df = df[df["group"] == group]


#     rois = df["roi"].unique()


#     fig, axes = plt.subplots(
#         2, 2,
#         figsize=(10, 8),
#         sharex=True,
#         sharey=True,
#         constrained_layout=True
#     )

#     axes = axes.ravel()


#     for ax, roi in zip(axes, rois):

#         roi_df = df[df["roi"] == roi]


#         # Average across subjects
#         roi_mean = (
#             roi_df
#             .groupby(
#                 ["amp_freq", "time"]
#             )["erpac_value"]
#             .mean()
#             .reset_index()
#         )


#         # Convert long format → matrix
#         erpac_matrix = (
#             roi_mean
#             .pivot(
#                 index="amp_freq",
#                 columns="time",
#                 values="erpac_value"
#             )
#             .values
#         )


#         amp_freqs = roi_mean["amp_freq"].unique()
#         times = roi_mean["time"].unique()


#         im = ax.imshow(
#             erpac_matrix,
#             aspect="auto",
#             origin="lower",
#             extent=[
#                 times[0],
#                 times[-1],
#                 amp_freqs[0],
#                 amp_freqs[-1]
#             ],
#             vmin=0.05,
#             vmax=0.175
#         )


#         ax.set_title(roi)
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel("Gamma (Hz)")


#     # Remove empty axes
#     for ax in axes[len(rois):]:
#         ax.remove()


#     title = (
#         f"{group + ' - ' if group else ''}"
#         f"{task_stage} - "
#         f"{coupling.replace('_', ' ').title()}"
#     )

#     fig.suptitle(title)

#     fig.colorbar(
#         im,
#         ax=axes.tolist(),
#         shrink=0.8,
#         label="ERPAC",
#         ticks=np.arange(0.05, 0.175, 0.025)
#     )

#     # Save figure
#     if save_dir is not None:

#         save_dir = Path(save_dir)
#         save_dir.mkdir(
#             parents=True,
#             exist_ok=True
#         )

#         group_name = group if group else "all"

#         fname = (
#             f"{group_name}_"
#             f"{task_stage}_"
#             f"{coupling}_"
#             "ERPAC.png"
#         )

#         fig.savefig(
#             save_dir / fname,
#             dpi=dpi,
#             bbox_inches="tight"
#         )

#     plt.show()
#     plt.close()


# def plot_group_erpac_timecourse(
#     erpac_df,
#     task_stage,
#     coupling,
#     mode="mean",
#     save_dir=None,
#     dpi=300,
# ):
#     """
#     Plot ROI ERPAC timecourses for Young and Older groups.

#     Parameters
#     ----------
#     erpac_df : pandas.DataFrame
#         Long-format ERPAC dataframe.

#     task_stage : str
#         Experimental stage (e.g. "plan", "go").

#     coupling : str
#         Coupling name (e.g. "theta_gamma").

#     mode : {"mean", "max"}
#         mean      -> mean ERPAC across gamma frequencies
#         max       -> maximum ERPAC across gamma frequencies

#     save_dir : str or Path | None
#         Directory for saving figure.

#     dpi : int
#         Figure resolution.
#     """

#     # -------------------------------------------------------
#     # Filter dataframe
#     # -------------------------------------------------------

#     df = erpac_df[
#         (erpac_df["task_stage"] == task_stage) &
#         (erpac_df["coupling"] == coupling)
#     ]

#     rois = sorted(df["roi"].unique())
#     groups = sorted(df["group"].unique())

#     y_min = 0.115
#     y_max = 0.15

#     # -------------------------------------------------------
#     # Create figure
#     # -------------------------------------------------------

#     fig, axes = plt.subplots(
#         2,
#         2,
#         figsize=(10, 8),
#         sharex=True,
#         sharey=True,
#         constrained_layout=True
#     )

#     axes = axes.ravel()

#     # -------------------------------------------------------
#     # Plot each ROI
#     # -------------------------------------------------------

#     for ax, roi in zip(axes, rois):

#         for group in groups:

#             roi_df = df[
#                 (df["roi"] == roi) &
#                 (df["group"] == group)
#             ]

#             # ---------------------------------------------
#             # Group-average ERPAC surface
#             # ---------------------------------------------

#             roi_mean = (
#                 roi_df
#                 .groupby(
#                     ["amp_freq", "time"]
#                 )["erpac_value"]
#                 .mean()
#                 .reset_index()
#             )

#             erpac_matrix = (
#                 roi_mean
#                 .pivot(
#                     index="amp_freq",
#                     columns="time",
#                     values="erpac_value"
#                 )
#                 .sort_index()
#             )

#             times = erpac_matrix.columns.values

#             matrix = erpac_matrix.values

#             # ---------------------------------------------
#             # Collapse frequency dimension
#             # ---------------------------------------------

#             if mode == "max":

#                 y = matrix.max(axis=0)
#                 ylabel = "Maximum ERPAC"

#             elif mode == "mean":

#                 y = matrix.mean(axis=0)
#                 ylabel = "Mean ERPAC"

#             else:

#                 raise ValueError(
#                     "mode must be 'max' or 'mean'"
#                 )

#             # ---------------------------------------------
#             # Plot
#             # ---------------------------------------------

#             ax.plot(
#                 times,
#                 y,
#                 linewidth=2,
#                 label=group
#             )

#         ax.set_title(roi)
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel(ylabel)

#         ax.axvline(
#             0,
#             color="k",
#             linestyle="--",
#             linewidth=1
#         )

#         ax.set_xlim(times[0], times[-1])
#         ax.set_ylim(y_min, y_max)

#         ax.legend()

#     # Remove unused axes

#     for ax in axes[len(rois):]:
#         ax.remove()

#     fig.suptitle(
#         f"{task_stage} | "
#         f"{coupling.replace('_', ' ').title()} | "
#         f"{mode.replace('_', ' ').title()}"
#     )

#     # -------------------------------------------------------
#     # Save
#     # -------------------------------------------------------

#     if save_dir is not None:

#         save_dir = Path(save_dir)
#         save_dir.mkdir(
#             parents=True,
#             exist_ok=True
#         )

#         fname = (
#             f"{task_stage}_"
#             f"{coupling}_"
#             f"{mode}_"
#             "timecourse.png"
#         )

#         fig.savefig(
#             save_dir / fname,
#             dpi=dpi,
#             bbox_inches="tight"
#         )

#     plt.show()
#     plt.close()