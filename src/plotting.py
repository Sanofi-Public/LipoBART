"""Plotting functions to visualize the results of our experiments."""
import os
import pickle
import re

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


def get_number_of_tail(lipid_string, lipid_name):
    c_string_list = re.findall(r"CCCC+", lipid_string)
    if "25A" in lipid_name:
        return len(c_string_list) - 1

    return len(c_string_list)


def get_count_charged_ion(lipid_string):
    charged_ion = re.findall(r"O\-+", lipid_string)

    return len(charged_ion)


def plot_family_distribution(data_file, ax):
    """Plot distribution of families on a subplot."""
    data = pd.read_csv(data_file)

    data.loc[:, "n_tail"] = data.apply(
        lambda row: get_number_of_tail(row["m1"], row["name"]), axis=1
    )
    data.loc[:, "n_zwitterion"] = data["m1"].apply(get_count_charged_ion)

    aggregated_data = (
        data.groupby(["family", "n_tail", "n_zwitterion"]).size().rename("count").reset_index()
    )
    count_df = data.groupby(["family", "y1"]).size().unstack()
    percentage_df = count_df.div(count_df.sum(axis=1), axis=0) * 100
    percentage_df_pd = percentage_df.reset_index()

    # Set styles
    plt.style.use("seaborn-v0_8")
    sns.set(palette="colorblind")
    matplotlib.rc("font", size=10)

    labels = ["0", "1", "2", "3", "4", "5", "6"]
    a = aggregated_data["n_zwitterion"].to_list()
    b = aggregated_data["n_tail"].to_list()
    c = aggregated_data["count"].to_list()
    bar_width = 0.20
    df = [c, a, b]

    colors = ["#a5c4b1", "#a5c4b1", "#a5c4b1"]
    color_bar = ["#fff9c9", "#efda6d", "#b64a47", "#754242"]
    columns = (
        "Family 0",
        "Family 1",
        "Family 2",
        "Family 3",
        "Family 4",
        "Family 5",
        "Family 6",
    )
    index = np.arange(len(labels))

    # --- plot barplot --- #
    value0 = percentage_df_pd[0].to_list()
    value1 = percentage_df_pd[1].to_list()
    value2 = percentage_df_pd[2].to_list()
    value3 = percentage_df_pd[3].to_list()

    ax.bar(labels, value0, color=color_bar[0], label="<1,000")
    ax.bar(labels, value1, color=color_bar[1], label="1,000-10,000", bottom=value0)
    ax.bar(
        labels,
        value2,
        color=color_bar[2],
        label="10,000-100,000",
        bottom=[value0[i] + value1[i] for i in range(len(value0))],
    )

    bottom = [value0[i] + value1[i] + value2[i] for i in range(len(value0))]
    bottom[0] = 100.0
    print(value3)
    print(bottom)
    ax.bar(labels, value3, color=color_bar[3], label=">100,000", bottom=bottom)
    # ax.bar(labels, value3, color=color_bar[3], label= '>100,000', bottom=[value0[i] + value1[i] + value2[i] for i in range(len(value0))])

    legend = ax.legend(
        loc="best",
        bbox_to_anchor=(-1.1, 0.9, 1, 0.2),
        title="RLU activity",
    )

    ax.table(
        cellText=df,
        rowLabels=[
            " Total Number of lipids ",
            " Number of tails ",
            " Number of zwitterions",
        ],
        rowColours=colors,
        colLabels=columns,
        loc="bottom",
        bbox=[0, -0.4, 1, 0.2],
    )

    ax.set_ylabel("%")
    # add x axis label as family
    ax.set_xlabel("Family")
    # ax.set_xticks()


def plot_tournament(data_file, pickle_file, save_path):
    """Plot the results of the tournament also plot distribution of families on the top Pickle file
    contains metrics of scores for different embeddings functions on 7 different folds for each of
    7 families of lipids.

    Extract pickle into a dictionary create a color map with family on y-axis, embedding method on
    x-axis and score as the color
    """
    data = pd.read_csv(data_file)

    data.loc[:, "n_tail"] = data.apply(
        lambda row: get_number_of_tail(row["m1"], row["name"]), axis=1
    )
    data.loc[:, "n_zwitterion"] = data["m1"].apply(get_count_charged_ion)

    aggregated_data = (
        data.groupby(["family", "n_tail", "n_zwitterion"]).size().rename("count").reset_index()
    )
    count_df = data.groupby(["family", "y1"]).size().unstack()
    percentage_df = count_df.div(count_df.sum(axis=1), axis=0) * 100

    # Set styles
    plt.style.use("seaborn-v0_8")
    sns.set(palette="colorblind")
    matplotlib.rc("font", size=10)

    labels = [
        "Family 0",
        "Family 1",
        "Family 2",
        "Family 3",
        "Family 4",
        "Family 5",
        "Family 6",
    ]
    a = aggregated_data["n_zwitterion"].to_list()
    b = aggregated_data["n_tail"].to_list()
    c = aggregated_data["count"].to_list()

    # --- set up figure --- #
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(7.5, 10), layout="constrained")
    barplot = axs[0]
    colormesh = axs[1]
    barplot.set_title("a", loc="left", fontweight="bold")
    colormesh.set_title("b", loc="left", fontweight="bold")

    # --- plot colormesh --- #
    with open(pickle_file, "rb") as f:
        results = pickle.load(f)
    embedding_methods = results.keys()
    families = list(range(7))
    values = np.zeros((len(embedding_methods), len(families)))
    for i, method in enumerate(embedding_methods):
        for j, family in enumerate(families):
            values[i, j] = results[method][j]

    im = axs[1].pcolormesh(
        families,
        embedding_methods,
        values,
        cmap=sns.light_palette("#b64a47", as_cmap=True),
    )
    fig.colorbar(im, ax=axs[0], label="Weighted F1 Score")
    axs[0].set_ylabel("Embedding Method")
    axs[0].set_xlabel("Family")
    # set the scale of the colorbar from 0 to 1
    im.set_clim(0, 1)

    return axs


def plot_colormesh(pickle_file, ax, fig, metric_label):
    # --- plot colormesh --- #
    with open(pickle_file, "rb") as f:
        results = pickle.load(f)
    embedding_methods = results.keys()
    families = list(range(7))
    values = np.zeros((len(embedding_methods), len(families)))
    for i, method in enumerate(embedding_methods):
        for j, family in enumerate(families):
            values[i, j] = results[method][j]

    # Remove rows with NaN
    # index = ~np.isnan(values).any(axis=1)
    # values = values[index]
    # families = np.array(families)[index]
    # replace iphos with MMB-FT in embedding_methods
    embedding_methods = [method if method != "iphos" else "MMB-FT" for method in embedding_methods]

    im = ax.pcolormesh(
        families,
        embedding_methods,
        values,
        cmap=sns.light_palette("#b64a47", as_cmap=True),
    )
    fig.colorbar(im, ax=ax, label=metric_label)
    ax.set_ylabel("Embedding Method")
    ax.set_xlabel("Family")
    # set the scale of the colorbar from 0 to 1
    im.set_clim(0, 1)
    # annotate colormesh boxes with values
    for i, method in enumerate(embedding_methods):
        for j, family in enumerate(families):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="w")

    return ax


def scatter_binary_vs_multiclass(binary_metrics, multiclass_metrics, subfig):
    """Plot a scatter plot with x-axis the binary metric score and y-axis the multiclass metric
    score color by the embedding method put the shape as the family test set remove those with
    np.nan.

    This function returns a subplot to be added to a group of other plots
    """
    with open(binary_metrics, "rb") as f:
        binary_results = pickle.load(f)
    with open(multiclass_metrics, "rb") as f:
        multiclass_results = pickle.load(f)

    # --- Plot --- #
    plt.style.use("seaborn-v0_8")
    sns.set(palette="colorblind")
    matplotlib.rc("font", size=10)
    # increase paddding on the subfigure left margin
    axs = subfig.subplots(3, 3, sharex=True, sharey=True)

    family_colors = ["blue", "orange", "green", "red", "purple", "brown"]
    family_to_color = {family: color for family, color in zip(range(1, 7), family_colors)}

    for i, method in enumerate(multiclass_results.keys()):
        binary_scores = []
        multiclass_scores = []
        families = []
        ax = axs[i // 3, i % 3]
        for family in range(7):
            binary_score = binary_results[method][family]
            multiclass_score = multiclass_results[method][family]
            if not np.isnan(binary_score) and not np.isnan(multiclass_score):
                binary_scores.append(binary_score)
                multiclass_scores.append(multiclass_score)
                families.append(family)
        ax.scatter(binary_scores, multiclass_scores, c=family_colors, alpha=0.5)
        # add a trendline
        z = np.polyfit(binary_scores, multiclass_scores, 1)
        p = np.poly1d(z)
        ax.plot(binary_scores, p(binary_scores), "r--", alpha=0.5)
        # add the correlation score for the trendline as an annotation
        corr = np.corrcoef(binary_scores, multiclass_scores)[0, 1]
        ax.annotate(f"Corr.: {corr:.2f}", xy=(0.05, 0.95), xycoords="axes fraction")
        # add titles to the subplots for method
        ax.set_title(method)

    axs[1, 1].axis("on")
    axs[1, 2].axis("on")

    # turn off axis for all other subplots
    for i in range(7, 9):
        axs[i // 3, i % 3].axis("off")

    # add axis titles to the subfigure
    # subfig.text(0.5, 0.01, 'Binary AUC Score', ha='center')
    # subfig.text(0.01, 0.5, 'Multiclass Accuracy Score', va='center', rotation='vertical')
    subfig.supxlabel("Binary AUC Score", x=0.5, ha="center")
    subfig.supylabel("Multi-class Accuracy", y=0.5, va="center", rotation="vertical")
    subfig.legend(
        handles=[
            matplotlib.patches.Patch(color=color, label=f"Family {family}")
            for family, color in family_to_color.items()
        ],
        loc="lower center",
        bbox_to_anchor=(0.7, 0.1),
        ncol=2,
        fontsize=10,
    )

    # add more space between the Axes and the subfigure edge

    return ax


def plot_colormesh_barplot(data_path, binary_metrics, save_path):
    """Just plot the colormesh and family distribution as subplots."""
    # Set styles
    plt.style.use("seaborn-v0_8")
    sns.set(palette="colorblind")
    matplotlib.rc("font", size=8)
    # --- set up figure --- #
    fig = plt.figure(layout="constrained", figsize=(7.5, 9))
    axs = fig.subplots(2, 1)
    barplot = axs[0]
    colormesh = axs[1]
    barplot.set_title("a", loc="left", fontweight="bold")
    colormesh.set_title("b", loc="left", fontweight="bold")

    # --- plot family distribution --- #
    plot_family_distribution(data_path, barplot)

    # --- plot colormesh --- #
    plot_colormesh(binary_metrics, colormesh, fig, "AUC")

    # --- save figure --- #
    plt.savefig(save_path, bbox_inches="tight")


def plot_colormesh_scatter(binary_metrics, multiclass_metrics, save_path):
    """Just plot the colormesh and scatter as subplots."""
    # Set styles
    plt.style.use("seaborn-v0_8")
    sns.set(palette="colorblind")
    matplotlib.rc("font", size=8)
    # --- set up figure --- #
    fig = plt.figure(layout="constrained", figsize=(7.5, 9))
    subfigs = fig.subfigures(nrows=2, ncols=1, wspace=1, hspace=0.05, height_ratios=[2, 3])
    colormesh = subfigs[0]
    scatter = subfigs[1]
    colormesh.suptitle("a", x=0.05, ha="left", fontweight="bold")
    scatter.suptitle("b", x=0.05, ha="left", fontweight="bold")

    # --- plot scatter --- #
    scatter_binary_vs_multiclass(binary_metrics, multiclass_metrics, scatter)

    # --- plot colormesh --- #
    axcolormesh = colormesh.subplots()
    plot_colormesh(multiclass_metrics, axcolormesh, colormesh, "Accuracy")

    # fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1, hspace=0.1, wspace=0.1)

    # --- save figure --- #
    plt.savefig(save_path, bbox_inches="tight")


def plot_all(data_path, binary_metrics, multiclass_metrics, save_path):
    """
    Combine the three functions above into a large plot with 4 subplots:
        1. (0,0)family distribution barplot
        2. (1,0)colormesh of binary distribution scores
        3. (0,1) scatter plot of binary vs multiclass
        4. (1,1) colormesh of multiclass distribution scores
    """
    # Set styles
    plt.style.use("seaborn-v0_8")
    sns.set(palette="colorblind")
    matplotlib.rc("font", size=8)
    # --- set up figure --- #
    fig = plt.figure(layout="constrained", figsize=(8.5, 11))
    subfigs = fig.subfigures(nrows=2, ncols=2, wspace=0.05, hspace=0.05, height_ratios=[3, 2])
    barplot = subfigs[0, 0]
    colormesh_binary = subfigs[1, 0]
    scatter = subfigs[0, 1]
    colormesh_multiclass = subfigs[1, 1]
    barplot.suptitle("a", x=0.05, ha="left", fontweight="bold")
    colormesh_binary.suptitle("b", x=0.05, ha="left", fontweight="bold")
    scatter.suptitle("c", x=0.05, ha="left", fontweight="bold")
    colormesh_multiclass.suptitle("d", x=0.05, ha="left", fontweight="bold")

    # --- plot family distribution --- #
    axbar = barplot.subplots()
    plot_family_distribution(data_path, axbar)

    # --- plot colormesh --- #
    axcolormesh = colormesh_binary.subplots()
    plot_colormesh(binary_metrics, axcolormesh, colormesh_binary, "AUC")

    # --- plot scatter --- #
    scatter_binary_vs_multiclass(binary_metrics, multiclass_metrics, scatter)

    # --- plot colormesh --- #
    axcolormesh = colormesh_multiclass.subplots()
    plot_colormesh(multiclass_metrics, axcolormesh, colormesh_multiclass, "Accuracy")

    # --- save figure --- #
    plt.savefig(save_path, bbox_inches="tight")


@hydra.main(config_path="../", config_name="config")
def main(cfg: DictConfig):
    # plot_colormesh_scatter(
    #     os.path.join(get_original_cwd(), cfg.plotting.binary_metrics),
    #     os.path.join(get_original_cwd(), cfg.plotting.multiclass_metrics),
    #     os.path.join(get_original_cwd(), cfg.plotting.save_path)
    # )
    plot_colormesh_barplot(
        os.path.join(get_original_cwd(), cfg.plotting.data_path),
        os.path.join(get_original_cwd(), cfg.plotting.binary_metrics),
        os.path.join(get_original_cwd(), cfg.plotting.save_path),
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
