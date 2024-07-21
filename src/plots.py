import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FILENAME = "without_pink_elephant-fp16_2_labels"
PRIMARY_CONCEPT = "Pink Elephant"
SECONDARY_CONCEPT = "Elephant"


def plot_lineplot(df: pd.DataFrame):
    # First plot
    plt.figure(figsize=(10, 6))
    for version in df["Version"].unique():
        subset = df[df["Version"] == version]
        plt.plot(
            subset["Threshold"], subset["CLIP Score Output"], marker="o", label=version
        )

    plt.title("CLIP Score Output Across Different Thresholds")
    plt.xlabel("Threshold")
    plt.ylabel("CLIP Score Output")
    plt.legend(title="Version")
    plt.grid(True)
    plt.savefig(f"plots/{FILENAME}/lineplot_clip_score.png")

    # Second plot
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.figure(figsize=(10, 6))
    for i, version in enumerate(df["Version"].unique()):
        subset = df[df["Version"] == version]
        color = default_colors[i % len(default_colors)]
        plt.plot(
            subset["Threshold"],
            subset["% Pink Elephant in Output"],
            marker="o",
            label=f"{version} - Output",
            color=color,
        )
        plt.plot(
            subset["Threshold"],
            subset["% Pink Elephant in Original"],
            linestyle="--",
            label=f"{version} - Original",
            color=color,
        )

    plt.title("% Pink Elephant in Output and Original Across Different Thresholds")
    plt.xlabel("Threshold")
    plt.ylabel("% Pink Elephant")
    plt.legend(title="Version")
    plt.grid(True)
    plt.savefig(f"plots/{FILENAME}/lineplot_comparison.png")
    plt.show()


def plot_scatterplot(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df[f"% {PRIMARY_CONCEPT} in Original"],
        df[f"% {PRIMARY_CONCEPT} in Output"],
        c="b",
        alpha=0.5,
    )
    plt.title(f"% {PRIMARY_CONCEPT} in Original vs. Output")
    plt.xlabel(f"% {PRIMARY_CONCEPT} in Original")
    plt.ylabel(f"% {PRIMARY_CONCEPT} in Output")
    plt.grid(True)
    plt.savefig(f"plots/{FILENAME}/scatterplot.png")


def plot_barplot(df: pd.DataFrame, threshold: float):
    subset = df[df["Threshold"] == threshold]

    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = range(len(subset))
    plt.bar(
        index,
        subset[f"% {PRIMARY_CONCEPT} in Original"],
        bar_width,
        label=f"% {PRIMARY_CONCEPT} in Original",
    )
    plt.bar(
        [i + bar_width for i in index],
        subset[f"% {PRIMARY_CONCEPT} in Output"],
        bar_width,
        label=f"% {PRIMARY_CONCEPT} in Output",
    )

    plt.title(
        f"Comparison of % {PRIMARY_CONCEPT} for Different Versions at Threshold {threshold}"
    )
    plt.xlabel("Version")
    plt.ylabel(f"% {PRIMARY_CONCEPT}")
    plt.xticks([i + bar_width / 2 for i in index], subset["Version"])
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{FILENAME}/barplot_threshold_{threshold}.png")


def plot_boxplot(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Version", y="CLIP Score Output", data=df)
    plt.title("Box Plot of CLIP Score Output Across Different Versions")
    plt.xlabel("Version")
    plt.ylabel("CLIP Score Output")
    plt.grid(True)
    plt.savefig(f"plots/{FILENAME}/boxplot.png")


def plot_violinplot(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Version", y=f"% {PRIMARY_CONCEPT} in Output", data=df)
    plt.title(f"Violin Plot of % {PRIMARY_CONCEPT} in Output Across Different Versions")
    plt.xlabel("Version")
    plt.ylabel(f"% {PRIMARY_CONCEPT} in Output")
    plt.grid(True)
    plt.savefig(f"plots/{FILENAME}/violinplot.png")


def plot_3d_scatterplot(df: pd.DataFrame):
    if "2_labels" in FILENAME:
        print("Cannot plot 3D scatter plot for 2 labels.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for version in df["Version"].unique():
        subset = df[df["Version"] == version]
        ax.scatter(
            subset["CLIP Score Output"],
            subset[f"% {PRIMARY_CONCEPT} in Output"],
            subset[f"% {SECONDARY_CONCEPT} in Output"],
            label=version,
        )

    ax.set_title(
        f"3D Scatter Plot of CLIP Score Output vs. % {PRIMARY_CONCEPT} in Output and % {SECONDARY_CONCEPT} in Output"
    )
    ax.set_xlabel("CLIP Score Output")
    ax.set_ylabel(f"% {PRIMARY_CONCEPT} in Output")
    ax.set_zlabel(f"% {SECONDARY_CONCEPT} in Output")
    ax.legend(title="Version")
    plt.savefig(f"plots/{FILENAME}/3d_scatterplot.png")


if __name__ == "__main__":
    df = pd.read_csv(f"results/{FILENAME}.csv")
    os.makedirs(f"plots/{FILENAME}", exist_ok=True)

    plot_lineplot(df)
    plot_scatterplot(df)
    plot_barplot(df, 0.5)
    plot_boxplot(df)
    plot_violinplot(df)
    plot_3d_scatterplot(df)
