import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Plots:
    def __init__(
        self, filename: str, labels: List[str] = []
    ) -> None:
        self.filename = filename
        self.primary_concept = labels[0]
        self.secondary_concept = labels[1]
        self.labels = labels

    def plot_lineplot(self, df: pd.DataFrame):
        # First plot
        plt.figure(figsize=(10, 6))
        for version in df["Version"].unique():
            subset = df[df["Version"] == version]
            plt.plot(
                subset["Threshold"],
                subset["CLIP Score Output"],
                marker="o",
                label=version,
            )

        plt.xlabel("Threshold")
        plt.ylabel("CLIP Score Output")
        plt.legend(title="Version")
        plt.grid(True)
        plt.savefig(f"plots/{self.filename}/lineplot_clip_score.png")

        # Second plot
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        plt.figure(figsize=(10, 6))
        for i, version in enumerate(df["Version"].unique()):
            subset = df[df["Version"] == version]
            color = default_colors[i % len(default_colors)]
            plt.plot(
                subset["Threshold"],
                subset[f"% {self.primary_concept} in Output"],
                marker="o",
                label=f"{version} - Output",
                color=color,
            )
            plt.plot(
                subset["Threshold"],
                subset[f"% {self.primary_concept} in Original"],
                linestyle="--",
                label=f"{version} - Original",
                color=color,
            )

        plt.xlabel("Threshold")
        plt.ylabel(f"% {self.primary_concept}")
        plt.legend(title="Version")
        plt.grid(True)
        plt.savefig(f"plots/{self.filename}/lineplot_comparison.png")
        plt.show()

    def plot_barplot(self, df: pd.DataFrame):
        for threshold in df["Threshold"].unique():
            subset = df[df["Threshold"] == threshold]

            plt.figure(figsize=(10, 6))
            bar_width = 0.35 if len(self.labels) <= 2 else 0.2
            index = range(len(subset))
            plt.bar(
                index,
                subset[f"% {self.primary_concept} in Original"],
                bar_width,
                label=f"% {self.primary_concept} in Original",
            )
            plt.bar(
                [i + bar_width for i in index],
                subset[f"% {self.primary_concept} in Output"],
                bar_width,
                label=f"% {self.primary_concept} in Output",
            )
            if len(self.labels) > 2:
                plt.bar(
                    [i + 2*bar_width for i in index],
                    subset[f"% {self.secondary_concept} in Original"],
                    bar_width,
                    label=f"% {self.secondary_concept} in Original",
                )
                plt.bar(
                    [i + 3*bar_width for i in index],
                    subset[f"% {self.secondary_concept} in Output"],
                    bar_width,
                    label=f"% {self.secondary_concept} in Output",
                )

            plt.xlabel("Version")
            plt.ylabel(f"% {self.primary_concept}")
            plt.xticks([i + bar_width * 1.5 for i in index], subset["Version"]) 
            plt.legend()
            plt.grid(True)
            plt.savefig(f"plots/{self.filename}/barplot_threshold_{threshold}.png")

    def plot_boxplot(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Version", y="CLIP Score Output", data=df)
        plt.xlabel("Version")
        plt.ylabel("CLIP Score Output")
        plt.grid(True)
        plt.savefig(f"plots/{self.filename}/boxplot.png")

    def plot_violinplot(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.violinplot(x="Version", y=f"% {self.primary_concept} in Output", data=df)
        plt.xlabel("Version")
        plt.ylabel(f"% {self.primary_concept} in Output")
        plt.grid(True)
        plt.savefig(f"plots/{self.filename}/violinplot.png")

    def plot_gender_difference(self, df: pd.DataFrame):
        versions = df["Version"].unique()
        fig_width = 8 + len(self.labels)
        for label in self.labels:
            df[f"{label} Difference"] = (
                df[f"% {label} in Output"] - df[f"% {label} in Original"]
            )

        for version in versions:
            df_version = df[df["Version"] == version]

            _, ax = plt.subplots(figsize=(fig_width, 6))
            bar_width = 0.8 / len(self.labels)
            index = np.arange(len(df_version))
            default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            for i, label in enumerate(self.labels):
                ax.bar(
                    index + i * bar_width,
                    df_version[f"{label} Difference"],
                    bar_width,
                    label=f"{label} Difference",
                    color=default_colors[i],
                )

            ax.set_xlabel("Threshold")
            ax.set_ylabel("Difference")
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(df_version["Threshold"], rotation=45, ha="right")
            ax.legend()

            plt.tight_layout()
            plt.show()
            plt.savefig(f"plots/{self.filename}/difference_plot_{version}.png")

    def run_plots(self):
        df = pd.read_csv(f"results/{self.filename}.csv")
        os.makedirs(f"plots/{self.filename}", exist_ok=True)
        self.plot_lineplot(df)
        self.plot_barplot(df)
        self.plot_boxplot(df)
        self.plot_violinplot(df)
        # self.plot_gender_difference(df, labels=[...])
