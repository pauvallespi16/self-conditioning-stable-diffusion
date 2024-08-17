import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Plots:
    def __init__(
        self, filename: str, primary_concept: str, secondary_concept: str = ""
    ) -> None:
        self.filename = filename
        self.primary_concept = primary_concept
        self.secondary_concept = secondary_concept

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

        plt.title("CLIP Score Output Across Different Thresholds")
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

        plt.title(
            f"% {self.primary_concept} in Output and Original Across Different Thresholds"
        )
        plt.xlabel("Threshold")
        plt.ylabel(f"% {self.primary_concept}")
        plt.legend(title="Version")
        plt.grid(True)
        plt.savefig(f"plots/{self.filename}/lineplot_comparison.png")
        plt.show()

    def plot_scatterplot(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        plt.scatter(
            df[f"% {self.primary_concept} in Original"],
            df[f"% {self.primary_concept} in Output"],
            c="b",
            alpha=0.5,
        )
        plt.title(f"% {self.primary_concept} in Original vs. Output")
        plt.xlabel(f"% {self.primary_concept} in Original")
        plt.ylabel(f"% {self.primary_concept} in Output")
        plt.grid(True)
        plt.savefig(f"plots/{self.filename}/scatterplot.png")

    def plot_barplot(self, df: pd.DataFrame):
        for threshold in df["Threshold"].unique():
            subset = df[df["Threshold"] == threshold]

            plt.figure(figsize=(10, 6))
            bar_width = 0.35
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

            plt.title(
                f"Comparison of % {self.primary_concept} for Different Versions at Threshold {threshold}"
            )
            plt.xlabel("Version")
            plt.ylabel(f"% {self.primary_concept}")
            plt.xticks([i + bar_width / 2 for i in index], subset["Version"])
            plt.legend()
            plt.grid(True)
            plt.savefig(f"plots/{self.filename}/barplot_threshold_{threshold}.png")

    def plot_boxplot(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Version", y="CLIP Score Output", data=df)
        plt.title("Box Plot of CLIP Score Output Across Different Versions")
        plt.xlabel("Version")
        plt.ylabel("CLIP Score Output")
        plt.grid(True)
        plt.savefig(f"plots/{self.filename}/boxplot.png")

    def plot_violinplot(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.violinplot(x="Version", y=f"% {self.primary_concept} in Output", data=df)
        plt.title(
            f"Violin Plot of % {self.primary_concept} in Output Across Different Versions"
        )
        plt.xlabel("Version")
        plt.ylabel(f"% {self.primary_concept} in Output")
        plt.grid(True)
        plt.savefig(f"plots/{self.filename}/violinplot.png")

    def run_plots(self):
        df = pd.read_csv(f"results/{self.filename}.csv")
        os.makedirs(f"plots/{self.filename}", exist_ok=True)

        self.plot_lineplot(df)
        self.plot_scatterplot(df)
        self.plot_barplot(df)
        self.plot_boxplot(df)
        self.plot_violinplot(df)


if __name__ == "__main__":
    primary_concept = "Pink Elephant"
    secondary_concept = "Something Else"
    filename = "without_pink_elephant_3_labels"
    plots = Plots(filename, primary_concept, secondary_concept)
    plots.run_plots()
