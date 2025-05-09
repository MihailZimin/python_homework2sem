import numpy as np
import matplotlib.pyplot as plt


class VisualiseData:
    pallete = {
        "pink-style": ["#FF9AA2", "#FFB7B2"],
        "bright-style": ["#2E86AB", "#A23B72"],
        "dark-style": ["#393E46", "#00ADB5"],
        "gradient-style": ["#003f5c", "#444e86"],
    }

    def __init__(self):
        self.colors = self.pallete["pink-style"]

    def plot_violin(
        self,
        axis: plt.Axes,
        data: np.ndarray,
    ) -> None:
        violin_parts = axis.violinplot(
            data,
            showmedians=True,
            orientation="horizontal"
        )
        for body in violin_parts["bodies"]:
            body.set_facecolor(self.colors[0])
            body.set_linewidth(2)
            body.set_edgecolor(self.colors[1])

        for part in violin_parts:
            if part == "bodies":
                continue

            violin_parts[part].set_edgecolor(self.colors[1])

    def plot_hist(
        self,
        axis: plt.Axes,
        data: np.ndarray,
    ) -> None:
        axis.hist(
            data,
            bins=50,
            color=self.colors[0],
            orientation="vertical",
            density=True,
            alpha=0.5,
            edgecolor=self.colors[1]
        )

    def plot_boxplot(
        self,
        axis: plt.Axes,
        data: np.ndarray,
    ) -> None:
        axis.boxplot(
            data,
            orientation="horizontal",
            patch_artist=True,
            boxprops=dict(facecolor=self.colors[0]),
            medianprops=dict(color=self.colors[1]),
        )
        axis.set_yticks([])

    map_of_functions = {
            "Violin": plot_violin,
            "Hist": plot_hist,
            "Boxplot": plot_boxplot
        }

    def plot_by_name(
        self,
        axis: plt.Axes,
        data: np.ndarray,
        name: str,
    ) -> None:
        if name not in self.map_of_functions:
            raise ValueError(f"No such plot name as: {name} exist")
        self.map_of_functions[name](self, axis, data)
