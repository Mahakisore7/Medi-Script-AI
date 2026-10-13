"""
Generates Figure 2: YOLOv8-nano medicine-field detection metrics
(medicine-only prototype, Bannari Amman dataset, 50 epochs).
Clean, minimal-ink horizontal bar chart, vector PDF output.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["pdf.fonttype"] = 42

NAVY = "#1f3a5f"
BLUE = "#3f6fa8"
LIGHT_GRID = "#e3e6ea"
INK = "#1a1a1a"
MUTED = "#6b7280"

metrics = ["Recall", "Precision", "mAP50-95", "mAP50"]
values = [80.6, 81.9, 46.2, 86.1]
colors = [BLUE, BLUE, "#9db6cf", NAVY]

fig, ax = plt.subplots(figsize=(6.6, 2.9))

bars = ax.barh(metrics, values, color=colors, height=0.58, zorder=3, edgecolor="none")

for bar, val in zip(bars, values):
    ax.text(
        val + 1.4,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}%",
        va="center",
        ha="left",
        fontsize=9.5,
        color=INK,
        fontweight="bold",
    )

ax.set_xlim(0, 100)
ax.set_xlabel("Score (%)", fontsize=9, color=MUTED)
ax.tick_params(axis="y", labelsize=10, colors=INK, length=0)
ax.tick_params(axis="x", labelsize=8.5, colors=MUTED)

for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(LIGHT_GRID)

ax.xaxis.grid(True, color=LIGHT_GRID, linewidth=0.9, zorder=0)
ax.set_axisbelow(True)

ax.set_title(
    "YOLOv8-nano medicine-field detection\n(medicine-only prototype, 50 epochs, Bannari Amman dataset)",
    fontsize=9.6,
    color=INK,
    loc="left",
    pad=10,
)

plt.tight_layout(pad=0.4)
plt.savefig("results_metrics.pdf", bbox_inches="tight")
print("saved results_metrics.pdf")
