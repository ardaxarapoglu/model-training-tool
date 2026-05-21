import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

data_raw = {
    "D01": [41.75, 33.06, 31.91, 15.01,  7.01,  4.02,  2.99],
    "D02": [44.04, 42.09, 34.09, 24.44,  5.03,  2.41,  3.19],
    "D03": [70.46, 70.91, 58.32, 15.72,  7.14,  1.75,  3.82],
    "D04": [43.56, 21.46, 19.90, 14.02,  3.55,  1.87,  1.29],
    "D05": [39.58, 30.43, 16.04,  6.43,  3.47,  2.57,  1.54],
    "D06": [34.81, 28.72, 15.01,  6.98,  2.62,  1.97,  1.53],
    "D07": [39.83, 34.49, 25.26, 16.20,  6.65,  2.51,  1.74],
    "D08": [35.51, 32.91, 29.64, 20.44,  5.64,  2.57,  1.79],
    "D09": [34.88, 22.90, 19.59, 13.31,  6.02,  3.42,  2.88],
    "D10": [57.70, 31.84, 17.17,  7.74,  3.58,  2.78,  1.70],
    "D11": [45.16, 33.18, 14.52,  8.77,  4.33,  2.89,  1.97],
    "D12": [50.26, 19.98,  7.90,  5.39,  3.93,  2.79,  1.70],
    "D13": [62.14, 40.18, 15.07,  7.42,  3.92,  2.55,  1.74],
    "D14": [60.83, 37.38, 17.57,  9.08,  2.69,  2.75,  1.74],
    "D15": [51.75, 20.01,  9.90,  6.69,  3.37,  3.36,  1.87],
    "D16": [45.26, 21.87,  8.42,  6.55,  2.95,  2.67,  1.87],
    "D17": [46.65, 19.03,  9.20,  4.40,  2.52,  2.04,  2.34],
}
frames  = ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7"]
exp_ids = list(data_raw.keys())
matrix  = np.array([data_raw[e] for e in exp_ids])   # (17, 7)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Pb Tenörü (%) — raw values", fontsize=13, fontweight="bold")

for ax, transpose, row_labels, col_labels, title in [
    (axes[0], False, exp_ids, frames,   "Experiments (rows) × Time Frames (cols)"),
    (axes[1], True,  frames,  exp_ids,  "Time Frames (rows) × Experiments (cols)"),
]:
    data = matrix if not transpose else matrix.T
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=75)
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="black" if 10 < v < 60 else "white")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("Pb %", fontsize=9)

plt.tight_layout()
plt.savefig("pb_distribution.png", dpi=150, bbox_inches="tight")
print("Saved pb_distribution.png")
