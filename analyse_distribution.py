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
frames    = ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7"]
exp_ids   = list(data_raw.keys())
all_vals  = [v for vals in data_raw.values() for v in vals]
by_frame  = [[data_raw[e][i] for e in exp_ids] for i in range(7)]

frame_colors = ["#1565C0","#1976D2","#1E88E5","#42A5F5","#90CAF9","#BBDEFB","#78909C"]

fig = plt.figure(figsize=(18, 22), facecolor="#f8f9fa")
fig.suptitle("PB Concentration Distribution — All Experiments × Time Frames",
             fontsize=15, fontweight="bold", y=0.99)
gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.32,
                      left=0.07, right=0.97, top=0.96, bottom=0.03)

# ── 1. Heatmap ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
matrix = np.array([data_raw[e] for e in exp_ids])
im = ax1.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=75)
ax1.set_xticks(range(7));  ax1.set_xticklabels(frames, fontsize=10)
ax1.set_yticks(range(17)); ax1.set_yticklabels(exp_ids, fontsize=9)
ax1.set_title("Heatmap — Pb Tenörü (%) per experiment × time frame",
              fontsize=11, fontweight="bold")
for i in range(17):
    for j in range(7):
        v = matrix[i, j]
        ax1.text(j, i, f"{v:.1f}", ha="center", va="center",
                 fontsize=7.5, fontweight="bold",
                 color="black" if 12 < v < 62 else "white")
fig.colorbar(im, ax=ax1, fraction=0.015, pad=0.01).set_label("Pb %", fontsize=9)

# ── 2. Box + jitter per time frame ──────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
bp = ax2.boxplot(by_frame, patch_artist=True, notch=False,
                 medianprops=dict(color="black", linewidth=2))
for patch, col in zip(bp["boxes"], frame_colors):
    patch.set_facecolor(col); patch.set_alpha(0.85)
np.random.seed(0)
for i, vals in enumerate(by_frame):
    ax2.scatter(np.random.normal(i + 1, 0.07, len(vals)), vals,
                color=frame_colors[i], edgecolors="#333", s=30, zorder=5, alpha=0.9)
ax2.set_xticklabels(frames)
ax2.set_xlabel("Time Frame", fontsize=10)
ax2.set_ylabel("Pb Tenörü (%)", fontsize=10)
ax2.set_title("Distribution per Time Frame", fontsize=11, fontweight="bold")
ax2.grid(axis="y", alpha=0.35)

# ── 3. Histogram – 3-class ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
bins = np.arange(0, 78, 2.5)
n3, _, _ = ax3.hist(all_vals, bins=bins, edgecolor="white", linewidth=0.6, color="#546E7A")
regions3 = [
    (0,  30, "#ef5350", "Bad\n(< 30)"),
    (30, 40, "#FFA726", "Acceptable\n(30 – 40)"),
    (40, 75, "#66BB6A", "Good\n(> 40)"),
]
for lo, hi, col, lbl in regions3:
    ax3.axvspan(lo, hi, alpha=0.18, color=col, zorder=0)
    ax3.text((lo + hi) / 2, max(n3) * 0.82, lbl,
             ha="center", fontsize=8.5, color=col, fontweight="bold")
for b in [30, 40]:
    ax3.axvline(b, color="#333", linewidth=1.6, linestyle="--", alpha=0.75)
c = [sum(1 for v in all_vals if lo <= v < hi) for lo, hi, _, _ in regions3]
c[-1] = sum(1 for v in all_vals if v >= 40)
tot = len(all_vals)
summary3 = (f"Bad     (<30): {c[0]:2d}  ({100*c[0]/tot:.0f}%)\n"
            f"Accept (30-40): {c[1]:2d}  ({100*c[1]/tot:.0f}%)\n"
            f"Good    (>40): {c[2]:2d}  ({100*c[2]/tot:.0f}%)")
ax3.text(0.97, 0.97, summary3, transform=ax3.transAxes, ha="right", va="top",
         fontsize=8.5, fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#aaa"))
ax3.set_xlabel("Pb Tenörü (%)", fontsize=10)
ax3.set_ylabel("Count (samples)", fontsize=9)
ax3.set_title("3-Class Candidate Boundaries", fontsize=11, fontweight="bold")
ax3.grid(axis="y", alpha=0.3)

# ── 4. Histogram – 5-class ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
n5, _, _ = ax4.hist(all_vals, bins=bins, edgecolor="white", linewidth=0.6, color="#546E7A")
regions5 = [
    (0,  10, "#b71c1c", "Very Bad\n(< 10)"),
    (10, 25, "#ef5350", "Bad\n(10 – 25)"),
    (25, 35, "#FFA726", "Acceptable\n(25 – 35)"),
    (35, 50, "#66BB6A", "Good\n(35 – 50)"),
    (50, 75, "#1b5e20", "Excellent\n(> 50)"),
]
for lo, hi, col, lbl in regions5:
    ax4.axvspan(lo, hi, alpha=0.2, color=col, zorder=0)
    ax4.text((lo + hi) / 2, max(n5) * 0.82, lbl,
             ha="center", fontsize=8, color=col, fontweight="bold")
for b in [10, 25, 35, 50]:
    ax4.axvline(b, color="#333", linewidth=1.6, linestyle="--", alpha=0.75)
c5 = [sum(1 for v in all_vals if lo <= v < hi) for lo, hi, _, _ in regions5]
c5[-1] = sum(1 for v in all_vals if v >= 50)
s5 = "\n".join(f"{lbl.replace(chr(10),' '):18s}: {cnt:2d} ({100*cnt/tot:.0f}%)"
               for (_, _, _, lbl), cnt in zip(regions5, c5))
ax4.text(0.97, 0.97, s5, transform=ax4.transAxes, ha="right", va="top",
         fontsize=8, fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#aaa"))
ax4.set_xlabel("Pb Tenörü (%)", fontsize=10)
ax4.set_ylabel("Count (samples)", fontsize=9)
ax4.set_title("5-Class Candidate Boundaries", fontsize=11, fontweight="bold")
ax4.grid(axis="y", alpha=0.3)

# ── 5. Strip chart ──────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
np.random.seed(1)
for j in range(7):
    vals = by_frame[j]
    ax5.scatter(vals, np.random.normal(j, 0.09, len(vals)),
                color=frame_colors[j], s=45, edgecolors="#333",
                linewidths=0.5, alpha=0.9, label=frames[j], zorder=3)
ax5.set_yticks(range(7)); ax5.set_yticklabels(frames)
ax5.set_xlabel("Pb Tenörü (%)", fontsize=10)
ax5.set_title("Strip Chart — all values per time frame", fontsize=11, fontweight="bold")
ax5.grid(axis="x", alpha=0.35)
for b in [30, 40]:
    ax5.axvline(b, color="#555", linewidth=1.3, linestyle="--", alpha=0.65)
ax5.legend(loc="lower right", fontsize=8, title="Frame")

# ── 6. Per-experiment trajectories ──────────────────────────────────
ax6 = fig.add_subplot(gs[3, :])
cmap = plt.get_cmap("tab20")
for i, (eid, vals) in enumerate(data_raw.items()):
    ax6.plot(frames, vals, marker="o", linewidth=1.8, markersize=5,
             color=cmap(i / 17), label=eid, alpha=0.85)
for lo, hi, col, _ in regions3:
    ax6.axhspan(lo, hi, alpha=0.07, color=col)
for b in [30, 40]:
    ax6.axhline(b, color="#555", linewidth=1.3, linestyle="--", alpha=0.65)
ax6.text(6.52, 31, "30", fontsize=8, color="#555")
ax6.text(6.52, 41, "40", fontsize=8, color="#555")
ax6.set_xlabel("Time Frame", fontsize=10)
ax6.set_ylabel("Pb Tenörü (%)", fontsize=10)
ax6.set_title("PB Concentration Trajectory per Experiment (Y1 → Y7)",
              fontsize=11, fontweight="bold")
ax6.grid(alpha=0.3)
ax6.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.85)

plt.savefig("pb_distribution.png", dpi=150, bbox_inches="tight")
print("Saved pb_distribution.png")

# ── console stats ───────────────────────────────────────────────────
print(f"\nAll {tot} samples:  min={min(all_vals):.2f}  max={max(all_vals):.2f}"
      f"  mean={np.mean(all_vals):.2f}  median={np.median(all_vals):.2f}")
print("\nPer-frame mean (std):")
for j, f in enumerate(frames):
    v = by_frame[j]
    print(f"  {f}: {np.mean(v):5.2f}  ±{np.std(v):.2f}  "
          f"  range [{min(v):.2f} – {max(v):.2f}]")
print(f"\n3-class  bad / acceptable / good:  {c[0]} / {c[1]} / {c[2]}")
print(f"         {100*c[0]/tot:.0f}%  /  {100*c[1]/tot:.0f}%  /  {100*c[2]/tot:.0f}%")
print(f"\n5-class  counts: {c5}  total={sum(c5)}")
