"""Supporting figures: roadmap flowchart (Week 1) and before/after cleaning (Week 2)."""
import json, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

steps = ["1. Data\nCollection", "2. Cleaning &\nPreprocessing", "3. Exploratory\nAnalysis", "4. Predictive\nModelling", "5. Optimisation", "6. Reporting &\nDecisions"]
fig, ax = plt.subplots(figsize=(11, 2.4)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 2)
for i, s in enumerate(steps):
    x = 0.2 + i * 1.95
    ax.add_patch(FancyBboxPatch((x, 0.5), 1.55, 1, boxstyle="round,pad=0.05", fc="#4c72b0", ec="none"))
    ax.text(x + .775, 1.0, s, ha="center", va="center", color="white", fontsize=9.5, weight="bold")
    if i < len(steps) - 1: ax.annotate("", xy=(x + 1.9, 1.0), xytext=(x + 1.62, 1.0), arrowprops=dict(arrowstyle="->", lw=1.6))
fig.savefig("figures/fig0_roadmap.png", dpi=150, bbox_inches="tight"); plt.close()

raw = pd.read_csv("data/raw_shipments.csv"); cl = pd.read_csv("data/clean_shipments.csv")
fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
for a, col, t in zip(ax, ["distance_km", "actual_hours", "shipping_cost"], ["Distance (km)", "Actual hours", "Shipping cost"]):
    a.boxplot([raw[col].dropna(), cl[col]], tick_labels=["Raw", "Clean"], showfliers=True, flierprops=dict(markersize=2)); a.set_title(t)
fig.tight_layout(); fig.savefig("figures/fig_w2_before_after.png", dpi=150); plt.close()
r = {c: {"raw_mean": round(raw[c].mean(), 2), "raw_max": round(raw[c].max(), 2), "raw_min": round(raw[c].min(), 2), "clean_mean": round(cl[c].mean(), 2), "clean_max": round(cl[c].max(), 2), "clean_min": round(cl[c].min(), 2)} for c in ["distance_km", "actual_hours", "shipping_cost", "weight_kg"]}
json.dump(r, open("data/before_after.json", "w"), indent=1); print(json.dumps(r))
