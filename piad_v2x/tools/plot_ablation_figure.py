"""Scale-ablation figure from invariant_experiment.json (the canonical within-detector
ablation): kinematic-scale-only vs multi-scale ROC-AUC per attack class, hard classes
highlighted. No compute -- reads the existing JSON.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESDIR = "experiments/results"
FIGDIR = "experiments/figures"
os.makedirs(FIGDIR, exist_ok=True)
d = json.load(open(os.path.join(RESDIR, "invariant_experiment.json")))

# order classes: hard (self-consistent) first, then the rest; drop the _1416 suffix
rows = []
for name, v in d.items():
    kin = v.get("rf_full_single_vehicle")
    ms = v.get("rf_full_plus_relational")
    if kin is None or ms is None:
        continue
    lift = v.get("lift", ms - kin)
    rows.append((name.replace("_1416", ""), kin, ms, lift, lift > 0.05))
rows.sort(key=lambda r: -r[3])                # by interaction-scale lift, descending

labels = [r[0] for r in rows]
kin = np.array([r[1] for r in rows]); ms = np.array([r[2] for r in rows])
hard = [r[4] for r in rows]                   # "hard" here = substantial lift (>0.05)

plt.rcParams.update({"font.size": 9, "figure.dpi": 300, "savefig.bbox": "tight"})
fig, ax = plt.subplots(figsize=(7.0, 2.9))
x = np.arange(len(labels)); w = 0.38
ax.bar(x - w/2, kin, w, label="kinematic scale only", color="#b07aa1")
ax.bar(x + w/2, ms, w, label="multi-scale (kinematic + interaction)", color="#2e7d32")
# annotate the lift on hard classes
for i, (k, m, h) in enumerate(zip(kin, ms, hard)):
    if h and (m - k) > 0.02:
        ax.annotate(f"+{m-k:.2f}", xy=(x[i], m + 0.01), ha="center", va="bottom",
                    fontsize=8, color="#2e7d32", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("held-out ROC-AUC")
ax.set_ylim(0.5, 1.05)
ax.axhline(0.5, color="grey", lw=0.6, ls=":")
ax.legend(fontsize=8, loc="lower left", ncol=1, framealpha=0.9)
ax.set_title("Within-detector scale ablation (VeReMi Extension, _1416)", fontsize=9)
ax.grid(axis="y", alpha=0.3)
# mark hard classes on the axis
for i, h in enumerate(hard):
    if h:
        ax.get_xticklabels()[i].set_color("crimson")
        ax.get_xticklabels()[i].set_fontweight("bold")
fig.savefig(os.path.join(FIGDIR, "fig_ablation.png"))
fig.savefig(os.path.join(FIGDIR, "fig_ablation.pdf"))
plt.close(fig)
print("wrote fig_ablation; hard-class lifts:",
      {r[0]: round(r[3], 3) for r in rows if r[4]})
