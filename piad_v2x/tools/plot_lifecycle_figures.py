"""Plot-only: regenerate the lifecycle figures from cached figure-data (no re-training).

Reads experiments/results/lifecycle_multiscale_figuredata.json (produced by
run_lifecycle_multiscale_figures.py) and re-renders the operating-curve and
Kaplan-Meier figures. Use this to tweak figure cosmetics without a compute run.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESDIR = "experiments/results"
FIGDIR = "experiments/figures"
os.makedirs(FIGDIR, exist_ok=True)
d = json.load(open(os.path.join(RESDIR, "lifecycle_multiscale_figuredata.json")))
sweep = d["sweep"]
op = d["operating_point"]
kx = np.array(d["km_curve"]["t"]); ky = np.array(d["km_curve"]["fraction_active"])
kmed = op["km_median_s"]
K_GRID = sorted({s["K"] for s in sweep})

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 300, "savefig.bbox": "tight"})

# ---- FIG 1: operating curve ----
fig, ax = plt.subplots(figsize=(3.4, 2.7))
markers = {3: "o", 5: "s", 8: "^"}
for K in K_GRID:
    pts = [s for s in sweep if s["K"] == K]
    xs = [s["false_revocation"] * 100 for s in pts]
    ys = [s["reduction"] * 100 for s in pts]
    ax.plot(xs, ys, marker=markers.get(K, "o"), ms=4, lw=1.2, label=f"K={K}")
ax.plot(op["false_revocation"] * 100, op["reduction"] * 100, "*", color="crimson",
        ms=15, zorder=5, label=f"operating point\n($\\tau$={op['tau']}, K={op['K']})")
ax.set_xlabel("benign false-revocation (%)")
ax.set_ylabel("accepted-malicious reduction (%)")
ax.legend(fontsize=7, loc="lower right")
ax.set_title("Isolation-outcome operating curve", fontsize=9)
fig.savefig(os.path.join(FIGDIR, "fig_tradeoff_multiscale.png"))
fig.savefig(os.path.join(FIGDIR, "fig_tradeoff_multiscale.pdf"))
plt.close(fig)

# ---- FIG 2: Kaplan-Meier ----
fig, ax = plt.subplots(figsize=(3.4, 2.7))
ax.step(kx, ky * 100, where="post", lw=1.6, color="#1f4e79")
if kmed is not None:
    ax.axhline(50, ls=":", color="grey", lw=1)
    ax.axvline(kmed, ls=":", color="crimson", lw=1)
    ax.plot([kmed], [50], "o", color="crimson", ms=6, label=f"median = {kmed:.0f} s")
    ax.legend(fontsize=8, loc="upper right")
ax.set_xlabel("time since attacker's first message (s)")
ax.set_ylabel("attacker pseudonyms\nnot yet isolated (%)")
ax.set_ylim(-2, 102)
xmax = float(np.percentile(kx, 99)) if len(kx) > 3 else float(kx.max())
ax.set_xlim(0, max(5.0, xmax))
ax.set_title("Time-to-isolate (Kaplan-Meier)", fontsize=9)
fig.savefig(os.path.join(FIGDIR, "fig_km_multiscale.png"))
fig.savefig(os.path.join(FIGDIR, "fig_km_multiscale.pdf"))
plt.close(fig)
print("re-rendered figures from cached figuredata")
