"""Figure: AUC vs attacker strength eps, multi-scale vs kinematic-scale-only.

One panel per scenario, with the 0.60 break line (pre-registration section 6). Reads
experiments/results/adaptive_robustness.json and writes experiments/figures/adaptive_robustness.png.

    python piad_v2x/tools/plot_adaptive_robustness.py
"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="experiments/results/adaptive_robustness.json")
    ap.add_argument("--out", default="experiments/figures/adaptive_robustness.png")
    args = ap.parse_args()
    data = json.load(open(args.inp))
    scenarios = data["scenarios"]
    names = list(scenarios.keys())
    n = len(names)
    cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    for i, nm in enumerate(names):
        ax = axes[i // cols][i % cols]
        c = scenarios[nm]
        eps = sorted(float(e) for e in c["single_vehicle"].keys())
        sv = [c["single_vehicle"][str(e)] for e in eps]
        rl = [c["relational"][str(e)] for e in eps]
        ax.plot(eps, rl, "o-", color="#1b7837", label="multi-scale (kinematic + interaction)")
        ax.plot(eps, sv, "s--", color="#762a83", label="kinematic scale only (ablation)")
        ax.axhline(data["break_auc"], color="crimson", ls=":", lw=1, label="break AUC 0.60")
        ax.set_title(nm, fontsize=10)
        ax.set_xlabel("attacker strength ε"); ax.set_ylabel("held-out ROC-AUC")
        ax.set_ylim(0.45, 1.0); ax.grid(alpha=0.3)
        gap = c.get("eps1_gap_ci95")
        if gap:
            ax.text(0.02, 0.06, f"ε=1 gap CI [{gap[0]:.2f},{gap[1]:.2f}]",
                    transform=ax.transAxes, fontsize=8)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    axes[0][0].legend(fontsize=8, loc="lower left")
    _tgt = list(scenarios.values())[0].get("target", "single")
    fig.suptitle("Adaptive robustness: multi-scale physics detector vs its kinematic-scale ablation "
                 f"under a goal-preserving white-box attacker (target={_tgt})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=140)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
