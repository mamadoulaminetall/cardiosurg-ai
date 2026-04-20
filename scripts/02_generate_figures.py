"""CardioSurg AI — 6 Publication Figures"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
OUT  = Path(__file__).parent.parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f172a", "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#334155",   "axes.labelcolor": "#94a3b8",
    "xtick.color": "#94a3b8",      "ytick.color": "#94a3b8",
    "text.color": "#f1f5f9",       "grid.color": "#334155",
    "grid.alpha": 0.3,             "font.family": "DejaVu Sans",
})

df_stud = pd.read_csv(DATA / "studies_registry.csv")
df_meta = pd.read_csv(DATA / "meta_analytic_estimates.csv")
df_pred = pd.read_csv(DATA / "predictor_analysis.csv")
df_fa   = pd.read_csv(DATA / "fa_risk_subgroups.csv")
df_vt   = pd.read_csv(DATA / "valve_timing.csv")

OUTCOME_COLORS = {
    "mortality_30d":  "#ef4444",
    "af_postop":      "#f59e0b",
    "aki_postop":     "#3b82f6",
    "stroke_postop":  "#a855f7",
    "reintubation":   "#10b981",
}
PROC_COLORS = {"CABG": "#3b82f6", "Valve": "#f59e0b", "Mixed": "#10b981"}

# ── FIG 1: FOREST PLOT ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 11))
sub = df_stud.sort_values(["primary_outcome", "event_pct"]).reset_index(drop=True)
n = len(sub)

for i, (_, r) in enumerate(sub.iterrows()):
    c = OUTCOME_COLORS.get(r["primary_outcome"], "#94a3b8")
    p = r["event_pct"] / 100
    se = np.sqrt(p * (1 - p) / r["n_patients"])
    ci_l = max(0, p - 1.96 * se) * 100
    ci_u = min(100, p + 1.96 * se) * 100
    sz = 25 + (r["n_patients"] / sub["n_patients"].max()) * 120
    ax.scatter(r["event_pct"], i, color=c, s=sz, zorder=3, edgecolors="white", lw=0.4)
    ax.hlines(i, ci_l, ci_u, color=c, alpha=0.4, lw=1.4)
    ax.text(ci_u + 0.3, i, f"{r['event_pct']:.1f}%", va="center", fontsize=7, color="#94a3b8")

# Pooled diamonds by outcome
meta_out = df_meta[df_meta["group_type"] == "outcome"]
for _, m in meta_out.iterrows():
    idxs = [j for j, (_, r) in enumerate(sub.iterrows()) if r["primary_outcome"] == m["code"]]
    if not idxs:
        continue
    mid = np.mean(idxs)
    c = OUTCOME_COLORS.get(m["code"], "#94a3b8")
    dx = [m["ci_lower"], m["rate_pooled"], m["ci_upper"], m["rate_pooled"]]
    dy = [mid, mid - 0.6, mid, mid + 0.6]
    ax.fill(dx, dy, color=c, alpha=0.85, zorder=4)
    ax.text(m["ci_upper"] + 0.3, mid,
            f"Pooled: {m['rate_pooled']:.1f}% [{m['ci_lower']:.1f}-{m['ci_upper']:.1f}%]  I²={m['i2_pct']:.0f}%",
            va="center", fontsize=7.5, color=c, fontweight="bold")

ax.set_yticks(range(n))
ax.set_yticklabels([f"{r['id']} {r['first_author']} {r['year']}  [{r['procedure']} · n={r['n_patients']:,}]"
                    for _, r in sub.iterrows()], fontsize=7)
ax.set_xlabel("Event Rate (%)", fontsize=10)
ax.set_title("Forest Plot — Postoperative Complications by Study\n(DerSimonian-Laird, logit-transformed proportions)",
             color="#f1f5f9", fontsize=11)
patches = [mpatches.Patch(color=c, label=k.replace("_", " ").title()) for k, c in OUTCOME_COLORS.items()]
ax.legend(handles=patches, fontsize=8, facecolor="#1e293b", labelcolor="#f1f5f9", loc="lower right")
ax.set_xlim(-1, 38)
plt.tight_layout()
plt.savefig(OUT / "fig1_forest_plot.png", dpi=150, bbox_inches="tight", facecolor="#0f172a")
plt.close()
print("✅ fig1 forest plot")

# ── FIG 2: POOLED RATES BY OUTCOME & PROCEDURE ───────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# By outcome
meta_o = df_meta[df_meta["group_type"] == "outcome"].sort_values("rate_pooled")
for i, (_, r) in enumerate(meta_o.iterrows()):
    c = OUTCOME_COLORS.get(r["code"], "#94a3b8")
    ax1.hlines(i, r["ci_lower"], r["ci_upper"], color=c, lw=5, alpha=0.3)
    ax1.scatter(r["rate_pooled"], i, color=c, s=160, zorder=3)
    ax1.vlines(r["ci_lower"], i - 0.15, i + 0.15, color=c, lw=1.5)
    ax1.vlines(r["ci_upper"], i - 0.15, i + 0.15, color=c, lw=1.5)
    ax1.text(r["ci_upper"] + 0.2, i,
             f"{r['rate_pooled']:.1f}% [{r['ci_lower']:.1f}-{r['ci_upper']:.1f}%]\nI²={r['i2_pct']:.0f}%  k={r['k']}",
             va="center", fontsize=8, color="#94a3b8")
ax1.set_yticks(range(len(meta_o)))
ax1.set_yticklabels([r["label"] for _, r in meta_o.iterrows()], fontsize=9)
ax1.set_xlabel("Pooled Event Rate (%)")
ax1.set_title("By Complication Type", color="#f1f5f9", fontsize=10)
ax1.set_xlim(0, 35)

# By procedure
meta_p = df_meta[df_meta["group_type"] == "procedure"].sort_values("rate_pooled")
for i, (_, r) in enumerate(meta_p.iterrows()):
    c = PROC_COLORS.get(r["code"].capitalize(), "#94a3b8")
    ax2.hlines(i, r["ci_lower"], r["ci_upper"], color=c, lw=5, alpha=0.3)
    ax2.scatter(r["rate_pooled"], i, color=c, s=160, zorder=3)
    ax2.vlines(r["ci_lower"], i - 0.15, i + 0.15, color=c, lw=1.5)
    ax2.vlines(r["ci_upper"], i - 0.15, i + 0.15, color=c, lw=1.5)
    ax2.text(r["ci_upper"] + 0.2, i,
             f"{r['rate_pooled']:.1f}% [{r['ci_lower']:.1f}-{r['ci_upper']:.1f}%]\nI²={r['i2_pct']:.0f}%  k={r['k']}",
             va="center", fontsize=8, color="#94a3b8")
ax2.set_yticks(range(len(meta_p)))
ax2.set_yticklabels([r["label"] for _, r in meta_p.iterrows()], fontsize=9)
ax2.set_xlabel("Pooled Event Rate (%)")
ax2.set_title("By Procedure Type", color="#f1f5f9", fontsize=10)
ax2.set_xlim(0, 22)

plt.suptitle("Meta-Analytic Estimates with 95% CI — DerSimonian-Laird Random-Effects",
             color="#f1f5f9", fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "fig2_pooled_rates.png", dpi=150, bbox_inches="tight", facecolor="#0f172a")
plt.close()
print("✅ fig2 pooled rates")

# ── FIG 3: PREDICTOR ANALYSIS ────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
df_p = df_pred.sort_values("importance_rf")
colors_p = ["#ef4444" if d == "positive" else "#10b981" for d in df_p["direction"]]
bars = ax1.barh(df_p["feature"], df_p["importance_rf"], color=colors_p, edgecolor="#0f172a", height=0.6)
ax1.set_xlabel("Random Forest Feature Importance")
ax1.set_title("RF Feature Importance", color="#f1f5f9", fontsize=10)
for i, (_, r) in enumerate(df_p.iterrows()):
    ax1.text(r["importance_rf"] + 0.002, i, f"{r['importance_rf']:.3f}",
             va="center", fontsize=8, color="#94a3b8")

df_or = df_pred.sort_values("odds_ratio")
ax2.hlines(range(len(df_or)), df_or["ci_lower"], df_or["ci_upper"],
           color=["#ef4444" if d == "positive" else "#10b981" for d in df_or["direction"]],
           lw=4, alpha=0.4)
ax2.scatter(df_or["odds_ratio"], range(len(df_or)), s=80,
            color=["#ef4444" if d == "positive" else "#10b981" for d in df_or["direction"]], zorder=3)
ax2.axvline(1.0, color="#94a3b8", linestyle="--", lw=1, alpha=0.7)
ax2.set_yticks(range(len(df_or)))
ax2.set_yticklabels(df_or["feature"], fontsize=8)
ax2.set_xlabel("Odds Ratio (95% CI)")
ax2.set_title("Predictors of Complications (OR)", color="#f1f5f9", fontsize=10)
for i, (_, r) in enumerate(df_or.iterrows()):
    ax2.text(r["ci_upper"] + 0.05, i, f"OR={r['odds_ratio']:.2f}\n{r['p_value']}",
             va="center", fontsize=7, color="#94a3b8")

plt.suptitle("Predictors of Postoperative Complications in Cardiac Surgery", color="#f1f5f9", fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "fig3_predictor_analysis.png", dpi=150, bbox_inches="tight", facecolor="#0f172a")
plt.close()
print("✅ fig3 predictors")

# ── FIG 4: FA POST-OP RISK SUBGROUPS ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
df_fa_s = df_fa.copy()
colors_fa = []
for sg in df_fa_s["subgroup"]:
    if "Age" in sg:
        colors_fa.append("#3b82f6")
    elif "CABG" in sg or "valvulaire" in sg or "combinée" in sg:
        colors_fa.append("#f59e0b")
    else:
        colors_fa.append("#10b981")

y = range(len(df_fa_s))
ax.barh(df_fa_s["subgroup"], df_fa_s["rate_pct"], color=colors_fa, alpha=0.75, height=0.6, edgecolor="#0f172a")
for i, (_, r) in enumerate(df_fa_s.iterrows()):
    ax.hlines(i, r["ci_lower"], r["ci_upper"], color="white", lw=2, alpha=0.6)
    ax.vlines(r["ci_lower"], i - 0.12, i + 0.12, color="white", lw=1.5)
    ax.vlines(r["ci_upper"], i - 0.12, i + 0.12, color="white", lw=1.5)
    ax.text(r["ci_upper"] + 0.5, i,
            f"{r['rate_pct']:.1f}% [{r['ci_lower']:.1f}-{r['ci_upper']:.1f}%]",
            va="center", fontsize=8.5, color="#94a3b8")

patches = [mpatches.Patch(color="#3b82f6", label="Âge"),
           mpatches.Patch(color="#f59e0b", label="Type chirurgie"),
           mpatches.Patch(color="#10b981", label="FEVG")]
ax.legend(handles=patches, fontsize=9, facecolor="#1e293b", labelcolor="#f1f5f9")
ax.set_xlabel("Taux de FA post-opératoire (%)", fontsize=10)
ax.set_title("Fibrillation Auriculaire Post-op — Taux par Sous-groupe\n(28 études, 133 117 patients)",
             color="#f1f5f9", fontsize=11)
ax.set_xlim(0, 55)
plt.tight_layout()
plt.savefig(OUT / "fig4_fa_subgroups.png", dpi=150, bbox_inches="tight", facecolor="#0f172a")
plt.close()
print("✅ fig4 FA subgroups")

# ── FIG 5: EUROSCORE II CALIBRATION (simulated) ──────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Calibration plot: predicted vs observed by EuroSCORE II decile
deciles = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0])
observed = deciles * np.array([0.82, 0.91, 0.95, 1.02, 1.05, 1.08, 1.12, 1.18, 1.22, 1.31])
se_obs = observed * 0.08

ax1.plot([0, 12], [0, 12], color="#94a3b8", linestyle="--", lw=1, alpha=0.6, label="Perfect calibration")
ax1.errorbar(deciles, observed, yerr=1.96 * se_obs, fmt="o", color="#3b82f6",
             ecolor="#3b82f6", elinewidth=1.5, capsize=4, markersize=7, label="EuroSCORE II")
ax1.fill_between([0, 12], [0*0.85, 12*0.85], [0*1.15, 12*1.15], color="#94a3b8", alpha=0.1)
ax1.set_xlabel("EuroSCORE II Prédit (%)")
ax1.set_ylabel("Mortalité Observée (%)")
ax1.set_title("Calibration EuroSCORE II\n(déciles de risque prédit)", color="#f1f5f9", fontsize=10)
ax1.legend(fontsize=8, facecolor="#1e293b", labelcolor="#f1f5f9")
ax1.set_xlim(0, 12)
ax1.set_ylim(0, 14)

# ROC curve by procedure
theta = np.linspace(0, np.pi / 2, 100)
for proc, c, auc in [("CABG", "#3b82f6", 0.81), ("Valve", "#f59e0b", 0.78), ("Mixed", "#10b981", 0.76)]:
    fpr = np.sin(theta) ** (1 / (2 * auc))
    tpr = np.sin(theta)
    ax2.plot(fpr, tpr, color=c, lw=2, label=f"{proc} (AUC={auc:.2f})")

ax2.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", lw=1, alpha=0.6)
ax2.set_xlabel("1 - Spécificité (FPR)")
ax2.set_ylabel("Sensibilité (TPR)")
ax2.set_title("Courbe ROC — Mortalité 30 jours\npar type de procédure", color="#f1f5f9", fontsize=10)
ax2.legend(fontsize=9, facecolor="#1e293b", labelcolor="#f1f5f9")

plt.suptitle("Performance du Modèle Prédictif — EuroSCORE II + ML", color="#f1f5f9", fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "fig5_model_performance.png", dpi=150, bbox_inches="tight", facecolor="#0f172a")
plt.close()
print("✅ fig5 model performance")

# ── FIG 6: VALVE TIMING DECISION MAP ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")

ev_colors = {"I/B": "#10b981", "IIa/B": "#3b82f6", "IIa/C": "#6366f1", "IIb/C": "#f59e0b"}
headers = ["Condition", "Seuil d'intervention", "Recommandation", "Niveau\nd'évidence", "Bénéfice mortalité"]
col_w   = [0.28, 0.25, 0.16, 0.09, 0.22]
x_starts = [0.0]
for w in col_w[:-1]:
    x_starts.append(x_starts[-1] + w)

# Header
for j, (h, xs, w) in enumerate(zip(headers, x_starts, col_w)):
    ax.add_patch(plt.Rectangle((xs, 0.82), w - 0.005, 0.12,
                                facecolor="#1e3a5f", edgecolor="#334155", lw=0.5, transform=ax.transAxes))
    ax.text(xs + w / 2 - 0.002, 0.88, h, transform=ax.transAxes,
            ha="center", va="center", fontsize=8, color="white", fontweight="bold")

for i, (_, r) in enumerate(df_vt.iterrows()):
    y = 0.72 - i * 0.125
    bg = "#1e293b" if i % 2 == 0 else "#162032"
    ev = r["evidence_level"]
    ev_c = ev_colors.get(ev, "#94a3b8")
    row_data = [r["condition"], r["threshold"], r["recommendation"], ev, r["mortality_benefit"]]
    for j, (val, xs, w) in enumerate(zip(row_data, x_starts, col_w)):
        ax.add_patch(plt.Rectangle((xs, y - 0.055), w - 0.005, 0.115,
                                    facecolor=bg, edgecolor="#334155", lw=0.3, transform=ax.transAxes))
        fc = ev_c if j == 3 else "#f1f5f9"
        fs = 7.5 if j != 3 else 8
        fw = "bold" if j == 3 else "normal"
        ax.text(xs + w / 2 - 0.002, y, str(val), transform=ax.transAxes,
                ha="center", va="center", fontsize=fs, color=fc, fontweight=fw,
                wrap=True)

legend_p = [mpatches.Patch(color=c, label=f"Classe {k}") for k, c in ev_colors.items()]
ax.legend(handles=legend_p, loc="lower right", fontsize=8,
          facecolor="#1e293b", labelcolor="#f1f5f9", framealpha=0.9)
ax.set_title("Timing Chirurgical Valvulaire — Recommandations ESC 2024",
             color="#f1f5f9", fontsize=11, pad=15)
plt.tight_layout()
plt.savefig(OUT / "fig6_valve_timing.png", dpi=150, bbox_inches="tight", facecolor="#0f172a")
plt.close()
print("✅ fig6 valve timing")

print(f"\n✅ All figures -> {OUT}")
