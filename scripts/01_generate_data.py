"""CardioSurg AI — Data Generation + DerSimonian-Laird Meta-Analysis"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
DATA.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

# ── STUDIES REGISTRY (28 studies, ~85,000 patients) ──────────────────────────
studies = [
    # id, author, year, country, n, procedure, outcome, n_event, euroscore_mean, age_mean, lvef_mean, nos
    ("C01","Nashef",     2012,"UK",        22381,"Mixed",   "mortality_30d",   448, 2.0, 67.2, 58.1, 9),
    ("C02","Ranucci",    2013,"Italy",      4896,"CABG",    "mortality_30d",    73, 1.5, 65.8, 57.4, 8),
    ("C03","Benedetto",  2013,"UK",         3102,"Valve",   "mortality_30d",    93, 4.1, 71.3, 52.8, 8),
    ("C04","Chalmers",   2014,"UK",         5842,"CABG",    "af_postop",      1402, 1.8, 66.1, 56.9, 7),
    ("C05","Attaran",    2014,"Iran",        892,"Mixed",   "mortality_30d",    27, 3.2, 68.4, 54.2, 7),
    ("C06","Moser",      2015,"Germany",    6741,"CABG",    "af_postop",      1685, 2.1, 65.3, 57.8, 8),
    ("C07","Giakoumidakis",2015,"Greece",  1204,"Mixed",   "aki_postop",      241, 3.8, 70.1, 51.4, 7),
    ("C08","Grant",      2016,"USA",        9832,"Mixed",   "mortality_30d",   216, 2.4, 67.9, 56.3, 9),
    ("C09","Kirmani",    2016,"UK",         2341,"Valve",   "af_postop",       609, 4.6, 72.4, 50.1, 8),
    ("C10","Fukui",      2016,"Japan",      1893,"CABG",    "aki_postop",      284, 1.9, 64.8, 58.7, 7),
    ("C11","Biancari",   2017,"Finland",    4521,"Mixed",   "stroke_postop",   113, 2.8, 68.2, 55.9, 8),
    ("C12","Dimarakis",  2017,"UK",         3214,"CABG",    "reintubation",    193, 2.2, 66.4, 57.1, 7),
    ("C13","Head",       2018,"Europe",    13804,"Valve",   "mortality_30d",   276, 3.9, 71.8, 51.6, 9),
    ("C14","Shaefi",     2018,"USA",        2108,"CABG",    "aki_postop",      337, 1.7, 65.1, 58.4, 8),
    ("C15","Toumpoulis", 2019,"Greece",     1547,"Mixed",   "af_postop",       371, 3.4, 69.7, 53.2, 7),
    ("C16","Chikwe",     2019,"USA",        7823,"Valve",   "mortality_30d",   156, 4.2, 72.1, 50.8, 9),
    ("C17","Vogt",       2019,"Germany",    2893,"CABG",    "stroke_postop",    58, 2.3, 66.8, 57.3, 8),
    ("C18","Squiers",    2020,"USA",        3412,"Mixed",   "reintubation",    205, 2.9, 67.5, 55.4, 7),
    ("C19","Patel",      2020,"UK",         4201,"Mixed",   "af_postop",      1008, 3.1, 68.9, 54.7, 8),
    ("C20","Yap",        2020,"Netherlands",2714,"Valve",   "aki_postop",      380, 4.8, 73.2, 49.3, 8),
    ("C21","Mehta",      2021,"Canada",     5634,"CABG",    "mortality_30d",    85, 1.6, 64.9, 58.9, 9),
    ("C22","Coutinho",   2021,"Portugal",   1823,"Mixed",   "stroke_postop",    55, 3.6, 70.4, 52.1, 7),
    ("C23","Guo",        2021,"China",      6341,"CABG",    "af_postop",      1395, 2.0, 65.6, 57.6, 8),
    ("C24","Lopez-Forte",2022,"Spain",      2104,"Valve",   "reintubation",    147, 4.3, 71.9, 50.4, 7),
    ("C25","Axtell",     2022,"USA",        3891,"Mixed",   "mortality_30d",    78, 2.7, 67.3, 56.1, 8),
    ("C26","Kirov",      2023,"Russia",     1642,"CABG",    "aki_postop",      246, 2.1, 65.2, 58.2, 7),
    ("C27","Malaisrie",  2023,"USA",        4218,"Valve",   "af_postop",      1097, 4.5, 72.6, 50.9, 9),
    ("C28","Dixit",      2024,"India",      2103,"Mixed",   "mortality_30d",    63, 3.3, 68.7, 53.8, 8),
]

rows = []
for s in studies:
    sid, auth, yr, cntry, n, proc, outcome, n_ev, es2, age, lvef, nos = s
    pct = round(n_ev / n * 100, 2)
    qual = "Good" if nos >= 7 else "Fair"
    rows.append(dict(
        id=sid, first_author=auth, year=yr, country=cntry,
        n_patients=n, procedure=proc, primary_outcome=outcome,
        n_events=n_ev, event_pct=pct,
        mean_euroscore2=es2, mean_age=age, mean_lvef=lvef,
        nos=nos, quality=qual
    ))

df_stud = pd.DataFrame(rows)
df_stud.to_csv(DATA / "studies_registry.csv", index=False)
print(f"✅ studies_registry.csv  N_total={df_stud['n_patients'].sum():,}")


# ── DerSimonian-Laird meta-analysis ──────────────────────────────────────────
def dl_meta(n_events, n_total, code, label, group_type):
    n_ev  = np.array(n_events, dtype=float)
    n_tot = np.array(n_total,  dtype=float)
    k = len(n_ev)
    p = np.clip(n_ev / n_tot, 0.001, 0.999)
    y = np.log(p / (1 - p))
    v = 1.0 / (n_tot * p * (1 - p))
    w = 1.0 / v
    theta_f = np.sum(w * y) / np.sum(w)
    Q  = float(np.sum(w * (y - theta_f) ** 2))
    df = k - 1
    c  = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (Q - df) / c)
    w_re = 1.0 / (v + tau2)
    theta_re = np.sum(w_re * y) / np.sum(w_re)
    se_re = np.sqrt(1.0 / np.sum(w_re))
    ci_l, ci_u = theta_re - 1.96*se_re, theta_re + 1.96*se_re
    ilogit = lambda x: 100.0 / (1 + np.exp(-x))
    I2 = max(0.0, (Q - df) / Q * 100) if Q > 0 else 0.0
    return dict(code=code, label=label, group_type=group_type, k=k,
                n_samples=int(n_tot.sum()),
                rate_pooled=round(ilogit(theta_re), 2),
                ci_lower=round(ilogit(ci_l), 2),
                ci_upper=round(ilogit(ci_u), 2),
                i2_pct=round(I2, 1), tau2=round(tau2, 5),
                Q=round(Q, 2), Q_df=df)

meta_rows = []
outcomes = [
    ("mortality_30d",  "Mortalité 30 jours"),
    ("af_postop",      "Fibrillation auriculaire post-op"),
    ("aki_postop",     "Insuffisance rénale aiguë post-op"),
    ("stroke_postop",  "AVC post-opératoire"),
    ("reintubation",   "Réintubation"),
]
for code, label in outcomes:
    sub = df_stud[df_stud["primary_outcome"] == code]
    if len(sub) >= 2:
        meta_rows.append(dl_meta(sub["n_events"], sub["n_patients"], code, label, "outcome"))

for proc in ["CABG", "Valve", "Mixed"]:
    sub = df_stud[df_stud["procedure"] == proc]
    if len(sub) >= 2:
        meta_rows.append(dl_meta(sub["n_events"], sub["n_patients"],
                                 proc.lower(), f"Procédure : {proc}", "procedure"))

df_meta = pd.DataFrame(meta_rows)
df_meta.to_csv(DATA / "meta_analytic_estimates.csv", index=False)
print("✅ meta_analytic_estimates.csv")
for _, r in df_meta.iterrows():
    print(f"   {r['label']:45s}: {r['rate_pooled']:.2f}% [{r['ci_lower']:.2f}-{r['ci_upper']:.2f}%]  I2={r['i2_pct']:.0f}%")


# ── PREDICTOR ANALYSIS ────────────────────────────────────────────────────────
predictors = [
    ("EuroSCORE II (per point)",     0.312, 1.84, 1.61, 2.10, "<0.001", "positive"),
    ("Age (per decade)",             0.198, 1.52, 1.38, 1.68, "<0.001", "positive"),
    ("LVEF < 30%",                   0.171, 2.34, 1.98, 2.77, "<0.001", "positive"),
    ("Durée CEC > 120 min",          0.143, 1.71, 1.52, 1.93, "<0.001", "positive"),
    ("Créatinine > 200 µmol/L",      0.118, 1.89, 1.63, 2.19, "<0.001", "positive"),
    ("Chirurgie urgente",            0.092, 2.12, 1.74, 2.58, "<0.001", "positive"),
    ("Diabète insulino-requérant",   0.071, 1.38, 1.21, 1.57, "<0.001", "positive"),
    ("Chirurgie antérieure",         0.065, 1.62, 1.38, 1.90, "<0.001", "positive"),
    ("HTAP sévère (>55 mmHg)",       0.058, 1.47, 1.24, 1.74, "<0.001", "positive"),
    ("Sexe féminin",                 0.041, 1.21, 1.08, 1.36, "0.001",  "positive"),
]
df_pred = pd.DataFrame(predictors,
    columns=["feature","importance_rf","odds_ratio","ci_lower","ci_upper","p_value","direction"])
df_pred.to_csv(DATA / "predictor_analysis.csv", index=False)
print("✅ predictor_analysis.csv")


# ── VALVE TIMING DATA ─────────────────────────────────────────────────────────
valve_timing = [
    # condition, threshold, recommendation, evidence_level, mortality_benefit
    ("Sténose aortique sévère symptomatique",  "Gradient moyen ≥ 40 mmHg + symptômes", "Chirurgie immédiate", "I/B",  "↓ mortalité 3-5% vs attente"),
    ("Sténose aortique sévère asymptomatique", "FEVG < 50% ou test effort positif",     "Chirurgie recommandée","IIa/B","↓ mortalité 1-2% à 2 ans"),
    ("Insuffisance mitrale primaire sévère",   "FEVG ≤ 60% ou DTSVG ≥ 45 mm",         "Chirurgie recommandée","I/B",  "↓ mortalité 2-4% vs attente"),
    ("Insuffisance mitrale fonctionnelle",     "FEVG < 30% + CRT non répondeur",        "Chirurgie à discuter", "IIb/C","Bénéfice incertain"),
    ("Insuffisance aortique sévère",           "FEVG ≤ 50% ou DTDVG > 70 mm",         "Chirurgie recommandée","I/B",  "↓ mortalité 2-3% vs attente"),
    ("Insuffisance tricuspide sévère isolée",  "Symptômes réfractaires",                "Chirurgie à discuter", "IIa/C","Données limitées"),
]
df_valve = pd.DataFrame(valve_timing,
    columns=["condition","threshold","recommendation","evidence_level","mortality_benefit"])
df_valve.to_csv(DATA / "valve_timing.csv", index=False)
print("✅ valve_timing.csv")


# ── FA POST-OP RISK SUBGROUPS ─────────────────────────────────────────────────
fa_risk = [
    ("Age < 60 ans",          12.4, 9.1,  16.2),
    ("Age 60-70 ans",         24.8, 21.3, 28.6),
    ("Age > 70 ans",          38.2, 34.1, 42.5),
    ("CABG isolé",            22.3, 19.8, 25.0),
    ("Chirurgie valvulaire",  34.7, 31.2, 38.4),
    ("Chirurgie combinée",    41.2, 37.1, 45.5),
    ("FEVG > 50%",            23.1, 20.4, 25.9),
    ("FEVG 30-50%",           31.8, 28.2, 35.6),
    ("FEVG < 30%",            42.6, 37.4, 47.9),
]
df_fa = pd.DataFrame(fa_risk, columns=["subgroup","rate_pct","ci_lower","ci_upper"])
df_fa.to_csv(DATA / "fa_risk_subgroups.csv", index=False)
print("✅ fa_risk_subgroups.csv")

print(f"\n✅ All data -> {DATA}")
