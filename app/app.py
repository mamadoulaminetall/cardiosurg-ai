"""CardioSurg AI — Clinical Decision Support Platform for Cardiac Surgery"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
from datetime import date

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSurg AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root { --bg: #0f172a; --card: #1e293b; --border: #334155; --accent: #3b82f6;
        --green: #10b981; --red: #ef4444; --yellow: #f59e0b; --text: #f1f5f9; --muted: #94a3b8; }
[data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid var(--border); }
h1,h2,h3 { color: var(--text) !important; }
.stButton>button { background: var(--accent); color: white; border: none;
                   border-radius: 8px; font-weight: 600; }
.stButton>button:hover { background: #2563eb; }
.metric-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
               padding: 16px; text-align: center; }
.risk-low    { color: #10b981; font-weight: 700; font-size: 1.6rem; }
.risk-medium { color: #f59e0b; font-weight: 700; font-size: 1.6rem; }
.risk-high   { color: #ef4444; font-weight: 700; font-size: 1.6rem; }
.risk-very-high { color: #dc2626; font-weight: 700; font-size: 1.6rem; }
.badge-green  { background:#064e3b; color:#6ee7b7; padding:3px 10px;
                border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-blue   { background:#1e3a5f; color:#93c5fd; padding:3px 10px;
                border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-orange { background:#78350f; color:#fcd34d; padding:3px 10px;
                border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-red    { background:#450a0a; color:#fca5a5; padding:3px 10px;
                border-radius:20px; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

DATA = Path(__file__).parent.parent / "data"

@st.cache_data
def load_data():
    return {
        "meta":  pd.read_csv(DATA / "meta_analytic_estimates.csv"),
        "pred":  pd.read_csv(DATA / "predictor_analysis.csv"),
        "valve": pd.read_csv(DATA / "valve_timing.csv"),
        "fa":    pd.read_csv(DATA / "fa_risk_subgroups.csv"),
        "stud":  pd.read_csv(DATA / "studies_registry.csv"),
    }

d = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 CardioSurg AI")
    st.markdown("<span class='badge-blue'>v1.0 · Avril 2026</span>", unsafe_allow_html=True)
    st.markdown("---")
    module = st.radio("Module", [
        "🧮 Calculateur de Risque",
        "📊 Prédicteur de Complications",
        "🔬 Timing Valvulaire",
        "📄 Compte-Rendu Opératoire",
    ])
    st.markdown("---")
    st.markdown("**Base de données**")
    st.metric("Études incluses", "28")
    st.metric("Patients totaux", "133 117")
    st.metric("Outcomes analysés", "5")
    st.markdown("---")
    st.markdown(
        "**Auteur :** Dr. Mamadou Lamine TALL, PhD  \n"
        "Research Engineer · Bioinformatics  \n"
        "**MedFlow AI** · Montpellier, France  \n"
        "[medflowailanding.streamlit.app](https://medflowailanding.streamlit.app)"
    )

# ════════════════════════════════════════════════════════════════════════════
# MODULE 1 — CALCULATEUR DE RISQUE
# ════════════════════════════════════════════════════════════════════════════
if module == "🧮 Calculateur de Risque":
    st.title("🧮 Calculateur de Risque Opératoire")
    st.markdown("**EuroSCORE II augmenté par ML** — méta-analyse 28 études, 133 117 patients")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Patient")
        age        = st.slider("Âge (années)", 30, 95, 67)
        sex_female = st.checkbox("Sexe féminin")
        diabetes   = st.checkbox("Diabète insulino-requérant")
        redo       = st.checkbox("Chirurgie cardiaque antérieure")
        urgent     = st.checkbox("Chirurgie urgente / émergente")
    with c2:
        st.markdown("#### Fonction cardiaque")
        lvef    = st.slider("FEVG (%)", 10, 75, 55)
        lvef_cat = "FEVG < 30%" if lvef < 30 else ("FEVG 30-50%" if lvef < 50 else "FEVG > 50%")
        creat   = st.slider("Créatininémie (µmol/L)", 50, 400, 90)
        pahp    = st.slider("PAPs (mmHg)", 15, 90, 30)
        euroscore = st.number_input("EuroSCORE II prédit (%)", 0.5, 25.0, 2.0, 0.1)
    with c3:
        st.markdown("#### Procédure")
        procedure = st.selectbox("Type de chirurgie", ["CABG isolé", "Chirurgie valvulaire", "Chirurgie combinée"])
        bypass_time = st.slider("Durée CEC estimée (min)", 40, 240, 90)

    # Risk calculation
    def compute_risk(age, lvef, creat, pahp, euroscore, bypass_time,
                     sex_female, diabetes, redo, urgent, procedure):
        # Logit baseline from meta-analysis
        baseline_logit = np.log(0.0212 / (1 - 0.0212))
        # Predictor coefficients from meta-analytic ORs
        logit = baseline_logit
        logit += 0.312 * (euroscore - 2.0) / 2.0
        logit += 0.198 * (age - 67) / 10
        if lvef < 30:  logit += np.log(2.34)
        elif lvef < 50: logit += np.log(1.45)
        if bypass_time > 120: logit += np.log(1.71)
        if creat > 200: logit += np.log(1.89)
        if urgent:   logit += np.log(2.12)
        if diabetes: logit += np.log(1.38)
        if redo:     logit += np.log(1.62)
        if pahp > 55: logit += np.log(1.47)
        if sex_female: logit += np.log(1.21)
        if procedure == "Chirurgie valvulaire": logit += 0.18
        elif procedure == "Chirurgie combinée": logit += 0.35
        p = 1 / (1 + np.exp(-logit)) * 100
        return round(p, 2)

    risk_mort = compute_risk(age, lvef, creat, pahp, euroscore, bypass_time,
                              sex_female, diabetes, redo, urgent, procedure)

    # AF risk
    af_base = 22.3 if "CABG" in procedure else (34.7 if "valvulaire" in procedure else 41.2)
    af_adj  = af_base
    if lvef < 30: af_adj *= 1.38
    elif lvef < 50: af_adj *= 1.18
    if age > 70: af_adj *= 1.28
    elif age > 60: af_adj *= 1.08
    af_risk = round(min(af_adj, 65), 1)

    # AKI risk
    aki_base = 15.85
    if creat > 200: aki_base *= 1.89
    elif creat > 130: aki_base *= 1.35
    if bypass_time > 120: aki_base *= 1.45
    aki_risk = round(min(aki_base, 55), 1)

    st.markdown("---")
    st.markdown("### Résultats du Calcul de Risque")

    def risk_color(r, thresholds, classes):
        for t, c in zip(thresholds, classes):
            if r < t: return c
        return classes[-1]

    def risk_badge(r, thresholds, labels):
        for t, l in zip(thresholds, labels):
            if r < t: return l
        return labels[-1]

    mort_class = risk_color(risk_mort, [2, 5, 10], ["risk-low","risk-medium","risk-high","risk-very-high"])
    mort_label = risk_badge(risk_mort, [2, 5, 10], ["Faible","Modéré","Élevé","Très élevé"])
    af_class   = risk_color(af_risk, [20, 30, 40], ["risk-low","risk-medium","risk-high","risk-very-high"])
    aki_class  = risk_color(aki_risk, [10, 20, 35], ["risk-low","risk-medium","risk-high","risk-very-high"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div style='color:#94a3b8;font-size:0.9rem'>Mortalité 30 jours</div>
            <div class='{mort_class}'>{risk_mort}%</div>
            <div style='color:#94a3b8;font-size:0.8rem'>{mort_label}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div style='color:#94a3b8;font-size:0.9rem'>FA post-opératoire</div>
            <div class='{af_class}'>{af_risk}%</div>
            <div style='color:#94a3b8;font-size:0.8rem'>Probabilité estimée</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div style='color:#94a3b8;font-size:0.9rem'>IRA post-opératoire</div>
            <div class='{aki_class}'>{aki_risk}%</div>
            <div style='color:#94a3b8;font-size:0.8rem'>Probabilité estimée</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'>
            <div style='color:#94a3b8;font-size:0.9rem'>EuroSCORE II</div>
            <div class='risk-medium'>{euroscore}%</div>
            <div style='color:#94a3b8;font-size:0.8rem'>Référence validée</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    with st.expander("📊 Contexte méta-analytique (28 études, 133 117 patients)"):
        st.markdown("""
| Complication | Taux poolé | IC 95% | I² |
|---|---|---|---|
| Mortalité 30 jours | 2.12% | 1.88–2.39% | 80% |
| Fibrillation auriculaire | 24.4% | 23.3–25.5% | 81% |
| Insuffisance rénale aiguë | 15.9% | 14.1–17.8% | 84% |
| AVC post-opératoire | 2.47% | 2.01–3.04% | 59% |
| Réintubation | 6.27% | 5.71–6.88% | 24% |
        """)

# ════════════════════════════════════════════════════════════════════════════
# MODULE 2 — PRÉDICTEUR DE COMPLICATIONS
# ════════════════════════════════════════════════════════════════════════════
elif module == "📊 Prédicteur de Complications":
    st.title("📊 Prédicteur Multi-Complications")
    st.markdown("**Probabilités individuelles basées sur méta-régression** — 10 prédicteurs validés")
    st.markdown("---")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Facteurs de risque")
        euroscore2 = st.slider("EuroSCORE II (%)", 0.5, 20.0, 2.0, 0.5)
        age_val    = st.slider("Âge (années)", 40, 90, 67)
        lvef_val   = st.slider("FEVG (%)", 10, 70, 55)
        creat_val  = st.slider("Créatinine (µmol/L)", 50, 350, 90)
        pahp_val   = st.slider("PAPs (mmHg)", 15, 80, 30)
    with c2:
        st.markdown("#### Comorbidités & Procédure")
        proc_val   = st.selectbox("Procédure", ["CABG", "Valve", "Combiné"])
        cec_val    = st.slider("Durée CEC (min)", 40, 240, 90)
        urgent_val = st.checkbox("Urgence")
        diabetes_v = st.checkbox("Diabète insulino-requérant")
        redo_val   = st.checkbox("Réintervention")
        sex_val    = st.checkbox("Sexe féminin")

    # Compute all 5 outcomes
    def logit_risk(baseline_rate, or_dict):
        bl = np.log(baseline_rate / (1 - baseline_rate))
        for key, (condition, log_or) in or_dict.items():
            if condition:
                bl += log_or
        return 1 / (1 + np.exp(-bl)) * 100

    mort_risk = logit_risk(0.0212, {
        "es":  (True,          0.312 * (euroscore2 - 2) / 2),
        "age": (True,          0.198 * (age_val - 67) / 10),
        "lvef":(lvef_val < 30, np.log(2.34)),
        "cec": (cec_val > 120, np.log(1.71)),
        "cr":  (creat_val > 200, np.log(1.89)),
        "urg": (urgent_val,    np.log(2.12)),
        "db":  (diabetes_v,    np.log(1.38)),
        "redo":(redo_val,      np.log(1.62)),
        "pahp":(pahp_val > 55, np.log(1.47)),
        "sex": (sex_val,       np.log(1.21)),
        "proc":(proc_val != "CABG", 0.25 if proc_val == "Valve" else 0.45),
    })

    af_base_r = {"CABG": 0.223, "Valve": 0.347, "Combiné": 0.412}[proc_val]
    af_risk_p = logit_risk(af_base_r, {
        "age": (True,          0.18 * (age_val - 67) / 10),
        "lvef":(lvef_val < 30, np.log(1.65)),
        "cec": (cec_val > 120, np.log(1.28)),
    })

    aki_risk_p = logit_risk(0.1585, {
        "cr":  (creat_val > 200, np.log(2.1)),
        "cr2": (creat_val > 130 and creat_val <= 200, np.log(1.45)),
        "cec": (cec_val > 120, np.log(1.52)),
        "lvef":(lvef_val < 30, np.log(1.38)),
        "es":  (True,          0.15 * (euroscore2 - 2) / 2),
    })

    stroke_risk = logit_risk(0.0247, {
        "age": (True,          0.22 * (age_val - 67) / 10),
        "redo":(redo_val,      np.log(1.55)),
        "pahp":(pahp_val > 55, np.log(1.32)),
        "proc":(proc_val != "CABG", 0.28),
    })

    reintub_risk = logit_risk(0.0627, {
        "lvef":(lvef_val < 30, np.log(2.1)),
        "cec": (cec_val > 120, np.log(1.45)),
        "urg": (urgent_val,    np.log(1.68)),
        "es":  (True,          0.18 * (euroscore2 - 2) / 2),
    })

    outcomes = [
        ("Mortalité 30 jours", round(mort_risk, 2), [2, 5, 10], "#ef4444"),
        ("FA post-opératoire", round(af_risk_p, 1), [20, 30, 45], "#f59e0b"),
        ("IRA post-opératoire", round(aki_risk_p, 1), [10, 20, 35], "#3b82f6"),
        ("AVC post-opératoire", round(stroke_risk, 2), [2, 4, 8], "#a855f7"),
        ("Réintubation", round(reintub_risk, 1), [5, 10, 20], "#10b981"),
    ]

    st.markdown("---")
    st.markdown("### Profil de Risque Individuel")

    cols = st.columns(5)
    for col, (name, val, thr, color) in zip(cols, outcomes):
        with col:
            risk_l = "Faible" if val < thr[0] else ("Modéré" if val < thr[1] else ("Élevé" if val < thr[2] else "Très élevé"))
            st.markdown(f"""<div class='metric-card'>
                <div style='color:#94a3b8;font-size:0.78rem;margin-bottom:6px'>{name}</div>
                <div style='color:{color};font-size:1.8rem;font-weight:700'>{val}%</div>
                <div style='color:#94a3b8;font-size:0.78rem'>{risk_l}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    # Predictor contribution chart
    st.markdown("### Contribution des Prédicteurs (Mortalité 30j)")
    preds = d["pred"].copy()
    preds = preds.sort_values("importance_rf", ascending=True)
    bar_colors = ["#ef4444" if row["direction"] == "positive" else "#10b981"
                  for _, row in preds.iterrows()]

    import plotly.graph_objects as go
    fig_pred = go.Figure(go.Bar(
        x=preds["importance_rf"],
        y=preds["feature"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"OR={r['odds_ratio']:.2f}" for _, r in preds.iterrows()],
        textposition="outside",
    ))
    fig_pred.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font_color="#94a3b8", height=350,
        xaxis_title="Importance RF",
        margin=dict(l=10, r=80, t=10, b=10),
    )
    st.plotly_chart(fig_pred, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# MODULE 3 — TIMING VALVULAIRE
# ════════════════════════════════════════════════════════════════════════════
elif module == "🔬 Timing Valvulaire":
    st.title("🔬 Aide à la Décision — Timing Chirurgical Valvulaire")
    st.markdown("**Recommandations ESC 2024** — 6 pathologies valvulaires")
    st.markdown("---")

    condition = st.selectbox("Pathologie valvulaire", d["valve"]["condition"].tolist())
    row = d["valve"][d["valve"]["condition"] == condition].iloc[0]

    ev_colors = {"I/B": "#10b981", "IIa/B": "#3b82f6", "IIa/C": "#6366f1", "IIb/C": "#f59e0b"}
    ev_color = ev_colors.get(row["evidence_level"], "#94a3b8")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
<div class='metric-card' style='margin-bottom:16px'>
    <div style='color:#94a3b8;font-size:0.85rem;margin-bottom:4px'>Condition</div>
    <div style='color:#f1f5f9;font-size:1.05rem;font-weight:600'>{row['condition']}</div>
</div>
<div class='metric-card' style='margin-bottom:16px'>
    <div style='color:#94a3b8;font-size:0.85rem;margin-bottom:4px'>Seuil d'intervention</div>
    <div style='color:#fbbf24;font-size:1rem'>{row['threshold']}</div>
</div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
<div class='metric-card' style='margin-bottom:16px;border-color:{ev_color}'>
    <div style='color:#94a3b8;font-size:0.85rem;margin-bottom:4px'>Recommandation</div>
    <div style='color:{ev_color};font-size:1.2rem;font-weight:700'>{row['recommendation']}</div>
</div>
<div class='metric-card' style='margin-bottom:16px'>
    <div style='color:#94a3b8;font-size:0.85rem;margin-bottom:4px'>Niveau d'évidence</div>
    <div style='color:{ev_color};font-size:1.5rem;font-weight:700'>{row['evidence_level']}</div>
</div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
<div class='metric-card' style='margin-top:8px'>
    <div style='color:#94a3b8;font-size:0.85rem;margin-bottom:4px'>Bénéfice mortalité attendu</div>
    <div style='color:#10b981;font-size:1.1rem'>{row['mortality_benefit']}</div>
</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Tableau complet — Toutes les pathologies")
    st.dataframe(
        d["valve"][["condition","recommendation","evidence_level","mortality_benefit"]],
        use_container_width=True, hide_index=True
    )

    st.markdown("")
    with st.expander("ℹ️ Classification ESC — Niveaux d'évidence"):
        st.markdown("""
| Classe | Signification | Couleur |
|---|---|---|
| **I/B** | Recommandation forte — bénéfice bien établi | 🟢 Vert |
| **IIa/B** | Recommandation probable — consensus en faveur | 🔵 Bleu |
| **IIa/C** | Recommandation probable — données limitées | 🟣 Violet |
| **IIb/C** | Bénéfice incertain — avis d'expert | 🟡 Jaune |
        """)

# ════════════════════════════════════════════════════════════════════════════
# MODULE 4 — COMPTE-RENDU OPÉRATOIRE
# ════════════════════════════════════════════════════════════════════════════
elif module == "📄 Compte-Rendu Opératoire":
    st.title("📄 Générateur de Compte-Rendu Opératoire")
    st.markdown("**Génération automatique PDF** — Chirurgie cardiaque")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Identité Patient")
        pat_nom    = st.text_input("Nom", "DUPONT")
        pat_prenom = st.text_input("Prénom", "Jean")
        pat_ddn    = st.date_input("Date de naissance", date(1955, 3, 15))
        pat_ipp    = st.text_input("IPP / N° dossier", "2026-04578")
        service    = st.text_input("Service", "Chirurgie Cardiaque — CHU Montpellier")
    with c2:
        st.markdown("#### Intervention")
        op_date    = st.date_input("Date d'intervention", date.today())
        chirurgien = st.text_input("Chirurgien principal", "Pr. Martin")
        anesthesiste = st.text_input("Anesthésiste", "Dr. Bernard")
        procedure_cr = st.selectbox("Procédure principale", [
            "CABG monotronculaire", "CABG bitronculaire", "CABG tritronculaire",
            "Remplacement valvulaire aortique (bioprothèse)",
            "Remplacement valvulaire aortique (mécanique)",
            "Plastie mitrale", "Remplacement valvulaire mitral",
            "Chirurgie combinée CABG + valvulaire",
            "Remplacement aortique (Bentall)",
        ])
        voie_abord = st.selectbox("Voie d'abord", [
            "Sternotomie médiane", "Thoracotomie antérolatérale droite",
            "Ministernotomie en J", "Ministernotomie en T inversé",
        ])

    st.markdown("#### Perfusion & CEC")
    c3, c4, c5 = st.columns(3)
    with c3:
        cec_dur     = st.number_input("Durée CEC (min)", 30, 300, 90)
        clamp_dur   = st.number_input("Durée clampage aortique (min)", 20, 200, 60)
    with c4:
        cardiopleg  = st.selectbox("Cardioplégie", ["Custodiol HTK", "Sang froid", "Bretschneider", "Calafiore"])
        perfusionniste = st.text_input("Perfusionniste", "M. Lefebvre")
    with c5:
        saignement  = st.number_input("Saignement estimé (mL)", 100, 3000, 400)
        transfusion = st.number_input("CGR transfusés", 0, 10, 0)

    st.markdown("#### Déroulé opératoire")
    incidents    = st.text_area("Incidents / particularités", "RAS")
    suites_imm   = st.text_area("Suites immédiates", "Extubation en salle de réveil. Hémodynamique stable.")
    recommandations = st.text_area("Recommandations post-op", "Surveillance télémétrie 48h. Anticoagulation curative J1.")

    st.markdown("---")
    if st.button("📄 Générer le Compte-Rendu PDF"):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
            from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

            buf = BytesIO()
            doc_cr = SimpleDocTemplate(buf, pagesize=A4,
                leftMargin=2.5*cm, rightMargin=2.5*cm,
                topMargin=2.5*cm, bottomMargin=2.5*cm)
            DARK_C = colors.HexColor("#1e3a5f")
            GRAY_C = colors.HexColor("#6b7280")
            TEXT_C = colors.HexColor("#1f2937")
            BORD_C = colors.HexColor("#e2e8f0")

            ss = getSampleStyleSheet()
            def Sc(n="Normal", **kw):
                return ParagraphStyle(n+str(id(kw)), parent=ss.get(n,ss["Normal"]), **kw)

            TIT = Sc("Title", fontSize=13, leading=17, textColor=DARK_C, alignment=TA_CENTER, spaceAfter=3)
            H1c = Sc("Heading1", fontSize=10, leading=13, textColor=DARK_C,
                     fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3)
            BOD = Sc("Normal", fontSize=9, leading=13, textColor=TEXT_C, spaceAfter=3, alignment=TA_JUSTIFY)
            SM  = Sc("Normal", fontSize=8, leading=11, textColor=GRAY_C, spaceAfter=2)

            def hr_c(): return HRFlowable(width="100%", thickness=0.4, color=BORD_C, spaceAfter=4)
            def row2(label, val):
                return [Paragraph(f"<b>{label}</b>", SM), Paragraph(str(val), SM)]

            stry = []
            stry.append(Paragraph("COMPTE-RENDU OPÉRATOIRE", TIT))
            stry.append(Paragraph("Chirurgie Cardiaque", Sc("Normal", fontSize=10,
                alignment=TA_CENTER, textColor=GRAY_C, spaceAfter=2)))
            stry.append(Paragraph(service, Sc("Normal", fontSize=9,
                alignment=TA_CENTER, textColor=GRAY_C, spaceAfter=6)))
            stry.append(hr_c())

            # Patient identity table
            stry.append(Paragraph("IDENTITÉ PATIENT", H1c))
            data_id = [
                ["Nom — Prénom", f"{pat_nom} {pat_prenom}", "Date naissance", str(pat_ddn)],
                ["IPP / Dossier", pat_ipp, "Date intervention", str(op_date)],
                ["Chirurgien", chirurgien, "Anesthésiste", anesthesiste],
            ]
            ts_id = TableStyle([
                ("BACKGROUND", (0,0),(0,-1), colors.HexColor("#f8fafc")),
                ("BACKGROUND", (2,0),(2,-1), colors.HexColor("#f8fafc")),
                ("FONTNAME",   (0,0),(0,-1), "Helvetica-Bold"),
                ("FONTNAME",   (2,0),(2,-1), "Helvetica-Bold"),
                ("FONTSIZE",   (0,0),(-1,-1), 8.5),
                ("LEADING",    (0,0),(-1,-1), 11),
                ("GRID",       (0,0),(-1,-1), 0.4, BORD_C),
                ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
                ("TOPPADDING", (0,0),(-1,-1), 4),
                ("BOTTOMPADDING",(0,0),(-1,-1), 4),
            ])
            stry.append(Table(data_id, colWidths=[3.5*cm,6*cm,3.5*cm,4*cm], style=ts_id))
            stry.append(Spacer(1, 0.3*cm))

            # Intervention
            stry.append(Paragraph("INTERVENTION", H1c))
            stry.append(Paragraph(f"<b>Procédure :</b> {procedure_cr}", BOD))
            stry.append(Paragraph(f"<b>Voie d'abord :</b> {voie_abord}", BOD))

            # CEC table
            stry.append(Paragraph("CIRCULATION EXTRACORPORELLE", H1c))
            data_cec = [
                [Paragraph("<b>Durée CEC</b>", SM), Paragraph(f"{cec_dur} min", SM),
                 Paragraph("<b>Clampage aortique</b>", SM), Paragraph(f"{clamp_dur} min", SM)],
                [Paragraph("<b>Cardioplégie</b>", SM), Paragraph(cardiopleg, SM),
                 Paragraph("<b>Perfusionniste</b>", SM), Paragraph(perfusionniste, SM)],
                [Paragraph("<b>Saignement estimé</b>", SM), Paragraph(f"{saignement} mL", SM),
                 Paragraph("<b>CGR transfusés</b>", SM), Paragraph(f"{transfusion}", SM)],
            ]
            ts_cec = TableStyle([
                ("BACKGROUND",(0,0),(0,-1), colors.HexColor("#f8fafc")),
                ("BACKGROUND",(2,0),(2,-1), colors.HexColor("#f8fafc")),
                ("FONTNAME",  (0,0),(0,-1), "Helvetica-Bold"),
                ("FONTNAME",  (2,0),(2,-1), "Helvetica-Bold"),
                ("FONTSIZE",  (0,0),(-1,-1), 8.5),
                ("LEADING",   (0,0),(-1,-1), 11),
                ("GRID",      (0,0),(-1,-1), 0.4, BORD_C),
                ("VALIGN",    (0,0),(-1,-1), "MIDDLE"),
                ("TOPPADDING",(0,0),(-1,-1), 4),
                ("BOTTOMPADDING",(0,0),(-1,-1), 4),
            ])
            stry.append(Table(data_cec, colWidths=[3.5*cm,6*cm,3.5*cm,4*cm], style=ts_cec))
            stry.append(Spacer(1, 0.3*cm))

            stry.append(Paragraph("DÉROULÉ OPÉRATOIRE", H1c))
            stry.append(Paragraph(incidents if incidents else "RAS", BOD))
            stry.append(Paragraph("SUITES IMMÉDIATES", H1c))
            stry.append(Paragraph(suites_imm, BOD))
            stry.append(Paragraph("RECOMMANDATIONS POST-OPÉRATOIRES", H1c))
            stry.append(Paragraph(recommandations, BOD))

            stry.append(Spacer(1, 1*cm))
            stry.append(hr_c())
            stry.append(Paragraph(
                f"Signé électroniquement par {chirurgien} le {op_date} — Généré via CardioSurg AI / MedFlow AI",
                Sc("Normal", fontSize=7.5, textColor=GRAY_C, alignment=TA_CENTER)))

            doc_cr.build(stry)
            buf.seek(0)

            filename = f"CR_{pat_nom}_{pat_prenom}_{op_date}.pdf"
            st.success(f"✅ Compte-rendu généré — {filename}")
            st.download_button(
                label="⬇️ Télécharger le PDF",
                data=buf,
                file_name=filename,
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Erreur lors de la génération : {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.8rem'>"
    "CardioSurg AI v1.0 · MedFlow AI · Mamadou Lamine TALL, PhD · Avril 2026<br>"
    "Usage clinique en aide à la décision uniquement — ne remplace pas le jugement médical"
    "</div>", unsafe_allow_html=True
)
