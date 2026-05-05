"""CardioSurg AI — Manuscript + Supplementary PDF (ReportLab)"""

from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

DATA = Path(__file__).parent.parent / "data"
FIGS = Path(__file__).parent.parent / "figures"
OUT  = Path(__file__).parent.parent / "manuscript"
OUT.mkdir(exist_ok=True)

df_stud = pd.read_csv(DATA / "studies_registry.csv")
df_meta = pd.read_csv(DATA / "meta_analytic_estimates.csv")
df_pred = pd.read_csv(DATA / "predictor_analysis.csv")
df_fa   = pd.read_csv(DATA / "fa_risk_subgroups.csv")
df_vt   = pd.read_csv(DATA / "valve_timing.csv")

DARK = colors.HexColor("#1e3a5f")
GRAY = colors.HexColor("#6b7280")
TEXT = colors.HexColor("#1f2937")
BORD = colors.HexColor("#e2e8f0")
LITE = colors.HexColor("#f8fafc")

styles = getSampleStyleSheet()
def S(name="Normal", **kw):
    return ParagraphStyle(name + str(id(kw)), parent=styles.get(name, styles["Normal"]), **kw)

TITLE  = S("Title",    fontSize=14, leading=18, textColor=DARK, alignment=TA_CENTER, spaceAfter=4)
TITLE2 = S("Normal",  fontSize=11, leading=14, textColor=DARK, alignment=TA_CENTER, spaceAfter=6)
H1     = S("Heading1", fontSize=11, leading=14, textColor=DARK, fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
H2     = S("Heading2", fontSize=10, leading=13, textColor=DARK, fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=3)
BODY   = S("Normal",  fontSize=9.5, leading=14, textColor=TEXT, spaceAfter=5, alignment=TA_JUSTIFY)
SMALL  = S("Normal",  fontSize=8,   leading=11, textColor=GRAY, spaceAfter=3)
CAP    = S("Normal",  fontSize=8.5, leading=12, textColor=GRAY, alignment=TA_CENTER,
           fontName="Helvetica-Oblique", spaceAfter=8)
AUTH   = S("Normal",  fontSize=9,   leading=13, textColor=GRAY, alignment=TA_CENTER, spaceAfter=3)

def hr(): return HRFlowable(width="100%", thickness=0.5, color=BORD, spaceAfter=6)

def stab(data, widths, font_size=8):
    ts = TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), DARK),
        ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), font_size),
        ("LEADING",       (0,0),(-1,-1), font_size + 2),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [LITE, colors.white]),
        ("GRID",          (0,0),(-1,-1), 0.4, BORD),
        ("TOPPADDING",    (0,0),(-1,-1), 3),
        ("BOTTOMPADDING", (0,0),(-1,-1), 3),
    ])
    return Table(data, colWidths=[w*cm for w in widths], style=ts, repeatRows=1)

def fig(fname, caption, w=14):
    p = FIGS / fname
    elems = []
    if p.exists():
        elems.append(Image(str(p), width=w*cm, height=w*cm*0.55))
        elems.append(Paragraph(caption, CAP))
    return elems

# ════════════════════════════════════════════════════════════════════════════
# MAIN MANUSCRIPT
# ════════════════════════════════════════════════════════════════════════════
doc = SimpleDocTemplate(str(OUT / "CardioSurgAI_Manuscript.pdf"),
    pagesize=A4, leftMargin=2.5*cm, rightMargin=2.5*cm,
    topMargin=2.5*cm, bottomMargin=2.5*cm)
story = []

# Title
story.append(Paragraph(
    "Predicting Postoperative Complications in Cardiac Surgery:<br/>"
    "A Systematic Review, Meta-Analysis of 28 Studies (133,117 Patients),<br/>"
    "and Machine Learning Decision-Support Platform", TITLE))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("Mamadou Lamine TALL, PhD", AUTH))
story.append(Paragraph("Aix Marseille Univ, IRD, MEPHI, APHM, IHU-Méditerranée Infection, Marseille, France | MedFlow AI", AUTH))
story.append(Paragraph("mamadoulaminetallgithub@gmail.com", AUTH))
story.append(Paragraph("Submitted: April 2026 | medRxiv Preprint", AUTH))
story.append(Spacer(1, 0.4*cm))
story.append(hr())

# Abstract
story.append(Paragraph("ABSTRACT", H1))
story.append(Paragraph(
    "<b>Background.</b> Cardiac surgery carries significant risk of postoperative complications, "
    "including 30-day mortality, atrial fibrillation (AF), acute kidney injury (AKI), stroke, and "
    "reintubation. EuroSCORE II remains the standard risk stratification tool, but its discriminative "
    "performance is limited and no unified clinical decision-support platform exists for French "
    "cardiac surgery units.", BODY))
story.append(Paragraph(
    "<b>Methods.</b> We conducted a systematic review and meta-analysis of 28 observational cohort "
    "studies published between 2012 and 2024, totalling 133,117 patients. DerSimonian-Laird "
    "random-effects models with logit-transformed proportions were used to pool event rates. "
    "Heterogeneity was assessed via I² and Cochran's Q. Predictors were derived from random forest "
    "feature importance and pooled odds ratios via meta-regression.", BODY))
story.append(Paragraph(
    "<b>Results.</b> Pooled 30-day mortality was 2.12% (95% CI: 1.88–2.39%; I²=80%). "
    "AF occurred in 24.4% of patients (I²=81%), AKI in 15.9% (I²=84%), stroke in 2.5% (I²=59%), "
    "and reintubation in 6.3% (I²=24%). EuroSCORE II (OR=1.84 per point), age (OR=1.52/decade), "
    "and LVEF<30% (OR=2.34) were the strongest independent predictors. Valve surgery carried the "
    "highest complication burden. A machine learning decision-support platform (CardioSurg AI) was "
    "developed integrating these findings.", BODY))
story.append(Paragraph(
    "<b>Conclusions.</b> Postoperative complication rates in cardiac surgery remain substantial and "
    "highly heterogeneous. CardioSurg AI provides the first unified SaaS platform combining "
    "EuroSCORE II augmentation, ML-based complication prediction, valve timing decision support, "
    "and operative report generation for French cardiac surgery teams.", BODY))
story.append(Paragraph(
    "<b>Keywords:</b> cardiac surgery, EuroSCORE II, meta-analysis, machine learning, atrial "
    "fibrillation, postoperative complications, clinical decision support", SMALL))
story.append(hr())

# Introduction
story.append(Paragraph("1. INTRODUCTION", H1))
story.append(Paragraph(
    "Cardiac surgery encompasses a broad spectrum of procedures—coronary artery bypass grafting "
    "(CABG), valve replacement and repair, combined procedures—each carrying distinct risk profiles. "
    "In France alone, approximately 45,000 cardiac surgery procedures are performed annually, with "
    "postoperative complication rates that significantly impact patient outcomes, length of stay, "
    "and healthcare costs.", BODY))
story.append(Paragraph(
    "EuroSCORE II, developed by Nashef et al. in 2012 from a pan-European registry of 22,381 "
    "patients, remains the gold standard for preoperative risk stratification. However, its "
    "discriminative ability has been questioned in contemporary cohorts, particularly for specific "
    "complication endpoints beyond 30-day mortality. Moreover, no platform currently integrates "
    "multi-outcome prediction, valve timing decision support, and automated operative report "
    "generation in a single clinical tool.", BODY))
story.append(Paragraph(
    "We conducted a comprehensive meta-analysis to quantify pooled rates of five major "
    "postoperative complications across 28 international cohorts, identify independent predictors, "
    "and develop CardioSurg AI—a Streamlit-based clinical decision-support platform targeting "
    "cardiac surgery units in French-speaking countries.", BODY))

# Methods
story.append(Paragraph("2. METHODS", H1))
story.append(Paragraph("2.1 Search Strategy and Study Selection", H2))
story.append(Paragraph(
    "We searched PubMed, Embase, and Web of Science for studies published between January 2012 "
    "and December 2024 reporting complication rates after adult cardiac surgery. Inclusion criteria: "
    "(1) adult patients (≥18 years); (2) CABG, valve, or combined procedures; (3) reporting at "
    "least one of five outcomes (30-day mortality, AF, AKI, stroke, reintubation); (4) ≥500 "
    "patients; (5) Newcastle-Ottawa Scale (NOS) score ≥7. Studies reporting only paediatric or "
    "transcatheter procedures were excluded.", BODY))
story.append(Paragraph("2.2 Statistical Analysis", H2))
story.append(Paragraph(
    "Event rates were pooled using DerSimonian-Laird random-effects models on logit-transformed "
    "proportions. Back-transformation used the inverse logit function. Heterogeneity was quantified "
    "by I² and Cochran's Q statistic. Subgroup analyses were performed by procedure type (CABG, "
    "valve, mixed). Predictor analysis combined random forest feature importance scores (500 trees, "
    "max depth=6) with pooled odds ratios from multivariate models reported in ≥3 studies. "
    "All analyses were performed in Python 3.11 (NumPy, pandas, scikit-learn).", BODY))
story.append(Paragraph("2.3 CardioSurg AI Platform", H2))
story.append(Paragraph(
    "The CardioSurg AI platform was developed using Streamlit (v1.32+) and integrates four modules: "
    "(1) EuroSCORE II + ML risk calculator with calibrated probability estimates; "
    "(2) multi-outcome complication predictor using meta-analytic priors; "
    "(3) valve timing decision support based on ESC 2024 guidelines; and "
    "(4) automated PDF operative report generator via ReportLab.", BODY))

# Results
story.append(Paragraph("3. RESULTS", H1))
story.append(Paragraph("3.1 Study Characteristics", H2))
story.append(Paragraph(
    f"Twenty-eight studies comprising 133,117 patients were included (Table 1). Studies were "
    f"conducted across 13 countries (2012–2024). Mean patient age ranged from 64.8 to 73.2 years; "
    f"mean LVEF from 49.3% to 58.9%; mean EuroSCORE II from 1.5% to 4.8%. Procedures included "
    f"CABG (n=10 studies), valve surgery (n=9), and mixed (n=9). NOS scores ranged from 7 to 9 "
    f"(all studies rated 'Good' quality).", BODY))

story.append(Paragraph("3.2 Pooled Complication Rates", H2))
meta_out = df_meta[df_meta["group_type"] == "outcome"]
for _, r in meta_out.iterrows():
    story.append(Paragraph(
        f"<b>{r['label']}:</b> pooled rate {r['rate_pooled']:.2f}% "
        f"(95% CI: {r['ci_lower']:.2f}–{r['ci_upper']:.2f}%; I²={r['i2_pct']:.0f}%; "
        f"k={r['k']} studies; N={r['n_samples']:,}).", BODY))

story.append(Paragraph("3.3 Predictors of Complications", H2))
story.append(Paragraph(
    "The strongest predictor of postoperative complications was LVEF<30% (OR=2.34, 95% CI: "
    "1.98–2.77), followed by emergency surgery (OR=2.12), EuroSCORE II per point (OR=1.84), "
    "creatinine >200 µmol/L (OR=1.89), and age per decade (OR=1.52). All predictors reached "
    "p<0.001 in meta-regression. Random forest feature importance ranked EuroSCORE II as the "
    "most informative single predictor (importance=0.312).", BODY))

# Figures
story.append(Paragraph("3.4 Figures", H2))
for fname, cap in [
    ("fig1_forest_plot.png",
     "Figure 1. Forest plot of postoperative complication rates across 28 studies "
     "(DerSimonian-Laird). Circle size proportional to sample size. Diamonds = pooled estimates per outcome."),
    ("fig2_pooled_rates.png",
     "Figure 2. Pooled event rates (95% CI) by complication type (left) and procedure type (right). "
     "I² and study count shown for each estimate."),
    ("fig3_predictor_analysis.png",
     "Figure 3. Predictors of postoperative complications. Left: random forest feature importance. "
     "Right: pooled odds ratios with 95% CI from meta-regression."),
    ("fig4_fa_subgroups.png",
     "Figure 4. Atrial fibrillation incidence by patient subgroup (age, procedure type, LVEF). "
     "Horizontal lines = 95% CI."),
    ("fig5_model_performance.png",
     "Figure 5. Model performance. Left: EuroSCORE II calibration by predicted risk decile. "
     "Right: ROC curves for 30-day mortality by procedure type."),
]:
    story += fig(fname, cap)
    story.append(Spacer(1, 0.3*cm))

# Discussion
story.append(Paragraph("4. DISCUSSION", H1))
story.append(Paragraph(
    "This meta-analysis provides the most comprehensive synthesis of postoperative complication "
    "rates in contemporary cardiac surgery, pooling 133,117 patients across 28 international "
    "cohorts. Our pooled 30-day mortality of 2.12% aligns with recent European registry data and "
    "confirms substantial heterogeneity (I²=80%), reflecting differences in case mix, centre "
    "volume, and era effects.", BODY))
story.append(Paragraph(
    "The high incidence of postoperative AF (24.4%) underscores its importance as a primary "
    "prevention target, particularly in patients undergoing valve surgery (34.7%) and those with "
    "reduced LVEF (42.6% for LVEF<30%). Current prophylactic strategies (amiodarone, beta-blockers, "
    "posterior pericardiotomy) reduce AF risk by 30–50% but are inconsistently applied.", BODY))
story.append(Paragraph(
    "CardioSurg AI addresses a critical gap in clinical practice: no unified digital tool exists "
    "for French cardiac surgery teams to simultaneously assess multi-outcome risk, time valve "
    "interventions per ESC 2024 guidelines, and generate standardised operative reports. "
    "The platform's ML component augments EuroSCORE II with patient-specific complication "
    "probabilities, enabling more nuanced preoperative counselling.", BODY))

# Limitations
story.append(Paragraph("4.1 Limitations", H2))
story.append(Paragraph(
    "All included studies are observational; residual confounding cannot be excluded. "
    "High heterogeneity for mortality and AF outcomes (I²>80%) limits the precision of pooled "
    "estimates. The ML predictor relies on aggregate-level meta-analytic coefficients rather than "
    "individual patient data, which will be incorporated in future platform versions as real-world "
    "data are collected.", BODY))

# Conclusion
story.append(Paragraph("5. CONCLUSION", H1))
story.append(Paragraph(
    "Postoperative complications in cardiac surgery remain a major clinical burden, with AF "
    "affecting one in four patients and mortality varying substantially by procedure type and "
    "patient profile. CardioSurg AI provides the first integrated, evidence-based decision-support "
    "platform for cardiac surgery teams, combining meta-analytic risk estimates, ML augmentation "
    "of EuroSCORE II, ESC 2024 valve timing guidance, and automated report generation. "
    "Prospective validation in French cardiac surgery centres is planned.", BODY))

story.append(hr())
story.append(Paragraph("CONFLICTS OF INTEREST", H2))
story.append(Paragraph("The author declares no conflicts of interest.", BODY))
story.append(Paragraph("FUNDING", H2))
story.append(Paragraph("No funding was received for this work.", BODY))
story.append(Paragraph("DATA AVAILABILITY", H2))
story.append(Paragraph(
    "All data, analysis scripts, and the CardioSurg AI platform are available at "
    "github.com/mamadoulaminetall under CC BY 4.0 license.", BODY))

story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY
# ════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("SUPPLEMENTARY DATA", TITLE))
story.append(Paragraph(
    "Predicting Postoperative Complications in Cardiac Surgery — CardioSurg AI", TITLE2))
story.append(Paragraph("Mamadou Lamine TALL, PhD · MedFlow AI · April 2026",
    S("Normal", fontSize=9, alignment=TA_CENTER, textColor=GRAY, spaceAfter=8)))
story.append(Spacer(1, 0.3*cm))

# Table S1
story.append(hr())
story.append(Paragraph("Table S1 — Characteristics of All 28 Included Studies", H1))
h1 = ["ID","Author","Year","Country","N","Procedure","Outcome","Events","Event%",
      "EuroSCORE II","Age","LVEF","NOS","Quality"]
rows1 = [h1]
for _, r in df_stud.iterrows():
    rows1.append([r["id"], r["first_author"], str(r["year"]), r["country"][:10],
                  f"{r['n_patients']:,}", r["procedure"], r["primary_outcome"].replace("_"," "),
                  str(r["n_events"]), f"{r['event_pct']:.1f}%",
                  f"{r['mean_euroscore2']:.1f}", f"{r['mean_age']:.1f}",
                  f"{r['mean_lvef']:.1f}", str(r["nos"]), r["quality"]])
story.append(stab(rows1, [0.6,1.3,0.7,1.2,0.9,1.1,1.8,0.9,0.8,1.1,0.7,0.7,0.6,0.7]))
story.append(Spacer(1, 0.4*cm))

# Table S2
story.append(hr())
story.append(Paragraph("Table S2 — Meta-Analytic Estimates (DerSimonian-Laird)", H1))
h2 = ["Code","Label","Type","k","N","Rate (%)","CI lower","CI upper","I² (%)","tau²","Q","df"]
rows2 = [h2]
for _, r in df_meta.iterrows():
    rows2.append([r["code"], r["label"][:22], r["group_type"], str(r["k"]),
                  f"{r['n_samples']:,}", f"{r['rate_pooled']:.2f}",
                  f"{r['ci_lower']:.2f}", f"{r['ci_upper']:.2f}",
                  f"{r['i2_pct']:.0f}", f"{r['tau2']:.5f}", f"{r['Q']:.2f}", str(r["Q_df"])])
story.append(stab(rows2, [1.1,2.8,1.1,0.5,1.1,1.1,1.1,1.1,0.9,1.0,0.9,0.5]))
story.append(Spacer(1, 0.4*cm))

# Table S3
story.append(hr())
story.append(Paragraph("Table S3 — Predictors of Postoperative Complications (RF + OR)", H1))
h3 = ["Feature","RF Importance","Odds Ratio","CI lower","CI upper","p-value","Direction"]
rows3 = [h3]
for _, r in df_pred.iterrows():
    rows3.append([r["feature"][:32], f"{r['importance_rf']:.3f}",
                  f"{r['odds_ratio']:.2f}", f"{r['ci_lower']:.2f}",
                  f"{r['ci_upper']:.2f}", str(r["p_value"]), r["direction"]])
story.append(stab(rows3, [4.0,1.4,1.2,1.1,1.1,0.9,1.1]))
story.append(Spacer(1, 0.4*cm))

# Table S4
story.append(hr())
story.append(Paragraph("Table S4 — Atrial Fibrillation Risk by Subgroup", H1))
h4 = ["Subgroup","Rate (%)","CI lower (%)","CI upper (%)"]
rows4 = [h4]
for _, r in df_fa.iterrows():
    rows4.append([r["subgroup"], f"{r['rate_pct']:.1f}", f"{r['ci_lower']:.1f}", f"{r['ci_upper']:.1f}"])
story.append(stab(rows4, [5.0, 2.0, 2.0, 2.0]))
story.append(Spacer(1, 0.4*cm))

# Table S5
story.append(hr())
story.append(Paragraph("Table S5 — Valve Timing Recommendations (ESC 2024)", H1))
h5 = ["Condition","Threshold","Recommendation","Evidence Level","Mortality Benefit"]
rows5 = [h5]
for _, r in df_vt.iterrows():
    rows5.append([r["condition"][:32], r["threshold"][:28], r["recommendation"],
                  r["evidence_level"], r["mortality_benefit"][:26]])
story.append(stab(rows5, [3.6,3.0,2.0,1.0,2.6]))
story.append(Spacer(1, 0.4*cm))

# Supplementary figures
story.append(PageBreak())
story.append(Paragraph("Supplementary Figures", H1))
for fname, cap in [
    ("fig6_valve_timing.png",
     "Figure S1. Valve timing decision map per ESC 2024 guidelines. Colour-coded by evidence class."),
]:
    story += fig(fname, cap, w=15)

doc.build(story)
print(f"✅ Manuscript PDF -> {OUT / 'CardioSurgAI_Manuscript.pdf'}")
