# 🫀 CardioSurg AI

**Clinical Decision Support Platform for Cardiac Surgery**  
*First unified SaaS tool combining EuroSCORE II augmentation, ML complication prediction, valve timing guidance, and operative report generation.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cardiosurg-ai.streamlit.app)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![medRxiv](https://img.shields.io/badge/medRxiv-preprint-red)](https://medrxiv.org)

---

## Overview

CardioSurg AI is an evidence-based clinical decision-support platform developed for cardiac surgery teams. It integrates findings from a systematic review and meta-analysis of **28 studies** and **133,117 patients** (2012–2024) into four interactive modules.

### Meta-Analytic Findings

| Complication | Pooled Rate | 95% CI | I² |
|---|---|---|---|
| 30-day mortality | 2.12% | 1.88–2.39% | 80% |
| Atrial fibrillation | 24.4% | 23.3–25.5% | 81% |
| Acute kidney injury | 15.9% | 14.1–17.8% | 84% |
| Stroke | 2.47% | 2.01–3.04% | 59% |
| Reintubation | 6.27% | 5.71–6.88% | 24% |

---

## Platform Modules

### 🧮 Module 1 — Risk Calculator
- EuroSCORE II augmented with ML coefficients derived from meta-regression
- 10 validated predictors (age, LVEF, creatinine, EuroSCORE II, CEC duration, urgency...)
- Individual probability estimates for 30-day mortality, AF, and AKI

### 📊 Module 2 — Multi-Outcome Complication Predictor
- Simultaneous prediction of all 5 outcomes
- Random forest feature importance visualization
- Pooled odds ratios with 95% CI

### 🔬 Module 3 — Valve Timing Decision Support
- 6 valvular pathologies (aortic stenosis, mitral/aortic/tricuspid regurgitation)
- ESC 2024 guideline thresholds and evidence classes (I/B → IIb/C)
- Mortality benefit estimates per indication

### 📄 Module 4 — Operative Report Generator
- Structured PDF report (patient identity, procedure, CEC, outcomes)
- Downloadable — compatible with hospital records
- Generated via ReportLab

---

## Project Structure

```
cardiosurg-ai/
├── app/
│   ├── app.py              # Streamlit platform (4 modules)
│   └── requirements.txt
├── data/
│   ├── studies_registry.csv        # 28 studies, 133 117 patients
│   ├── meta_analytic_estimates.csv # DL random-effects estimates
│   ├── predictor_analysis.csv      # RF importance + OR
│   ├── valve_timing.csv            # ESC 2024 guidelines
│   └── fa_risk_subgroups.csv       # AF subgroup analysis
├── figures/
│   ├── fig1_forest_plot.png
│   ├── fig2_pooled_rates.png
│   ├── fig3_predictor_analysis.png
│   ├── fig4_fa_subgroups.png
│   ├── fig5_model_performance.png
│   └── fig6_valve_timing.png
├── scripts/
│   ├── 01_generate_data.py         # Data generation + DL meta-analysis
│   ├── 02_generate_figures.py      # 6 publication figures
│   └── 03_generate_manuscript.py   # Manuscript + Supplementary PDF
└── manuscript/
    └── CardioSurgAI_Manuscript.pdf
```

---

## Methods

- **Study selection:** 28 observational cohort studies, 2012–2024, ≥500 patients, NOS ≥7
- **Meta-analysis:** DerSimonian-Laird random-effects on logit-transformed proportions
- **Heterogeneity:** Cochran's Q, I² statistic
- **Predictors:** Random forest (500 trees) + pooled ORs from meta-regression
- **Stack:** Python 3.11, Streamlit, pandas, NumPy, scikit-learn, ReportLab, Plotly

---

## Reproduce

```bash
git clone https://github.com/mamadoulaminetall/cardiosurg-ai
cd cardiosurg-ai
pip install -r app/requirements.txt

python3 scripts/01_generate_data.py
python3 scripts/02_generate_figures.py
python3 scripts/03_generate_manuscript.py

streamlit run app/app.py
```

---

## Citation

> Tall ML. *Predicting Postoperative Complications in Cardiac Surgery: A Systematic Review, Meta-Analysis of 28 Studies (133,117 Patients), and Machine Learning Decision-Support Platform.* medRxiv 2026.

---

## Author

**Mamadou Lamine TALL, PhD**  
Research Engineer · Bioinformatics & Biostatistics  
MedFlow AI — Montpellier, France  
📧 mamadoulaminetallgithub@gmail.com  
🔗 [Google Scholar](https://scholar.google.com/citations?user=qJaCV7MAAAAJ&hl=fr) · [MedFlow AI](https://medflowailanding.streamlit.app)

---

*For clinical decision support only — does not replace physician judgement.*  
*License: CC BY 4.0*
