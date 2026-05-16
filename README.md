# 🇯🇵 Japan Tourism Demand Forecasting

Climate-adjusted tourism demand forecasting across all 47 Japanese prefectures (2016–2024) using nonlinear machine learning models. Tourism demand is measured as total overnight guests across accommodation facilities.

## 📊 Key Results

- Pooled XGBoost RMSE: 0.0959 (log scale) — ~35% lower than independently trained segmented models
- Pooled model outperforms segmented models in 41 out of 47 prefectures
- SARIMAX collapses under COVID disruption (RMSE: 0.9028 vs 0.0386 pre-COVID)

## 🧠 Modeling Approach

This project compares classical time-series models with nonlinear machine learning approaches:

- **XGBoost** (primary model) — pooled cross-prefecture training with climate-adjusted features
- **CatBoost** (comparison model) — pooled setup, confirms generalization advantage is model-agnostic
- **SARIMAX** (baseline) — trained on national-level aggregated data

Note: SARIMAX is trained on national-level aggregated data, while XGBoost and CatBoost operate on pooled prefecture-level data. This makes the comparison conservative — SARIMAX has an easier modeling task yet still underperforms significantly, especially under structural demand shocks.

## ⚙️ Features

- Autoregressive features (lag variables, rolling statistics)
- Seasonal encoding (sin/cos transformation)
- Climate variables (temperature, precipitation)
- Regime indicators (COVID impact)

## 🧪 Key Insight

Tourism demand is not purely seasonal — it exhibits:

- Strong short-term momentum
- Climate-sensitive deviations
- Nonlinear and interaction-driven dynamics

SARIMAX phase-wise performance confirms classical model limitations:

| Phase | SARIMAX RMSE | Status |
|---|---|---|
| Pre-COVID | 0.0386 | Competitive |
| COVID Disruption | 0.9028 | Collapse |
| Post-COVID | 0.7647 | Fails to recover |

Pooled XGBoost maintained robustness across all three phases.

## 🖥️ Interactive Dashboard

An interactive Streamlit dashboard is included:

- Pooled vs segmented model performance comparison
- Phase-wise SARIMAX structural break analysis
- Feature importance and category breakdown
- Peak month classification insights
- Pipeline and reproducibility overview

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🔬 Research Context

This project was developed as part of a MEXT 2027 scholarship research preparation.

Research Question: *Does pooled, climate-adjusted machine learning forecasting outperform prefecture-level segmented models for Japanese tourism demand, and how robust is this advantage across seasonal patterns and post-COVID structural changes?*

Research Direction:
- Pooled vs segmented cross-prefecture forecasting
- Climate-adjusted tourism demand modeling
- Structural break analysis across pre-COVID, disruption, and recovery phases
- Congestion-risk detection as applied policy extension

## 🚀 Future Work

- Deeper analysis of prefecture-level heterogeneity in pooled vs segmented performance
- Congestion-risk detection framework built on forecast outputs
- Demand redistribution simulation for proactive regional tourism management
- Uncertainty-aware forecasting
