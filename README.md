# 🇯🇵 Japan Tourism Demand Forecasting

Climate-adjusted tourism demand forecasting across all 47 Japanese prefectures (2016–2024) using nonlinear machine learning models.

## 📊 Key Results

- XGBoost RMSE: 0.098 (log scale) — ~50% lower than SARIMA baseline
- Baseline improvement: ~41% over seasonal naive model
- Coverage: ~98% of prefectures show improved predictions

## 🧠 Modeling Approach

This project compares classical time-series models with nonlinear machine learning approaches:

- **XGBoost** (primary model) — captures nonlinear interactions and pooled regional patterns
- **CatBoost** (comparison model) — alternative gradient boosting framework
- **SARIMA** (baseline) — classical seasonal autoregressive model

Note: SARIMA is trained on national-level aggregated data, while XGBoost and CatBoost operate on pooled prefecture-level data. This makes the comparison conservative — SARIMA has an easier modeling task yet still underperforms significantly.

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

This explains why nonlinear boosting models significantly outperform classical time-series approaches.

## 🖥️ Interactive Dashboard

An interactive Streamlit dashboard is included:

- Model performance comparison (XGBoost vs CatBoost vs SARIMA)
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

Research Direction:
- Nonlinear vs classical model comparison
- Climate-adjusted tourism forecasting
- Spatio-temporal pooled learning vs aggregated modeling

## 🚀 Future Work

- Prefecture-level SARIMAX / hierarchical models
- Uncertainty-aware forecasting
- Congestion-risk and overtourism modeling
