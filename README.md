# 🏭 Steel Processing Temperature Forecast

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-Model-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Tested-red)

> Predictive modeling pipeline for optimizing energy consumption in steelmaking by forecasting final batch temperatures with high accuracy and interpretability.

## 📖 Overview
This project delivers a production-ready regression model to predict the final temperature of steel batches during the ladle refining process. Developed for the **"Steel Bird" Metallurgical Plant**, the model enables precise control over heating cycles, directly reducing electricity consumption and stabilizing the technological process.

## 🎯 Business Context & Objectives
Steel refining involves multiple cycles of electrode heating, material addition, and inert gas blowing. Inaccurate temperature control leads to energy waste and quality deviations. The primary goal was to build a transparent, high-accuracy forecasting tool to support decision-making and energy optimization.

**Key Objectives:**
- ✅ Perform EDA & clean relational data from a 7-table SQLite database
- ✅ Engineer domain-relevant features (energy, durations, material counts)
- ✅ Compare Tree-based, Boosting, and Neural Network architectures
- ✅ Achieve **MAE < 6.8°C** on unseen data
- ✅ Provide actionable business recommendations based on model insights

## 📊 Dataset
Data is stored in an SQLite database (`ds-plus-final.db`) containing historical process logs:
| Table | Description |
|-------|-------------|
| `data_arc` | Electrode heating cycles, power consumption |
| `data_temp` | Timestamped temperature measurements |
| `data_bulk` / `data_bulk_time` | Volume & timing of bulk material additions |
| `data_wire` / `data_wire_time` | Volume & timing of wire material additions |
| `data_gas` | Inert gas blowing volume |

**Target Variable:** `final_temperature` (last measurement per batch)  
**Sample Size:** ~2,400 aggregated batches after anomaly removal & cleaning

## 🛠 Methodology & Pipeline
1. **Data Ingestion & Cleaning**  
   - Handled missing values (treated as "material not added")
   - Removed physical anomalies (`temperature < 1500°C`)
   - Filtered batches with `<2` temperature readings
2. **Feature Engineering**  
   - Aggregated raw logs by `key` (batch ID)
   - Created 15+ domain features: `heating_duration_sum`, `energy_consumed_sum`, `processing_duration`, `power_factor_mean`, material counts/volumes, timing ratios
   - Applied `log1p` transformation for skewed distributions
3. **Modeling & Validation**  
   - Compared: `DecisionTreeRegressor`, `LightGBM`, `CatBoostRegressor`, `PyTorch MLP`
   - Used 5-fold CV + `RandomizedSearchCV` for hyperparameter tuning
   - Primary metric: **MAE** (Mean Absolute Error)
4. **Interpretation**  
   - SHAP values for feature importance & interaction analysis
   - VIF & Phi-k correlation matrices to ensure feature independence

## 📈 Results & Model Performance
| Model | MAE (Val) | RMSE (Val) | R² (Val) |
|:------|:----------|:-----------|:---------|
| Decision Tree | 7.64 | 11.45 | 0.473 |
| LightGBM | 6.49 | 9.08 | 0.669 |
| **CatBoostRegressor** | **6.22** | **8.65** | **0.700** |
| PyTorch MLP (Best) | 6.22 | 8.56 | 0.706 |

✅ **Final Production Choice:** `CatBoostRegressor`  
*(Selected for optimal balance of accuracy, training speed, robustness to unscaled data, and built-in interpretability)*

## 💡 Key Insights & Business Recommendations
🔍 **Top Predictors:** `initial_temperature`, `heating_duration_sum`, `processing_duration`, `energy_consumed_sum`

📋 **Actionable Recommendations:**
1. **Standardize Initial Heating:** Tighten control over `initial_temperature` (most influential feature) to reduce downstream variability.
2. **Optimize Heating Cycles:** Monitor `heating_duration_sum` and `processing_duration` to avoid energy overconsumption while maintaining target temperature.
3. **Continuous Model Retraining:** Implement a monitoring pipeline to retrain the model quarterly as raw materials or equipment degrade over time.


## 🚀 Getting Started
```bash
# 1. Clone repository
git clone https://github.com/your-username/Steel-Processing-Temperature-Forecast.git
cd Steel-Processing-Temperature-Forecast

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook notebooks/steel_temperature_forecast.ipynb
