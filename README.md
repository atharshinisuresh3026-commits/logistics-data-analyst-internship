# Logistics Data Analyst Internship – Last-Mile Delivery Analytics

Python project for the YuvaIntern Logistics Data Analyst Internship (4 weeks).
It analyses a **simulated** shipment dataset (structure modelled on public datasets such as Olist and DataCo) to
predict delivery time and optimise carrier allocation.

## Pipeline
| Step | Script | Output |
|------|--------|--------|
| 1. Generate dirty sample data | `src/generate_data.py` | `data/raw_shipments.csv` |
| 2. Clean & preprocess (Week 2) | `src/clean.py` | `data/clean_shipments.csv`, `data/clean_log.json` |
| 3. EDA & visualisation (Week 3) | `src/eda.py` | `figures/fig1–fig6`, `data/eda_results.json` |
| 4. Modelling & optimisation (Week 4) | `src/model.py` | `figures/fig7–fig9`, `data/model_results.json` |
| Extra figures | `src/extra_figs.py` | roadmap and before/after charts |

## Run
```bash
pip install -r requirements.txt
python src/generate_data.py && python src/clean.py && python src/eda.py && python src/model.py && python src/extra_figs.py
```

## Key results (simulated data)
- Delivery-time model: MAE about 0.52 h vs 1.42 h for the "planned hours" baseline; R² about 0.91.
- Carrier re-allocation (linear programme): expected late rate 11.9% -> 9.6% at slightly lower cost.

Reports for each week are in `reports/`. Data are synthetic; results illustrate the method, not a real company.
