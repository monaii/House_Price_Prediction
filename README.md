# House Price Prediction

End-to-end ML pipeline predicting California house prices with strong results.

## Results

- `RMSE`: 0.4392 (original scale)
- `MAE`: 0.2793
- `R²`: 0.8528
- `Baseline RMSE`: 1.1583 → `Improvement`: 62.1% (target exceeded)

## Quickstart

- Install: `pip install -r requirements.txt`
- Train & evaluate: `python src/test_models.py` (writes `results/final_project_report.txt`, saves models in `models/`)
- Production demo: `python src/ml_pipeline.py` (runs a sample single prediction)

## Project Structure

```
src/
  data_loader.py        # Load/save dataset
  eda_analysis.py       # EDA and plots
  data_preprocessor.py  # Feature engineering, scaling, outliers, target log
  model_trainer.py      # Model training, CV, tuning, report
  ml_pipeline.py        # Prediction pipeline (load preprocessor/model, predict)
  test_preprocessing.py # Preprocessing sanity check
  test_models.py        # Full training + final report
results/
  final_project_report.txt, plots
models/
  preprocessor.pkl, best_model.pkl, others
```

## Notes

- Best model: XGBoost (top CV RMSE)
- Consistent preprocessing: saved preprocessor ensures identical train/inference transforms

## Requirements

- Python 3.8+
- See `requirements.txt` for exact package versions