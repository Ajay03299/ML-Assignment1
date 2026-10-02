# ML Assignment 1 – Polynomial Regression (BT2024079)

Two separate problems, each with its own script, model, predictions and results.

| Problem | Script | Inputs | Chosen model | 5-fold CV MSE | 5-fold CV R² |
|---|---|---|---|---|---|
| var1 – Steam turbine Net Power Score | `src/var1.py` | x1–x6 | Degree 5 + Lasso (α = 0.01) | 0.3381 | 0.9652 |
| var2 – Thermal Anomaly Score | `src/var2.py` | x1–x3 | Degree 8 + Ridge (α = 0.1) | 0.2384 | 0.9952 |

## Approach (same pipeline, run separately per problem)
1. **Data checks**: missing values, duplicates, input ranges, values clipped at ±1.
2. **Degree search**: `PolynomialFeatures(d)` (all terms with total power ≤ d) followed by a plain least-squares, Ridge or Lasso fit. Each combination is scored with 5-fold cross-validation (MSE). The search stops once the number of terms exceeds the number of training rows.
3. **Degree choice**: the lowest degree within 5% of the best CV MSE. This parsimony rule guards against overfitting.
4. **Outlier check**: an IQR rule on `y`, plus robust z-scores of the out-of-fold residuals. Points are removed only if dropping them improves CV. For both problems they don't, so all data is kept.
5. **Final model**: refit on all training data and save to `models/`.
6. **Inference**: predict the test set and write the result to `predictions/`.

## Structure
```
data/          train/test CSVs + sample submission
src/var1.py    Problem 1: training + inference
src/var2.py    Problem 2: training + inference
models/        saved trained models (.joblib)
predictions/   BT2024079_pred_var1.csv, BT2024079_pred_var2.csv
results/var1/  CV table, summary, degree-vs-MSE plot, residual plot
results/var2/  same for Problem 2
```

## How to run
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/var1.py                # train + predict (~30 s)
python src/var2.py                # train + predict (~3 min)

python src/var1.py --predict      # inference only, from the saved model
python src/var2.py --predict
```
