"""
Problem 2 (var2): Subterranean Thermal Anomaly Score  |  Roll No: BT2024079
Inputs: x1..x3 (East-West, North-South, depth offsets)  ->  Target: y

Usage (from repo root):
    python src/var2.py            # train + inference
    python src/var2.py --predict  # inference only, using the saved model
"""
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------- settings ----------------
PRB = "var2"
ROLL = "BT2024079"
TRAIN_CSV = f"data/{ROLL}_train_{PRB}.csv"
TEST_CSV = f"data/{ROLL}_test_{PRB}.csv"
PRED_CSV = f"predictions/{ROLL}_pred_{PRB}.csv"
MODEL_PATH = f"models/{PRB}_model.joblib"
RES_DIR = f"results/{PRB}"

DEGREES = range(1, 21)                 # problem says "up to degree 20"
RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 30, 100]
LASSO_ALPHAS = [0.0003, 0.001, 0.003, 0.01, 0.03]
TOL = 0.05                             # parsimony: lowest degree within 5% of best CV MSE
SEED = 42


def build_model(degree, kind, alpha):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    if kind == "ols":
        return make_pipeline(poly, LinearRegression())
    if kind == "ridge":
        return make_pipeline(poly, StandardScaler(), Ridge(alpha=alpha))
    return make_pipeline(poly, StandardScaler(), Lasso(alpha=alpha, max_iter=50000))


def data_checks(train, test):
    """Missing values, duplicates, ranges, clipped (+-1) values."""
    X = train.drop(columns="y")
    print(f"train {train.shape}, test {test.shape}")
    print(f"missing values: train={train.isna().sum().sum()}, test={test.isna().sum().sum()}")
    print(f"duplicate rows: {train.duplicated().sum()}, duplicate inputs: {X.duplicated().sum()}")
    print(f"input range: train [{X.min().min()}, {X.max().max()}], test [{test.min().min()}, {test.max().max()}]")
    print(f"% values clipped at +-1: train {(X.abs() == 1).values.mean():.1%}, test {(test.abs() == 1).values.mean():.1%}")


def outlier_check(X, y, spec, kf):
    """
    1) Target-level: IQR rule on y.
    2) Model-level: robust z-score of out-of-fold residuals (MAD scale).
    3) Test whether dropping |z| > 3 points from the training folds improves CV MSE.
    """
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    n_iqr = int(((y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)).sum())

    res = y - cross_val_predict(build_model(*spec), X, y, cv=kf)
    mad = 1.4826 * np.median(np.abs(res - np.median(res)))
    z = res / mad
    n_z3 = int((np.abs(z) > 3).sum())

    def cv_mse(drop):
        errs = []
        for tr, va in kf.split(X):
            m = build_model(*spec).fit(X[tr], y[tr])
            if drop:
                r = y[tr] - m.predict(X[tr])
                s = 1.4826 * np.median(np.abs(r - np.median(r)))
                keep = np.abs(r / s) <= 3
                m = build_model(*spec).fit(X[tr][keep], y[tr][keep])
            errs.extend((m.predict(X[va]) - y[va]) ** 2)
        return float(np.mean(errs))

    mse_keep, mse_drop = cv_mse(False), cv_mse(True)
    print(f"y IQR outliers: {n_iqr} (tails of the polynomial, not errors)")
    print(f"residual |z|>3: {n_z3} of {len(y)} (Gaussian expectation ~{0.0027 * len(y):.1f}), "
          f"residual kurtosis {pd.Series(res).kurt():.3f}")
    print(f"CV MSE keep all = {mse_keep:.4f} | drop |z|>3 = {mse_drop:.4f}")
    remove = mse_drop < mse_keep * 0.99    # only remove if it clearly helps
    print("decision:", "remove outliers" if remove else "keep all points (no genuine outliers)")

    # residual plot
    plt.figure(figsize=(6, 4))
    plt.scatter(y - res, res, s=8, alpha=0.6)
    plt.axhline(0, c="k", lw=1)
    for k in (3, -3):
        plt.axhline(k * mad, c="r", ls="--", lw=1)
    plt.xlabel("Out-of-fold prediction")
    plt.ylabel("Residual (y - prediction)")
    plt.title(f"{PRB}: residuals (red = +-3 robust SD)")
    plt.tight_layout()
    plt.savefig(f"{RES_DIR}/residuals.png", dpi=150)
    plt.close()

    return (np.abs(z) <= 3) if remove else np.ones(len(y), bool)


def degree_search(X, y, kf):
    max_terms = int(0.8 * len(X))     # rows in a CV training fold
    rows = []
    for d in DEGREES:
        n_terms = PolynomialFeatures(d, include_bias=False).fit(X[:1]).n_output_features_
        if n_terms > max_terms:
            print(f"degree {d}: {n_terms} terms > {max_terms} training rows -> stop")
            break
        cands = [("ols", None)] + [("ridge", a) for a in RIDGE_ALPHAS] + [("lasso", a) for a in LASSO_ALPHAS]
        for kind, alpha in cands:
            p = cross_val_predict(build_model(d, kind, alpha), X, y, cv=kf)
            rows.append({"degree": d, "model": kind, "alpha": alpha,
                         "cv_mse": mean_squared_error(y, p), "cv_r2": r2_score(y, p)})
        b = min((r for r in rows if r["degree"] == d), key=lambda r: r["cv_mse"])
        print(f"degree {d:2d} ({n_terms:4d} terms): CV MSE {b['cv_mse']:.4f}  R2 {b['cv_r2']:.4f}  "
              f"[{b['model']}, alpha={b['alpha']}]")
    res = pd.DataFrame(rows)
    res.to_csv(f"{RES_DIR}/cv_results.csv", index=False)

    ok = res[res.cv_mse <= (1 + TOL) * res.cv_mse.min()]
    best = res[res.degree == ok.degree.min()].sort_values("cv_mse").iloc[0]

    # degree-vs-error plot
    ols = res[res.model == "ols"].set_index("degree").cv_mse
    plt.figure(figsize=(6, 4))
    plt.plot(ols.index, ols.values, "o-", label="Plain polynomial (OLS)")
    plt.plot(res.groupby("degree").cv_mse.min(), "s-", label="Best with Ridge/Lasso")
    plt.axvline(best.degree, c="gray", ls="--", label=f"chosen degree = {int(best.degree)}")
    plt.yscale("log")
    plt.xlabel("Polynomial degree")
    plt.ylabel("5-fold CV MSE (log)")
    plt.title(f"{PRB}: degree selection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RES_DIR}/degree_vs_mse.png", dpi=150)
    plt.close()
    alpha = None if pd.isna(best.alpha) else float(best.alpha)
    return (int(best.degree), best.model, alpha), best


def train():
    for d in ("models", "predictions", RES_DIR):
        os.makedirs(d, exist_ok=True)
    train_df, test_df = pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)
    X, y = train_df.drop(columns="y").values, train_df["y"].values
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    print(f"\n===== {PRB}: 1. data checks =====")
    data_checks(train_df, test_df)

    print(f"\n===== {PRB}: 2. degree + regularisation search (5-fold CV) =====")
    spec, row = degree_search(X, y, kf)

    print(f"\n===== {PRB}: 3. outlier check =====")
    keep = outlier_check(X, y, spec, kf)
    if not keep.all():                 # re-run the search on the cleaned data
        X, y = X[keep], y[keep]
        spec, row = degree_search(X, y, kf)

    d, kind, alpha = spec

    print(f"\n===== {PRB}: 4. final model =====")
    model = build_model(d, kind, alpha).fit(X, y)
    tp = model.predict(X)
    print(f"chosen: degree={d}, model={kind}, alpha={alpha}")
    print(f"CV    MSE={row.cv_mse:.4f}  R2={row.cv_r2:.4f}")
    print(f"train MSE={mean_squared_error(y, tp):.4f}  R2={r2_score(y, tp):.4f}")
    joblib.dump(model, MODEL_PATH)
    pd.DataFrame([{"problem": PRB, "degree": d, "model": kind, "alpha": alpha,
                   "cv_mse": round(row.cv_mse, 4), "cv_r2": round(row.cv_r2, 4),
                   "train_mse": round(mean_squared_error(y, tp), 4), "train_r2": round(r2_score(y, tp), 4)}]
                 ).to_csv(f"{RES_DIR}/summary.csv", index=False)
    print(f"saved model -> {MODEL_PATH}")


def predict():
    print(f"\n===== {PRB}: inference =====")
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_CSV)
    pred = model.predict(test_df.values)
    pd.DataFrame({"y": pred}).to_csv(PRED_CSV, index=False)
    print(f"{len(pred)} predictions (range {pred.min():.2f} to {pred.max():.2f}) -> {PRED_CSV}")


if __name__ == "__main__":
    if "--predict" not in sys.argv:
        train()
    predict()
