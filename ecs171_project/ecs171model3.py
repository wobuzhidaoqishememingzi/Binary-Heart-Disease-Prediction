import os
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
RAW_CSV = r"C:\Users\victo\OneDrive\Desktop\CS\ecs171\heart_statlog_cleveland_hungary_final.csv"
PROJECT_DIR = os.path.dirname(RAW_CSV)
PROCESSED_CSV = os.path.join(PROJECT_DIR, "heart_disease_processed.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs_model3")
TARGET_COL = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CONTINUOUS_COLS = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]
CATEGORICAL_COLS = [
    "sex", "chest pain type", "fasting blood sugar",
    "resting ecg", "exercise angina", "ST slope"
]
ZERO_AS_MISSING_COLS = ["cholesterol", "resting bp s", "ST slope"]
ONEHOT_COLS = ["chest pain type", "ST slope", "resting ecg"]
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_text(path: str, s: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

def assert_cols_exist(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] missing columns: {missing}\nExisting: {df.columns.tolist()}")
#数据
def make_processed_csv(raw_csv: str, out_csv: str) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    # drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    # validate target
    uniq = set(df[TARGET_COL].unique())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"Target not binary {{0,1}}. Unique={uniq}")

    # ensure required cols exist
    assert_cols_exist(df, CONTINUOUS_COLS, "continuous")
    assert_cols_exist(df, CATEGORICAL_COLS, "categorical")
    assert_cols_exist(df, ONEHOT_COLS, "onehot")
    # zeros -> NaN for implicit missing
    for c in ZERO_AS_MISSING_COLS:
        zc = (df[c] == 0).sum()
        if zc > 0:
            df.loc[df[c] == 0, c] = np.nan
    # impute
    for c in CONTINUOUS_COLS:
        df[c] = df[c].fillna(df[c].median())

    for c in CATEGORICAL_COLS:
        mode = df[c].mode(dropna=True)
        if len(mode) == 0:
            raise ValueError(f"Cannot compute mode for {c}")
        df[c] = df[c].fillna(mode.iloc[0])
    # cast categorical + target to int
    for c in CATEGORICAL_COLS + [TARGET_COL]:
        df[c] = df[c].astype(int)
    # feature engineering BEFORE scaling
    df_original = df.copy()

    df["age_group"] = pd.cut(
        df_original["age"],
        bins=[0, 40, 55, 70, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)

    df["heart_rate_reserve"] = (220 - df_original["age"]) - df_original["max heart rate"]
    df["chol_risk"] = (df_original["cholesterol"] > 200).astype(int)
    df["bp_risk"] = (df_original["resting bp s"] > 140).astype(int)
    df["oldpeak_abnormal"] = (df_original["oldpeak"] > 0).astype(int)
    # one-hot encoding
    df = pd.get_dummies(df, columns=ONEHOT_COLS, drop_first=True)
    # scaling continuous + heart_rate_reserve
    scale_cols = CONTINUOUS_COLS + ["heart_rate_reserve"]
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    df.to_csv(out_csv, index=False)
    return df

#andom forest
def run_model3(df: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL].astype(int)
    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    # baseline RF
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    # metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics_txt = "\n".join([
        "=== Model 3: Random Forest (Baseline) ===",
        f"Test Accuracy : {acc:.4f}",
        f"Test Precision: {prec:.4f}",
        f"Test Recall   : {rec:.4f}",
        f"Test F1       : {f1:.4f}",
        "",
        "--- classification_report ---",
        report
    ])
    write_text(os.path.join(out_dir, "metrics.txt"), metrics_txt)
    write_text(os.path.join(out_dir, "confusion_matrix.txt"), np.array2string(cm))
    # CV (F1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="f1")
    cv_txt = "\n".join([
        "=== 5-Fold Stratified CV (Baseline RF) ===",
        f"F1 mean: {cv_scores.mean():.4f}",
        f"F1 std : {cv_scores.std():.4f}",
        f"All scores: {np.round(cv_scores, 4).tolist()}",
    ])
    write_text(os.path.join(out_dir, "cv_results.txt"), cv_txt)
    # GridSearch tuning (F1)
    param_grid = {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 3, 5],
        "max_features": ["sqrt", "log2", None],
    }
    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X, y)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_score = float(grid.best_score_)

    with open(os.path.join(out_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"best_params": best_params, "best_cv_f1": best_score},
            f, ensure_ascii=False, indent=2
        )

    # feature importance
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances.to_csv(os.path.join(out_dir, "feature_importance.csv"), header=["importance"])

    top15 = importances.head(15)
    top15_txt = "=== Top 15 Feature Importances (Best RF) ===\n" + "\n".join(
        [f"{k}: {v:.6f}" for k, v in top15.items()]
    )
    write_text(os.path.join(out_dir, "feature_importance_top15.txt"), top15_txt)


# =============================================================================
# 3) 主程序
# =============================================================================

def main():
    print("RAW_CSV:", RAW_CSV)
    print("PROCESSED_CSV:", PROCESSED_CSV)
    print("OUT_DIR:", OUT_DIR)

    print("\n[1/2] Building processed dataset...")
    df = make_processed_csv(RAW_CSV, PROCESSED_CSV)
    print("Processed shape:", df.shape)
    print("Saved:", PROCESSED_CSV)

    print("\n[2/2] Running Model 3 (Random Forest)...")
    run_model3(df, OUT_DIR)

    print("\n✅ ALL DONE.")
    print("Outputs saved to:", OUT_DIR)
    print(" - metrics.txt")
    print(" - confusion_matrix.txt")
    print(" - cv_results.txt")
    print(" - best_params.json")
    print(" - feature_importance.csv / feature_importance_top15.txt")


if __name__ == "__main__":
    main()
