"""
Compare Multiple ML Models for Loan Default Prediction
Models: LightGBM, XGBoost, CatBoost, Random Forest, Gradient Boosting, Logistic Regression
Metric: F1 Score (default class), ROC-AUC, Precision, Recall
"""

import numpy as np
import pandas as pd
import joblib
import warnings
import time
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                              recall_score, classification_report, confusion_matrix)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
N = 50000

print("=" * 65)
print("   MODEL COMPARISON — Loan Default Prediction")
print("   Metric Focus: F1 Score (Default Class)")
print("=" * 65)

# ── Generate Synthetic Dataset (same as before) ───────────────────────────────
print("\n[1] Generating dataset...")

age_years        = np.random.normal(40, 12, N).clip(21, 70)
employment_years = np.random.exponential(5, N).clip(0, 40)
income           = np.random.lognormal(11, 0.6, N).clip(15000, 1500000)
credit_amount    = np.random.lognormal(13, 0.8, N).clip(45000, 4500000)
annuity          = credit_amount / np.random.uniform(24, 96, N)
ext1             = np.random.beta(3, 2, N)
ext2             = np.random.beta(3, 2, N)
ext3             = np.random.beta(3, 2, N)
children         = np.random.choice([0,1,2,3,4], N, p=[0.4,0.3,0.2,0.07,0.03])
fam_members      = children + np.random.choice([1,2], N, p=[0.3,0.7])
region_rating    = np.random.choice([1,2,3], N, p=[0.2,0.5,0.3])

gender        = np.random.choice(["M","F"], N, p=[0.36,0.64])
education     = np.random.choice(
    ["Higher education","Secondary","Incomplete higher","Lower secondary","Academic degree"],
    N, p=[0.28,0.56,0.10,0.05,0.01])
income_type   = np.random.choice(
    ["Working","Commercial associate","Pensioner","State servant","Unemployed"],
    N, p=[0.52,0.23,0.18,0.06,0.01])
housing_type  = np.random.choice(
    ["House / apartment","With parents","Municipal apartment","Rented apartment","Office apartment"],
    N, p=[0.72,0.12,0.08,0.06,0.02])
family_status = np.random.choice(
    ["Married","Single / not married","Civil marriage","Separated","Widow"],
    N, p=[0.64,0.15,0.10,0.07,0.04])
own_car       = np.random.choice(["Y","N"], N, p=[0.34,0.66])
own_realty    = np.random.choice(["Y","N"], N, p=[0.69,0.31])
occupation    = np.random.choice(
    ["Laborers","Core staff","Accountants","Managers","Drivers",
     "Sales staff","Cleaning staff","Cooking staff","Medicine staff"],
    N, p=[0.27,0.15,0.10,0.09,0.09,0.08,0.06,0.05,0.11])
org_type      = np.random.choice(
    ["Business Entity Type 3","School","Government","Religion","Other",
     "Medicine","Self-employed","Transport: type 4","Construction"],
    N, p=[0.18,0.12,0.10,0.04,0.15,0.07,0.13,0.06,0.15])

logit = (
    - 2.5
    - 1.8 * ext1 - 2.2 * ext2 - 1.9 * ext3
    + 0.4 * (credit_amount / income)
    + 0.3 * (annuity / income)
    - 0.03 * age_years
    - 0.06 * employment_years
    + 0.15 * (region_rating == 3).astype(float)
    + 0.2  * (income_type == "Unemployed").astype(float)
    - 0.2  * (education == "Higher education").astype(float)
    + 0.1  * (housing_type == "With parents").astype(float)
    + 0.08 * children
)
prob_default = 1 / (1 + np.exp(-logit))
target = (np.random.random(N) < prob_default).astype(int)

df = pd.DataFrame({
    "CODE_GENDER": gender,
    "DAYS_BIRTH": -(age_years*365).astype(int),
    "DAYS_EMPLOYMENT": -(employment_years*365).astype(int),
    "CNT_CHILDREN": children,
    "CNT_FAM_MEMBERS": fam_members,
    "AMT_INCOME_TOTAL": income.round(2),
    "AMT_CREDIT": credit_amount.round(2),
    "AMT_ANNUITY": annuity.round(2),
    "AMT_GOODS_PRICE": (credit_amount*0.85).round(2),
    "EXT_SOURCE_1": ext1.round(4),
    "EXT_SOURCE_2": ext2.round(4),
    "EXT_SOURCE_3": ext3.round(4),
    "NAME_EDUCATION_TYPE": education,
    "NAME_INCOME_TYPE": income_type,
    "NAME_HOUSING_TYPE": housing_type,
    "NAME_FAMILY_STATUS": family_status,
    "FLAG_OWN_CAR": own_car,
    "FLAG_OWN_REALTY": own_realty,
    "REGION_RATING_CLIENT": region_rating,
    "REGION_POPULATION_RELATIVE": np.random.uniform(0.001, 0.073, N).round(4),
    "OCCUPATION_TYPE": occupation,
    "ORGANIZATION_TYPE": org_type,
    "TARGET": target,
})

# Feature engineering
df["CREDIT_INCOME_RATIO"]  = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
df["AGE_YEARS"]            = -df["DAYS_BIRTH"] / 365
df["EMPLOYMENT_YEARS"]     = df["DAYS_EMPLOYMENT"].apply(lambda x: -x/365 if x < 0 else 0)
df["EXT_SOURCE_MEAN"]      = df[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]].mean(axis=1)
df["EXT_SOURCE_MIN"]       = df[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]].min(axis=1)
df["EXT_SOURCE_MAX"]       = df[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]].max(axis=1)
df["CHILDREN_RATIO"]       = df["CNT_CHILDREN"] / (df["CNT_FAM_MEMBERS"] + 1)
df["INCOME_PER_PERSON"]    = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
df["CREDIT_TERM"]          = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)
df["INCOME_CREDIT_RATIO"]  = df["AMT_INCOME_TOTAL"] / (df["AMT_CREDIT"] + 1)

CATEGORICAL = ["CODE_GENDER","NAME_FAMILY_STATUS","NAME_EDUCATION_TYPE",
               "NAME_HOUSING_TYPE","NAME_INCOME_TYPE","OCCUPATION_TYPE",
               "ORGANIZATION_TYPE","FLAG_OWN_CAR","FLAG_OWN_REALTY"]

le_dict = {}
for col in CATEGORICAL:
    le = LabelEncoder()
    df[col] = df[col].astype(str).fillna("Unknown")
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Add missing values realistically
for col in ["EXT_SOURCE_1","EXT_SOURCE_3","OCCUPATION_TYPE"]:
    mask = np.random.random(N) < 0.05
    df.loc[mask, col] = np.nan

feature_cols = [c for c in df.columns if c != "TARGET"]
X = df[feature_cols].values
y = df["TARGET"].values

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Dataset: {N:,} samples | Default rate: {y.mean():.1%}")
print(f"   Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} | Features: {X_train.shape[1]}")

# ── Define All Models ─────────────────────────────────────────────────────────
print("\n[2] Setting up models...")

models = {}

# LightGBM
try:
    import lightgbm as lgb
    models["LightGBM"] = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        num_leaves=63, min_child_samples=30, subsample=0.8,
        colsample_bytree=0.8, class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=-1)
    print("   ✅ LightGBM")
except ImportError:
    print("   ❌ LightGBM not available")

# XGBoost
try:
    import xgboost as xgb
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    models["XGBoost"] = xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42, n_jobs=-1,
        eval_metric="auc", verbosity=0, use_label_encoder=False)
    print("   ✅ XGBoost")
except ImportError:
    print("   ❌ XGBoost not available")

# CatBoost
try:
    from catboost import CatBoostClassifier
    models["CatBoost"] = CatBoostClassifier(
        iterations=400, learning_rate=0.05, depth=6,
        auto_class_weights="Balanced",
        random_seed=42, verbose=0)
    print("   ✅ CatBoost")
except ImportError:
    print("   ❌ CatBoost not available")

# Random Forest
models["Random Forest"] = RandomForestClassifier(
    n_estimators=300, max_depth=10,
    class_weight="balanced", random_state=42, n_jobs=-1)
print("   ✅ Random Forest")

# Gradient Boosting
models["Gradient Boosting"] = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05,
    max_depth=5, subsample=0.8, random_state=42)
print("   ✅ Gradient Boosting")

# Logistic Regression
models["Logistic Regression"] = LogisticRegression(
    class_weight="balanced", max_iter=1000,
    random_state=42, n_jobs=-1)
print("   ✅ Logistic Regression")

# ── Train & Evaluate All Models ───────────────────────────────────────────────
print("\n[3] Training & evaluating all models...")
print("-" * 65)

results = []
trained_models = {}
THRESHOLD = 0.35

for name, model in models.items():
    print(f"\n   🔄 Training {name}...")
    start = time.time()

    try:
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= THRESHOLD).astype(int)

        f1_default  = f1_score(y_test, y_pred, pos_label=1)
        f1_macro    = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        auc         = roc_auc_score(y_test, y_prob)
        precision   = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall      = recall_score(y_test, y_pred, pos_label=1)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        results.append({
            "Model"           : name,
            "F1 (Default)"    : round(f1_default, 4),
            "F1 (Macro)"      : round(f1_macro, 4),
            "F1 (Weighted)"   : round(f1_weighted, 4),
            "ROC-AUC"         : round(auc, 4),
            "Precision"       : round(precision, 4),
            "Recall"          : round(recall, 4),
            "True Positives"  : int(tp),
            "False Negatives" : int(fn),
            "Train Time (s)"  : round(elapsed, 1),
        })
        trained_models[name] = (model, y_prob)

        print(f"      ✅ Done in {elapsed:.1f}s")
        print(f"         F1 (Default): {f1_default:.4f} | AUC: {auc:.4f} | "
              f"Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"         Caught {tp} defaulters | Missed {fn} defaulters")

    except Exception as e:
        print(f"      ❌ Failed: {e}")

# ── Results Table ─────────────────────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("   RESULTS — Sorted by F1 Score (Default Class)")
print("=" * 65)

df_results = pd.DataFrame(results).sort_values("F1 (Default)", ascending=False).reset_index(drop=True)
df_results.index += 1  # rank from 1

print(f"\n{'Rank':<5} {'Model':<22} {'F1(Default)':<13} {'ROC-AUC':<10} {'Precision':<11} {'Recall':<9} {'Time(s)'}")
print("-" * 85)
for idx, row in df_results.iterrows():
    medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "  "
    print(f"{medal} {idx:<3} {row['Model']:<22} {row['F1 (Default)']:<13} "
          f"{row['ROC-AUC']:<10} {row['Precision']:<11} {row['Recall']:<9} {row['Train Time (s)']}")

# ── Winner ────────────────────────────────────────────────────────────────────
best = df_results.iloc[0]
print(f"\n{'='*65}")
print(f"  🏆 BEST MODEL: {best['Model']}")
print(f"     F1 Score (Default Class) : {best['F1 (Default)']}")
print(f"     ROC-AUC                  : {best['ROC-AUC']}")
print(f"     Precision                : {best['Precision']}")
print(f"     Recall                   : {best['Recall']}")
print(f"{'='*65}")

# ── Detailed report for best model ───────────────────────────────────────────
best_name  = best["Model"]
best_model, best_prob = trained_models[best_name]
best_pred  = (best_prob >= THRESHOLD).astype(int)

print(f"\n  Detailed Classification Report — {best_name}:")
print(classification_report(y_test, best_pred, target_names=["Repaid", "Default"]))

# ── Save best model ───────────────────────────────────────────────────────────
print(f"\n  Saving best model ({best_name}) as loan_model.pkl ...")
bundle = {
    "model"          : best_model,
    "label_encoders" : le_dict,
    "imputer"        : imputer,
    "feature_cols"   : feature_cols,
    "auc_score"      : float(best["ROC-AUC"]),
    "f1_score"       : float(best["F1 (Default)"]),
    "threshold"      : THRESHOLD,
    "categorical"    : CATEGORICAL,
    "best_model_name": best_name,
    "use_lgbm"       : best_name == "LightGBM",
    "all_results"    : df_results.to_dict(orient="records"),
}
joblib.dump(bundle, "loan_model.pkl")

# Save comparison CSV
df_results.to_csv("model_comparison.csv", index=False)

print(f"\n  ✅ loan_model.pkl  — best model saved")
print(f"  ✅ model_comparison.csv — full comparison saved")
print(f"\n  Run: streamlit run app.py")
print("=" * 65)
