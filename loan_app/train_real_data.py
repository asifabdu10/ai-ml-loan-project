"""
Train on REAL Home Credit Default Risk data from Kaggle.
Place all CSV files in the same folder as this script, then run:
    python train_real_data.py

Uses: application_train.csv, bureau.csv, previous_application.csv,
      installments_payments.csv, POS_CASH_balance.csv, credit_card_balance.csv
"""

import pandas as pd
import numpy as np
import joblib
import warnings
import os
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.impute import SimpleImputer

try:
    import lightgbm as lgb
    USE_LGBM = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    USE_LGBM = False

# ── File paths (edit if your CSV files are in a different folder) ─────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def p(filename):
    return os.path.join(DATA_DIR, filename)

print("=" * 60)
print("  Home Credit Default Risk — Real Data Training")
print("=" * 60)

# ── Step 1: Load main application data ───────────────────────────────────────
print("\n[1/6] Loading application_train.csv ...")
app = pd.read_csv(p("application_train.csv"))
print(f"      Shape: {app.shape}")
print(f"      Default rate: {app['TARGET'].mean():.2%}")

# ── Step 2: Aggregate bureau.csv ─────────────────────────────────────────────
print("[2/6] Loading bureau.csv ...")
bureau = pd.read_csv(p("bureau.csv"))
print(f"      Shape: {bureau.shape}")

bureau_agg = bureau.groupby("SK_ID_CURR").agg(
    BUREAU_LOAN_COUNT        = ("SK_ID_BUREAU", "count"),
    BUREAU_CREDIT_SUM        = ("AMT_CREDIT_SUM", "sum"),
    BUREAU_CREDIT_SUM_DEBT   = ("AMT_CREDIT_SUM_DEBT", "sum"),
    BUREAU_CREDIT_SUM_MAX    = ("AMT_CREDIT_SUM", "max"),
    BUREAU_OVERDUE_COUNT     = ("CREDIT_DAY_OVERDUE", lambda x: (x > 0).sum()),
    BUREAU_ACTIVE_LOANS      = ("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
    BUREAU_CLOSED_LOANS      = ("CREDIT_ACTIVE", lambda x: (x == "Closed").sum()),
    BUREAU_AVG_DAYS_CREDIT   = ("DAYS_CREDIT", "mean"),
).reset_index()

app = app.merge(bureau_agg, on="SK_ID_CURR", how="left")
print(f"      Merged bureau features: {bureau_agg.shape[1]-1}")

# ── Step 3: Aggregate previous_application.csv ────────────────────────────────
print("[3/6] Loading previous_application.csv ...")
prev = pd.read_csv(p("previous_application.csv"))
print(f"      Shape: {prev.shape}")

prev_agg = prev.groupby("SK_ID_CURR").agg(
    PREV_APP_COUNT           = ("SK_ID_PREV", "count"),
    PREV_APP_APPROVED        = ("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
    PREV_APP_REFUSED         = ("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
    PREV_AMT_CREDIT_MEAN     = ("AMT_CREDIT", "mean"),
    PREV_AMT_ANNUITY_MEAN    = ("AMT_ANNUITY", "mean"),
    PREV_DAYS_DECISION_MEAN  = ("DAYS_DECISION", "mean"),
).reset_index()
prev_agg["PREV_APPROVAL_RATE"] = (
    prev_agg["PREV_APP_APPROVED"] / (prev_agg["PREV_APP_COUNT"] + 1)
)

app = app.merge(prev_agg, on="SK_ID_CURR", how="left")
print(f"      Merged prev_application features: {prev_agg.shape[1]-1}")

# ── Step 4: Aggregate installments_payments.csv ───────────────────────────────
print("[4/6] Loading installments_payments.csv ...")
inst = pd.read_csv(p("installments_payments.csv"))
print(f"      Shape: {inst.shape}")

inst["PAYMENT_DIFF"]  = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]
inst["DAYS_PAST_DUE"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
inst["DAYS_PAST_DUE"] = inst["DAYS_PAST_DUE"].clip(lower=0)

inst_agg = inst.groupby("SK_ID_CURR").agg(
    INST_PAYMENT_DIFF_MEAN   = ("PAYMENT_DIFF", "mean"),
    INST_PAYMENT_DIFF_SUM    = ("PAYMENT_DIFF", "sum"),
    INST_DAYS_PAST_DUE_MEAN  = ("DAYS_PAST_DUE", "mean"),
    INST_DAYS_PAST_DUE_MAX   = ("DAYS_PAST_DUE", "max"),
    INST_LATE_PAYMENTS       = ("DAYS_PAST_DUE", lambda x: (x > 0).sum()),
    INST_COUNT               = ("SK_ID_PREV", "count"),
).reset_index()

app = app.merge(inst_agg, on="SK_ID_CURR", how="left")
print(f"      Merged installments features: {inst_agg.shape[1]-1}")

# ── Step 5: Aggregate POS_CASH_balance.csv ────────────────────────────────────
print("[5/6] Loading POS_CASH_balance.csv ...")
pos = pd.read_csv(p("POS_CASH_balance.csv"))
print(f"      Shape: {pos.shape}")

pos_agg = pos.groupby("SK_ID_CURR").agg(
    POS_MONTHS_BALANCE_MEAN  = ("MONTHS_BALANCE", "mean"),
    POS_CNT_INSTALMENT_MEAN  = ("CNT_INSTALMENT", "mean"),
    POS_SK_DPD_MEAN          = ("SK_DPD", "mean"),
    POS_SK_DPD_MAX           = ("SK_DPD", "max"),
    POS_SK_DPD_DEF_MEAN      = ("SK_DPD_DEF", "mean"),
    POS_COUNT                = ("SK_ID_PREV", "count"),
).reset_index()

app = app.merge(pos_agg, on="SK_ID_CURR", how="left")
print(f"      Merged POS_CASH features: {pos_agg.shape[1]-1}")

# ── Step 6: Feature Engineering ───────────────────────────────────────────────
print("[6/6] Feature engineering ...")

# Core engineered features
app["CREDIT_INCOME_RATIO"]   = app["AMT_CREDIT"]  / (app["AMT_INCOME_TOTAL"] + 1)
app["ANNUITY_INCOME_RATIO"]  = app["AMT_ANNUITY"] / (app["AMT_INCOME_TOTAL"] + 1)
app["CREDIT_TERM"]           = app["AMT_CREDIT"]  / (app["AMT_ANNUITY"] + 1)
app["AGE_YEARS"]             = -app["DAYS_BIRTH"] / 365
app["EMPLOYMENT_YEARS"]      = app["DAYS_EMPLOYED"].apply(lambda x: -x/365 if x < 0 else 0)
app["EXT_SOURCE_MEAN"]       = app[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]].mean(axis=1)
app["EXT_SOURCE_MIN"]        = app[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]].min(axis=1)
app["EXT_SOURCE_MAX"]        = app[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]].max(axis=1)
app["CHILDREN_RATIO"]        = app["CNT_CHILDREN"] / (app["CNT_FAM_MEMBERS"] + 1)
app["INCOME_PER_PERSON"]     = app["AMT_INCOME_TOTAL"] / (app["CNT_FAM_MEMBERS"] + 1)
app["GOODS_CREDIT_RATIO"]    = app["AMT_GOODS_PRICE"] / (app["AMT_CREDIT"] + 1)
app["INCOME_CREDIT_RATIO"]   = app["AMT_INCOME_TOTAL"] / (app["AMT_CREDIT"] + 1)
app["DAYS_EMPLOYED_PERCENT"] = app["DAYS_EMPLOYED"] / (app["DAYS_BIRTH"] + 1)

# Bureau ratio
if "BUREAU_CREDIT_SUM" in app.columns:
    app["BUREAU_DEBT_RATIO"] = app["BUREAU_CREDIT_SUM_DEBT"] / (app["BUREAU_CREDIT_SUM"] + 1)

# Encode categorical columns
CATEGORICAL = [
    "CODE_GENDER","NAME_CONTRACT_TYPE","FLAG_OWN_CAR","FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START","ORGANIZATION_TYPE",
    "FONDKAPREMONT_MODE","HOUSETYPE_MODE","WALLSMATERIAL_MODE","EMERGENCYSTATE_MODE",
]

print("      Encoding categorical features...")
le_dict = {}
for col in CATEGORICAL:
    if col in app.columns:
        le = LabelEncoder()
        app[col] = app[col].astype(str).fillna("Unknown")
        app[col] = le.fit_transform(app[col])
        le_dict[col] = le

# Drop ID and target
drop_cols = ["SK_ID_CURR", "TARGET"]
feature_cols = [c for c in app.columns if c not in drop_cols]

print(f"      Total features: {len(feature_cols)}")

# ── Prepare X, y ─────────────────────────────────────────────────────────────
X = app[feature_cols].values
y = app["TARGET"].values

print("\n      Imputing missing values...")
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

# ── Train/Test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train size : {X_train.shape[0]:,}")
print(f"      Test size  : {X_test.shape[0]:,}")

# ── Train Model ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
if USE_LGBM:
    print("  Training LightGBM on REAL data...")
    model = lgb.LGBMClassifier(
        n_estimators      = 1000,
        learning_rate     = 0.02,
        max_depth         = 7,
        num_leaves        = 63,
        min_child_samples = 50,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 0.1,
        class_weight      = "balanced",
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1,
    )
    model.fit(
        X_train, y_train,
        eval_set    = [(X_test, y_test)],
        callbacks   = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
else:
    print("  Training GradientBoosting on REAL data (LightGBM not available)...")
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=5, random_state=42
    )
    model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.35).astype(int)
auc    = roc_auc_score(y_test, y_prob)

print(f"\n{'='*60}")
print(f"  ✅ REAL DATA TRAINING COMPLETE")
print(f"     ROC-AUC Score : {auc:.4f}")
print(f"     Features Used : {len(feature_cols)}")
print(f"     Training Rows : {X_train.shape[0]:,}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Repaid","Default"]))

# ── Save ──────────────────────────────────────────────────────────────────────
bundle = {
    "model"         : model,
    "label_encoders": le_dict,
    "imputer"       : imputer,
    "feature_cols"  : feature_cols,
    "auc_score"     : round(auc, 4),
    "threshold"     : 0.35,
    "categorical"   : CATEGORICAL,
    "use_lgbm"      : USE_LGBM,
    "trained_on"    : "real_kaggle_data",
    "n_train"       : X_train.shape[0],
}
out_path = os.path.join(DATA_DIR, "loan_model.pkl")
joblib.dump(bundle, out_path)
print(f"\n  ✅ Model saved: {out_path}")
print(f"     Run: streamlit run app.py")
print("="*60)
