"""
Generate synthetic Home Credit style dataset and train model.
Run this if you don't have application_train.csv yet.
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

np.random.seed(42)
N = 50000  # 50k synthetic samples

print("=" * 55)
print("  Generating Synthetic Home Credit Dataset...")
print("=" * 55)

# ── Generate realistic synthetic data ────────────────────────────────────────
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

gender           = np.random.choice(["M","F"], N, p=[0.36,0.64])
contract_type    = np.random.choice(["Cash loans","Revolving loans"], N, p=[0.9,0.1])
education        = np.random.choice(
    ["Higher education","Secondary","Incomplete higher","Lower secondary","Academic degree"],
    N, p=[0.28,0.56,0.10,0.05,0.01]
)
income_type      = np.random.choice(
    ["Working","Commercial associate","Pensioner","State servant","Unemployed"],
    N, p=[0.52,0.23,0.18,0.06,0.01]
)
housing_type     = np.random.choice(
    ["House / apartment","With parents","Municipal apartment","Rented apartment","Office apartment"],
    N, p=[0.72,0.12,0.08,0.06,0.02]
)
family_status    = np.random.choice(
    ["Married","Single / not married","Civil marriage","Separated","Widow"],
    N, p=[0.64,0.15,0.10,0.07,0.04]
)
own_car          = np.random.choice(["Y","N"], N, p=[0.34,0.66])
own_realty       = np.random.choice(["Y","N"], N, p=[0.69,0.31])
flag_phone       = np.random.choice([0,1], N, p=[0.28,0.72])
flag_email       = np.random.choice([0,1], N, p=[0.71,0.29])
flag_work_phone  = np.random.choice([0,1], N, p=[0.79,0.21])
flag_doc3        = np.random.choice([0,1], N, p=[0.18,0.82])
reg_not_work     = np.random.choice([0,1], N, p=[0.77,0.23])

# ── Realistic TARGET based on risk factors ────────────────────────────────────
logit = (
    - 2.5
    - 1.8 * ext1
    - 2.2 * ext2
    - 1.9 * ext3
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

print(f"  Samples         : {N:,}")
print(f"  Default rate    : {target.mean():.1%}  (realistic: ~8-10%)")

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame({
    "CODE_GENDER"               : gender,
    "DAYS_BIRTH"                : -(age_years * 365).astype(int),
    "DAYS_EMPLOYMENT"           : -(employment_years * 365).astype(int),
    "CNT_CHILDREN"              : children,
    "CNT_FAM_MEMBERS"           : fam_members,
    "AMT_INCOME_TOTAL"          : income.round(2),
    "AMT_CREDIT"                : credit_amount.round(2),
    "AMT_ANNUITY"               : annuity.round(2),
    "AMT_GOODS_PRICE"           : (credit_amount * 0.85).round(2),
    "EXT_SOURCE_1"              : ext1.round(4),
    "EXT_SOURCE_2"              : ext2.round(4),
    "EXT_SOURCE_3"              : ext3.round(4),
    "NAME_CONTRACT_TYPE"        : contract_type,
    "NAME_EDUCATION_TYPE"       : education,
    "NAME_INCOME_TYPE"          : income_type,
    "NAME_HOUSING_TYPE"         : housing_type,
    "NAME_FAMILY_STATUS"        : family_status,
    "FLAG_OWN_CAR"              : own_car,
    "FLAG_OWN_REALTY"           : own_realty,
    "FLAG_PHONE"                : flag_phone,
    "FLAG_EMAIL"                : flag_email,
    "FLAG_WORK_PHONE"           : flag_work_phone,
    "FLAG_DOCUMENT_3"           : flag_doc3,
    "REG_CITY_NOT_WORK_CITY"    : reg_not_work,
    "LIVE_CITY_NOT_WORK_CITY"   : np.random.choice([0,1], N, p=[0.82,0.18]),
    "REGION_RATING_CLIENT"      : region_rating,
    "REGION_RATING_CLIENT_W_CITY": region_rating + np.random.choice([0,1], N, p=[0.85,0.15]),
    "REGION_POPULATION_RELATIVE": np.random.uniform(0.001, 0.073, N).round(4),
    "OCCUPATION_TYPE"           : np.random.choice(
        ["Laborers","Core staff","Accountants","Managers","Drivers",
         "Sales staff","Cleaning staff","Cooking staff","Medicine staff"],
        N, p=[0.27,0.15,0.10,0.09,0.09,0.08,0.06,0.05,0.11]
    ),
    "ORGANIZATION_TYPE"         : np.random.choice(
        ["Business Entity Type 3","School","Government","Religion","Other",
         "Medicine","Self-employed","Transport: type 4","Construction"],
        N, p=[0.18,0.12,0.10,0.04,0.15,0.07,0.13,0.06,0.15]
    ),
    "TARGET": target,
})

# Introduce ~5% missing values on EXT_SOURCE columns (realistic)
for col in ["EXT_SOURCE_1", "EXT_SOURCE_3", "OCCUPATION_TYPE"]:
    mask = np.random.random(N) < 0.05
    df.loc[mask, col] = np.nan

print(f"  Columns         : {df.shape[1]}")
print(f"  Missing values  : {df.isnull().sum().sum():,}")

# ── Feature engineering ───────────────────────────────────────────────────────
df["CREDIT_INCOME_RATIO"]  = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
df["AGE_YEARS"]            = -df["DAYS_BIRTH"] / 365
df["EMPLOYMENT_YEARS"]     = df["DAYS_EMPLOYMENT"].apply(lambda x: -x/365 if x < 0 else 0)
df["EXT_SOURCE_MEAN"]      = df[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]].mean(axis=1)
df["CHILDREN_RATIO"]       = df["CNT_CHILDREN"] / (df["CNT_FAM_MEMBERS"] + 1)
df["CREDIT_TERM"]          = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)

# ── Encode categoricals ───────────────────────────────────────────────────────
CATEGORICAL = [
    "CODE_GENDER","NAME_FAMILY_STATUS","NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE","NAME_INCOME_TYPE","OCCUPATION_TYPE",
    "ORGANIZATION_TYPE","NAME_CONTRACT_TYPE","FLAG_OWN_CAR","FLAG_OWN_REALTY",
]

le_dict = {}
for col in CATEGORICAL:
    le = LabelEncoder()
    df[col] = df[col].astype(str).fillna("Unknown")
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

feature_cols = [c for c in df.columns if c != "TARGET"]
X = df[feature_cols].values
y = df["TARGET"].values

# ── Impute ────────────────────────────────────────────────────────────────────
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

# ── Train / Test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "=" * 55)
print("  Training Model...")
print("=" * 55)

# Try LightGBM first, fallback to GradientBoosting
try:
    import lightgbm as lgb
    print("  Algorithm : LightGBM (best for tabular data)")
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
except ImportError:
    print("  Algorithm : Gradient Boosting (LightGBM not installed)")
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    )

model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.35).astype(int)   # lower threshold → catch more defaults
auc    = roc_auc_score(y_test, y_prob)

print(f"\n  ROC-AUC Score  : {auc:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Repaid (0)","Default (1)"]))

# ── Save bundle ───────────────────────────────────────────────────────────────
bundle = {
    "model"          : model,
    "label_encoders" : le_dict,
    "imputer"        : imputer,
    "feature_cols"   : feature_cols,
    "auc_score"      : round(auc, 4),
    "threshold"      : 0.35,
    "categorical"    : CATEGORICAL,
}
joblib.dump(bundle, "loan_model.pkl")

print(f"\n{'='*55}")
print(f"  ✅ Model saved  : loan_model.pkl")
print(f"  ROC-AUC        : {auc:.4f}")
print(f"  Features used  : {len(feature_cols)}")
print(f"{'='*55}")
