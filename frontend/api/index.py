from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import joblib
import os
import heapq

app = FastAPI(title="AI Loan Eligibility API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model bundle dynamically relative to this file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "loan_model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Model path not found: {MODEL_PATH}")
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

bundle = load_model()

# ── Request Models ───────────────────────────────────────────────────────────
class ApplicantData(BaseModel):
    name: str = Field(..., example="Rajan Kumar")
    age: float = Field(..., ge=18, le=100, example=35)
    gender: str = Field("M", example="M")
    education: str = Field("Secondary", example="Secondary")
    family_status: str = Field("Married", example="Married")
    children: int = Field(0, ge=0, example=0)
    family_members: int = Field(2, ge=1, example=2)
    own_car: str = Field("N", example="N")
    own_realty: str = Field("N", example="N")
    income_type: str = Field("Working", example="Working")
    monthly_income: float = Field(..., ge=0, example=50000)
    occupation: str = Field("Laborers", example="Laborers")
    employment_months: int = Field(36, ge=0, example=36)
    housing_type: str = Field("House / apartment", example="House / apartment")
    region_rating: int = Field(2, ge=1, le=3, example=2)
    loan_amount: float = Field(..., ge=1000, example=500000)
    loan_purpose: str = Field("Home Loan", example="Home Loan")
    tenure_months: int = Field(60, ge=1, example=60)
    ext1: float = Field(0.5, ge=0.0, le=1.0, example=0.6)
    ext2: float = Field(0.5, ge=0.0, le=1.0, example=0.65)
    ext3: float = Field(0.5, ge=0.0, le=1.0, example=0.55)

# ── A* Search Pre-Screening ──────────────────────────────────────────────────
class LoanEligibilityNode:
    def __init__(self, criteria_checked, score, remaining_criteria):
        self.criteria_checked = criteria_checked
        self.score = score
        self.remaining = remaining_criteria
        self.f = 0
    def __lt__(self, other):
        return self.f < other.f

def heuristic(remaining, weights):
    return sum(weights.get(c, 0) for c in remaining)

def astar_loan_check(data: dict):
    weights = {
        "ext_source_score": 30,
        "repayment_history": 25,
        "income_stability": 20,
        "debt_to_income": 15,
        "employment_status": 10,
    }
    criteria_order = list(weights.keys())

    def evaluate(criterion, d):
        if criterion == "ext_source_score":
            m = d.get("ext_source_mean", 0.5)
            if m >= 0.7:   return weights[criterion],        f"Excellent ({m:.2f})"
            elif m >= 0.55: return weights[criterion]*0.75,  f"Good ({m:.2f})"
            elif m >= 0.4:  return weights[criterion]*0.45,  f"Fair ({m:.2f})"
            else:           return 0,                         f"Poor ({m:.2f})"
        elif criterion == "repayment_history":
            years = d.get("employment_years", 0)
            own   = d.get("own_realty", "N")
            sc = 0
            if years >= 5:   sc += weights[criterion]*0.6
            elif years >= 2: sc += weights[criterion]*0.4
            elif years >= 1: sc += weights[criterion]*0.2
            if own == "Y":   sc += weights[criterion]*0.4
            return min(sc, weights[criterion]), f"{years:.1f} yrs employment, realty={own}"
        elif criterion == "income_stability":
            mo = d.get("employment_months", 0)
            if mo >= 36:   return weights[criterion],       f"Stable ({mo} months)"
            elif mo >= 12: return weights[criterion]*0.7,   f"Moderate ({mo} months)"
            elif mo >= 6:  return weights[criterion]*0.4,   f"Short ({mo} months)"
            else:          return 0,                         f"Insufficient ({mo} months)"
        elif criterion == "debt_to_income":
            dti = d.get("annuity_income_ratio", 1)*100
            if dti <= 20:   return weights[criterion],       f"Low DTI ({dti:.1f}%)"
            elif dti <= 35: return weights[criterion]*0.65,  f"Moderate DTI ({dti:.1f}%)"
            elif dti <= 50: return weights[criterion]*0.3,   f"High DTI ({dti:.1f}%)"
            else:           return 0,                         f"Very High DTI ({dti:.1f}%)"
        elif criterion == "employment_status":
            it = d.get("income_type", "").lower()
            if it in ["state servant","pensioner"]:          return weights[criterion],      f"Stable ({it})"
            elif it in ["working","commercial associate"]:   return weights[criterion]*0.8,  f"Employed ({it})"
            elif it == "self-employed":                      return weights[criterion]*0.6,  "Self-employed"
            else:                                            return 0,                        f"Other ({it})"
        return 0, "Unknown"

    start = LoanEligibilityNode([], 0, criteria_order[:])
    start.f = heuristic(criteria_order, weights)
    heap = [(start.f, 0, start)]
    best_path, criteria_results, counter = [], {}, 0

    while heap:
        _, _, node = heapq.heappop(heap)
        if not node.remaining:
            best_path = node.criteria_checked
            break
        nc = node.remaining[0]
        earned, detail = evaluate(nc, data)
        criteria_results[nc] = {"earned": earned, "max": weights[nc], "detail": detail}
        child = LoanEligibilityNode(
            node.criteria_checked + [nc],
            node.score + earned,
            node.remaining[1:]
        )
        child.f = child.score + heuristic(child.remaining, weights)
        counter += 1
        heapq.heappush(heap, (-child.f, counter, child))

    final_score = sum(v["earned"] for v in criteria_results.values())
    return final_score, best_path, criteria_results, weights

# ── ML Prediction ────────────────────────────────────────────────────────────
def predict_with_model(ap: dict):
    if bundle is None:
        return None
    model        = bundle["model"]
    le_dict      = bundle["label_encoders"]
    imputer      = bundle["imputer"]
    feature_cols = bundle["feature_cols"]
    threshold    = bundle.get("threshold", 0.35)
    categorical  = bundle.get("categorical", [])

    income   = ap["monthly_income"] * 12
    credit   = ap["loan_amount"]
    annuity  = credit / max(ap["tenure_months"], 1)
    emp_y    = ap["employment_months"] / 12
    ext_mean = (ap.get("ext1",0.5) + ap.get("ext2",0.5) + ap.get("ext3",0.5)) / 3

    row = {
        "CODE_GENDER": ap.get("gender","M"),
        "DAYS_BIRTH": -int(ap["age"]*365),
        "DAYS_EMPLOYMENT": -int(emp_y*365),
        "CNT_CHILDREN": ap.get("children",0),
        "CNT_FAM_MEMBERS": ap.get("family_members",2),
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": credit*0.85,
        "EXT_SOURCE_1": ap.get("ext1", ext_mean),
        "EXT_SOURCE_2": ap.get("ext2", ext_mean),
        "EXT_SOURCE_3": ap.get("ext3", ext_mean),
        "NAME_CONTRACT_TYPE": "Cash loans",
        "NAME_EDUCATION_TYPE": ap.get("education","Secondary"),
        "NAME_INCOME_TYPE": ap.get("income_type","Working"),
        "NAME_HOUSING_TYPE": ap.get("housing_type","House / apartment"),
        "NAME_FAMILY_STATUS": ap.get("family_status","Married"),
        "FLAG_OWN_CAR": ap.get("own_car","N"),
        "FLAG_OWN_REALTY": ap.get("own_realty","N"),
        "FLAG_PHONE": 1, "FLAG_EMAIL": 0, "FLAG_WORK_PHONE": 1,
        "FLAG_DOCUMENT_3": 1, "REG_CITY_NOT_WORK_CITY": 0,
        "LIVE_CITY_NOT_WORK_CITY": 0,
        "REGION_RATING_CLIENT": ap.get("region_rating",2),
        "REGION_RATING_CLIENT_W_CITY": ap.get("region_rating",2),
        "REGION_POPULATION_RELATIVE": 0.025,
        "OCCUPATION_TYPE": ap.get("occupation","Laborers"),
        "ORGANIZATION_TYPE": "Business Entity Type 3",
        "CREDIT_INCOME_RATIO": credit/(income+1),
        "ANNUITY_INCOME_RATIO": annuity/(income+1),
        "AGE_YEARS": ap["age"],
        "EMPLOYMENT_YEARS": emp_y,
        "EXT_SOURCE_MEAN": ext_mean,
        "CHILDREN_RATIO": ap.get("children",0)/max(ap.get("family_members",2),1),
        "CREDIT_TERM": credit/(annuity+1),
    }

    for col in categorical:
        if col in row and col in le_dict:
            le  = le_dict[col]
            val = str(row[col])
            row[col] = le.transform([val])[0] if val in list(le.classes_) else 0

    X = np.array([[row.get(f, 0) for f in feature_cols]], dtype=float)
    X = imputer.transform(X)

    prob_default = model.predict_proba(X)[0][1]
    decision = ("REJECTED" if prob_default >= threshold else
                "CONDITIONAL" if prob_default >= threshold*0.6 else "APPROVED")

    max_mult = max(0.3, 1.0 - prob_default*1.5)
    max_loan = min(credit, income*5*max_mult)
    rate     = round(8.5 + prob_default*12, 1)
    mr       = rate/100/12
    n        = ap.get("tenure_months", 60)
    emi      = (max_loan * mr * (1+mr)**n / ((1+mr)**n - 1)) if mr > 0 else max_loan/n

    monthly_inc = max(ap["monthly_income"], 1)
    dti = annuity / monthly_inc

    strengths, risks = [], []
    if ext_mean >= 0.6:            strengths.append("Strong external credit score")
    if emp_y >= 3:                 strengths.append(f"Stable employment ({emp_y:.1f} yrs)")
    if income >= 500000:           strengths.append("High annual income")
    if dti <= 0.3:                 strengths.append("Manageable Debt-to-Income (DTI) ratio")
    if ap.get("own_realty")=="Y":  strengths.append("Owns property (collateral)")
    if ext_mean < 0.4:             risks.append("Low external credit score")
    if emp_y < 1:                  risks.append("Very short employment history")
    if dti > 0.4:                  risks.append(f"High Debt-to-Income (DTI) ratio ({dti*100:.1f}%)")
    if credit > income*4:          risks.append("Loan amount very high vs income")
    if ap.get("children",0) >= 3:  risks.append("High number of dependents")
    if not strengths: strengths.append("No major strengths identified")
    if not risks:     risks.append("No major risk factors identified")
    if not risks:     risks.append("No major risk factors identified")

    return {
        "decision": decision,
        "confidence": round((1-prob_default)*100 if decision=="APPROVED" else prob_default*100, 1),
        "prob_default": round(prob_default*100, 2),
        "max_eligible_amount": round(max_loan),
        "suggested_interest_rate": f"{rate}%",
        "estimated_emi": round(emi),
        "key_strengths": strengths[:3],
        "key_risks": risks[:3],
        "conditions": (["Submit last 3 months salary slips","KYC documents required",
                        "ITR for last 2 years (if self-employed)"] if decision!="REJECTED" else []),
        "officer_note": (
            f"Applicant has a {prob_default*100:.1f}% estimated default probability. "
            f"{'Application meets lending criteria.' if decision=='APPROVED' else 'Application carries elevated risk.' if decision=='CONDITIONAL' else 'Does not meet minimum risk thresholds.'}"
        ),
    }

# ── API Endpoints ────────────────────────────────────────────────────────────
@app.get("/api/info")
def get_info():
    if bundle is None:
        return {"loaded": False, "error": f"loan_model.pkl not found at {MODEL_PATH}"}
    return {
        "loaded": True,
        "auc_score": bundle.get("auc_score", 0),
        "feature_count": len(bundle.get("feature_cols", [])),
        "algorithm": "LightGBM" if bundle.get("use_lgbm", True) else "GradientBoosting"
    }

@app.post("/api/predict")
def predict_loan(applicant: ApplicantData):
    ap_dict = applicant.dict()
    
    # Calculate helper parameters
    annuity_income = (ap_dict["loan_amount"] / max(ap_dict["tenure_months"], 1)) / max(ap_dict["monthly_income"], 1)
    
    # Enrich input dict
    ap_dict.update({
        "ext_source_mean": (ap_dict["ext1"] + ap_dict["ext2"] + ap_dict["ext3"]) / 3,
        "employment_years": ap_dict["employment_months"] / 12,
        "annuity_income_ratio": annuity_income,
    })

    # 1. Pre-Screening A* Search
    score, path, criteria_results, weights = astar_loan_check(ap_dict)
    
    # 2. ML Prediction
    ml_result = predict_with_model(ap_dict)
    
    if ml_result is None:
        ml_result = {
            "decision": "CONDITIONAL",
            "confidence": 50.0,
            "prob_default": 50.0,
            "max_eligible_amount": int(ap_dict["loan_amount"] * 0.5),
            "suggested_interest_rate": "12.0%",
            "estimated_emi": int(ap_dict["loan_amount"] * 0.02),
            "key_strengths": ["Pre-screening completed"],
            "key_risks": ["ML Model binary not loaded"],
            "conditions": ["Verify ID"],
            "officer_note": "ML Model was not loaded. Outputting rule-based pre-screening fallback."
        }

    return {
        "pre_screening": {
            "score": score,
            "path": path,
            "results": criteria_results,
            "weights": weights
        },
        "ml_result": ml_result
    }
