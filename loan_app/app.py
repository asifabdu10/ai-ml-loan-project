import streamlit as st
import heapq
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="AI Loan Eligibility Checker",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    h1, h2, h3 { color: #1a2f5a; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "loan_model.pkl")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

bundle = load_model()

# ── A* Search ─────────────────────────────────────────────────────────────────
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

def astar_loan_check(data):
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

# ── ML Prediction ─────────────────────────────────────────────────────────────
def predict_with_model(ap):
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

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# 🏦 AI Loan Eligibility Checker")

if bundle:
    auc = bundle.get("auc_score", 0)
    nf  = len(bundle.get("feature_cols",[]))
    st.success(f"✅ AI Model loaded — ROC-AUC: **{auc:.4f}** | Features: **{nf}**")
else:
    st.error("❌ loan_model.pkl not found. Run `python generate_and_train.py` first.")

st.divider()

with st.sidebar:
    st.markdown("### 🤖 Model Info")
    if bundle:
        st.metric("ROC-AUC Score", f"{bundle.get('auc_score',0):.4f}")
        st.metric("Features Used", len(bundle.get("feature_cols",[])))
        st.metric("Algorithm", "LightGBM")
    st.divider()
    st.markdown("### 📊 Pre-Screening Weights")
    st.markdown("""
| Criterion | Weight |
|---|---|
| Avg External Score | 30 pts |
| Repayment History | 25 pts |
| Income Stability | 20 pts |
| Debt-to-Income | 15 pts |
| Employment Type | 10 pts |
    """)


col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 👤 Personal Information")
    name           = st.text_input("Applicant Name", placeholder="e.g. Rajan Kumar")
    age            = st.number_input("Age", 21, 70, 35)
    gender         = st.selectbox("Gender", ["M","F"])
    education      = st.selectbox("Education Level", [
        "Higher education","Secondary","Incomplete higher","Lower secondary","Academic degree"])
    family_status  = st.selectbox("Family Status", [
        "Married","Single / not married","Civil marriage","Separated","Widow"])
    children       = st.number_input("Number of Children", 0, 10, 0)
    family_members = st.number_input("Total Family Members", 1, 15, 2)
    own_car        = st.selectbox("Owns a Car?", ["N","Y"])
    own_realty     = st.selectbox("Owns Property/House?", ["N","Y"])

with col2:
    st.markdown("#### 💼 Employment & Income")
    income_type       = st.selectbox("Income Type", [
        "Working","Commercial associate","State servant","Pensioner","Self-employed","Unemployed"])
    monthly_income    = st.number_input("Monthly Income (Rs.)", 0, 2000000, 50000, step=5000)
    
    if income_type != "Unemployed" and monthly_income > 0:
        occupation = st.selectbox("Occupation", [
            "Laborers","Core staff","Accountants","Managers","Drivers","Sales staff","Medicine staff","Cleaning staff"])
    else:
        occupation = "Unknown"
        
    unspecified_tenure = st.checkbox("Unspecified / Irregular / Temporary Tenure", help="Check if tenure cannot be predicted or is temporary")
    if unspecified_tenure:
        employment_years = 0.0
    else:
        employment_years = st.number_input("Tenure in Current Job (Years)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
    employment_months = int(employment_years * 12)
    housing_type      = st.selectbox("Housing Type", [
        "House / apartment","With parents","Municipal apartment","Rented apartment","Office apartment"])
    region_rating     = st.selectbox("Region Rating", [1,2,3], index=1, help="1=Best, 3=Worst")

st.divider()
st.markdown("#### 🏠 Loan Request")
lc1, lc2, lc3 = st.columns(3)
with lc1: loan_amount   = st.number_input("Loan Amount (Rs.)", 10000, 50000000, 500000, step=50000)
with lc2: loan_purpose  = st.selectbox("Loan Purpose", [
    "Home Loan","Personal Loan","Vehicle Loan","Education Loan","Business Loan","Medical Loan","Gold Loan"])
with lc3: tenure_months = st.selectbox("Tenure (months)", [12,24,36,48,60,84,120,180,240,360])

st.divider()
st.markdown("#### 📊 External Credit Scores (Bureau / CIBIL)")
ec1, ec2, ec3 = st.columns(3)
with ec1: ext1 = st.slider("Alternative Credit Score (Ext 1)", 0.0, 1.0, 0.60, 0.01, help="0=worst, 1=best")
with ec2: ext2 = st.slider("Primary Bureau Score (Ext 2)", 0.0, 1.0, 0.65, 0.01)
with ec3: ext3 = st.slider("Behavioral Risk Score (Ext 3)", 0.0, 1.0, 0.55, 0.01)

cibil_approx   = int(300 + ((ext1+ext2+ext3)/3)*600)
annuity_income = (loan_amount/max(tenure_months,1)) / max(monthly_income,1)
dti_pct        = round(annuity_income*100, 1)
st.caption(f"Approximate CIBIL equivalent: **{cibil_approx}** (300–900) | Debt-to-Income: **{dti_pct}%**")

st.divider()

if st.button("🔍 Check Loan Eligibility", type="primary", use_container_width=True):
    applicant_data = {
        "name": name or "Applicant", "age": age, "gender": gender,
        "monthly_income": monthly_income, "income_type": income_type,
        "occupation": occupation, "employment_months": employment_months,
        "education": education, "family_status": family_status,
        "children": children, "family_members": max(family_members, children+1),
        "own_car": own_car, "own_realty": own_realty,
        "housing_type": housing_type, "region_rating": region_rating,
        "loan_amount": loan_amount, "tenure_months": tenure_months,
        "ext1": ext1, "ext2": ext2, "ext3": ext3,
        "ext_source_mean": (ext1+ext2+ext3)/3,
        "employment_years": employment_months/12,
        "annuity_income_ratio": annuity_income,
    }

    with st.spinner("Running A* search..."):
        score, path, criteria_results, weights = astar_loan_check(applicant_data)

    st.markdown("## 📊 Rule-Based Pre-Screening Analysis")
    card = "background:#ffffff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.15);text-align:center;margin:8px 0;"
    vs   = "font-size:2rem;font-weight:700;"
    ls   = "font-size:0.85rem;color:#555;margin-top:4px;"

    m1,m2,m3,m4 = st.columns(4)
    with m1: st.markdown(f'<div style="{card}"><div style="{vs}color:#4a90d9">{score:.0f}/100</div><div style="{ls}">Pre-Screening Score</div></div>', unsafe_allow_html=True)
    with m2:
        grade = "A" if score>=80 else "B" if score>=65 else "C" if score>=50 else "D"
        gc    = "#28a745" if score>=80 else "#ffc107" if score>=65 else "#fd7e14" if score>=50 else "#dc3545"
        st.markdown(f'<div style="{card}"><div style="{vs}color:{gc}">{grade}</div><div style="{ls}">Grade</div></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div style="{card}"><div style="{vs}color:#6f42c1">Rs.{monthly_income:,}</div><div style="{ls}">Monthly Income</div></div>', unsafe_allow_html=True)
    with m4: st.markdown(f'<div style="{card}"><div style="{vs}color:#e83e8c">{dti_pct}%</div><div style="{ls}">DTI Ratio</div></div>', unsafe_allow_html=True)

    st.markdown("### 🔎 Criteria Evaluation Order")
    path_cols = st.columns(len(path))
    for i,(step,col) in enumerate(zip(path, path_cols)):
        r    = criteria_results[step]
        pct  = r["earned"]/r["max"]*100 if r["max"]>0 else 0
        icon = "✅" if pct>=80 else "⚠️" if pct>=40 else "❌"
        with col:
            sb = "background:#fff;border-radius:8px;padding:10px 16px;margin:4px 0;border-left:3px solid #4a90d9;"
            st.markdown(
                f'<div style="{sb}">{icon} <b style="color:#1a2f5a">Step {i+1}</b>'
                f'<br><span style="color:#333;font-size:0.85rem">{step.replace("_"," ").title()}</span>'
                f'<br><small style="color:#666">{r["earned"]:.0f}/{r["max"]} pts</small></div>',
                unsafe_allow_html=True)

    st.markdown("### 📋 Criteria Breakdown")
    for crit, res in criteria_results.items():
        pct = res["earned"]/res["max"]*100 if res["max"]>0 else 0
        st.markdown(f"**{crit.replace('_',' ').title()}** — {res['detail']}")
        st.progress(pct/100)
        st.caption(f"{res['earned']:.1f} / {res['max']} points ({pct:.0f}%)")

    st.markdown("### ⚖️ Pre-Screening Verdict")
    if score >= 75:
        st.markdown('<div style="background:linear-gradient(135deg,#d4edda,#c3e6cb);border-left:5px solid #28a745;border-radius:10px;padding:20px;"><h3 style="color:#155724">✅ LIKELY ELIGIBLE</h3><p style="color:#155724">Strong profile. See ML decision below.</p></div>', unsafe_allow_html=True)
    elif score >= 50:
        st.markdown('<div style="background:linear-gradient(135deg,#fff3cd,#ffeeba);border-left:5px solid #ffc107;border-radius:10px;padding:20px;"><h3 style="color:#856404">⚠️ CONDITIONAL</h3><p style="color:#856404">Mixed signals. See ML decision below.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background:linear-gradient(135deg,#f8d7da,#f5c6cb);border-left:5px solid #dc3545;border-radius:10px;padding:20px;"><h3 style="color:#721c24">❌ LIKELY NOT ELIGIBLE</h3><p style="color:#721c24">Does not meet criteria.</p></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("## 🤖 ML Model Decision (AI)")

    with st.spinner("Running AI Model..."):
        ai = predict_with_model(applicant_data)

    if ai:
        decision = ai["decision"]
        if decision == "APPROVED":
            st.markdown(f'<div style="background:linear-gradient(135deg,#d4edda,#c3e6cb);border-left:5px solid #28a745;border-radius:10px;padding:24px;"><h2 style="color:#155724">✅ LOAN APPROVED</h2><p style="color:#155724">Confidence: {ai["confidence"]}% | Default Risk: {ai["prob_default"]}%</p></div>', unsafe_allow_html=True)
        elif decision == "CONDITIONAL":
            st.markdown(f'<div style="background:linear-gradient(135deg,#fff3cd,#ffeeba);border-left:5px solid #ffc107;border-radius:10px;padding:24px;"><h2 style="color:#856404">⚠️ CONDITIONAL APPROVAL</h2><p style="color:#856404">Confidence: {ai["confidence"]}% | Default Risk: {ai["prob_default"]}%</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background:linear-gradient(135deg,#f8d7da,#f5c6cb);border-left:5px solid #dc3545;border-radius:10px;padding:24px;"><h2 style="color:#721c24">❌ LOAN REJECTED</h2><p style="color:#721c24">Default Risk: {ai["prob_default"]}% (too high)</p></div>', unsafe_allow_html=True)

        st.divider()
        a1,a2,a3 = st.columns(3)
        with a1: st.metric("Max Eligible Amount", f"Rs.{ai['max_eligible_amount']:,}")
        with a2: st.metric("Interest Rate", ai["suggested_interest_rate"])
        with a3: st.metric("Estimated EMI", f"Rs.{ai['estimated_emi']:,}/month")

        prob = ai["prob_default"]
        gc2  = "#28a745" if prob<20 else "#ffc107" if prob<40 else "#dc3545"
        st.markdown(f"""
        <div style="background:white;border-radius:10px;padding:16px;margin:8px 0;box-shadow:0 2px 6px rgba(0,0,0,0.1)">
            <div style="color:#333;font-weight:600;margin-bottom:8px">🎯 Default Risk Probability</div>
            <div style="background:#eee;border-radius:10px;height:22px;overflow:hidden">
                <div style="background:{gc2};width:{min(prob,100)}%;height:100%;border-radius:10px"></div>
            </div>
            <div style="color:{gc2};font-size:1.3rem;font-weight:700;margin-top:6px">{prob}%</div>
        </div>""", unsafe_allow_html=True)

        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("**💪 Key Strengths**")
            for s in ai["key_strengths"]: st.markdown(f"• {s}")
        with rc2:
            st.markdown("**⚠️ Key Risks**")
            for r in ai["key_risks"]: st.markdown(f"• {r}")

        if ai["conditions"]:
            st.markdown("**📋 Required Documents**")
            for c in ai["conditions"]: st.markdown(f"• {c}")

        st.info(f"🏦 **Officer Note:** {ai['officer_note']}")
    else:
        st.warning("Model not loaded. Run `python generate_and_train.py` first.")

st.divider()
st.caption("⚠️ For informational purposes only. Final decisions subject to bank policy.")