import React from 'react';
import { 
  TrendingUp, AlertCircle, CheckCircle2, XCircle, 
  HelpCircle, Sparkles, Milestone, Scale, FileSpreadsheet, FileCheck, Info
} from 'lucide-react';

const ResultsDashboard = ({ results, isLoading, formData }) => {
  if (isLoading) {
    return (
      <div className="dashboard-section">
        <div className="loading-container">
          <div className="spinner large-spinner"></div>
          <div style={{ textAlign: 'center' }}>
            <h3 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.25rem', marginBottom: '0.25rem' }}>AI is Evaluating Loan Profile</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Running A* heuristic search and model inference...</p>
          </div>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="dashboard-section">
        <div className="placeholder-dashboard">
          <TrendingUp size={48} className="placeholder-icon" />
          <h2>Decision Intelligence Console</h2>
          <p>Fill out the applicant details on the left and trigger evaluation to see the decision pipeline in action.</p>
        </div>
      </div>
    );
  }

  const { pre_screening, ml_result } = results;
  const score = pre_screening.score;
  const grade = score >= 80 ? 'A' : score >= 65 ? 'B' : score >= 50 ? 'C' : 'D';
  const gradeColor = score >= 80 ? 'var(--color-success)' : score >= 65 ? 'var(--color-warning)' : score >= 50 ? '#fd7e14' : 'var(--color-danger)';
  
  // DTI calculation
  const annuity = formData.loan_amount / Math.max(formData.tenure_months, 1);
  const dti = ((annuity / Math.max(formData.monthly_income, 1)) * 100).toFixed(1);

  // Verdict config
  let verdictClass = 'rejected';
  let verdictTitle = '❌ LIKELY NOT ELIGIBLE';
  let verdictDesc = 'Applicant profile does not meet the minimum rule-based pre-screening thresholds.';
  if (score >= 75) {
    verdictClass = 'eligible';
    verdictTitle = '✅ LIKELY ELIGIBLE';
    verdictDesc = 'Strong applicant profile. Meets rule-based criteria. Machine learning verdict is shown below.';
  } else if (score >= 50) {
    verdictClass = 'conditional';
    verdictTitle = '⚠️ CONDITIONAL';
    verdictDesc = 'Applicant profile has mixed signals. Subject to higher machine learning validation.';
  }

  const getStepIcon = (earned, max) => {
    const pct = (earned / max) * 100;
    if (pct >= 80) return <CheckCircle2 size={16} style={{ color: 'var(--color-success)' }} />;
    if (pct >= 40) return <AlertCircle size={16} style={{ color: 'var(--color-warning)' }} />;
    return <XCircle size={16} style={{ color: 'var(--color-danger)' }} />;
  };

  const getStepClass = (earned, max) => {
    const pct = (earned / max) * 100;
    if (pct >= 80) return 'success';
    if (pct >= 40) return 'warning';
    return 'danger';
  };

  return (
    <div className="dashboard-section">
      <div className="results-container">
        
        {/* SECTION 1: PRE-SCREENING SUMMARY */}
        <div>
          <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.5rem', marginBottom: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
            Pre-Screening Report
          </h2>
          
          <div className="stats-grid" style={{ marginBottom: '1.5rem' }}>
            <div className="stat-card">
              <div className="stat-label">Pre-Screening Score</div>
              <div className="stat-val" style={{ color: 'var(--color-primary)' }}>{score.toFixed(0)}/100</div>
            </div>
            
            <div className="stat-card" style={{ borderColor: gradeColor }}>
              <div className="stat-label">Grade</div>
              <div className="stat-val" style={{ color: gradeColor }}>{grade}</div>
            </div>
            
            <div className="stat-card">
              <div className="stat-label">Monthly Income</div>
              <div className="stat-val" style={{ color: 'var(--color-success)', fontSize: '1.4rem', marginTop: '0.5rem' }}>
                Rs.{formData.monthly_income.toLocaleString()}
              </div>
            </div>
            
            <div className="stat-card">
              <div className="stat-label">DTI Ratio</div>
              <div className="stat-val" style={{ color: parseFloat(dti) > 40 ? 'var(--color-danger)' : 'var(--color-success)' }}>
                {dti}%
              </div>
            </div>
          </div>

          <div className={`verdict-banner ${verdictClass}`} style={{ marginBottom: '2rem' }}>
            <div className="verdict-header">
              <h3>{verdictTitle}</h3>
            </div>
            <p>{verdictDesc}</p>
          </div>
        </div>

        {/* SECTION 2: A* EVALUATION PATH */}
        <div className="dashboard-card">
          <div className="card-title">
            <Milestone size={18} />
            <span>A* Search Criteria Evaluation Order</span>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
            The A* algorithm explores the decision path dynamically, evaluating the most informative criteria first.
          </p>
          <div className="eval-path">
            {pre_screening.path.map((step, idx) => {
              const res = pre_screening.results[step];
              return (
                <div key={step} className={`eval-step ${getStepClass(res.earned, res.max)}`}>
                  <div className="eval-step-header">
                    <span>STEP {idx + 1}</span>
                    {getStepIcon(res.earned, res.max)}
                  </div>
                  <div className="eval-step-name" title={step.replace(/_/g, ' ')}>
                    {step.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                  </div>
                  <div className="eval-step-pts">
                    {res.earned.toFixed(0)}/{res.max} pts
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* SECTION 3: CRITERIA BREAKDOWN */}
        <div className="dashboard-card">
          <div className="card-title">
            <Scale size={18} />
            <span>Criteria Point Breakdown</span>
          </div>
          <div className="criteria-list">
            {Object.entries(pre_screening.results).map(([criteria, result]) => {
              const pct = (result.earned / result.max) * 100;
              let fillCol = 'var(--color-success)';
              if (pct < 40) fillCol = 'var(--color-danger)';
              else if (pct < 80) fillCol = 'var(--color-warning)';
              
              return (
                <div key={criteria}>
                  <div className="criteria-item-label">
                    <span style={{ textTransform: 'capitalize' }}>{criteria.replace(/_/g, ' ')}</span>
                    <span style={{ fontFamily: 'var(--font-heading)', fontWeight: '600' }}>
                      {result.earned.toFixed(1)} / {result.max} ({pct.toFixed(0)}%)
                    </span>
                  </div>
                  <div className="progress-bar-bg">
                    <div className="progress-bar-fill" style={{ width: `${pct}%`, backgroundColor: fillCol }}></div>
                  </div>
                  <div className="criteria-desc">{result.detail}</div>
                </div>
              );
            })}
          </div>
        </div>

        {/* SECTION 4: MACHINE LEARNING DECISION */}
        <div>
          <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.5rem', marginBottom: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
            <Sparkles size={20} style={{ display: 'inline', marginRight: '0.5rem', verticalAlign: 'text-bottom', color: '#818cf8' }} />
            AI Machine Learning Verdict
          </h2>
          
          <div className="dashboard-card" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            
            {/* Decision Banner */}
            <div className={`verdict-banner ${ml_result.decision.toLowerCase()}`} style={{ borderLeftWidth: '6px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <div style={{ fontSize: '0.8rem', fontWeight: '800', textTransform: 'uppercase', opacity: 0.8, letterSpacing: '1px', marginBottom: '0.25rem' }}>
                    Final AI Decision
                  </div>
                  <h3 style={{ fontSize: '1.75rem', fontFamily: 'var(--font-heading)', fontWeight: '800' }}>
                    LOAN {ml_result.decision}
                  </h3>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: '600', opacity: 0.8 }}>
                    Confidence / Risk
                  </div>
                  <div style={{ fontSize: '1.5rem', fontWeight: '800', fontFamily: 'var(--font-heading)' }}>
                    {ml_result.decision === 'APPROVED' ? `${ml_result.confidence}% Match` : `${ml_result.confidence}% Risk`}
                  </div>
                </div>
              </div>
            </div>

            {/* Metrics Row */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
              <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '1rem', textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: '600' }}>Max Eligible Amount</div>
                <div style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--text-main)', marginTop: '0.25rem' }}>
                  Rs.{ml_result.max_eligible_amount.toLocaleString()}
                </div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '1rem', textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: '600' }}>Interest Rate</div>
                <div style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--text-main)', marginTop: '0.25rem' }}>
                  {ml_result.suggested_interest_rate}
                </div>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '1rem', textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: '600' }}>Estimated EMI</div>
                <div style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--text-main)', marginTop: '0.25rem' }}>
                  Rs.{ml_result.estimated_emi.toLocaleString()}/mo
                </div>
              </div>
            </div>

            {/* Default Risk probability slider */}
            <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '1.25rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
                <span style={{ fontWeight: '600' }}>Estimated Default Risk Probability</span>
                <span style={{ fontWeight: '700', color: ml_result.prob_default >= 40 ? 'var(--color-danger)' : ml_result.prob_default >= 20 ? 'var(--color-warning)' : 'var(--color-success)' }}>
                  {ml_result.prob_default}%
                </span>
              </div>
              <div className="progress-bar-bg" style={{ height: '10px' }}>
                <div 
                  className="progress-bar-fill" 
                  style={{ 
                    width: `${ml_result.prob_default}%`, 
                    backgroundColor: ml_result.prob_default >= 40 ? 'var(--color-danger)' : ml_result.prob_default >= 20 ? 'var(--color-warning)' : 'var(--color-success)' 
                  }}
                ></div>
              </div>
            </div>

            {/* Strengths Risks bullet points */}
            <div className="strengths-risks-grid">
              <div>
                <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.35rem', fontWeight: '700' }}>
                  <CheckCircle2 size={15} style={{ color: 'var(--color-success)' }} /> Key Strengths
                </h4>
                <ul className="points-list strengths">
                  {ml_result.key_strengths.map((str, i) => (
                    <li key={i}>
                      <CheckCircle2 size={14} />
                      <span>{str}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.35rem', fontWeight: '700' }}>
                  <AlertCircle size={15} style={{ color: 'var(--color-danger)' }} /> Risk Factors
                </h4>
                <ul className="points-list risks">
                  {ml_result.key_risks.map((risk, i) => (
                    <li key={i}>
                      <AlertCircle size={14} />
                      <span>{risk}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Document Checklist */}
            {ml_result.conditions.length > 0 && (
              <div>
                <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.35rem', fontWeight: '700' }}>
                  <FileCheck size={15} style={{ color: 'var(--color-primary)' }} /> Required Documents / Conditions
                </h4>
                <ul className="points-list" style={{ listStyle: 'none' }}>
                  {ml_result.conditions.map((cond, i) => (
                    <li key={i} style={{ fontSize: '0.9rem', color: 'var(--text-main)' }}>
                      <span style={{ display: 'inline-block', width: '6px', height: '6px', backgroundColor: 'var(--color-primary)', borderRadius: '50%', marginRight: '0.5rem', alignSelf: 'center' }}></span>
                      <span>{cond}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Officer Note */}
            <div className="officer-note-box">
              <Info size={16} />
              <div>
                <strong>Underwriter Note:</strong> {ml_result.officer_note}
              </div>
            </div>
            
          </div>
        </div>

        <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: '1rem' }}>
          Disclaimer: This system provides an algorithmic pre-screening evaluation. Final decisions are subject to credit bureau confirmation, bank lending policies, and physical verification.
        </div>

      </div>
    </div>
  );
};

export default ResultsDashboard;
