import React, { useState } from 'react';
import { User, DollarSign, Shield, FileText, Check } from 'lucide-react';

const LoanForm = ({ formData, setFormData, onSubmit, isSubmitting, activeTab, setActiveTab }) => {
  const [unspecifiedTenure, setUnspecifiedTenure] = useState(false);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    let finalVal = type === 'checkbox' ? checked : value;
    
    // Parse numeric fields
    if (type === 'number') {
      finalVal = parseFloat(value) || 0;
    }

    setFormData((prev) => ({
      ...prev,
      [name]: finalVal,
    }));
  };

  const handleTenureCheckbox = (e) => {
    const checked = e.target.checked;
    setUnspecifiedTenure(checked);
    setFormData((prev) => ({
      ...prev,
      employment_months: checked ? 0 : 36,
    }));
  };

  const calculateCibil = () => {
    const avgExt = (formData.ext1 + formData.ext2 + formData.ext3) / 3;
    return Math.round(300 + avgExt * 600);
  };

  const calculateDti = () => {
    const annuity = formData.loan_amount / Math.max(formData.tenure_months, 1);
    const monthlyInc = Math.max(formData.monthly_income, 1);
    return ((annuity / monthlyInc) * 100).toFixed(1);
  };

  const renderTabIcon = (tab) => {
    switch (tab) {
      case 'profile': return <User size={18} />;
      case 'financials': return <DollarSign size={18} />;
      case 'loan': return <FileText size={18} />;
      case 'bureau': return <Shield size={18} />;
      default: return null;
    }
  };

  return (
    <div className="form-section">
      <div className="form-tabs">
        {['profile', 'financials', 'loan', 'bureau'].map((tab) => (
          <button
            key={tab}
            id={`tab-btn-${tab}`}
            type="button"
            className={`tab-btn ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {renderTabIcon(tab)}
            <span style={{ textTransform: 'capitalize' }}>{tab}</span>
          </button>
        ))}
      </div>

      <div className="form-content">
        {/* PROFILE TAB */}
        <div className={`tab-pane ${activeTab === 'profile' ? 'active' : ''}`}>
          <div className="form-group">
            <label htmlFor="name-input">Applicant Full Name</label>
            <input
              id="name-input"
              type="text"
              name="name"
              value={formData.name}
              onChange={handleChange}
              placeholder="e.g. Rajan Kumar"
              className="form-control"
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div className="form-group">
              <label htmlFor="age-input">Age</label>
              <input
                id="age-input"
                type="number"
                name="age"
                min="21"
                max="70"
                value={formData.age}
                onChange={handleChange}
                className="form-control"
              />
            </div>
            <div className="form-group">
              <label htmlFor="gender-select">Gender</label>
              <select
                id="gender-select"
                name="gender"
                value={formData.gender}
                onChange={handleChange}
                className="form-control"
              >
                <option value="M">Male</option>
                <option value="F">Female</option>
              </select>
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="education-select">Education Level</label>
            <select
              id="education-select"
              name="education"
              value={formData.education}
              onChange={handleChange}
              className="form-control"
            >
              <option value="Secondary">Secondary / secondary special</option>
              <option value="Higher education">Higher education</option>
              <option value="Incomplete higher">Incomplete higher</option>
              <option value="Lower secondary">Lower secondary</option>
              <option value="Academic degree">Academic degree</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="family-status-select">Family Status</label>
            <select
              id="family-status-select"
              name="family_status"
              value={formData.family_status}
              onChange={handleChange}
              className="form-control"
            >
              <option value="Married">Married</option>
              <option value="Single / not married">Single / not married</option>
              <option value="Civil marriage">Civil marriage</option>
              <option value="Separated">Separated</option>
              <option value="Widow">Widow</option>
            </select>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div className="form-group">
              <label htmlFor="children-input">Children</label>
              <input
                id="children-input"
                type="number"
                name="children"
                min="0"
                max="10"
                value={formData.children}
                onChange={handleChange}
                className="form-control"
              />
            </div>
            <div className="form-group">
              <label htmlFor="family-members-input">Family Members</label>
              <input
                id="family-members-input"
                type="number"
                name="family_members"
                min="1"
                max="15"
                value={formData.family_members}
                onChange={handleChange}
                className="form-control"
              />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '0.5rem' }}>
            <div className="form-group">
              <label htmlFor="own-car-select">Owns Car?</label>
              <select
                id="own-car-select"
                name="own_car"
                value={formData.own_car}
                onChange={handleChange}
                className="form-control"
              >
                <option value="N">No</option>
                <option value="Y">Yes</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="own-realty-select">Owns Property?</label>
              <select
                id="own-realty-select"
                name="own_realty"
                value={formData.own_realty}
                onChange={handleChange}
                className="form-control"
              >
                <option value="N">No</option>
                <option value="Y">Yes</option>
              </select>
            </div>
          </div>
        </div>

        {/* FINANCIALS TAB */}
        <div className={`tab-pane ${activeTab === 'financials' ? 'active' : ''}`}>
          <div className="form-group">
            <label htmlFor="income-type-select">Income Type</label>
            <select
              id="income-type-select"
              name="income_type"
              value={formData.income_type}
              onChange={handleChange}
              className="form-control"
            >
              <option value="Working">Working</option>
              <option value="Commercial associate">Commercial associate</option>
              <option value="State servant">State servant</option>
              <option value="Pensioner">Pensioner</option>
              <option value="Self-employed">Self-employed</option>
              <option value="Unemployed">Unemployed</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="monthly-income-input">Monthly Income (Rs.)</label>
            <input
              id="monthly-income-input"
              type="number"
              name="monthly_income"
              min="0"
              value={formData.monthly_income}
              onChange={handleChange}
              className="form-control"
            />
          </div>

          {formData.income_type !== 'Unemployed' && formData.monthly_income > 0 && (
            <div className="form-group">
              <label htmlFor="occupation-select">Occupation</label>
              <select
                id="occupation-select"
                name="occupation"
                value={formData.occupation}
                onChange={handleChange}
                className="form-control"
              >
                <option value="Laborers">Laborers</option>
                <option value="Core staff">Core staff</option>
                <option value="Accountants">Accountants</option>
                <option value="Managers">Managers</option>
                <option value="Drivers">Drivers</option>
                <option value="Sales staff">Sales staff</option>
                <option value="Medicine staff">Medicine staff</option>
                <option value="Cleaning staff">Cleaning staff</option>
                <option value="Unknown">Other / Unspecified</option>
              </select>
            </div>
          )}

          <div className="form-group">
            <label>Current Job Tenure</label>
            <div className="checkbox-group" onClick={() => handleTenureCheckbox({ target: { checked: !unspecifiedTenure } })}>
              <input
                type="checkbox"
                checked={unspecifiedTenure}
                onChange={handleTenureCheckbox}
              />
              <div className="checkbox-custom">
                {unspecifiedTenure && <Check size={14} />}
              </div>
              <span className="checkbox-label">Temporary / Irregular Tenure</span>
            </div>
            
            {!unspecifiedTenure && (
              <div style={{ marginTop: '1rem' }}>
                <label htmlFor="employment-years-input">Tenure in Current Job (Years)</label>
                <input
                  id="employment-years-input"
                  type="number"
                  name="employment_months"
                  step="0.5"
                  min="0"
                  value={formData.employment_months / 12}
                  onChange={(e) => {
                    const yrs = parseFloat(e.target.value) || 0;
                    setFormData((prev) => ({
                      ...prev,
                      employment_months: Math.round(yrs * 12),
                    }));
                  }}
                  className="form-control"
                />
              </div>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="housing-type-select">Housing Type</label>
            <select
              id="housing-type-select"
              name="housing_type"
              value={formData.housing_type}
              onChange={handleChange}
              className="form-control"
            >
              <option value="House / apartment">House / apartment</option>
              <option value="With parents">With parents</option>
              <option value="Municipal apartment">Municipal apartment</option>
              <option value="Rented apartment">Rented apartment</option>
              <option value="Office apartment">Office apartment</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="region-rating-select">Region Rating</label>
            <select
              id="region-rating-select"
              name="region_rating"
              value={formData.region_rating}
              onChange={handleChange}
              className="form-control"
            >
              <option value="1">1 (Best Region)</option>
              <option value="2">2 (Average Region)</option>
              <option value="3">3 (Riskier Region)</option>
            </select>
          </div>
        </div>

        {/* LOAN REQUEST TAB */}
        <div className={`tab-pane ${activeTab === 'loan' ? 'active' : ''}`}>
          <div className="form-group">
            <label htmlFor="loan-amount-input">Loan Amount (Rs.)</label>
            <input
              id="loan-amount-input"
              type="number"
              name="loan_amount"
              min="10000"
              value={formData.loan_amount}
              onChange={handleChange}
              className="form-control"
            />
          </div>

          <div className="form-group">
            <label htmlFor="loan-purpose-select">Loan Purpose</label>
            <select
              id="loan-purpose-select"
              name="loan_purpose"
              value={formData.loan_purpose}
              onChange={handleChange}
              className="form-control"
            >
              <option value="Home Loan">Home Loan</option>
              <option value="Personal Loan">Personal Loan</option>
              <option value="Vehicle Loan">Vehicle Loan</option>
              <option value="Education Loan">Education Loan</option>
              <option value="Business Loan">Business Loan</option>
              <option value="Medical Loan">Medical Loan</option>
              <option value="Gold Loan">Gold Loan</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="tenure-months-select">Tenure (Months)</label>
            <select
              id="tenure-months-select"
              name="tenure_months"
              value={formData.tenure_months}
              onChange={handleChange}
              className="form-control"
            >
              {[12, 24, 36, 48, 60, 84, 120, 180, 240, 360].map((t) => (
                <option key={t} value={t}>{t} Months</option>
              ))}
            </select>
          </div>

          <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              <span>Debt-to-Income (DTI) Ratio:</span>
              <span style={{ fontWeight: '700', color: parseFloat(calculateDti()) > 40 ? 'var(--color-danger)' : 'var(--color-success)' }}>
                {calculateDti()}%
              </span>
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
              Values below 35% are preferred by lenders.
            </p>
          </div>
        </div>

        {/* BUREAU TAB */}
        <div className={`tab-pane ${activeTab === 'bureau' ? 'active' : ''}`}>
          <div className="form-group slider-container">
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <label htmlFor="ext1-slider">Alternative Credit Score (Ext 1)</label>
              <span style={{ fontFamily: 'var(--font-heading)', fontWeight: '600' }}>{formData.ext1.toFixed(2)}</span>
            </div>
            <input
              id="ext1-slider"
              type="range"
              name="ext1"
              min="0.0"
              max="1.0"
              step="0.01"
              value={formData.ext1}
              onChange={(e) => setFormData((prev) => ({ ...prev, ext1: parseFloat(e.target.value) }))}
              className="slider-control"
            />
            <div className="slider-values">
              <span>0.0 (Worst)</span>
              <span>1.0 (Best)</span>
            </div>
          </div>

          <div className="form-group slider-container">
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <label htmlFor="ext2-slider">Primary Bureau Score (Ext 2)</label>
              <span style={{ fontFamily: 'var(--font-heading)', fontWeight: '600' }}>{formData.ext2.toFixed(2)}</span>
            </div>
            <input
              id="ext2-slider"
              type="range"
              name="ext2"
              min="0.0"
              max="1.0"
              step="0.01"
              value={formData.ext2}
              onChange={(e) => setFormData((prev) => ({ ...prev, ext2: parseFloat(e.target.value) }))}
              className="slider-control"
            />
            <div className="slider-values">
              <span>0.0 (Worst)</span>
              <span>1.0 (Best)</span>
            </div>
          </div>

          <div className="form-group slider-container">
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <label htmlFor="ext3-slider">Behavioral Risk Score (Ext 3)</label>
              <span style={{ fontFamily: 'var(--font-heading)', fontWeight: '600' }}>{formData.ext3.toFixed(2)}</span>
            </div>
            <input
              id="ext3-slider"
              type="range"
              name="ext3"
              min="0.0"
              max="1.0"
              step="0.01"
              value={formData.ext3}
              onChange={(e) => setFormData((prev) => ({ ...prev, ext3: parseFloat(e.target.value) }))}
              className="slider-control"
            />
            <div className="slider-values">
              <span>0.0 (Worst)</span>
              <span>1.0 (Best)</span>
            </div>
          </div>

          <div style={{ marginTop: '1.5rem', padding: '1.25rem', background: 'rgba(59, 130, 246, 0.05)', borderRadius: '12px', border: '1px solid rgba(59, 130, 246, 0.15)', textAlign: 'center' }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: '700', letterSpacing: '0.5px' }}>
              Estimated CIBIL Equivalent
            </div>
            <div style={{ fontSize: '2.5rem', fontWeight: '800', fontFamily: 'var(--font-heading)', margin: '0.25rem 0', color: calculateCibil() >= 700 ? 'var(--color-success)' : calculateCibil() >= 550 ? 'var(--color-warning)' : 'var(--color-danger)' }}>
              {calculateCibil()}
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Range: 300 to 900 (Calculated from average bureau ratings)
            </div>
          </div>
        </div>
      </div>

      <div className="form-footer">
        <button
          id="btn-check-eligibility"
          type="button"
          className="btn-submit"
          onClick={onSubmit}
          disabled={isSubmitting}
        >
          {isSubmitting ? (
            <>
              <div className="spinner"></div>
              <span>Analyzing Profile...</span>
            </>
          ) : (
            <>
              <span>🔍 Check Loan Eligibility</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default LoanForm;
