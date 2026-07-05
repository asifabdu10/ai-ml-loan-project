import React, { useState, useEffect } from 'react';
import { Landmark, Activity, CheckCircle, AlertTriangle } from 'lucide-react';
import LoanForm from './LoanForm';
import ResultsDashboard from './ResultsDashboard';
import './App.css';

const initialFormData = {
  name: 'Rajan Kumar',
  age: 35,
  gender: 'M',
  education: 'Secondary',
  family_status: 'Married',
  children: 0,
  family_members: 2,
  own_car: 'N',
  own_realty: 'N',
  income_type: 'Working',
  monthly_income: 50000,
  occupation: 'Laborers',
  employment_months: 36,
  housing_type: 'House / apartment',
  region_rating: 2,
  loan_amount: 500000,
  loan_purpose: 'Home Loan',
  tenure_months: 60,
  ext1: 0.60,
  ext2: 0.65,
  ext3: 0.55,
};

function App() {
  const [formData, setFormData] = useState(initialFormData);
  const [results, setResults] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [apiInfo, setApiInfo] = useState(null);
  const [activeTab, setActiveTab] = useState('profile');

  // Fetch model metadata on load
  useEffect(() => {
    const fetchApiInfo = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/info');
        if (response.ok) {
          const data = await response.json();
          setApiInfo(data);
        } else {
          setApiInfo({ loaded: false, error: 'Server error' });
        }
      } catch (err) {
        setApiInfo({ loaded: false, error: 'Cannot connect to API' });
      }
    };
    fetchApiInfo();
  }, []);

  const handleEvaluate = async () => {
    setIsSubmitting(true);
    // Smooth scroll results to top on mobile
    if (window.innerWidth <= 1024) {
      setTimeout(() => {
        const resultsEl = document.getElementById('results-anchor');
        if (resultsEl) {
          resultsEl.scrollIntoView({ behavior: 'smooth' });
        }
      }, 100);
    }
    
    try {
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('API server returned an error');
      }

      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Error during evaluation:', error);
      alert(
        'Failed to connect to the backend server. Please make sure that the Python FastAPI backend is running at http://localhost:8000'
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <>
      {/* Header */}
      <header className="app-header">
        <div className="logo-container">
          <Landmark size={28} className="logo-icon" />
          <h1 className="app-title">AI Loan Eligibility Checker</h1>
        </div>
        

      </header>

      {/* Main Container */}
      <main className="app-main">
        {/* Sidebar Inputs */}
        <LoanForm
          formData={formData}
          setFormData={setFormData}
          onSubmit={handleEvaluate}
          isSubmitting={isSubmitting}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
        />

        {/* Dashboard Results */}
        <div id="results-anchor">
          <ResultsDashboard
            results={results}
            isLoading={isSubmitting}
            formData={formData}
          />
        </div>
      </main>
    </>
  );
}

export default App;
