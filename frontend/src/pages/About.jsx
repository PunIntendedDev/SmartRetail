import React from 'react';
import PageHeader from '../components/PageHeader';
import { Github, Linkedin, FileText, GitPullRequest, Code2, Server } from 'lucide-react';

const About = () => {
  const techStack = [
    'Python 3.11', 'Scikit-Learn', 'React 19', 'Vite', 'Tailwind CSS', 'Recharts', 'Framer Motion', 'Pandas', 'NumPy', 'Joblib'
  ];

  return (
    <div className="space-y-8">
      <PageHeader
        title="About Project & System Architecture"
        subtitle="Technical documentation, algorithms overview, and developer credentials."
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left 2 Cols: Docs */}
        <div className="lg:col-span-2 space-y-8">
          {/* Project Objectives */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h3 className="text-base font-bold text-slate-800 tracking-tight mb-3">
              🎯 Project Objective
            </h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              SmartRetail AI is an end-to-end analytical framework built to segment transaction databases into customer value tiers and forecast next-quarter spends (LTV). It acts as a modular decision support application, highlighting high-value cohorts and identifying standard clients for promotional campaign distribution.
            </p>
          </div>

          {/* Core Pipeline Timeline */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h3 className="text-base font-bold text-slate-800 tracking-tight mb-5">
              🔄 Processing Pipeline Flowchart
            </h3>
            <div className="relative pl-6 border-l-2 border-slate-200 space-y-6">
              <div className="relative">
                <div className="absolute -left-[31px] top-1 w-2.5 h-2.5 rounded-full bg-blue-600 border border-white" />
                <h4 className="text-xs font-bold text-slate-800">Phase 1: Ingestion & Ledger Validation</h4>
                <p className="text-[10px] text-slate-500 mt-1">Filters cancellations, duplicates, non-positive values, and null CustomerIDs from transaction tables.</p>
              </div>
              <div className="relative">
                <div className="absolute -left-[31px] top-1 w-2.5 h-2.5 rounded-full bg-blue-600 border border-white" />
                <h4 className="text-xs font-bold text-slate-800">Phase 2: Operational Splits & RFM Extraction</h4>
                <p className="text-[10px] text-slate-500 mt-1">Locks variables to Dec-Aug (Months 1-9) features; locks spending targets to Sep-Dec (Months 10-12) to avoid future data leakage.</p>
              </div>
              <div className="relative">
                <div className="absolute -left-[31px] top-1 w-2.5 h-2.5 rounded-full bg-blue-600 border border-white" />
                <h4 className="text-xs font-bold text-slate-800">Phase 3: Unsupervised PCA & Supervised LDA</h4>
                <p className="text-[10px] text-slate-500 mt-1">Fits PCA on spend compositions, dropping Other_Spend_Pct to avoid perfect multicollinearity. Fits LDA to establish separation boundaries.</p>
              </div>
              <div className="relative">
                <div className="absolute -left-[31px] top-1 w-2.5 h-2.5 rounded-full bg-blue-600 border border-white" />
                <h4 className="text-xs font-bold text-slate-800">Phase 4: ML Classifier & Regressor Tuning</h4>
                <p className="text-[10px] text-slate-500 mt-1">Runs GridSearchCV, fitting MLPClassifier (32-16 layers, ReLU) and MLPRegressor on scaled training arrays.</p>
              </div>
              <div className="relative">
                <div className="absolute -left-[31px] top-1 w-2.5 h-2.5 rounded-full bg-blue-600 border border-white" />
                <h4 className="text-xs font-bold text-slate-800">Phase 5: SaaS Frontend Deployment</h4>
                <p className="text-[10px] text-slate-500 mt-1">Deploys standard React dashboard, loading serialized joblib binaries offline.</p>
              </div>
            </div>
          </div>

          {/* Folder structure mapping */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h3 className="text-base font-bold text-slate-800 tracking-tight mb-3 flex items-center gap-2">
              <Code2 className="text-blue-600" size={18} />
              Platform Directory Blueprint
            </h3>
            <pre className="p-4 bg-slate-50 border border-slate-100 rounded-xl text-[10px] text-slate-600 font-mono leading-normal overflow-x-auto">
{`project-root/
│
├── config/              # YAML training configs
├── data/                # Raw Online Retail.xlsx splits & processed CSVs
├── models/              # Serialized scaler, PCA, and ML joblib models
├── outputs/             # Pre-saved training figures & metric plots
├── src/                 # Python pipeline: Preprocessing, PCA, ML modeling
│   ├── preprocessing.py
│   ├── dimensionality_reduction.py
│   └── classification.py
├── frontend/            # React Vite & Tailwind SaaS dashboard
│   ├── src/
│   │   ├── components/  # Reusable UI widgets
│   │   ├── pages/       # Dashboard routes
│   │   └── services/    # REST API preparation client
└── app.py               # Streamlit alternative server entry`}
            </pre>
          </div>
        </div>

        {/* Right 1 Col: Tech Stack & Dev Details */}
        <div className="space-y-6">
          {/* Tech stack */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h3 className="text-base font-bold text-slate-800 tracking-tight mb-4">
              🛠️ Tech Stack & Libraries
            </h3>
            <div className="flex flex-wrap gap-2">
              {techStack.map((tech) => (
                <span key={tech} className="bg-slate-50 border border-slate-200/60 rounded-full px-3 py-1 text-xs font-semibold text-slate-600">
                  {tech}
                </span>
              ))}
            </div>
          </div>

          {/* Dev credentials */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-4">
            <h3 className="text-base font-bold text-slate-800 tracking-tight">
              👤 Developer Details
            </h3>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between border-b border-slate-100 pb-2">
                <span className="text-slate-500 font-semibold">Project Role</span>
                <span className="text-slate-800 font-bold">ML Architect & UI/UX Developer</span>
              </div>
              <div className="flex justify-between border-b border-slate-100 pb-2">
                <span className="text-slate-500 font-semibold">Course Module</span>
                <span className="text-slate-800 font-bold">Final Year Project (FYP)</span>
              </div>
              <div className="flex justify-between pb-2">
                <span className="text-slate-500 font-semibold">Database Scope</span>
                <span className="text-slate-800 font-bold">541,909 Invoice Log Sheets</span>
              </div>
            </div>

            {/* Social link buttons */}
            <div className="pt-4 border-t border-slate-100 flex flex-col gap-2">
              <a
                href="https://github.com"
                target="_blank"
                rel="noreferrer"
                className="flex items-center justify-center gap-2 border border-slate-200 hover:bg-slate-50 font-bold text-slate-700 text-xs py-2 px-4 rounded-xl transition-all"
              >
                <Github size={14} />
                Explore Repository
              </a>
              <a
                href="https://linkedin.com"
                target="_blank"
                rel="noreferrer"
                className="flex items-center justify-center gap-2 border border-slate-200 hover:bg-slate-50 font-bold text-slate-700 text-xs py-2 px-4 rounded-xl transition-all"
              >
                <Linkedin size={14} className="text-blue-600" />
                LinkedIn Profile
              </a>
              <a
                href="https://github.com"
                target="_blank"
                rel="noreferrer"
                className="flex items-center justify-center gap-2 border border-slate-200 hover:bg-slate-50 font-bold text-slate-700 text-xs py-2 px-4 rounded-xl transition-all"
              >
                <FileText size={14} />
                Documentation PDF
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
