import React from 'react';
import { ArrowRight, Database, Brush, Activity, Expand, Shuffle, Brain, Sparkles } from 'lucide-react';

const PipelineDiagram = () => {
  const steps = [
    { name: 'Dataset', icon: Database, desc: 'UCI raw ledger' },
    { name: 'Cleaning', icon: Brush, desc: 'Invoice validation' },
    { name: 'Feature Eng.', icon: Activity, desc: 'Months 1-9 RFM' },
    { name: 'Scaling', icon: Expand, desc: 'Train StandardScaler' },
    { name: 'PCA / LDA', icon: Shuffle, desc: 'Collinearity drop' },
    { name: 'ML Modeling', icon: Brain, desc: 'MLP Classifier/Regressor' },
    { name: 'Prediction', icon: Sparkles, desc: 'LTV segment inference' },
  ];

  return (
    <div className="w-full bg-white border border-slate-200 rounded-2xl p-6 shadow-sm overflow-hidden">
      <h3 className="text-base font-bold text-slate-800 tracking-tight mb-6">
        🔧 Operational Machine Learning Pipeline Flow
      </h3>
      <div className="flex flex-col xl:flex-row items-center justify-between gap-4 xl:gap-2">
        {steps.map((step, index) => {
          const StepIcon = step.icon;
          return (
            <React.Fragment key={index}>
              <div className="flex flex-col items-center p-4 bg-slate-50 border border-slate-100 rounded-2xl w-full xl:w-[12%] text-center shadow-xs hover:border-blue-500 hover:bg-blue-50/20 transition-all group">
                <div className="p-2.5 bg-white text-blue-600 rounded-xl mb-3 shadow-xs border border-slate-100 group-hover:bg-blue-600 group-hover:text-white transition-all">
                  <StepIcon size={20} className="stroke-[2.2]" />
                </div>
                <div className="font-bold text-slate-800 text-xs tracking-tight">
                  {step.name}
                </div>
                <div className="text-[10px] text-slate-400 mt-1 font-medium leading-normal">
                  {step.desc}
                </div>
              </div>
              {index < steps.length - 1 && (
                <div className="text-slate-300 xl:rotate-0 rotate-90 my-2 xl:my-0">
                  <ArrowRight size={18} className="stroke-[2.5] text-slate-400" />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};

export default PipelineDiagram;
