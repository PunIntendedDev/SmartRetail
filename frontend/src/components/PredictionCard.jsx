import React from 'react';
import StatBadge from './StatBadge';
import { Sparkles, TrendingUp, AlertTriangle, Lightbulb } from 'lucide-react';

const PredictionCard = ({ prediction, confidence, futureSpend, riskLevel, recommendation }) => {
  const isHighValue = prediction === 'High Value';
  const themeColor = isHighValue ? 'green' : 'yellow';

  const cardColors = {
    green: 'border-green-500 shadow-green-500/5',
    yellow: 'border-yellow-500 shadow-yellow-500/5',
  };

  return (
    <div className={`bg-white border border-slate-200 border-t-4 ${cardColors[themeColor]} rounded-2xl p-6 shadow-sm space-y-6`}>
      <h3 className="text-base font-bold text-slate-800 tracking-tight flex items-center gap-2">
        <Sparkles size={18} className="text-blue-600" />
        Diagnostic Inference Result
      </h3>

      <div className="grid grid-cols-2 gap-4">
        {/* Customer Segment */}
        <div className="p-4 bg-slate-50 border border-slate-100 rounded-xl">
          <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
            Customer Segment
          </div>
          <div className="mt-1.5">
            <StatBadge type={isHighValue ? 'success' : 'warning'} label={isHighValue ? '💎 High Value' : '👤 Standard'} />
          </div>
        </div>

        {/* Prediction Confidence */}
        <div className="p-4 bg-slate-50 border border-slate-100 rounded-xl">
          <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider flex items-center gap-1">
            <TrendingUp size={12} />
            Probability Rating
          </div>
          <div className="text-lg font-extrabold text-slate-900 mt-1">
            {confidence}%
          </div>
        </div>

        {/* Forecast LTV Spend */}
        <div className="p-4 bg-slate-50 border border-slate-100 rounded-xl">
          <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
            Future Spend (Forecast)
          </div>
          <div className="text-lg font-extrabold text-slate-900 mt-1">
            ${futureSpend}
          </div>
        </div>

        {/* Churn Risk Assessment */}
        <div className="p-4 bg-slate-50 border border-slate-100 rounded-xl">
          <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider flex items-center gap-1">
            <AlertTriangle size={12} />
            Risk Level
          </div>
          <div className="mt-1.5">
            <StatBadge type={isHighValue ? 'success' : 'warning'} label={isHighValue ? 'Low Risk' : 'Medium Risk'} />
          </div>
        </div>
      </div>

      {/* Suggested Campaign Action */}
      <div className="p-4 bg-blue-50/50 border border-blue-100 rounded-xl flex gap-3 items-start">
        <div className="p-1.5 bg-blue-600 text-white rounded-lg mt-0.5">
          <Lightbulb size={16} />
        </div>
        <div>
          <div className="text-[10px] uppercase font-bold text-blue-600 tracking-wider">
            Suggested Campaign Recommendation
          </div>
          <p className="text-xs font-semibold text-slate-700 mt-1 leading-normal">
            {recommendation}
          </p>
        </div>
      </div>
    </div>
  );
};

export default PredictionCard;
