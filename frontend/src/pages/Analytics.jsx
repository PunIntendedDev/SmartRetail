import React from 'react';
import PageHeader from '../components/PageHeader';
import MetricCard from '../components/MetricCard';
import ChartCard from '../components/ChartCard';
import { classificationComparisons, regressionComparisons, featureImportance, predictedVsActualData, residualData } from '../utils/mockData';
import { ResponsiveContainer, LineChart, Line, BarChart, Bar, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { BarChart3, Binary, Percent, AreaChart, GitMerge } from 'lucide-react';

const Analytics = () => {
  const rocCurveMock = [
    { fpr: 0.0, tpr: 0.0 },
    { fpr: 0.02, tpr: 0.4 },
    { fpr: 0.05, tpr: 0.75 },
    { fpr: 0.1, tpr: 0.9 },
    { fpr: 0.2, tpr: 0.96 },
    { fpr: 0.4, tpr: 0.99 },
    { fpr: 0.7, tpr: 1.0 },
    { fpr: 1.0, tpr: 1.0 },
  ];

  const prCurveMock = [
    { recall: 0.0, precision: 1.0 },
    { recall: 0.2, precision: 1.0 },
    { recall: 0.5, precision: 0.98 },
    { recall: 0.8, precision: 0.95 },
    { recall: 0.9, precision: 0.88 },
    { recall: 0.95, precision: 0.82 },
    { recall: 1.0, precision: 0.75 },
  ];

  return (
    <div className="space-y-8">
      <PageHeader
        title="Model Performance Audit"
        subtitle="Benchmark evaluation statistics, validation matrices, and residual diagnostics across splits."
      />

      {/* Model Sections */}
      <div>
        <h3 className="text-base font-extrabold text-slate-900 tracking-tight mb-4 flex items-center gap-2">
          <Percent className="text-blue-600" size={18} />
          High-Value Segment Classification
        </h3>
        
        {/* Metric grids */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
          <MetricCard title="Test Accuracy" value="100.0%" description="Correct segment predictions" color="green" />
          <MetricCard title="Test Precision" value="100.0%" description="Ratio of true high-value positives" color="green" />
          <MetricCard title="Test Recall" value="100.0%" description="Ratio of captured high-value class" color="purple" />
          <MetricCard title="Test F1-Score" value="1.000" description="Harmonic mean rating" color="purple" />
          <MetricCard title="Test ROC AUC" value="1.000" description="Area under curve rating" color="blue" />
        </div>

        {/* Classification comparisons table */}
        <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm mb-8">
          <h4 className="text-sm font-bold text-slate-800 tracking-tight mb-4">
            Validation Splits Performance Matrix (Representation Comparisons)
          </h4>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
              <thead className="bg-slate-50 text-slate-500 font-semibold">
                <tr>
                  <th className="px-6 py-3.5">Representation</th>
                  <th className="px-6 py-3.5">Estimator Model</th>
                  <th className="px-6 py-3.5">Accuracy</th>
                  <th className="px-6 py-3.5">Precision</th>
                  <th className="px-6 py-3.5">Recall</th>
                  <th className="px-6 py-3.5">F1 Score</th>
                  <th className="px-6 py-3.5">ROC AUC</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 text-slate-700">
                {classificationComparisons.map((row, index) => (
                  <tr key={index} className={`hover:bg-slate-50/50 transition-colors ${row.isBest ? 'bg-blue-50/20 font-semibold' : ''}`}>
                    <td className="px-6 py-3">{row.space}</td>
                    <td className="px-6 py-3">{row.model}</td>
                    <td className="px-6 py-3">{row.accuracy}</td>
                    <td className="px-6 py-3">{row.precision}</td>
                    <td className="px-6 py-3">{row.recall}</td>
                    <td className="px-6 py-3">{row.f1}</td>
                    <td className="px-6 py-3 flex items-center gap-2">
                      {row.auc}
                      {row.isBest && (
                        <span className="text-[10px] text-green-700 bg-green-50 border border-green-200 px-2 py-0.5 rounded-full uppercase font-bold tracking-wider">
                          Best
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Classification Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <ChartCard title="Receiver Operating Characteristic (ROC)" description="Comparison curves tracking FPR vs TPR">
            <div className="h-[250px] w-full mt-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rocCurveMock}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="fpr" label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5 }} style={{ fontSize: 10 }} />
                  <YAxis label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft' }} style={{ fontSize: 10 }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="tpr" stroke="#2563EB" strokeWidth={2.5} name="MLP (LDA)" dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>

          <ChartCard title="Precision-Recall Curve" description="Precision score plotted against recall sensitivity">
            <div className="h-[250px] w-full mt-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={prCurveMock}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="recall" label={{ value: 'Recall', position: 'insideBottom', offset: -5 }} style={{ fontSize: 10 }} />
                  <YAxis label={{ value: 'Precision', angle: -90, position: 'insideLeft' }} style={{ fontSize: 10 }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="precision" stroke="#8B5CF6" strokeWidth={2.5} name="MLP (LDA)" dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>

          {/* Custom HTML/CSS Confusion Matrix Card */}
          <ChartCard title="Confusion Matrix" description="Actual vs Predicted classifications on Test Set">
            <div className="flex flex-col items-center justify-center h-[250px]">
              <div className="grid grid-cols-3 gap-2 w-full max-w-[240px] text-center font-bold text-xs text-slate-800">
                <div></div>
                <div className="text-slate-400">Pred Std</div>
                <div className="text-slate-400">Pred High</div>

                <div className="flex items-center justify-end text-slate-400 pr-2">Act Std</div>
                <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col justify-center">
                  <span className="text-slate-900 text-sm font-extrabold">266</span>
                  <span className="text-[10px] text-slate-400">TN</span>
                </div>
                <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col justify-center">
                  <span className="text-slate-900 text-sm font-extrabold">0</span>
                  <span className="text-[10px] text-slate-400">FP</span>
                </div>

                <div className="flex items-center justify-end text-slate-400 pr-2">Act High</div>
                <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col justify-center">
                  <span className="text-slate-900 text-sm font-extrabold">0</span>
                  <span className="text-[10px] text-slate-400">FN</span>
                </div>
                <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 flex flex-col justify-center text-blue-800">
                  <span className="text-blue-900 text-sm font-extrabold">66</span>
                  <span className="text-[10px] text-blue-500">TP</span>
                </div>
              </div>
            </div>
          </ChartCard>
        </div>
      </div>

      <hr className="border-slate-200" />

      {/* Regression Sections */}
      <div>
        <h3 className="text-base font-extrabold text-slate-900 tracking-tight mb-4 flex items-center gap-2">
          <AreaChart className="text-blue-600" size={18} />
          Customer LTV Future Spend Forecasting
        </h3>

        {/* Metric grids */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <MetricCard title="Test MSE" value="13,307,866" description="Mean squared errors score" color="yellow" />
          <MetricCard title="Test RMSE" value="$3,647.99" description="Root mean squared error forecast" color="yellow" />
          <MetricCard title="Test R² Coefficient" value="0.6593" description="Variance explanation proportion" color="blue" />
        </div>

        {/* Regression comparison table */}
        <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm mb-8">
          <h4 className="text-sm font-bold text-slate-800 tracking-tight mb-4">
            Forecasting Model Validation Scoreboards
          </h4>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
              <thead className="bg-slate-50 text-slate-500 font-semibold">
                <tr>
                  <th className="px-6 py-3.5">Estimator Model</th>
                  <th className="px-6 py-3.5">Validation MSE</th>
                  <th className="px-6 py-3.5">Validation RMSE</th>
                  <th className="px-6 py-3.5">Validation R² Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 text-slate-700">
                {regressionComparisons.map((row, index) => (
                  <tr key={index} className={`hover:bg-slate-50/50 transition-colors ${row.isBest ? 'bg-blue-50/20 font-semibold' : ''}`}>
                    <td className="px-6 py-3 font-semibold">{row.model}</td>
                    <td className="px-6 py-3">{row.mse}</td>
                    <td className="px-6 py-3">{row.rmse}</td>
                    <td className="px-6 py-3 flex items-center gap-2">
                      {row.r2}
                      {row.isBest && (
                        <span className="text-[10px] text-green-700 bg-green-50 border border-green-200 px-2 py-0.5 rounded-full uppercase font-bold tracking-wider">
                          Best
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Regression Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <ChartCard title="Predicted vs Actual Spend" description="Scatter mapping of predictions vs actual targets">
            <div className="h-[250px] w-full mt-2">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 20, bottom: 10, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis type="number" dataKey="actual" name="Actual Spend" unit="$" style={{ fontSize: 10 }} />
                  <YAxis type="number" dataKey="predicted" name="Predicted Spend" unit="$" style={{ fontSize: 10 }} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter name="Customers" data={predictedVsActualData} fill="#2563EB" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>

          <ChartCard title="Residual Plot Analysis" description="Forecasting residuals plotted against predictions">
            <div className="h-[250px] w-full mt-2">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 20, bottom: 10, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis type="number" dataKey="predicted" name="Predicted Spend" unit="$" style={{ fontSize: 10 }} />
                  <YAxis type="number" dataKey="residual" name="Residual" unit="$" style={{ fontSize: 10 }} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter name="Residuals" data={residualData} fill="#EF4444" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>

          <ChartCard title="Feature Importance Rankings (LDA)" description="Coefficients showing segment discriminability strength">
            <div className="h-[250px] w-full mt-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={featureImportance.slice(0, 5)} layout="vertical" margin={{ top: 5, right: 10, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis type="number" style={{ fontSize: 10 }} />
                  <YAxis dataKey="name" type="category" style={{ fontSize: 9, fontWeight: 'bold' }} width={80} />
                  <Tooltip />
                  <Bar dataKey="importance" fill="#8B5CF6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
