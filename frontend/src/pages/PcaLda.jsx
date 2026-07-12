import React from 'react';
import PageHeader from '../components/PageHeader';
import ChartCard from '../components/ChartCard';
import MetricCard from '../components/MetricCard';
import { pcaScreeData, pcaScatterData, ldaScatterData } from '../utils/mockData';
import { ResponsiveContainer, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ScatterChart, Scatter, ReferenceLine } from 'recharts';
import { Binary, ToggleLeft, HelpCircle } from 'lucide-react';

const PcaLda = () => {
  return (
    <div className="space-y-8">
      <PageHeader
        title="Dimensionality Reduction Insights"
        subtitle="Deconstruct categorical spend multicollinearity and maximize customer class separation boundaries."
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Explained Component KPIs */}
        <MetricCard title="Selected PCA Components" value="4 Components" description="Explained variance threshold >= 90%" color="blue" />
        <MetricCard title="Variance Retained" value="93.26%" description="Information ratio preserved" color="green" />
        <MetricCard title="Top LDA Discriminant" value="Product Diversity" description="Coefficient score: +1.1471" color="purple" />
      </div>

      {/* Explanation Cards and Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Side: Explanations */}
        <div className="lg:col-span-1 space-y-6">
          <h3 className="text-base font-bold text-slate-800 tracking-tight flex items-center gap-2">
            <HelpCircle className="text-blue-600" size={18} />
            Mathematical Rationale
          </h3>

          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2">Standard Scaling</h4>
            <p className="text-xs text-slate-500 leading-relaxed">
              Standardizes features by removing the mean and scaling to unit variance. Scalers are fitted strictly on training splits to prevent validation and test set information leakage.
            </p>
          </div>

          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2">Principal Component Analysis (PCA)</h4>
            <p className="text-xs text-slate-500 leading-relaxed">
              Unsupervised method projecting category spend percentages onto orthogonal coordinates containing the maximum variance. Excludes Other_Spend_Pct to avoid perfect multicollinearity.
            </p>
          </div>

          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2">Linear Discriminant Analysis (LDA)</h4>
            <p className="text-xs text-slate-500 leading-relaxed">
              Supervised projection technique focused on finding an axis that maximizes high-value class separability relative to within-class dispersion, defining a 1D boundary score.
            </p>
          </div>

          <div className="bg-yellow-50 border border-yellow-200 rounded-2xl p-6">
            <h4 className="text-xs font-bold text-yellow-800 uppercase tracking-wider mb-2 flex items-center gap-1">
              💡 Multicollinearity Choice
            </h4>
            <p className="text-xs text-yellow-900 leading-relaxed">
              Since all six product category spent percentages sum to 100%, they exhibit perfect multicollinearity. One category (Other_Spend_Pct) is intentionally excluded only to avoid redundant information, utilizing the remaining five category features for PCA.
            </p>
          </div>
        </div>

        {/* Right Side: Charts */}
        <div className="lg:col-span-2 space-y-6">
          <h3 className="text-base font-bold text-slate-800 tracking-tight flex items-center gap-2">
            <ToggleLeft className="text-blue-600" size={18} />
            Projection Visualization
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Scree Plot */}
            <ChartCard title="PCA Explained Variance Scree Chart" description="Individual and cumulative explained ratios per PC">
              <div className="h-[250px] w-full mt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={pcaScreeData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis dataKey="name" style={{ fontSize: 10 }} />
                    <YAxis style={{ fontSize: 10 }} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="ratio" stroke="#2563EB" strokeWidth={2.5} name="Individual Ratio" dot={{ r: 4 }} />
                    <Line type="monotone" dataKey="cumulative" stroke="#10B981" strokeWidth={2} name="Cumulative Ratio" strokeDasharray="4 4" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </ChartCard>

            {/* PCA Scatter Plot */}
            <ChartCard title="PCA Component Scatter Map" description="First two PCs showing unsupervised customer clusters">
              <div className="h-[250px] w-full mt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis type="number" dataKey="pc1" name="PC1" style={{ fontSize: 10 }} />
                    <YAxis type="number" dataKey="pc2" name="PC2" style={{ fontSize: 10 }} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter name="Standard Customer" data={pcaScatterData.filter(d => d.isHighValue === 0)} fill="#94A3B8" />
                    <Scatter name="High Value" data={pcaScatterData.filter(d => d.isHighValue === 1)} fill="#2563EB" />
                    <Legend />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </ChartCard>
          </div>

          {/* LDA Projection */}
          <ChartCard title="LDA Supervised Projection (1D Separation)" description="Projections on LD1 axis showing clear class division boundary">
            <div className="h-[200px] w-full mt-2">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis type="number" dataKey="ld1" name="LD1 Coordinate" style={{ fontSize: 10 }} />
                  <YAxis type="number" dataKey="y" name="Distribution Spread" style={{ fontSize: 10 }} domain={[-1, 1]} hide />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <ReferenceLine x={0} stroke="#EF4444" strokeDasharray="3 3" label={{ value: 'Separation Boundary', position: 'top', fill: '#EF4444', fontSize: 10 }} />
                  <Scatter name="Standard Customer" data={ldaScatterData.filter(d => d.isHighValue === 0)} fill="#94A3B8" />
                  <Scatter name="High Value" data={ldaScatterData.filter(d => d.isHighValue === 1)} fill="#2563EB" />
                  <Legend />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>
        </div>
      </div>
    </div>
  );
};

export default PcaLda;
