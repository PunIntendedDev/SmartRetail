import React, { useState } from 'react';
import PageHeader from '../components/PageHeader';
import MetricCard from '../components/MetricCard';
import DataTable from '../components/DataTable';
import ChartCard from '../components/ChartCard';
import StatBadge from '../components/StatBadge';
import { sampleDataset, histogramData } from '../utils/mockData';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { Database, Filter, ArrowUpDown } from 'lucide-react';

const DatasetExplorer = () => {
  const [selectedFeature, setSelectedFeature] = useState('Recency');

  const columns = [
    { key: 'customerId', label: 'Customer ID', sortable: true },
    { key: 'recency', label: 'Recency (Days)', sortable: true },
    { key: 'frequency', label: 'Frequency (Orders)', sortable: true },
    { key: 'monetary', label: 'Monetary ($)', sortable: true, render: (val) => `$${val.toLocaleString(undefined, { minimumFractionDigits: 2 })}` },
    { key: 'averageSpend', label: 'Avg Spend ($)', sortable: true, render: (val) => `$${val.toLocaleString(undefined, { minimumFractionDigits: 2 })}` },
    { key: 'diversity', label: 'Product Diversity', sortable: true },
    { key: 'basketSize', label: 'Avg Basket Size', sortable: true },
    { key: 'qtyPerOrder', label: 'Avg Qty/Order', sortable: true },
    { key: 'isHighValue', label: 'Segment Type', sortable: false, render: (val) => <StatBadge type={val === 1 ? 'success' : 'slate'} label={val === 1 ? '💎 High Value' : '👤 Standard'} /> },
  ];

  return (
    <div className="space-y-8">
      <PageHeader
        title="Dataset Explorer & Auditor"
        subtitle="Search, sort, and filter preprocessed customer registers splits."
      />

      {/* Database Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard title="Rows Count" value="3,312" description="Total parsed customers" color="blue" />
        <MetricCard title="Columns count" value="11" description="Processed attributes variables" color="purple" />
        <MetricCard title="Features variables" value="13" description="Base engineered predictors" color="yellow" />
        <MetricCard title="Missing values count" value="0" description="Complete record entries" color="green" />
      </div>

      {/* Main Interactive Table */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
        <div className="flex items-center gap-2 mb-4 border-b border-slate-100 pb-3">
          <Database className="text-blue-600" size={18} />
          <h3 className="text-base font-bold text-slate-800 tracking-tight">
            Customer Profile Splits Registry
          </h3>
        </div>
        <DataTable columns={columns} data={sampleDataset} filterKey="isHighValue" />
      </div>

      {/* Interactive Histogram Explorer */}
      <ChartCard title="Attribute Distribution Auditor" description="Explore histogram frequency metrics grouped by loyalty segment">
        <div className="flex gap-4 items-center mb-6">
          <label className="text-xs font-semibold text-slate-500">Choose Distribution Feature:</label>
          <select
            value={selectedFeature}
            onChange={(e) => setSelectedFeature(e.target.value)}
            className="px-3 py-1.5 text-xs border border-slate-200 bg-white rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent font-medium"
          >
            <option value="Recency">Recency (Days since purchase)</option>
            <option value="Frequency">Frequency (Orders Count)</option>
            <option value="Monetary">Monetary (Total Spends)</option>
          </select>
        </div>

        <div className="h-[300px] w-full mt-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={histogramData[selectedFeature]} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="range" style={{ fontSize: 10 }} />
              <YAxis style={{ fontSize: 10 }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" fill="#94A3B8" name="Standard Customer" radius={[4, 4, 0, 0]} />
              <Bar dataKey="isHighValue" fill="#2563EB" name="High Value" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>
    </div>
  );
};

export default DatasetExplorer;
