import React from 'react';
import PageHeader from '../components/PageHeader';
import MetricCard from '../components/MetricCard';
import PipelineDiagram from '../components/PipelineDiagram';
import { Users, FileSpreadsheet, Trophy, CircleDollarSign, Percent, LineChart, FileText, CheckCircle2 } from 'lucide-react';
import { quickStats, recentActivity } from '../utils/mockData';
import { motion } from 'framer-motion';

const Dashboard = () => {
  return (
    <div className="space-y-8">
      {/* Page Header */}
      <PageHeader
        title="SmartRetail AI Dashboard"
        subtitle="Executive behavior metrics, transaction auditing, and forecasting model summaries."
      />

      {/* Quick Statistics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        <MetricCard
          title="Total Customers"
          value={quickStats.totalCustomers}
          description="Unique registers"
          icon={Users}
          color="blue"
        />
        <MetricCard
          title="Total Features"
          value={quickStats.totalFeatures}
          description="Inference inputs"
          icon={FileSpreadsheet}
          color="purple"
        />
        <MetricCard
          title="Best Classifier"
          value={quickStats.bestClassifier}
          description="Trained model"
          icon={Trophy}
          color="green"
        />
        <MetricCard
          title="Best Regressor"
          value={quickStats.bestRegressor}
          description="Forecasting model"
          icon={CircleDollarSign}
          color="yellow"
        />
        <MetricCard
          title="Classification Accuracy"
          value={quickStats.classificationAccuracy}
          description="LDA validation set"
          icon={Percent}
          color="green"
        />
        <MetricCard
          title="Regression R²"
          value={quickStats.regressionR2}
          description="Test set score"
          icon={LineChart}
          color="blue"
        />
      </div>

      {/* Pipeline Flow Section */}
      <PipelineDiagram />

      {/* Split Section: Activity Logs vs Project Status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Side: Recent Activity Timeline */}
        <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
          <h3 className="text-base font-bold text-slate-800 tracking-tight mb-5">
            📋 Pipeline Activity Log
          </h3>
          <div className="relative pl-6 border-l border-slate-200 space-y-6">
            {recentActivity.map((log, index) => (
              <div key={index} className="relative">
                <div className="absolute -left-[31px] top-1 w-2.5 h-2.5 rounded-full bg-blue-600 border border-white" />
                <div className="flex justify-between items-start">
                  <span className="text-xs font-bold text-slate-800">{log.event}</span>
                  <span className="text-[10px] text-slate-400 font-semibold">{log.time}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Side: Project Status */}
        <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm flex flex-col justify-between">
          <div>
            <h3 className="text-base font-bold text-slate-800 tracking-tight mb-4 flex items-center gap-2">
              <CheckCircle2 className="text-green-600 stroke-[2.2]" size={18} />
              Platform Status Overview
            </h3>
            <p className="text-xs text-slate-500 leading-relaxed mb-4">
              All preprocessed customer features are computed strictly using Months 1-9 transaction parameters (Recency, Frequency, Monetary). Target spend parameters are calculated using Months 10-12 to prevent target leaks.
            </p>
            <div className="space-y-3">
              <div className="flex justify-between text-xs font-semibold py-1.5 border-b border-slate-100">
                <span className="text-slate-500">Classification Threshold</span>
                <span className="text-slate-800">$1,629.40 (Monetary 80th Percentile)</span>
              </div>
              <div className="flex justify-between text-xs font-semibold py-1.5 border-b border-slate-100">
                <span className="text-slate-500">Training Segment Distribution</span>
                <span className="text-slate-800">80.0% Standard | 20.0% High-Value</span>
              </div>
              <div className="flex justify-between text-xs font-semibold py-1.5 border-b border-slate-100">
                <span className="text-slate-500">Fitted Components Count</span>
                <span className="text-slate-800">4 PCA Dimensions (Variance: 93.26%)</span>
              </div>
            </div>
          </div>

          <div className="mt-6 flex justify-between items-center text-xs p-3 bg-slate-50 border border-slate-100 rounded-xl">
            <span className="font-semibold text-slate-500 flex items-center gap-1.5">
              <FileText size={14} className="text-blue-600" />
              Pre-trained Binary Artifacts
            </span>
            <span className="font-bold text-green-700 bg-green-50 border border-green-200 rounded-full px-2 py-0.5">
              Loaded Offline
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
