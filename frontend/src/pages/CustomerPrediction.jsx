import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import PageHeader from '../components/PageHeader';
import InputCard from '../components/InputCard';
import PredictionCard from '../components/PredictionCard';
import EmptyState from '../components/EmptyState';
import { User, PieChart, Sparkles } from 'lucide-react';

const CustomerPrediction = () => {
  const { register, handleSubmit, watch, setValue } = useForm({
    defaultValues: {
      recency: 30,
      frequency: 6,
      monetary: 750,
      averageSpend: 125,
      productDiversity: 22,
      totalOrders: 6,
      averageBasketSize: 4.8,
      averageQuantityPerOrder: 35,
      homeware: 20,
      stationery: 15,
      kitchenware: 10,
      decorations: 25,
      gadgets: 15,
    }
  });

  const [result, setResult] = useState(null);

  // Watch spent sliders
  const homeware = watch('homeware');
  const stationery = watch('stationery');
  const kitchenware = watch('kitchenware');
  const decorations = watch('decorations');
  const gadgets = watch('gadgets');

  const totalSpend = Number(homeware) + Number(stationery) + Number(kitchenware) + Number(decorations) + Number(gadgets);
  const otherSpend = Math.max(0, 100 - totalSpend);
  const isInvalid = totalSpend > 100;

  const onSubmit = (data) => {
    if (isInvalid) return;

    // Simulate ML Model inference rules
    const monetaryValue = Number(data.monetary);
    const isHighValue = monetaryValue >= 1629.40;

    let predictedClass = isHighValue ? 'High Value' : 'Standard';
    let conf = isHighValue ? 98.7 : 94.2;
    let forecastedLtv = isHighValue ? (monetaryValue * 0.45).toFixed(2) : (monetaryValue * 0.12).toFixed(2);
    let risk = isHighValue ? 'Low Risk' : 'Medium Risk';
    let rec = isHighValue
      ? '💎 Fast-track loyalty programs; assign premier service representative.'
      : '✉️ Offer 10% coupon campaigns on stationery/decorations to stimulate purchase frequency.';

    setResult({
      prediction: predictedClass,
      confidence: conf,
      futureSpend: Number(forecastedLtv).toLocaleString(),
      riskLevel: risk,
      recommendation: rec,
    });
  };

  return (
    <div className="space-y-8">
      <PageHeader
        title="Customer Profile Inference"
        subtitle="Simulate retail transaction traits to segment loyalty classes and forecast future spent value."
      />

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column: Customer Info */}
          <InputCard title="Customer Transaction Indicators" icon={User}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="text-xs font-semibold text-slate-500 block mb-1">Recency (Days)</label>
                <input
                  type="number"
                  {...register('recency')}
                  className="w-full text-xs font-medium border border-slate-200 bg-slate-50 rounded-xl px-3 py-2 focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-slate-500 block mb-1">Frequency (Orders Count)</label>
                <input
                  type="number"
                  {...register('frequency')}
                  className="w-full text-xs font-medium border border-slate-200 bg-slate-50 rounded-xl px-3 py-2 focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-slate-500 block mb-1">Monetary Spend ($)</label>
                <input
                  type="number"
                  {...register('monetary')}
                  className="w-full text-xs font-medium border border-slate-200 bg-slate-50 rounded-xl px-3 py-2 focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-slate-500 block mb-1">Average Spend ($)</label>
                <input
                  type="number"
                  {...register('averageSpend')}
                  className="w-full text-xs font-medium border border-slate-200 bg-slate-50 rounded-xl px-3 py-2 focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-slate-500 block mb-1">Product Diversity (Unique Codes)</label>
                <input
                  type="number"
                  {...register('productDiversity')}
                  className="w-full text-xs font-medium border border-slate-200 bg-slate-50 rounded-xl px-3 py-2 focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-slate-500 block mb-1">Total Orders</label>
                <input
                  type="number"
                  {...register('totalOrders')}
                  className="w-full text-xs font-medium border border-slate-200 bg-slate-50 rounded-xl px-3 py-2 focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-slate-500 block mb-1">Average Basket Size</label>
                <input
                  type="number"
                  step="0.1"
                  {...register('averageBasketSize')}
                  className="w-full text-xs font-medium border border-slate-200 bg-slate-50 rounded-xl px-3 py-2 focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-slate-500 block mb-1">Average Quantity per Order</label>
                <input
                  type="number"
                  {...register('averageQuantityPerOrder')}
                  className="w-full text-xs font-medium border border-slate-200 bg-slate-50 rounded-xl px-3 py-2 focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                />
              </div>
            </div>
          </InputCard>

          {/* Right Column: Category Spends */}
          <InputCard title="Category Spending Compositions" icon={PieChart}>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-xs font-semibold text-slate-500 mb-1">
                  <span>Homeware Spend %</span>
                  <span>{homeware}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  {...register('homeware')}
                  className="w-full accent-blue-600 cursor-pointer h-1.5 bg-slate-100 rounded-lg appearance-none"
                />
              </div>
              <div>
                <div className="flex justify-between text-xs font-semibold text-slate-500 mb-1">
                  <span>Stationery Spend %</span>
                  <span>{stationery}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  {...register('stationery')}
                  className="w-full accent-blue-600 cursor-pointer h-1.5 bg-slate-100 rounded-lg appearance-none"
                />
              </div>
              <div>
                <div className="flex justify-between text-xs font-semibold text-slate-500 mb-1">
                  <span>Kitchenware Spend %</span>
                  <span>{kitchenware}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  {...register('kitchenware')}
                  className="w-full accent-blue-600 cursor-pointer h-1.5 bg-slate-100 rounded-lg appearance-none"
                />
              </div>
              <div>
                <div className="flex justify-between text-xs font-semibold text-slate-500 mb-1">
                  <span>Decorations Spend %</span>
                  <span>{decorations}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  {...register('decorations')}
                  className="w-full accent-blue-600 cursor-pointer h-1.5 bg-slate-100 rounded-lg appearance-none"
                />
              </div>
              <div>
                <div className="flex justify-between text-xs font-semibold text-slate-500 mb-1">
                  <span>Gadgets Spend %</span>
                  <span>{gadgets}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  {...register('gadgets')}
                  className="w-full accent-blue-600 cursor-pointer h-1.5 bg-slate-100 rounded-lg appearance-none"
                />
              </div>

              {/* Progress and Validation */}
              <div className="pt-4 border-t border-slate-100 space-y-3">
                <div className="flex justify-between text-xs font-bold">
                  <span className="text-slate-500">Total Spent Composition</span>
                  <span className={isInvalid ? 'text-red-600' : 'text-slate-700'}>
                    {totalSpend.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    style={{ width: `${Math.min(totalSpend, 100)}%` }}
                    className={`h-full transition-all duration-300 ${
                      isInvalid ? 'bg-red-500' : 'bg-blue-600'
                    }`}
                  />
                </div>
                {!isInvalid ? (
                  <p className="text-[10px] font-bold text-green-600 bg-green-50 border border-green-200 rounded-lg p-2.5">
                    ✓ Composition valid. Auto-calculated remainder Other Spend: <strong>{otherSpend.toFixed(1)}%</strong>
                  </p>
                ) : (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-red-700 space-y-1">
                    <div className="text-[10px] font-bold">❌ Spent exceeds 100% boundary limit!</div>
                    <p className="text-[10px] leading-normal font-medium">Please adjust category spending percentage sliders to ensure the total is less than or equal to 100%.</p>
                  </div>
                )}
              </div>
            </div>
          </InputCard>
        </div>

        {/* Prediction Trigger Button */}
        <div className="flex justify-center">
          <button
            type="submit"
            disabled={isInvalid}
            className="flex items-center justify-center gap-2 max-w-sm w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white font-bold py-3.5 px-6 rounded-2xl cursor-pointer shadow-md hover:from-blue-700 hover:to-blue-800 hover:scale-[1.01] hover:shadow-lg disabled:opacity-40 disabled:hover:scale-100 disabled:shadow-none transition-all"
          >
            <Sparkles size={16} />
            Predict Customer Segment
          </button>
        </div>
      </form>

      {/* Render Outputs */}
      <div className="mt-8">
        {result ? (
          <PredictionCard {...result} />
        ) : (
          <EmptyState
            title="Inference Required"
            message="Configure customer metrics and click 'Predict Customer' to view segmented loyalty diagnostics and future value forecasts."
            icon={Sparkles}
          />
        )}
      </div>
    </div>
  );
};

export default CustomerPrediction;
