import React from 'react';
import { motion } from 'framer-motion';

const MetricCard = ({ title, value, description, icon: Icon, color = 'blue' }) => {
  const colorMap = {
    blue: 'border-blue-600 text-blue-600',
    purple: 'border-purple-600 text-purple-600',
    yellow: 'border-yellow-500 text-yellow-500',
    green: 'border-green-600 text-green-600',
    red: 'border-red-600 text-red-600',
  };

  return (
    <motion.div
      whileHover={{ y: -6, boxShadow: '0 12px 30px rgba(0, 0, 0, 0.08)' }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className={`bg-white border border-slate-200 border-t-4 ${colorMap[color] || 'border-blue-600'} rounded-2xl p-6 shadow-sm`}
    >
      <div className="flex justify-between items-start">
        <div>
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block mb-1">
            {title}
          </span>
          <span className="text-3xl font-extrabold text-slate-900 leading-none">
            {value}
          </span>
        </div>
        {Icon && (
          <div className={`p-2 rounded-xl bg-slate-50 text-slate-600`}>
            <Icon size={22} className="stroke-[2px]" />
          </div>
        )}
      </div>
      {description && (
        <p className="text-xs text-slate-500 mt-3 font-medium">
          {description}
        </p>
      )}
    </motion.div>
  );
};

export default MetricCard;
