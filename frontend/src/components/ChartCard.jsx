import React from 'react';
import { motion } from 'framer-motion';

const ChartCard = ({ title, description, children, className = '' }) => {
  return (
    <motion.div
      whileHover={{ y: -4, boxShadow: '0 10px 25px rgba(0, 0, 0, 0.05)' }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className={`bg-white border border-slate-200 rounded-2xl p-6 shadow-sm ${className}`}
    >
      {title && (
        <div className="mb-4">
          <h3 className="text-base font-bold text-slate-800 tracking-tight leading-snug">
            {title}
          </h3>
          {description && (
            <p className="text-xs text-slate-500 mt-1">
              {description}
            </p>
          )}
        </div>
      )}
      <div className="w-full">
        {children}
      </div>
    </motion.div>
  );
};

export default ChartCard;
