import React from 'react';

const SectionTitle = ({ children, className = '' }) => {
  return (
    <h2 className={`text-lg font-bold text-slate-800 tracking-tight mb-4 ${className}`}>
      {children}
    </h2>
  );
};

export default SectionTitle;
