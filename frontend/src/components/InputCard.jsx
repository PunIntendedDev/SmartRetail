import React from 'react';

const InputCard = ({ title, icon: Icon, children }) => {
  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
      <h3 className="text-sm font-bold text-slate-800 tracking-tight flex items-center gap-2 mb-4">
        {Icon && <Icon size={16} className="text-blue-600 stroke-[2.2]" />}
        {title}
      </h3>
      <div className="space-y-4">
        {children}
      </div>
    </div>
  );
};

export default InputCard;
