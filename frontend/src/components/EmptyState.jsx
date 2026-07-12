import React from 'react';
import { HelpCircle } from 'lucide-react';

const EmptyState = ({ title = 'No Data Available', message = 'There is no information to display here at the moment.', icon: Icon = HelpCircle }) => {
  return (
    <div className="flex flex-col items-center justify-center p-8 text-center bg-white border border-dashed border-slate-300 rounded-2xl min-h-[300px]">
      <div className="p-3 bg-slate-50 text-slate-400 rounded-full mb-4">
        <Icon size={32} className="stroke-[1.5]" />
      </div>
      <h3 className="text-base font-bold text-slate-800 tracking-tight mb-1">
        {title}
      </h3>
      <p className="text-xs text-slate-500 max-w-sm">
        {message}
      </p>
    </div>
  );
};

export default EmptyState;
