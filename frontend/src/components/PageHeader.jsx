import React from 'react';

const PageHeader = ({ title, subtitle, children }) => {
  return (
    <div className="flex flex-col md:flex-row md:items-center md:justify-between border-b border-slate-200 pb-5 mb-8">
      <div>
        <h1 className="text-2xl md:text-3xl font-extrabold text-slate-900 tracking-tight leading-none">
          {title}
        </h1>
        {subtitle && (
          <p className="text-sm text-slate-500 mt-2 font-normal">
            {subtitle}
          </p>
        )}
      </div>
      {children && (
        <div className="mt-4 md:mt-0 flex items-center gap-3">
          {children}
        </div>
      )}
    </div>
  );
};

export default PageHeader;
