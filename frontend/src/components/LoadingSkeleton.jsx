import React from 'react';

const LoadingSkeleton = ({ type = 'card', count = 1 }) => {
  const renderSkeleton = () => {
    if (type === 'table') {
      return (
        <div className="space-y-4 w-full animate-pulse border border-slate-200 rounded-2xl p-6 bg-white">
          <div className="h-6 bg-slate-200 rounded-lg w-1/4 mb-6"></div>
          {[...Array(5)].map((_, i) => (
            <div key={i} className="flex gap-4 items-center">
              <div className="h-4 bg-slate-100 rounded-lg w-12"></div>
              <div className="h-4 bg-slate-100 rounded-lg flex-1"></div>
              <div className="h-4 bg-slate-100 rounded-lg w-20"></div>
              <div className="h-4 bg-slate-100 rounded-lg w-16"></div>
            </div>
          ))}
        </div>
      );
    }

    if (type === 'chart') {
      return (
        <div className="animate-pulse border border-slate-200 rounded-2xl p-6 bg-white w-full h-[320px] flex flex-col justify-end">
          <div className="h-6 bg-slate-200 rounded-lg w-1/3 mb-auto"></div>
          <div className="flex items-end gap-3 h-[200px] w-full px-4">
            <div className="h-[40%] bg-slate-100 rounded-t-lg flex-1"></div>
            <div className="h-[75%] bg-slate-100 rounded-t-lg flex-1"></div>
            <div className="h-[55%] bg-slate-100 rounded-t-lg flex-1"></div>
            <div className="h-[90%] bg-slate-100 rounded-t-lg flex-1"></div>
            <div className="h-[30%] bg-slate-100 rounded-t-lg flex-1"></div>
          </div>
        </div>
      );
    }

    // Default to card skeleton
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 w-full">
        {[...Array(count)].map((_, i) => (
          <div key={i} className="animate-pulse border border-slate-200 rounded-2xl p-6 bg-white shadow-sm flex flex-col justify-between h-[140px]">
            <div className="h-4 bg-slate-200 rounded-lg w-2/3"></div>
            <div className="h-8 bg-slate-200 rounded-lg w-1/2 my-2"></div>
            <div className="h-3 bg-slate-100 rounded-lg w-3/4"></div>
          </div>
        ))}
      </div>
    );
  };

  return <>{renderSkeleton()}</>;
};

export default LoadingSkeleton;
