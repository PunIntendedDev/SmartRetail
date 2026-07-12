import React from 'react';
import { Menu, Activity, ShieldCheck, Github } from 'lucide-react';

const Navbar = ({ setSidebarOpen }) => {
  return (
    <header className="sticky top-0 z-30 flex items-center justify-between h-16 px-6 bg-white border-b border-slate-200 shadow-xs">
      {/* Mobile Toggle Button */}
      <button
        onClick={() => setSidebarOpen(true)}
        className="p-1.5 rounded-lg border border-slate-200 text-slate-500 hover:bg-slate-50 lg:hidden transition-colors"
      >
        <Menu size={18} />
      </button>

      {/* Breadcrumb / Title Context */}
      <div className="hidden md:flex items-center gap-2">
        <span className="text-xs font-semibold text-slate-400">SmartRetail Platform</span>
        <span className="text-slate-300">/</span>
        <span className="text-xs font-bold text-slate-600">Enterprise AI Engine</span>
      </div>

      {/* Toolbar / Global Statuses */}
      <div className="flex items-center gap-4">
        {/* API Pipeline Status Indicators */}
        <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider text-slate-500 border border-slate-200 rounded-full px-3 py-1 bg-slate-50">
          <Activity size={12} className="text-green-500 animate-pulse" />
          <span>Local Engine Status: Connected</span>
        </div>

        {/* GitHub link */}
        <a
          href="https://github.com"
          target="_blank"
          rel="noreferrer"
          className="p-1.5 rounded-lg text-slate-500 hover:bg-slate-50 border border-slate-200 transition-colors"
        >
          <Github size={16} />
        </a>
      </div>
    </header>
  );
};

export default Navbar;
