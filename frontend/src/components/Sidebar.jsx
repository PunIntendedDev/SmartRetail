import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, UserCheck, BarChart3, Binary, Database, Info, X } from 'lucide-react';
import { motion } from 'framer-motion';

const Sidebar = ({ isOpen, setIsOpen }) => {
  const menuItems = [
    { name: 'Dashboard', path: '/', icon: LayoutDashboard },
    { name: 'Customer Prediction', path: '/prediction', icon: UserCheck },
    { name: 'Model Performance', path: '/analytics', icon: BarChart3 },
    { name: 'PCA & LDA', path: '/pca-lda', icon: Binary },
    { name: 'Dataset Explorer', path: '/explorer', icon: Database },
    { name: 'About', path: '/about', icon: Info },
  ];

  return (
    <>
      {/* Mobile Drawer Overlay */}
      {isOpen && (
        <div
          onClick={() => setIsOpen(false)}
          className="fixed inset-0 z-40 bg-slate-900/40 backdrop-blur-xs lg:hidden"
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 flex flex-col w-64 bg-slate-900 border-r border-slate-800 text-slate-400 transform transition-transform duration-300 lg:translate-x-0 lg:static lg:h-screen ${
          isOpen ? 'translate-x-0' : '-translate-x-0'
        }`}
      >
        {/* Header Branding */}
        <div className="flex items-center justify-between h-16 px-6 border-b border-slate-800 bg-slate-950/20">
          <div className="flex flex-col">
            <span className="text-base font-extrabold text-white tracking-tight flex items-center gap-1.5">
              🛍️ SmartRetail AI
            </span>
            <span className="text-[10px] text-slate-500 font-semibold tracking-wider uppercase mt-0.5">
              Customer Behavior Analytics
            </span>
          </div>
          <button
            onClick={() => setIsOpen(false)}
            className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white lg:hidden transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 px-4 py-6 space-y-1.5 overflow-y-auto">
          {menuItems.map((item) => {
            const IconComponent = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={() => setIsOpen(false)}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-4 py-3 text-xs font-semibold rounded-xl transition-all relative ${
                    isActive
                      ? 'text-white bg-blue-600/10 border border-blue-600/30'
                      : 'border border-transparent hover:text-white hover:bg-slate-800/50'
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    {isActive && (
                      <motion.div
                        layoutId="active-indicator"
                        className="absolute left-0 w-1 h-6 bg-blue-600 rounded-r-full"
                        transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                      />
                    )}
                    <IconComponent
                      size={18}
                      className={`stroke-[2px] ${isActive ? 'text-blue-500' : 'text-slate-400'}`}
                    />
                    {item.name}
                  </>
                )}
              </NavLink>
            );
          })}
        </nav>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-slate-800 text-[10px] text-slate-500 font-medium text-center">
          Presentation Edition v1.1.0
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
