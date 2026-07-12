import React, { useState, useMemo } from 'react';
import { ChevronUp, ChevronDown, Search, ChevronLeft, ChevronRight } from 'lucide-react';
import StatBadge from './StatBadge';

const DataTable = ({ columns, data, filterKey = 'isHighValue' }) => {
  const [search, setSearch] = useState('');
  const [sortField, setSortField] = useState('');
  const [sortDirection, setSortDirection] = useState('asc');
  const [currentPage, setCurrentPage] = useState(1);
  const [filterClass, setFilterClass] = useState('all');
  const itemsPerPage = 8;

  // Handle Search and Filter
  const filteredData = useMemo(() => {
    return data.filter(item => {
      const matchesSearch = Object.values(item).some(val => 
        val.toString().toLowerCase().includes(search.toLowerCase())
      );
      
      const matchesFilter = 
        filterClass === 'all' || 
        (filterClass === 'high' && item[filterKey] === 1) ||
        (filterClass === 'standard' && item[filterKey] === 0);

      return matchesSearch && matchesFilter;
    });
  }, [data, search, filterClass, filterKey]);

  // Handle Sorting
  const sortedData = useMemo(() => {
    if (!sortField) return filteredData;

    return [...filteredData].sort((a, b) => {
      const aValue = a[sortField];
      const bValue = b[sortField];

      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
      }
      
      return sortDirection === 'asc'
        ? aValue.toString().localeCompare(bValue.toString())
        : bValue.toString().localeCompare(aValue.toString());
    });
  }, [filteredData, sortField, sortDirection]);

  // Handle Pagination
  const paginatedData = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    return sortedData.slice(startIndex, startIndex + itemsPerPage);
  }, [sortedData, currentPage]);

  const totalPages = Math.ceil(sortedData.length / itemsPerPage);

  const requestSort = (key) => {
    let direction = 'asc';
    if (sortField === key && sortDirection === 'asc') {
      direction = 'desc';
    }
    setSortField(key);
    setSortDirection(direction);
    setCurrentPage(1);
  };

  const getSortIcon = (key) => {
    if (sortField !== key) return null;
    return sortDirection === 'asc' ? <ChevronUp size={14} /> : <ChevronDown size={14} />;
  };

  return (
    <div className="w-full">
      {/* Table Actions Toolbar */}
      <div className="flex flex-col md:flex-row gap-4 items-center justify-between mb-5">
        <div className="relative w-full md:max-w-xs">
          <span className="absolute inset-y-0 left-0 pl-3 flex items-center text-slate-400">
            <Search size={16} />
          </span>
          <input
            type="text"
            placeholder="Search customer records..."
            value={search}
            onChange={(e) => { setSearch(e.target.value); setCurrentPage(1); }}
            className="pl-9 pr-4 py-2 w-full text-xs border border-slate-200 rounded-xl bg-slate-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
          />
        </div>

        <div className="flex gap-2 w-full md:w-auto">
          <select
            value={filterClass}
            onChange={(e) => { setFilterClass(e.target.value); setCurrentPage(1); }}
            className="px-3 py-2 text-xs border border-slate-200 rounded-xl bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="all">All Segments</option>
            <option value="high">High Value Only</option>
            <option value="standard">Standard Only</option>
          </select>
        </div>
      </div>

      {/* Main Table responsive scroll container */}
      <div className="overflow-x-auto border border-slate-200 rounded-2xl bg-white shadow-sm">
        <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
          <thead className="bg-slate-50 text-slate-500 font-semibold">
            <tr>
              {columns.map((col) => (
                <th
                  key={col.key}
                  onClick={() => col.sortable && requestSort(col.key)}
                  className={`px-6 py-4 cursor-pointer select-none ${col.sortable ? 'hover:bg-slate-100' : ''}`}
                >
                  <div className="flex items-center gap-1.5">
                    {col.label}
                    {col.sortable && getSortIcon(col.key)}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 text-slate-700">
            {paginatedData.length > 0 ? (
              paginatedData.map((row, index) => (
                <tr key={index} className="hover:bg-slate-50/50 transition-colors">
                  {columns.map((col) => (
                    <td key={col.key} className="px-6 py-3.5 whitespace-nowrap font-medium">
                      {col.render ? col.render(row[col.key], row) : row[col.key]}
                    </td>
                  ))}
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={columns.length} className="px-6 py-12 text-center text-slate-400">
                  No matching customer profiles found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-5">
          <span className="text-xs text-slate-500 font-medium">
            Showing Page {currentPage} of {totalPages} ({filteredData.length} records found)
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
              disabled={currentPage === 1}
              className="p-1.5 border border-slate-200 rounded-lg hover:bg-slate-50 disabled:opacity-40 disabled:hover:bg-transparent transition-all"
            >
              <ChevronLeft size={16} />
            </button>
            <button
              onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
              disabled={currentPage === totalPages}
              className="p-1.5 border border-slate-200 rounded-lg hover:bg-slate-50 disabled:opacity-40 disabled:hover:bg-transparent transition-all"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataTable;
