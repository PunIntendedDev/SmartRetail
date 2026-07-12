import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import DashboardLayout from './layouts/DashboardLayout';
import Dashboard from './pages/Dashboard';
import CustomerPrediction from './pages/CustomerPrediction';
import Analytics from './pages/Analytics';
import PcaLda from './pages/PcaLda';
import DatasetExplorer from './pages/DatasetExplorer';
import About from './pages/About';

function App() {
  return (
    <BrowserRouter>
      <DashboardLayout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/prediction" element={<CustomerPrediction />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/pca-lda" element={<PcaLda />} />
          <Route path="/explorer" element={<DatasetExplorer />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </DashboardLayout>
    </BrowserRouter>
  );
}

export default App;
