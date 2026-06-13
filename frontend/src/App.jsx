import React from 'react';
import { BrowserRouter as Router, Routes, Route, Outlet } from 'react-router-dom';
import Navbar from './components/Navbar';
import HeroSection from './components/HeroSection';
import BenefitsSection from './components/BenefitsSection';
import DashboardLayout from './components/DashboardLayout';
import FleetMonitoring from './pages/FleetMonitoring';
import EngineAnalysis from './pages/EngineAnalysis';
import Explainability from './pages/Explainability';

const LandingPage = () => (
  <>
    <Navbar />
    <HeroSection />
    <BenefitsSection />
  </>
);

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dashboard" element={<DashboardLayout />}>
          <Route index element={<FleetMonitoring />} />
          <Route path="engine" element={<EngineAnalysis />} />
          <Route path="explain" element={<Explainability />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
