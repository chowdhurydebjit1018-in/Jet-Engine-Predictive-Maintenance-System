import React from 'react';
import { NavLink, Outlet, Link } from 'react-router-dom';
import { LayoutDashboard, Activity, FileSearch, ArrowLeft } from 'lucide-react';
import '../styles/Dashboard.css';

const DashboardLayout = () => {
  return (
    <div className="dashboard-layout">
      {/* Sidebar - Collapsed Icon Only */}
      <aside className="dashboard-sidebar glass-card">
        <div className="sidebar-logo">
          <Link to="/" style={{ textDecoration: 'none' }}>
            <div className="brand-name-small" style={{ fontWeight: 700, fontSize: '20px', color: 'var(--text-primary)' }}>AP</div>
          </Link>
        </div>
        
        <nav className="sidebar-nav">
          <NavLink to="/dashboard" end className={({isActive}) => isActive ? "nav-item active" : "nav-item"} title="Fleet Monitoring">
            <LayoutDashboard size={24} />
          </NavLink>
          
          <NavLink to="/dashboard/engine" className={({isActive}) => isActive ? "nav-item active" : "nav-item"} title="Engine Analysis">
            <Activity size={24} />
          </NavLink>
          
          <NavLink to="/dashboard/explain" className={({isActive}) => isActive ? "nav-item active" : "nav-item"} title="Explainability">
            <FileSearch size={24} />
          </NavLink>
        </nav>

        <div className="sidebar-bottom">
          <Link to="/" className="nav-item" title="Back to Home">
            <ArrowLeft size={24} />
          </Link>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="dashboard-content">
        <Outlet />
      </main>
    </div>
  );
};

export default DashboardLayout;
