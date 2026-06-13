import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="container navbar-container">
        <div className="navbar-left">
          <span className="brand-name">AeroPredict AI</span>
        </div>
        
        <div className="navbar-center">
          <Link to="/dashboard" className="nav-link">Dashboard</Link>
          <a href="#resources" className="nav-link">Resources</a>
          <a href="#work" className="nav-link">Our Work</a>
        </div>
        
        <div className="navbar-right">
          <a href="#faq" className="nav-link">FAQ</a>
          <Link to="/dashboard"><button className="download-btn">Start Predicting</button></Link>
          <button className="hamburger-menu">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M4 6H20M4 12H20M4 18H20" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
