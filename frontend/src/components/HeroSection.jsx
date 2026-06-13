import React from 'react';
import { Link } from 'react-router-dom';
import './HeroSection.css';
import FeatureCard from './FeatureCard';

const HeroSection = () => {
  return (
    <section className="hero-section">
      <div className="hero-background">
        <div className="noise-texture"></div>
        <div className="radial-gradient-top"></div>
        <div className="radial-gradient-bottom"></div>
      </div>
      
      <div className="container hero-container">
        <div className="hero-content">
          <h1 className="hero-headline">
            Predictive maintenance,<br />
            <span className="text-glow">beautifully precise.</span>
          </h1>
          
          <p className="hero-description">
            Harness the power of AI to forecast jet engine anomalies before they happen. 
            Reduce downtime, optimize maintenance schedules, and elevate fleet reliability.
          </p>
          
          <div className="hero-actions">
            <Link to="/dashboard"><button className="btn-primary glass-card">Start Predicting</button></Link>
            <button className="btn-secondary">View Documentation</button>
          </div>
        </div>

        <div className="feature-grid">
          <FeatureCard 
            title="Real-time Telemetry" 
            description="Process thousands of sensor readings per second with zero latency."
            orbColor="purple"
            icon="⚡"
          />
          <FeatureCard 
            title="Anomaly Detection" 
            description="Our advanced models predict failures with 99.8% accuracy."
            orbColor="orange"
            icon="🎯"
          />
          <FeatureCard 
            title="Fleet Overview" 
            description="Monitor all your assets from a single, beautiful dashboard."
            orbColor="green"
            icon="🌐"
          />
          <FeatureCard 
            title="Automated Reports" 
            description="Generate compliance and maintenance reports instantly."
            orbColor="pink"
            icon="📊"
          />
        </div>

        <div className="partner-logos">
          <p className="trusted-by">TRUSTED BY LEADING AIRLINES</p>
          <div className="logo-strip">
            <div className="logo-placeholder-text">Skyways</div>
            <div className="logo-placeholder-text">AeroGlobal</div>
            <div className="logo-placeholder-text">JetStream</div>
            <div className="logo-placeholder-text">Horizon</div>
            <div className="logo-placeholder-text">AviaTech</div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
