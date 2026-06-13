import React from 'react';
import './FeatureCard.css';

const FeatureCard = ({ title, description, orbColor, icon }) => {
  const orbStyle = {
    background: `radial-gradient(circle, var(--accent-${orbColor}) 0%, transparent 70%)`
  };

  return (
    <div className="glass-card feature-card">
      <div className="feature-orb" style={orbStyle}></div>
      <div className="feature-icon">{icon}</div>
      <h3 className="feature-title">{title}</h3>
      <p className="feature-description">{description}</p>
    </div>
  );
};

export default FeatureCard;
