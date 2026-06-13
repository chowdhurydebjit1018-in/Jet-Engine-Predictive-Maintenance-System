import React from 'react';
import './BenefitCard.css';

const BenefitCard = ({ title, description, icon }) => {
  return (
    <div className="glass-card benefit-card">
      <div className="benefit-icon-wrapper">
        <span className="benefit-icon">{icon}</span>
      </div>
      <h3 className="benefit-title">{title}</h3>
      <p className="benefit-description">{description}</p>
    </div>
  );
};

export default BenefitCard;
