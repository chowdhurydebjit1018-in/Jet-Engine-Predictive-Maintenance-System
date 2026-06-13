import React from 'react';
import './BenefitsSection.css';
import BenefitCard from './BenefitCard';

const BenefitsSection = () => {
  return (
    <section className="benefits-section" id="product">
      <div className="container">
        <div className="benefits-header">
          <div className="section-label">WHY AEROPREDICT</div>
          <h2 className="section-heading">Transform Maintenance from Reactive to Proactive</h2>
          <p className="section-subheading">
            Our platform provides end-to-end visibility and unparalleled foresight, empowering your engineering teams to keep fleets in the air.
          </p>
        </div>

        <div className="benefits-grid">
          <BenefitCard 
            title="Increase Asset Lifespan" 
            description="By identifying micro-stressors early, you can perform preventative maintenance that adds years to the lifecycle of your engines."
            icon="⏳"
          />
          <BenefitCard 
            title="Reduce AOG Incidents" 
            description="Aircraft On Ground (AOG) situations cost millions. Our predictive models reduce unplanned downtime by up to 45%."
            icon="✈️"
          />
          <BenefitCard 
            title="Optimize Inventory" 
            description="Know exactly which parts will fail and when. Keep your spare parts inventory lean without risking availability."
            icon="📦"
          />
        </div>
      </div>
    </section>
  );
};

export default BenefitsSection;
