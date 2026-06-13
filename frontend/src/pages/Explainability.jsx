import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const mockFeatureImportance = [
  { feature: 'Sensor 11 (Static pressure at HPC outlet)', importance: 0.18 },
  { feature: 'Sensor 4 (Total temp at LPT outlet)', importance: 0.15 },
  { feature: 'Sensor 15 (Bypass Ratio)', importance: 0.12 },
  { feature: 'Sensor 9 (Physical core speed)', importance: 0.10 },
  { feature: 'Sensor 14 (Corrected core speed)', importance: 0.08 },
  { feature: 'Sensor 2 (Total temp at LPC outlet)', importance: 0.07 },
  { feature: 'Sensor 3 (Total temp at HPC outlet)', importance: 0.06 },
  { feature: 'Sensor 8 (Physical fan speed)', importance: 0.05 },
  { feature: 'Sensor 13 (Corrected fan speed)', importance: 0.04 },
  { feature: 'Sensor 21 (Bleed Enthalpy)', importance: 0.03 }
];

const mockLocalShap = [
  { feature: 'Base Value', value: 75.0, type: 'base' },
  { feature: 'Sensor 11 (High)', value: -12.5, type: 'negative' },
  { feature: 'Sensor 4 (High)', value: -8.2, type: 'negative' },
  { feature: 'Sensor 15 (Low)', value: 5.1, type: 'positive' },
  { feature: 'Sensor 9 (Normal)', value: 2.3, type: 'positive' },
  { feature: 'Sensor 14 (High)', value: -4.7, type: 'negative' }
];

const Explainability = () => {
  return (
    <div>
      <div className="dashboard-header">
        <h1 className="dashboard-title">Model Explainability (SHAP)</h1>
        <p className="dashboard-subtitle">Understand what drives the AI predictions for engine health.</p>
      </div>

      <div className="chart-card glass-card">
        <div className="chart-header">
          <h2 className="chart-title">Global Feature Importance</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginTop: '8px' }}>
            The top 10 sensors that most strongly influence the Remaining Useful Life predictions across the entire fleet.
          </p>
        </div>
        <div style={{ height: 400, width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={mockFeatureImportance} layout="vertical" margin={{ top: 5, right: 30, left: 200, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="rgba(0,0,0,0.05)" />
              <XAxis type="number" tick={{fill: 'var(--text-secondary)'}} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="feature" tick={{fill: 'var(--text-secondary)'}} axisLine={false} tickLine={false} width={250} />
              <Tooltip cursor={{fill: 'rgba(0,0,0,0.02)'}} contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
              <Bar dataKey="importance" fill="var(--accent-purple)" radius={[0, 4, 4, 0]} barSize={20} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="dashboard-grid cols-2">
        <div className="chart-card glass-card">
          <div className="chart-header">
            <h2 className="chart-title">Local Prediction Explanation</h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginTop: '8px' }}>
              How specific sensor readings shifted the prediction for Engine #1.
            </p>
          </div>
          <div style={{ height: 300, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mockLocalShap} layout="vertical" margin={{ top: 5, right: 30, left: 120, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="rgba(0,0,0,0.05)" />
                <XAxis type="number" tick={{fill: 'var(--text-secondary)'}} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="feature" tick={{fill: 'var(--text-secondary)'}} axisLine={false} tickLine={false} width={150} />
                <Tooltip cursor={{fill: 'rgba(0,0,0,0.02)'}} />
                <Bar dataKey="value" barSize={20} radius={[4, 4, 4, 4]}>
                  {mockLocalShap.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.type === 'base' ? 'var(--text-muted)' : entry.type === 'positive' ? 'var(--accent-green)' : 'var(--accent-pink)'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="chart-card glass-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
           <h2 className="chart-title" style={{ marginBottom: '16px' }}>Prediction Summary</h2>
           <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
             <div style={{ padding: '16px', background: 'rgba(107, 91, 255, 0.05)', borderRadius: '12px' }}>
               <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>Base Expected RUL</span>
               <div style={{ fontSize: '24px', fontWeight: 600, color: 'var(--text-primary)' }}>75.0 cycles</div>
             </div>
             <div style={{ padding: '16px', background: 'rgba(255, 124, 181, 0.05)', borderRadius: '12px' }}>
               <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>Total Negative Impact</span>
               <div style={{ fontSize: '24px', fontWeight: 600, color: 'var(--accent-pink)' }}>-25.4 cycles</div>
             </div>
             <div style={{ padding: '16px', background: 'rgba(184, 231, 106, 0.1)', borderRadius: '12px' }}>
               <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>Total Positive Impact</span>
               <div style={{ fontSize: '24px', fontWeight: 600, color: 'var(--accent-green)' }}>+7.4 cycles</div>
             </div>
             <div style={{ padding: '16px', border: '1px solid var(--border)', borderRadius: '12px' }}>
               <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>Final Predicted RUL</span>
               <div style={{ fontSize: '28px', fontWeight: 600, color: 'var(--text-primary)' }}>57.0 cycles</div>
             </div>
           </div>
        </div>
      </div>
    </div>
  );
};

export default Explainability;
