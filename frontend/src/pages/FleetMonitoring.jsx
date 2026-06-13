import React, { useState, useEffect } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from 'recharts';

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="glass-card" style={{ padding: '12px', background: 'rgba(255,255,255,0.9)' }}>
        <p style={{ margin: 0, fontWeight: 600 }}>Engine #{data.id}</p>
        <p style={{ margin: 0, fontSize: '14px', color: 'var(--text-secondary)' }}>Cycles: {data.currentCycles}</p>
        <p style={{ margin: 0, fontSize: '14px', color: 'var(--text-secondary)' }}>RUL: {data.predictedRUL}</p>
        <p style={{ margin: 0, fontSize: '14px', color: 'var(--text-secondary)' }}>Health: {data.healthScore}%</p>
      </div>
    );
  }
  return null;
};

const FleetMonitoring = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/fleet')
      .then(res => res.json())
      .then(json => {
        setData(json.fleet);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setError("Failed to connect to AI Model Backend. Is it running?");
        setLoading(false);
      });
  }, []);
  
  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <h2 style={{ color: 'var(--accent-purple)' }}>Loading Live AI Predictions...</h2>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '24px', background: 'rgba(255, 124, 181, 0.1)', border: '1px solid var(--accent-pink)', borderRadius: '12px', color: 'var(--accent-pink)' }}>
        <h2>Error</h2>
        <p>{error}</p>
      </div>
    );
  }

  const criticalEngines = data.filter(d => d.status === 'Critical');
  const warningEngines = data.filter(d => d.status === 'Warning');
  const attentionEngines = [...criticalEngines, ...warningEngines].sort((a, b) => a.predictedRUL - b.predictedRUL);

  const getStatusColor = (status) => {
    if (status === 'Critical') return 'var(--accent-pink)';
    if (status === 'Warning') return 'var(--accent-orange)';
    return 'var(--accent-green)';
  };

  return (
    <div>
      <div className="dashboard-header">
        <h1 className="dashboard-title">Fleet Monitoring</h1>
        <p className="dashboard-subtitle">Real-time AI predictions for all active engines.</p>
      </div>

      <div className="dashboard-grid cols-3">
        <div className="stat-card glass-card">
          <span className="stat-title">Total Active Engines</span>
          <span className="stat-value">{data.length}</span>
          <span className="stat-footer">Analyzed by Multi-Task TCN</span>
        </div>
        <div className="stat-card glass-card">
          <span className="stat-title">Engines in Warning</span>
          <span className="stat-value" style={{ color: 'var(--accent-orange)' }}>{warningEngines.length}</span>
          <span className="stat-footer">RUL between 16-30 cycles</span>
        </div>
        <div className="stat-card glass-card">
          <span className="stat-title">Engines in Critical State</span>
          <span className="stat-value" style={{ color: 'var(--accent-pink)' }}>{criticalEngines.length}</span>
          <span className="stat-footer">RUL ≤ 15 cycles</span>
        </div>
      </div>

      <div className="chart-card glass-card">
        <div className="chart-header">
          <h2 className="chart-title">Fleet Health Overview (RUL vs Cycles)</h2>
        </div>
        <div style={{ height: 400, width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(0,0,0,0.05)" />
              <XAxis dataKey="currentCycles" type="number" name="Current Cycles" tick={{fill: 'var(--text-secondary)'}} tickLine={false} axisLine={false} />
              <YAxis dataKey="predictedRUL" type="number" name="Predicted RUL" tick={{fill: 'var(--text-secondary)'}} tickLine={false} axisLine={false} />
              <Tooltip content={<CustomTooltip />} cursor={{strokeDasharray: '3 3'}} />
              <ReferenceLine y={30} stroke="var(--accent-orange)" strokeDasharray="3 3" label={{ position: 'top', value: 'Warning Threshold', fill: 'var(--accent-orange)', fontSize: 12 }} />
              <ReferenceLine y={15} stroke="var(--accent-pink)" strokeDasharray="3 3" label={{ position: 'top', value: 'Critical Threshold', fill: 'var(--accent-pink)', fontSize: 12 }} />
              <Scatter name="Engines" data={data}>
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getStatusColor(entry.status)} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="chart-card glass-card">
        <div className="chart-header">
          <h2 className="chart-title">Engines Requiring Immediate Attention</h2>
        </div>
        <div className="data-table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Engine ID</th>
                <th>Status</th>
                <th>Predicted RUL</th>
                <th>Health Score</th>
                <th>Current Cycles</th>
              </tr>
            </thead>
            <tbody>
              {attentionEngines.map(engine => (
                <tr key={engine.id}>
                  <td style={{ fontWeight: 500 }}>#{engine.id}</td>
                  <td>
                    <span className={`status-badge ${engine.status.toLowerCase()}`}>
                      {engine.status === 'Critical' ? '🔴' : '🟡'} {engine.status}
                    </span>
                  </td>
                  <td style={{ fontWeight: 600, color: getStatusColor(engine.status) }}>{engine.predictedRUL}</td>
                  <td>{engine.healthScore}%</td>
                  <td>{engine.currentCycles}</td>
                </tr>
              ))}
              {attentionEngines.length === 0 && (
                <tr>
                  <td colSpan="5" style={{ textAlign: 'center', padding: '32px' }}>No engines currently require attention.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default FleetMonitoring;
