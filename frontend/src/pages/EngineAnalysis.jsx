import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ReferenceArea } from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card" style={{ padding: '12px', background: 'rgba(255,255,255,0.9)' }}>
        <p style={{ margin: '0 0 8px', fontWeight: 600 }}>Cycle {label}</p>
        {payload.map((p, index) => (
          <p key={index} style={{ margin: 0, fontSize: '14px', color: p.color }}>
            {p.name}: {p.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

const EngineAnalysis = () => {
  const [engineId, setEngineId] = useState(1);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    fetch(`http://localhost:8000/api/engine/${engineId}`)
      .then(res => {
        if (!res.ok) throw new Error("Failed to fetch engine data");
        return res.json();
      })
      .then(json => {
        setData(json);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setError("Failed to connect to AI Model Backend.");
        setLoading(false);
      });
  }, [engineId]);

  if (loading || !data) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <h2 style={{ color: 'var(--accent-purple)' }}>Loading AI Analysis...</h2>
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

  const { prediction, recommendation, history, current_cycle } = data;

  return (
    <div>
      <div className="dashboard-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 className="dashboard-title">Individual Engine Analysis</h1>
          <p className="dashboard-subtitle">Deep dive into health, RUL, and sensor telemetry.</p>
        </div>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <span style={{ color: 'var(--text-secondary)' }}>Select Engine ID:</span>
          <select 
            value={engineId} 
            onChange={(e) => setEngineId(Number(e.target.value))}
            style={{ padding: '8px 16px', borderRadius: '8px', border: '1px solid var(--border)', background: 'transparent', outline: 'none' }}
          >
            {[...Array(100)].map((_, i) => (
              <option key={i+1} value={i+1}>Engine #{i+1}</option>
            ))}
          </select>
        </div>
      </div>

      <div className={`glass-card`} style={{ padding: '24px', marginBottom: '32px', border: `1px solid var(--accent-${recommendation.color === 'green' ? 'green' : recommendation.color === 'orange' ? 'orange' : 'pink'})` }}>
        <h2 style={{ fontSize: '24px', fontWeight: 600, margin: '0 0 12px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          {recommendation.emoji} Status: {recommendation.level}
        </h2>
        <h4 style={{ fontSize: '18px', fontWeight: 500, margin: '0 0 8px', color: 'var(--text-secondary)' }}>{recommendation.message}</h4>
        <p style={{ margin: 0, color: 'var(--text-secondary)' }}><strong style={{ color: 'var(--text-primary)' }}>Recommended Action:</strong> {recommendation.action}</p>
      </div>

      <div className="dashboard-grid cols-4">
        <div className="stat-card glass-card">
          <span className="stat-title">Predicted RUL</span>
          <span className="stat-value">{parseInt(prediction.rul)} <span style={{fontSize: '16px', fontWeight: 400, color: 'var(--text-secondary)'}}>cycles</span></span>
        </div>
        <div className="stat-card glass-card">
          <span className="stat-title">Health Score</span>
          <span className="stat-value">{prediction.health_score.toFixed(1)}%</span>
        </div>
        <div className="stat-card glass-card">
          <span className="stat-title">Failure Prob (30d)</span>
          <span className="stat-value">{prediction.failure_probability.toFixed(1)}%</span>
        </div>
        <div className="stat-card glass-card">
          <span className="stat-title">Current Cycle</span>
          <span className="stat-value">{current_cycle}</span>
        </div>
      </div>

      <div className="chart-card glass-card">
        <div className="chart-header">
          <h2 className="chart-title">Degradation Timeline</h2>
        </div>
        <div style={{ height: 400, width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(0,0,0,0.05)" />
              <XAxis dataKey="cycle" name="Operating Cycle" tick={{fill: 'var(--text-secondary)'}} tickLine={false} axisLine={false} />
              <YAxis yAxisId="left" tick={{fill: 'var(--text-secondary)'}} tickLine={false} axisLine={false} />
              <YAxis yAxisId="right" orientation="right" tick={{fill: 'var(--text-secondary)'}} tickLine={false} axisLine={false} />
              <Tooltip content={<CustomTooltip />} cursor={{strokeDasharray: '3 3'}} />
              <Legend verticalAlign="top" height={36} />
              <ReferenceArea yAxisId="right" y1={0} y2={30} fill="var(--accent-pink)" fillOpacity={0.05} />
              <ReferenceArea yAxisId="right" y1={30} y2={70} fill="var(--accent-orange)" fillOpacity={0.05} />
              <ReferenceArea yAxisId="right" y1={70} y2={100} fill="var(--accent-green)" fillOpacity={0.05} />
              
              <Line yAxisId="left" type="monotone" dataKey="rul" name="Predicted RUL" stroke="var(--accent-purple)" strokeWidth={3} dot={false} />
              <Line yAxisId="right" type="monotone" dataKey="health_score" name="Health Score (%)" stroke="var(--accent-green)" strokeWidth={2} strokeDasharray="5 5" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      <div className="dashboard-grid cols-2">
        <div className="chart-card glass-card">
          <div className="chart-header">
            <h2 className="chart-title">Sensor_2 Trend</h2>
          </div>
          <div style={{ height: 250, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(0,0,0,0.05)" />
                <XAxis dataKey="cycle" hide />
                <YAxis domain={['auto', 'auto']} hide />
                <Tooltip content={<CustomTooltip />} />
                <Line type="monotone" dataKey="s2" stroke="var(--accent-orange)" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="chart-card glass-card">
          <div className="chart-header">
            <h2 className="chart-title">Sensor_3 Trend</h2>
          </div>
          <div style={{ height: 250, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(0,0,0,0.05)" />
                <XAxis dataKey="cycle" hide />
                <YAxis domain={['auto', 'auto']} hide />
                <Tooltip content={<CustomTooltip />} />
                <Line type="monotone" dataKey="s3" stroke="var(--accent-pink)" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EngineAnalysis;
