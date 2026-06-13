/**
 * CervicalAI — React Frontend Application
 * Adaptive Grid System for Desktop & Mobile viewports
 * Implements: Dashboard, Image Analysis, Risk Assessment, History, Report, Settings
 */

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { classifyImage, assessRisk, generateReport, checkHealth } from './api.js'
import './styles.css'

// ─── Constants ───────────────────────────────────────────────

const SIPAKMED_CLASSES = {
  0: { name: 'Dyskeratotic', risk: 'high', color: 'var(--red)' },
  1: { name: 'Koilocytotic', risk: 'high', color: 'var(--red)' },
  2: { name: 'Metaplastic', risk: 'medium', color: 'var(--amber)' },
  3: { name: 'Parabasal', risk: 'medium', color: 'var(--amber)' },
  4: { name: 'Superficial-Intermediate', risk: 'low', color: 'var(--green)' },
}

const RISK_COLORS = {
  Low: 'var(--green)',
  Moderate: 'var(--amber)',
  High: 'var(--red)',
  Critical: 'var(--red)',
}

const INITIAL_RISK_FORM = {
  age: 30, num_sexual_partners: 1, first_sexual_intercourse: 18,
  num_pregnancies: 0, smokes: 0, smokes_years: 0, smokes_packs_year: 0,
  hormonal_contraceptives: 0, hormonal_contraceptives_years: 0,
  iud: 0, iud_years: 0, stds: 0, stds_number: 0,
  stds_condylomatosis: 0, stds_hpv: 0, stds_hiv: 0, stds_syphilis: 0,
  dx_cancer: 0, dx_cin: 0, dx_hpv: 0,
  patient_name: '', patient_id: '',
}

// ─── Hooks ───────────────────────────────────────────────────

function useClock() {
  const [time, setTime] = useState('')
  useEffect(() => {
    const tick = () => {
      const now = new Date()
      setTime(now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }))
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])
  return time
}

function useGreeting() {
  const h = new Date().getHours()
  if (h < 12) return 'GOOD MORNING'
  if (h < 17) return 'GOOD AFTERNOON'
  return 'GOOD EVENING'
}

// ─── Small UI Components ──────────────────────────────────────

function SignalIcon() {
  return (
    <div className="signal-icon">
      <span /><span /><span /><span />
    </div>
  )
}

function Toast({ message, visible }) {
  return (
    <div className={`app-toast ${visible ? 'visible' : ''}`}>
      {message}
    </div>
  )
}

function LoadingDots({ text = 'Processing...' }) {
  return (
    <div className="ai-loading">
      <div className="typing-dots">
        <span /><span /><span />
      </div>
      <div className="loading-text">{text}</div>
    </div>
  )
}

function RiskBadge({ tier, score }) {
  const color = RISK_COLORS[tier] || 'var(--amber)'
  return (
    <div className="risk-badge" style={{ borderColor: color, background: `${color}15` }}>
      <div className="rb-score" style={{ color }}>{Math.round(score * 100)}%</div>
      <div className="rb-tier" style={{ color }}>{tier?.toUpperCase()} RISK</div>
    </div>
  )
}

function ProgressBar({ value, max = 1, color = 'var(--cyan)' }) {
  return (
    <div className="progress-track">
      <div className="progress-fill" style={{ width: `${(value / max) * 100}%`, background: color }} />
    </div>
  )
}

function SectionHeader({ title, right }) {
  return (
    <div className="sec-header">
      <div className="sec-title">{title}</div>
      {right && <div className="sec-time">{right}</div>}
    </div>
  )
}

// ─── STATUS BAR ───────────────────────────────────────────────

function StatusBar({ backendOk }) {
  const time = useClock()
  return (
    <div className="status-bar">
      <div className="status-left">
        <span className="app-logo">CervicalAI</span>
        <span className="ai-badge">FASTVIT+XGB</span>
        {backendOk !== null && (
          <span className={`conn-dot ${backendOk ? 'ok' : 'err'}`} title={backendOk ? 'Backend connected' : 'Backend offline'} />
        )}
      </div>
      <div className="status-right">
        <SignalIcon />
        <span className="status-time">{time}</span>
      </div>
    </div>
  )
}

// ─── BOTTOM NAV ───────────────────────────────────────────────

const NAV_ITEMS = [
  { id: 'dashboard', icon: '🏥', label: 'Overview' },
  { id: 'image', icon: '🔬', label: 'Image AI' },
  { id: 'risk', icon: '📊', label: 'Risk AI' },
  { id: 'report', icon: '📋', label: 'Report' },
  { id: 'settings', icon: '⚙️', label: 'Settings' },
]

function BottomNav({ current, onNav, hasImageResult, hasRiskResult }) {
  return (
    <nav className="bottom-nav">
      {NAV_ITEMS.map(item => (
        <button
          key={item.id}
          className={`nav-btn ${current === item.id ? 'active' : ''} ${item.id === 'report' ? 'report-nav' : ''}`}
          onClick={() => onNav(item.id)}
        >
          <span className="nav-icon">{item.icon}</span>
          <span className="nav-label">{item.label}</span>
          {item.id === 'image' && hasImageResult && <span className="notif-badge" />}
          {item.id === 'risk' && hasRiskResult && <span className="notif-badge" />}
        </button>
      ))}
    </nav>
  )
}

// ─── VIEW: DASHBOARD ─────────────────────────────────────────

function DashboardView({ imageResult, riskResult, onNav }) {
  const greeting = useGreeting()
  const hasBoth = imageResult && riskResult
  const combined = hasBoth
    ? riskResult.risk_score * 0.6 + imageResult.risk_score * 0.4
    : null

  return (
    <div className="view active" id="view-dashboard">
      {/* Greeting */}
      <div className="greeting">
        <div className="greeting-sub">{greeting}</div>
        <div className="greeting-name">
          CervicalAI <span>Detection Framework</span>
        </div>
      </div>

      {/* Responsive Dashboard Grid Matrix */}
      <div className="dashboard-grid-layout" style={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: '20px' }}>
        
        {/* Left Matrix: Analytics Data */}
        <div className="dashboard-main-col" style={{ flex: '1 1 500px', minWidth: '320px' }}>
          {/* Combined Score Card */}
          <div className="score-card">
            <div className="score-circle">
              <svg viewBox="0 0 70 70">
                <defs>
                  <linearGradient id="scoreGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="var(--cyan)" />
                    <stop offset="100%" stopColor="var(--purple)" />
                  </linearGradient>
                </defs>
                <circle fill="none" stroke="rgba(0,212,255,.1)" strokeWidth="6" cx="35" cy="35" r="29" />
                <circle
                  fill="none" stroke="url(#scoreGrad)" strokeWidth="6"
                  cx="35" cy="35" r="29"
                  strokeDasharray="182"
                  strokeDashoffset={combined !== null ? 182 - combined * 182 : 182}
                  strokeLinecap="round"
                  transform="rotate(-90 35 35)"
                  style={{ transition: 'stroke-dashoffset 1s ease' }}
                />
              </svg>
              <div className="score-num">
                {combined !== null ? `${Math.round(combined * 100)}%` : '--'}
              </div>
            </div>
            <div className="score-info">
              <h3>Combined AI Risk Index
                {combined !== null && (
                  <span className={`tag tag-${combined < 0.35 ? 'green' : combined < 0.6 ? 'amber' : 'red'}`}>
                    {combined < 0.35 ? 'Low' : combined < 0.6 ? 'Moderate' : 'High'}
                  </span>
                )}
              </h3>
              <p>
                {!hasBoth
                  ? 'Complete both Stage 1 (Risk Assessment) and Stage 2 (Image Analysis) for combined scoring.'
                  : 'Weighted combination: 60% clinical risk + 40% cytological image classification.'}
              </p>
              {combined !== null && (
                <span className="score-grade">
                  XGBoost: {Math.round(riskResult.risk_score * 100)}% · FastViT: {Math.round(imageResult.risk_score * 100)}%
                </span>
              )}
            </div>
          </div>

          {/* Two Stage Status */}
          <SectionHeader title="Detection Stages" />
          <div className="vitals-grid">
            {/* Stage 1 */}
            <div className="vital-mini clickable" onClick={() => onNav('risk')} style={{ cursor: 'pointer' }}>
              <div className="vm-icon">📊</div>
              <div>
                <span className="vm-val" style={{ fontSize: riskResult ? '1rem' : '1.25rem' }}>
                  {riskResult ? `${Math.round(riskResult.risk_score * 100)}%` : '—'}
                </span>
              </div>
              <div className="vm-label">XGBoost Risk</div>
              <div className="vm-trend" style={{ color: riskResult ? RISK_COLORS[riskResult.risk_tier] : 'var(--text3)' }}>
                {riskResult ? riskResult.risk_tier : '→ Not run'}
              </div>
              {riskResult && (
                <div className="vm-bar" style={{
                  background: `linear-gradient(90deg,${RISK_COLORS[riskResult.risk_tier]},transparent)`,
                  width: `${riskResult.risk_score * 100}%`
                }} />
              )}
            </div>

            {/* Stage 2 */}
            <div className="vital-mini clickable" onClick={() => onNav('image')} style={{ cursor: 'pointer' }}>
              <div className="vm-icon">🔬</div>
              <div>
                <span className="vm-val" style={{ fontSize: imageResult ? '1rem' : '1.25rem' }}>
                  {imageResult ? `${Math.round(imageResult.confidence * 100)}%` : '—'}
                </span>
              </div>
              <div className="vm-label">FastViT Confidence</div>
              <div className="vm-trend" style={{ color: imageResult ? SIPAKMED_CLASSES[imageResult.predicted_class]?.color : 'var(--text3)' }}>
                {imageResult ? imageResult.class_name : '→ No image'}
              </div>
              {imageResult && (
                <div className="vm-bar" style={{
                  background: `linear-gradient(90deg,${SIPAKMED_CLASSES[imageResult.predicted_class]?.color || 'var(--cyan)'},transparent)`,
                  width: `${imageResult.confidence * 100}%`
                }} />
              )}
            </div>

            {/* SHAP */}
            <div className="vital-mini">
              <div className="vm-icon">🧬</div>
              <div><span className="vm-val">{riskResult ? riskResult.features_analyzed : '—'}</span></div>
              <div className="vm-label">SHAP Features</div>
              <div className="vm-trend norm">{riskResult ? '→ Analyzed' : '→ Pending'}</div>
            </div>

            {/* GradCAM */}
            <div className="vital-mini">
              <div className="vm-icon">🎯</div>
              <div><span className="vm-val">{imageResult ? '5' : '—'}</span></div>
              <div className="vm-label">GradCAM Regions</div>
              <div className="vm-trend norm">{imageResult ? '→ Generated' : '→ Pending'}</div>
            </div>
          </div>
        </div>

        {/* Right Matrix: AI Insights & Quick Links */}
        <div className="dashboard-side-col" style={{ flex: '1 1 350px', minWidth: '320px' }}>
          <SectionHeader title="AI Insights" />
          <div id="alertsList">
            {!imageResult && !riskResult && (
              <div className="alert-item">
                <div className="alert-icon">ℹ️</div>
                <div className="alert-text">
                  <p>No analysis run yet</p>
                  <span>Upload a pap smear image or fill the risk form to begin analysis</span>
                </div>
              </div>
            )}
            {riskResult && (
              <div className={`alert-item ${riskResult.risk_tier === 'Low' ? 'ok' : riskResult.risk_tier === 'Critical' ? 'critical' : ''}`}>
                <div className="alert-icon">{riskResult.risk_tier === 'Low' ? '✅' : riskResult.risk_tier === 'Critical' ? '🚨' : '⚠️'}</div>
                <div className="alert-text">
                  <p>Stage 1: {riskResult.risk_tier} Clinical Risk ({riskResult.urgency})</p>
                  <span>{riskResult.recommendation}</span>
                </div>
              </div>
            )}
            {imageResult && (
              <div className={`alert-item ${imageResult.risk_level === 'low' ? 'ok' : imageResult.risk_level === 'high' ? 'critical' : ''}`}>
                <div className="alert-icon">
                  {imageResult.risk_level === 'low' ? '✅' : imageResult.risk_level === 'high' ? '🚨' : '⚠️'}
                </div>
                <div className="alert-text">
                  <p>Stage 2: {imageResult.class_name} Cells ({Math.round(imageResult.confidence * 100)}% confidence)</p>
                  <span>{imageResult.description}</span>
                </div>
              </div>
            )}
            {hasBoth && (
              <div className="alert-item ok">
                <div className="alert-icon">📊</div>
                <div className="alert-text">
                  <p>Combined report ready for generation</p>
                  <span>Go to Report tab to download full PDF with SHAP and GradCAM explanations</span>
                </div>
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
            <button className="btn-half btn-outline" onClick={() => onNav('image')}>🔬 Analyze Image</button>
            <button className="btn-half btn-outline" onClick={() => onNav('risk')}>📊 Risk Form</button>
          </div>
        </div>
      </div>
      <div style={{ height: 16 }} />
    </div>
  )
}

// ─── VIEW: IMAGE ANALYSIS ─────────────────────────────────────

function ImageView({ result, onResult, showToast }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showGradcam, setShowGradcam] = useState(true)
  const fileInputRef = useRef()
  const dropRef = useRef()

  const handleFile = useCallback((f) => {
    if (!f) return
    if (!f.type.startsWith('image/')) { showToast('⚠️ Please upload an image file'); return }
    setFile(f)
    setError(null)
    const url = URL.createObjectURL(f)
    setPreview(url)
  }, [showToast])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    dropRef.current?.classList.remove('drag-over')
    const f = e.dataTransfer.files[0]
    handleFile(f)
  }, [handleFile])

  const handleAnalyze = async () => {
    if (!file) { showToast('Please select a pap smear image first'); return }
    setLoading(true)
    setError(null)
    try {
      const res = await classifyImage(file)
      onResult(res.data)
      showToast('✅ Image classified successfully')
    } catch (err) {
      setError(err.message)
      showToast('❌ Classification failed: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="view active" id="view-image">
      <div className="ai-header">
        <div className="ai-title">Stage 2 — <span>FastViT</span> Image Analysis</div>
        <div className="ai-sub">
          Upload a Pap smear cytological image for AI classification into SipakMed categories.
          GradCAM saliency maps highlight diagnostically relevant cell regions.
        </div>
      </div>

      {/* Adaptive Side-by-Side Split Workspace */}
      <div className="image-grid-layout" style={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: '20px' }}>
        
        <div className="image-upload-col" style={{ flex: '1 1 400px', minWidth: '320px' }}>
          <SectionHeader title="Upload Pap Smear Image" />
          <div
            ref={dropRef}
            className={`upload-zone ${preview ? 'has-preview' : ''}`}
            onClick={() => !preview && fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); dropRef.current?.classList.add('drag-over') }}
            onDragLeave={() => dropRef.current?.classList.remove('drag-over')}
            onDrop={handleDrop}
          >
            {preview ? (
              <div className="preview-wrap">
                <img
                  src={showGradcam && result?.gradcam_image ? result.gradcam_image : preview}
                  alt="Pap smear preview"
                  className="preview-img"
                />
                {result?.gradcam_image && (
                  <div className="gradcam-toggle">
                    <button
                      className={`gc-btn ${!showGradcam ? 'active' : ''}`}
                      onClick={(e) => { e.stopPropagation(); setShowGradcam(false) }}
                    >Original</button>
                    <button
                      className={`gc-btn ${showGradcam ? 'active' : ''}`}
                      onClick={(e) => { e.stopPropagation(); setShowGradcam(true) }}
                    >GradCAM</button>
                  </div>
                )}
                <button
                  className="clear-img-btn"
                  onClick={(e) => { e.stopPropagation(); setFile(null); setPreview(null); onResult(null) }}
                >✕ Clear</button>
              </div>
            ) : (
              <div className="upload-placeholder">
                <div className="upload-icon">🔬</div>
                <div className="upload-text">Drop pap smear image here</div>
                <div className="upload-sub">or tap to browse · JPG, PNG, TIFF supported</div>
                <div className="upload-hint">SipakMed categories: Dyskeratotic · Koilocytotic · Metaplastic · Parabasal · Superficial-Intermediate</div>
              </div>
            )}
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => handleFile(e.target.files[0])}
          />

          {file && !loading && (
            <button className="btn-full btn-cyan" onClick={handleAnalyze} style={{ marginTop: 10 }}>
              🔍 Classify with FastViT
            </button>
          )}

          {loading && (
            <div className="ai-response">
              <div className="air-header">
                <div className="air-avatar">🔬</div>
                <div>
                  <div className="air-name">FastViT-T8</div>
                  <div style={{ fontSize: '.58rem', color: 'var(--text3)', fontFamily: 'var(--mono)' }}>Analyzing cytological image...</div>
                </div>
              </div>
              <LoadingDots text="Running FastViT inference + GradCAM generation..." />
            </div>
          )}

          {error && (
            <div className="ai-response">
              <div className="air-header">
                <div className="air-avatar">⚠️</div>
                <div className="air-name" style={{ color: 'var(--amber)' }}>Error</div>
              </div>
              <div className="air-body">{error}</div>
            </div>
          )}
        </div>

        <div className="image-result-col" style={{ flex: '1 1 400px', minWidth: '320px' }}>
          {result && !loading && (
            <div className="ai-response" style={{ marginTop: 0 }}>
              <div className="air-header">
                <div className="air-avatar">🔬</div>
                <div>
                  <div className="air-name">FastViT Classification Result</div>
                  <div style={{ fontSize: '.58rem', color: 'var(--text3)', fontFamily: 'var(--mono)' }}>
                    {result.model} · {result.dataset}
                  </div>
                </div>
                <div className="air-tag">STAGE 2</div>
              </div>
              <div className="air-body">
                <div className="severity-bar">
                  <div className="sev-dots">
                    {[0, 1, 2, 3, 4].map(i => {
                      const dots = result.risk_level === 'low' ? 1 : result.risk_level === 'medium' ? 3 : 5
                      const cls = i < dots ? `active-${result.risk_level === 'low' ? 'low' : result.risk_level === 'medium' ? 'med' : 'high'}` : ''
                      return <div key={i} className={`sev-dot ${cls}`} />
                    })}
                  </div>
                  <span className="sev-label" style={{ color: SIPAKMED_CLASSES[result.predicted_class]?.color }}>
                    {result.risk_level?.toUpperCase()} RISK
                  </span>
                </div>

                <div className="r-section">
                  <div className="r-label"><span>Predicted Cell Type</span></div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <span style={{ fontSize: '1.2rem', fontWeight: 700, color: SIPAKMED_CLASSES[result.predicted_class]?.color }}>
                      {result.class_name}
                    </span>
                    <span className={`tag tag-${result.risk_level === 'low' ? 'green' : result.risk_level === 'medium' ? 'amber' : 'red'}`}>
                      {Math.round(result.confidence * 100)}% confidence
                    </span>
                  </div>
                  <p style={{ fontSize: '.78rem', color: 'var(--text2)', lineHeight: 1.6 }}>{result.description}</p>
                </div>

                <div className="r-section">
                  <div className="r-label"><span>Class Probability Distribution</span></div>
                  {result.class_probabilities?.map(cp => (
                    <div key={cp.class_id} style={{ marginBottom: 6 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                        <span style={{ fontSize: '.7rem', color: cp.class_name === result.class_name ? 'var(--cyan)' : 'var(--text2)', fontFamily: 'var(--mono)' }}>
                          {cp.class_name}
                        </span>
                        <span style={{ fontSize: '.7rem', fontFamily: 'var(--mono)', fontWeight: 700, color: SIPAKMED_CLASSES[cp.class_id]?.color }}>
                          {(cp.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <ProgressBar value={cp.probability} color={SIPAKMED_CLASSES[cp.class_id]?.color || 'var(--cyan)'} />
                    </div>
                  ))}
                </div>
              </div>
              <div className="disclaimer">
                ⚠️ FastViT-T8 trained on SipakMed dataset. Results require clinical validation by a pathologist.
              </div>
            </div>
          )}
        </div>

      </div>
      <div style={{ height: 16 }} />
    </div>
  )
}

// ─── VIEW: RISK ASSESSMENT ────────────────────────────────────

function RiskView({ result, onResult, showToast }) {
  const [form, setForm] = useState(INITIAL_RISK_FORM)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const handleSubmit = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await assessRisk(form)
      onResult(res.data)
      showToast('✅ Risk assessment complete')
    } catch (err) {
      setError(err.message)
      showToast('❌ Assessment failed: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  const Field = ({ label, field, type = 'number', min, max, step = 1, binary = false }) => (
    <div className="setting-row">
      <div className="setting-info">
        <p>{label}</p>
        {binary && <span>0 = No, 1 = Yes</span>}
      </div>
      {binary ? (
        <div className={`toggle ${form[field] ? 'on' : ''}`} onClick={() => set(field, form[field] ? 0 : 1)} />
      ) : (
        <input
          className="risk-input"
          type={type}
          min={min} max={max} step={step}
          value={form[field]}
          onChange={(e) => set(field, parseFloat(e.target.value) || 0)}
        />
      )}
    </div>
  )

  return (
    <div className="view active" id="view-risk">
      <div className="ai-header">
        <div className="ai-title">Stage 1 — <span>XGBoost</span> Risk Assessment</div>
        <div className="ai-sub">
          Enter patient clinical data for AI-powered risk stratification.
          SHAP values explain each feature's contribution to the prediction.
        </div>
      </div>

      {/* Adaptive 2-Column Form Structure */}
      <div className="form-grid-layout" style={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: '0 20px' }}>
        
        {/* Form Column Left */}
        <div className="form-column" style={{ flex: '1 1 400px', minWidth: '320px' }}>
          {/* Patient info */}
          <div className="settings-section">
            <div className="settings-label">Patient Information</div>
            <div className="settings-group">
              <div className="setting-row">
                <div className="setting-info"><p>Patient Name</p><span>For report generation</span></div>
                <input className="risk-input text-input" type="text" placeholder="Optional"
                  value={form.patient_name} onChange={e => set('patient_name', e.target.value)} />
              </div>
              <div className="setting-row">
                <div className="setting-info"><p>Patient ID</p><span>Reference number</span></div>
                <input className="risk-input text-input" type="text" placeholder="Optional"
                  value={form.patient_id} onChange={e => set('patient_id', e.target.value)} />
              </div>
            </div>
          </div>

          {/* Demographics */}
          <div className="settings-section">
            <div className="settings-label">Demographics & Reproductive History</div>
            <div className="settings-group">
              <Field label="Age (years)" field="age" min={13} max={90} />
              <Field label="Sexual Partners" field="num_sexual_partners" min={0} max={50} />
              <Field label="First Intercourse (age)" field="first_sexual_intercourse" min={10} max={40} />
              <Field label="Pregnancies" field="num_pregnancies" min={0} max={20} />
            </div>
          </div>

          {/* Smoking */}
          <div className="settings-section">
            <div className="settings-label">Smoking History</div>
            <div className="settings-group">
              <Field label="Smoker" field="smokes" binary />
              {form.smokes === 1 && <>
                <Field label="Smoking Duration (years)" field="smokes_years" min={0} max={60} />
                <Field label="Packs Per Year" field="smokes_packs_year" min={0} max={60} />
              </>}
            </div>
          </div>
        </div>

        {/* Form Column Right */}
        <div className="form-column" style={{ flex: '1 1 400px', minWidth: '320px' }}>
          {/* Contraceptives */}
          <div className="settings-section">
            <div className="settings-label">Contraceptives & IUD</div>
            <div className="settings-group">
              <Field label="Hormonal Contraceptives" field="hormonal_contraceptives" binary />
              {form.hormonal_contraceptives === 1 && (
                <Field label="Duration (years)" field="hormonal_contraceptives_years" min={0} max={40} />
              )}
              <Field label="IUD" field="iud" binary />
              {form.iud === 1 && (
                <Field label="IUD Duration (years)" field="iud_years" min={0} max={30} />
              )}
            </div>
          </div>

          {/* STDs */}
          <div className="settings-section">
            <div className="settings-label">STD History</div>
            <div className="settings-group">
              <Field label="Has STD" field="stds" binary />
              {form.stds === 1 && <>
                <Field label="Number of STDs" field="stds_number" min={0} max={10} />
                <Field label="Condylomatosis" field="stds_condylomatosis" binary />
                <Field label="HPV Infection" field="stds_hpv" binary />
                <Field label="HIV Status" field="stds_hiv" binary />
                <Field label="Syphilis" field="stds_syphilis" binary />
              </>}
            </div>
          </div>

          {/* Prior Diagnoses */}
          <div className="settings-section">
            <div className="settings-label">Prior Diagnoses</div>
            <div className="settings-group">
              <Field label="Cancer Diagnosis" field="dx_cancer" binary />
              <Field label="CIN Diagnosis" field="dx_cin" binary />
              <Field label="HPV Diagnosis" field="dx_hpv" binary />
            </div>
          </div>
        </div>

      </div>

      <button className="btn-full btn-cyan" onClick={handleSubmit} disabled={loading}>
        {loading ? '⏳ Analyzing...' : '📊 Run XGBoost Analysis'}
      </button>

      {loading && (
        <div className="ai-response" style={{ marginTop: 12 }}>
          <div className="air-header">
            <div className="air-avatar">📊</div>
            <div className="air-name">XGBoost + SHAP</div>
          </div>
          <LoadingDots text="Running XGBoost prediction + SHAP explainability..." />
        </div>
      )}

      {error && (
        <div className="ai-response" style={{ marginTop: 12 }}>
          <div className="air-header">
            <div className="air-avatar">⚠️</div>
            <div className="air-name" style={{ color: 'var(--amber)' }}>Error</div>
          </div>
          <div className="air-body">{error}</div>
        </div>
      )}

      {result && !loading && (
        <div className="ai-response" style={{ marginTop: 12 }}>
          <div className="air-header">
            <div className="air-avatar">📊</div>
            <div>
              <div className="air-name">XGBoost Risk Assessment</div>
              <div style={{ fontSize: '.58rem', color: 'var(--text3)', fontFamily: 'var(--mono)' }}>
                {result.model_used}
              </div>
            </div>
            <div className="air-tag">STAGE 1</div>
          </div>
          <div className="air-body">
            <RiskBadge tier={result.risk_tier} score={result.risk_score} />

            <div className="r-section" style={{ marginTop: 12 }}>
              <div className="r-label"><span>Clinical Recommendation</span></div>
              <div className="recommendation-box">
                <span className={`tag tag-${result.urgency === 'Routine' ? 'green' : result.urgency === 'Immediate' ? 'red' : 'amber'}`}>
                  {result.urgency}
                </span>
                <p style={{ marginTop: 6, fontSize: '.78rem', color: 'var(--text)', lineHeight: 1.6 }}>
                  {result.recommendation}
                </p>
              </div>
            </div>

            {result.top_risk_factors?.length > 0 && (
              <div className="r-section">
                <div className="r-label"><span>Top Risk Factors (SHAP)</span></div>
                {result.top_risk_factors.map((rf, i) => (
                  <div key={i} className="shap-row">
                    <span className="shap-name">{rf.display_name}</span>
                    <span className="shap-val" style={{ color: rf.shap_value > 0 ? 'var(--red)' : 'var(--cyan)' }}>
                      {rf.shap_value > 0 ? '↑' : '↓'} {Math.abs(rf.shap_value).toFixed(4)}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {result.shap_chart && (
              <div className="r-section">
                <div className="r-label"><span>SHAP Waterfall Chart</span></div>
                <img src={result.shap_chart} alt="SHAP waterfall chart" className="shap-chart-img" />
              </div>
            )}
          </div>
          <div className="disclaimer">
            ⚠️ XGBoost trained on Kaggle Cervical Cancer Dataset (Ranzeet013). Clinical use requires physician review.
          </div>
        </div>
      )}
      <div style={{ height: 16 }} />
    </div>
  )
}

// ─── VIEW: REPORT ─────────────────────────────────────────────

function ReportView({ imageResult, riskResult, showToast }) {
  const [loading, setLoading] = useState(false)
  const [physician, setPhysician] = useState('')
  const [patientAge, setPatientAge] = useState('')

  const hasBoth = imageResult && riskResult
  const combined = hasBoth ? riskResult.risk_score * 0.6 + imageResult.risk_score * 0.4 : null

  const handleGenerate = async () => {
    if (!imageResult && !riskResult) {
      showToast('⚠️ Please run at least one analysis first')
      return
    }
    setLoading(true)
    try {
      const payload = {
        patient_name: riskResult?.patient_name || imageResult?.patient_name || 'Anonymous Patient',
        patient_id: riskResult?.patient_id || undefined,
        patient_age: patientAge ? parseFloat(patientAge) : undefined,
        referring_physician: physician || 'N/A',
        risk_score: riskResult?.risk_score,
        risk_tier: riskResult?.risk_tier,
        urgency: riskResult?.urgency,
        recommendation: riskResult?.recommendation,
        top_risk_factors: riskResult?.top_risk_factors,
        shap_chart: riskResult?.shap_chart,
        model_used: riskResult?.model_used,
        input_features: riskResult?.input_features,
        image_class_name: imageResult?.class_name,
        image_risk_level: imageResult?.risk_level,
        image_confidence: imageResult?.confidence,
        image_description: imageResult?.description,
        class_probabilities: imageResult?.class_probabilities,
        gradcam_image: imageResult?.gradcam_image,
        combined_risk: combined,
      }
      await generateReport(payload)
      showToast('✅ PDF report downloaded!')
    } catch (err) {
      showToast('❌ Report generation failed: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="view active" id="view-report">
      <div className="ai-header">
        <div className="ai-title">📋 <span>Clinical</span> Report</div>
        <div className="ai-sub">
          Generate a comprehensive PDF report with SHAP waterfall charts, GradCAM heatmaps,
          and clinical recommendations. Suitable for physician review.
        </div>
      </div>

      <SectionHeader title="Analysis Summary" />
      <div className="vitals-grid" style={{ marginBottom: 12 }}>
        <div className={`vital-mini ${riskResult ? '' : 'dimmed'}`}>
          <div className="vm-icon">📊</div>
          <div className="vm-label">XGBoost Risk</div>
          <div style={{ fontSize: '.9rem', fontWeight: 700, color: riskResult ? RISK_COLORS[riskResult.risk_tier] : 'var(--text3)' }}>
            {riskResult ? `${riskResult.risk_tier} (${Math.round(riskResult.risk_score * 100)}%)` : 'Not run'}
          </div>
          <div className="vm-trend" style={{ color: riskResult ? 'var(--green)' : 'var(--amber)' }}>
            {riskResult ? '✓ Complete' : '○ Pending'}
          </div>
        </div>
        <div className={`vital-mini ${imageResult ? '' : 'dimmed'}`}>
          <div className="vm-icon">🔬</div>
          <div className="vm-label">FastViT Image</div>
          <div style={{ fontSize: '.9rem', fontWeight: 700, color: imageResult ? SIPAKMED_CLASSES[imageResult.predicted_class]?.color : 'var(--text3)' }}>
            {imageResult ? imageResult.class_name : 'Not run'}
          </div>
          <div className="vm-trend" style={{ color: imageResult ? 'var(--green)' : 'var(--amber)' }}>
            {imageResult ? '✓ Complete' : '○ Pending'}
          </div>
        </div>
        <div className="vital-mini">
          <div className="vm-icon">🧬</div>
          <div className="vm-label">SHAP Chart</div>
          <div style={{ fontSize: '.85rem', fontWeight: 700, color: riskResult?.shap_chart ? 'var(--green)' : 'var(--text3)' }}>
            {riskResult?.shap_chart ? 'Available' : 'Pending'}
          </div>
        </div>
        <div className="vital-mini">
          <div className="vm-icon">🎯</div>
          <div className="vm-label">GradCAM</div>
          <div style={{ fontSize: '.85rem', fontWeight: 700, color: imageResult?.gradcam_image ? 'var(--green)' : 'var(--text3)' }}>
            {imageResult?.gradcam_image ? 'Available' : 'Pending'}
          </div>
        </div>
      </div>

      <div className="settings-section">
        <div className="settings-label">Report Details</div>
        <div className="settings-group">
          <div className="setting-row">
            <div className="setting-info"><p>Referring Physician</p><span>Optional</span></div>
            <input className="risk-input text-input" type="text" placeholder="Dr. Name"
              value={physician} onChange={e => setPhysician(e.target.value)} />
          </div>
          <div className="setting-row">
            <div className="setting-info"><p>Patient Age</p><span>If not from form</span></div>
            <input className="risk-input" type="number" min={13} max={90} placeholder="—"
              value={patientAge} onChange={e => setPatientAge(e.target.value)} />
          </div>
        </div>
      </div>

      {combined !== null && (
        <div className="score-card" style={{ marginBottom: 12 }}>
          <div className="score-circle" style={{ width: 60, height: 60, flexShrink: 0 }}>
            <svg viewBox="0 0 70 70" width="60" height="60">
              <circle fill="none" stroke="rgba(0,212,255,.1)" strokeWidth="6" cx="35" cy="35" r="29" />
              <circle fill="none" stroke="var(--cyan)" strokeWidth="6" cx="35" cy="35" r="29"
                strokeDasharray="182" strokeDashoffset={182 - combined * 182}
                strokeLinecap="round" transform="rotate(-90 35 35)" />
            </svg>
            <div className="score-num" style={{ fontSize: '.85rem' }}>{Math.round(combined * 100)}%</div>
          </div>
          <div className="score-info">
            <h3>Combined Risk Index</h3>
            <p>60% XGBoost clinical + 40% FastViT cytological</p>
          </div>
        </div>
      )}

      {!hasBoth && (
        <div className="alert-item">
          <div className="alert-icon">ℹ️</div>
          <div className="alert-text">
            <p>For a complete report, run both analyses</p>
            <span>Stage 1 (Risk Form) + Stage 2 (Image Upload) can be done independently</span>
          </div>
        </div>
      )}

      <button
        className="btn-full btn-cyan"
        onClick={handleGenerate}
        disabled={loading || (!imageResult && !riskResult)}
        style={{ marginTop: 12 }}
      >
        {loading ? '⏳ Generating PDF...' : '📥 Download PDF Report'}
      </button>

      {loading && <LoadingDots text="Building clinical report with SHAP + GradCAM..." />}

      <div className="disclaimer" style={{ marginTop: 12, borderRadius: 10, padding: 12 }}>
        ⚠️ This AI-generated report is for clinical decision support only. Must be reviewed by a
        qualified healthcare professional before any clinical decisions are made.
      </div>
      <div style={{ height: 16 }} />
    </div>
  )
}

// ─── VIEW: SETTINGS ───────────────────────────────────────────

function SettingsView({ showToast, onClearResults }) {
  const [apiUrl, setApiUrl] = useState(import.meta.env.VITE_API_URL || '')

  return (
    <div className="view active" id="view-settings">
      <div className="profile-head">
        <div className="profile-avatar">🏥</div>
        <div className="profile-details">
          <h3>CervicalAI v2.0</h3>
          <p>Explainable Two-Stage Detection</p>
          <span className="profile-badge">XGBoost + FastViT + SHAP + GradCAM</span>
        </div>
      </div>

      <div className="settings-section">
        <div className="settings-label">Backend Configuration</div>
        <div className="settings-group">
          <div className="api-input-wrap">
            <div style={{ fontSize: '.7rem', color: 'var(--text2)', marginBottom: 6 }}>
              API Base URL <span style={{ color: 'var(--text3)', fontSize: '.58rem' }}>
                (Render backend URL)
              </span>
            </div>
            <input className="api-input" type="text"
              placeholder="https://cervicalai-backend.onrender.com"
              value={apiUrl}
              onChange={e => setApiUrl(e.target.value)}
            />
            <button className="api-save-btn" onClick={() => showToast('ℹ️ Set VITE_API_URL in .env to persist')}>
              INFO: Set via VITE_API_URL env var
            </button>
          </div>
        </div>
      </div>

      <div className="settings-section">
        <div className="settings-label">Model Information</div>
        <div className="settings-group">
          <div className="setting-row">
            <div className="setting-icon">📊</div>
            <div className="setting-info"><p>Stage 1 Model</p><span>XGBoost + SHAP TreeExplainer</span></div>
            <div className="setting-val">v2.0</div>
          </div>
          <div className="setting-row">
            <div className="setting-icon">🔬</div>
            <div className="setting-info"><p>Stage 2 Model</p><span>FastViT-T8 (6M params)</span></div>
            <div className="setting-val">v2.0</div>
          </div>
          <div className="setting-row">
            <div className="setting-icon">📁</div>
            <div className="setting-info"><p>XGB Dataset</p><span>Kaggle: ranzeet013/cervical-cancer-dataset</span></div>
            <div className="setting-val">858 pts</div>
          </div>
          <div className="setting-row">
            <div className="setting-icon">🖼️</div>
            <div className="setting-info"><p>Image Dataset</p><span>Kaggle: SipakMed (prahladmehandiratta)</span></div>
            <div className="setting-val">4,049 imgs</div>
          </div>
        </div>
      </div>

      <div className="settings-section">
        <div className="settings-label">Data & Session</div>
        <div className="settings-group">
          <div className="setting-row" onClick={onClearResults} style={{ cursor: 'pointer' }}>
            <div className="setting-icon">🗑️</div>
            <div className="setting-info"><p>Clear Session Results</p><span>Remove current analysis data</span></div>
            <div className="setting-arrow">›</div>
          </div>
          <div className="setting-row">
            <div className="setting-icon">ℹ️</div>
            <div className="setting-info"><p>About</p><span>CervicalAI v2.0 · MIT License</span></div>
            <div className="setting-arrow">›</div>
          </div>
        </div>
      </div>

      <div className="disclaimer" style={{ borderRadius: 10, padding: 12, marginTop: 8 }}>
        ⚠️ This software is a research tool and NOT a certified medical device.
        Always consult a licensed physician for medical diagnosis and treatment decisions.
      </div>
      <div style={{ height: 16 }} />
    </div>
  )
}

// ─── ROOT APP ─────────────────────────────────────────────────

export default function App() {
  const [view, setView] = useState('dashboard')
  const [imageResult, setImageResult] = useState(null)
  const [riskResult, setRiskResult] = useState(null)
  const [toast, setToast] = useState({ message: '', visible: false })
  const [backendOk, setBackendOk] = useState(null)
  const scrollRef = useRef()
  const toastTimer = useRef()

  useEffect(() => {
    checkHealth()
      .then(() => setBackendOk(true))
      .catch(() => setBackendOk(false))
  }, [])

  const showToast = useCallback((msg) => {
    setToast({ message: msg, visible: true })
    clearTimeout(toastTimer.current)
    toastTimer.current = setTimeout(() => setToast(t => ({ ...t, visible: false })), 3500)
  }, [])

  const navigateTo = useCallback((v) => {
    setView(v)
    if (scrollRef.current) scrollRef.current.scrollTop = 0
  }, [])

  const clearResults = useCallback(() => {
    setImageResult(null)
    setRiskResult(null)
    showToast('✅ Session cleared')
  }, [showToast])

  return (
    <div className="app-viewport-container" style={{ width: '100%', maxWidth: '1280px', margin: '0 auto', padding: '0 10px' }}>
      <div className="bg-mesh" />
      <div className="grid-lines" />

      <StatusBar backendOk={backendOk} />

      <div className="app-scroll" ref={scrollRef}>
        {view === 'dashboard' && (
          <DashboardView imageResult={imageResult} riskResult={riskResult} onNav={navigateTo} />
        )}
        {view === 'image' && (
          <ImageView result={imageResult} onResult={setImageResult} showToast={showToast} />
        )}
        {view === 'risk' && (
          <RiskView result={riskResult} onResult={setRiskResult} showToast={showToast} />
        )}
        {view === 'report' && (
          <ReportView imageResult={imageResult} riskResult={riskResult} showToast={showToast} />
        )}
        {view === 'settings' && (
          <SettingsView showToast={showToast} onClearResults={clearResults} />
        )}
      </div>

      <BottomNav
        current={view}
        onNav={navigateTo}
        hasImageResult={!!imageResult}
        hasRiskResult={!!riskResult}
      />

      <Toast message={toast.message} visible={toast.visible} />
    </div>
  )
}