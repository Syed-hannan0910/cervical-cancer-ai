/**
 * CervicalAI API Service
 * Handles all communication with the FastAPI backend
 */

import axios from 'axios'

// Base URL: use env variable in production, proxy in dev
const API_BASE = import.meta.env.VITE_API_URL || ''

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000, // 2 min timeout for model inference
})

// Request interceptor for logging
api.interceptors.request.use(config => {
  console.debug(`[API] ${config.method?.toUpperCase()} ${config.url}`)
  return config
})

// Response interceptor for error normalization
api.interceptors.response.use(
  res => res,
  err => {
    const msg = err.response?.data?.detail || err.message || 'Network error'
    const status = err.response?.status || 0
    console.error(`[API Error ${status}]:`, msg)
    return Promise.reject({ message: msg, status })
  }
)

// ─── Health Check ─────────────────────────────────────────────
export const checkHealth = () => api.get('/api/health')

// ─── Stage 2: Image Classification ───────────────────────────
export const classifyImage = (file) => {
  const formData = new FormData()
  formData.append('file', file)
  return api.post('/api/predict/image', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      const pct = Math.round((e.loaded * 100) / e.total)
      console.debug(`Upload: ${pct}%`)
    }
  })
}

// ─── Stage 1: Clinical Risk Assessment ───────────────────────
export const assessRisk = (riskData) =>
  api.post('/api/risk/assess', riskData)

// ─── PDF Report Generation ────────────────────────────────────
export const generateReport = async (reportData) => {
  const response = await api.post('/api/report/generate', reportData, {
    responseType: 'blob',
  })
  // Create download link
  const url = window.URL.createObjectURL(new Blob([response.data], { type: 'application/pdf' }))
  const a = document.createElement('a')
  a.href = url
  a.download = `CervicalAI_Report_${new Date().toISOString().slice(0, 10)}.pdf`
  document.body.appendChild(a)
  a.click()
  window.URL.revokeObjectURL(url)
  document.body.removeChild(a)
}

export const previewReport = (reportData) =>
  api.post('/api/report/preview', reportData)

export default api
