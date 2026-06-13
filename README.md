# CervicalAI — Explainable Two-Stage Cervical Cancer Detection Framework

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react)](https://react.dev)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture Overview

```
CervicalAI
├── Stage 1 — XGBoost Clinical Risk Stratification
│   ├── Dataset: Kaggle ranzeet013/cervical-cancer-dataset (858 patients, 36 features)
│   ├── Model: XGBoost + SHAP TreeExplainer
│   └── Output: Risk score (0–1), tier, SHAP waterfall chart
│
└── Stage 2 — FastViT Cytological Image Classification
    ├── Dataset: Kaggle SipakMed (4,049 pap smear images, 5 classes)
    ├── Model: FastViT-T8 (6M params) + GradCAM
    └── Output: Cell type, confidence, GradCAM saliency map
```

**Combined Risk Index** = 0.6 × XGBoost score + 0.4 × FastViT image risk

### SipakMed Cell Categories

| ID | Class | Risk |
|----|-------|------|
| 0 | Dyskeratotic | 🔴 High |
| 1 | Koilocytotic (HPV-infected) | 🔴 High |
| 2 | Metaplastic | 🟡 Medium |
| 3 | Parabasal | 🟡 Medium |
| 4 | Superficial-Intermediate (Normal) | 🟢 Low |

---

## Project Structure

```
cervical-cancer-ai/
├── backend/                          # FastAPI backend (deploy on Render)
│   ├── app/
│   │   ├── main.py                   # FastAPI app + CORS + lifespan
│   │   ├── models/
│   │   │   ├── fastvit_model.py      # FastViT-T8 + GradCAM
│   │   │   └── xgboost_model.py      # XGBoost + SHAP
│   │   ├── routers/
│   │   │   ├── predict.py            # POST /api/predict/image
│   │   │   ├── risk.py               # POST /api/risk/assess
│   │   │   ├── report.py             # POST /api/report/generate
│   │   │   └── health.py             # GET /api/health
│   │   └── tests/
│   │       └── test_api.py           # Full pytest test suite
│   ├── data/                         # Downloaded Kaggle datasets
│   ├── models/                       # Trained model weights (.pt, .pkl)
│   ├── static/                       # Served static files
│   ├── train_models.py               # Complete training pipeline
│   └── requirements.txt
│
├── frontend/                         # React + Vite frontend (deploy on Vercel)
│   ├── src/
│   │   ├── App.jsx                   # Main React app (all views)
│   │   ├── api.js                    # Axios API service layer
│   │   ├── styles.css                # Complete design system CSS
│   │   └── main.jsx                  # React DOM entry point
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   └── vercel.json                   # Vercel deployment config
│
├── render.yaml                       # Render deployment config
└── README.md
```

---

## Quick Start (Local Development)

### Prerequisites
- Python 3.10+
- Node.js 18+
- 8GB RAM recommended (for model training)

### 1. Clone & Setup Backend

```bash
git clone https://github.com/yourusername/cervical-cancer-ai.git
cd cervical-cancer-ai/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate.bat     # Windows

# Install dependencies
pip install -r requirements.txt

# Create required directories
mkdir -p data models static/reports static/gradcam
```

### 2. Train Models (Optional — Skip to use demo mode)

#### Step 2a: Set Up Kaggle API

1. Go to [kaggle.com](https://www.kaggle.com) → Account → API → **Create New Token**
2. Download `kaggle.json` → place at `~/.kaggle/kaggle.json`
3. Run: `chmod 600 ~/.kaggle/kaggle.json`

#### Step 2b: Download & Train

```bash
# Download both datasets and train both models
python train_models.py --stage all --download --epochs-xgb 500 --epochs-vit 50

# OR: Train individually
python train_models.py --stage xgboost --download   # Stage 1 only (~2 min)
python train_models.py --stage fastvit --download   # Stage 2 only (~2h CPU / 20min GPU)

# Check training metrics
cat models/training_metrics.json
```

> **Note:** Without trained weights, the app runs in **demo mode** using a randomly initialized FastViT and rule-based clinical scoring. This demonstrates the full UI and API flow.

#### Expected Training Performance (after full training)

| Model | Metric | Expected Value |
|-------|--------|----------------|
| XGBoost | AUC-ROC | ~0.88–0.93 |
| XGBoost | F1-Score | ~0.72–0.82 |
| FastViT-T8 | Accuracy | ~0.90–0.94 |
| FastViT-T8 | Macro F1 | ~0.88–0.92 |

### 3. Run Backend

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs  
Health check: http://localhost:8000/api/health

### 4. Setup & Run Frontend

```bash
cd frontend

# Install dependencies
npm install

# Create .env.local
echo "VITE_API_URL=http://localhost:8000" > .env.local

# Start dev server
npm run dev
```

Frontend: http://localhost:5173

---

## Deployment Guide

### Backend → Render

#### Option A: Via render.yaml (Recommended)

1. Push code to GitHub
2. Go to [render.com](https://render.com) → New → **Blueprint**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` and creates the service

#### Option B: Manual Setup

1. Render Dashboard → **New Web Service**
2. Connect GitHub repo → Select `backend/` as root directory
3. Configure:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free (or Starter for always-on)

4. Add Environment Variables:
   ```
   PYTHONUNBUFFERED=1
   ALLOWED_ORIGINS=https://your-app.vercel.app,http://localhost:5173
   MODEL_DIR=/opt/render/project/src/models
   ```

5. Deploy → Copy your backend URL: `https://cervicalai-backend.onrender.com`

> **Free Tier Note:** Render free tier spins down after 15 minutes of inactivity. First request after sleep takes ~30s. Upgrade to Starter ($7/mo) for always-on.

#### Uploading Trained Models to Render

Option 1 — **Commit models to git** (simplest, <100MB):
```bash
git add backend/models/
git commit -m "Add trained model weights"
git push
```

Option 2 — **Render Persistent Disk** (paid plans, uncomment in render.yaml):
```yaml
disk:
  name: cervicalai-models
  mountPath: /models
  sizeGB: 5
```

Option 3 — **Download on startup** (add to `main.py` lifespan):
```python
import gdown  # or use requests to fetch from Google Drive / S3
gdown.download("YOUR_GDRIVE_LINK", "models/fastvit_sipakmed.pt")
```

---

### Frontend → Vercel

1. Go to [vercel.com](https://vercel.com) → **New Project**
2. Import your GitHub repo
3. Configure:
   - **Framework Preset:** Vite
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`

4. Add Environment Variable:
   ```
   VITE_API_URL = https://cervicalai-backend.onrender.com
   ```
   *(Use your actual Render backend URL)*

5. Click **Deploy**

6. After deploy, copy your Vercel URL and add it to Render's `ALLOWED_ORIGINS`

---

## API Reference

### Image Classification
```http
POST /api/predict/image
Content-Type: multipart/form-data

file: <pap smear image JPG/PNG/TIFF>
```
Returns: predicted class, confidence, GradCAM heatmap (base64), class probabilities

### Clinical Risk Assessment
```http
POST /api/risk/assess
Content-Type: application/json

{
  "age": 35,
  "num_sexual_partners": 3,
  "stds_hpv": 1,
  "dx_hpv": 1,
  ... (20 clinical features)
}
```
Returns: risk score, tier, SHAP values, SHAP waterfall chart (base64), recommendation

### PDF Report
```http
POST /api/report/generate
Content-Type: application/json

{
  "patient_name": "...",
  "risk_score": 0.72,
  "risk_tier": "High",
  "gradcam_image": "data:image/png;base64,...",
  "shap_chart": "data:image/png;base64,...",
  ...
}
```
Returns: PDF binary stream (downloads automatically)

Full interactive docs: `http://your-backend/docs`

---

## Running Tests

```bash
cd backend
pip install pytest pytest-asyncio httpx

# Run all tests
pytest app/tests/test_api.py -v

# Run specific test class
pytest app/tests/test_api.py::TestImageClassification -v
pytest app/tests/test_api.py::TestRiskAssessment -v
pytest app/tests/test_api.py::TestReport -v

# With coverage
pip install pytest-cov
pytest app/tests/ --cov=app --cov-report=html
```

Expected output (demo mode without trained weights):
```
PASSED  test_health_endpoint
PASSED  test_classify_image_success
PASSED  test_risk_assessment_success
PASSED  test_low_risk_patient
PASSED  test_shap_values_structure
PASSED  test_generate_report_minimal
...
```

---

## Explainability

### SHAP (SHapley Additive exPlanations)
- **Algorithm:** TreeExplainer (exact Shapley values for tree-based models)
- **Output:** Per-feature contribution values + waterfall chart
- **Key features:** HPV diagnosis, STD history, smoking, age at first intercourse
- **Reference:** Lundberg & Lee, 2017 — "A Unified Approach to Interpreting Model Predictions"

### GradCAM (Gradient-weighted Class Activation Mapping)
- **Target layer:** Last depthwise conv in FastViT Stage 4
- **Output:** Heatmap overlaid on original pap smear image
- **Highlights:** Abnormal nuclear morphology, koilocytic halos, dyskeratotic regions
- **Reference:** Selvaraju et al., 2017 — "Grad-CAM: Visual Explanations from Deep Networks"

---

## Datasets

| Dataset | Source | Size | Use |
|---------|--------|------|-----|
| Cervical Cancer Risk Factors | [Kaggle: ranzeet013](https://www.kaggle.com/datasets/ranzeet013/cervical-cancer-dataset) | 858 patients, 36 features | XGBoost training |
| SipakMed Cytological Images | [Kaggle: prahladmehandiratta](https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed) | 4,049 images, 5 classes | FastViT training |

---

## ⚠️ Medical Disclaimer

**CervicalAI is a RESEARCH TOOL and NOT a certified medical device.**

This software is intended solely for academic research and as a clinical decision support demonstration. It does NOT constitute a medical diagnosis. Any clinical application requires:

1. Review and validation by a licensed pathologist or gynecologist
2. Regulatory approval (FDA 510(k), CE marking, etc.)
3. Clinical validation on prospective patient populations
4. Integration with comprehensive clinical workflows

**In case of medical emergency, contact emergency services immediately.**

---

## License

MIT License — See [LICENSE](LICENSE) for details.

## Citation

If you use CervicalAI in research:
```bibtex
@software{cervicalai2024,
  title={CervicalAI: Explainable Two-Stage Cervical Cancer Detection},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/cervical-cancer-ai}
}
```
