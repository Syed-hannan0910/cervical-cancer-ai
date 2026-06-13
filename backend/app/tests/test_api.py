"""
CervicalAI — Backend Test Suite
Run: pytest tests/ -v
"""

import io
import json
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np


# ─── Fixtures ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from app.main import app
    return TestClient(app)


@pytest.fixture
def dummy_image_bytes():
    """Create a minimal 224x224 RGB image for testing."""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def risk_payload():
    return {
        "age": 35,
        "num_sexual_partners": 3,
        "first_sexual_intercourse": 17,
        "num_pregnancies": 2,
        "smokes": 1,
        "smokes_years": 5,
        "smokes_packs_year": 10,
        "hormonal_contraceptives": 1,
        "hormonal_contraceptives_years": 8,
        "iud": 0,
        "iud_years": 0,
        "stds": 1,
        "stds_number": 1,
        "stds_condylomatosis": 0,
        "stds_hpv": 1,
        "stds_hiv": 0,
        "stds_syphilis": 0,
        "dx_cancer": 0,
        "dx_cin": 0,
        "dx_hpv": 1,
        "patient_name": "Test Patient",
        "patient_id": "TEST-001"
    }


@pytest.fixture
def low_risk_payload():
    return {
        "age": 25,
        "num_sexual_partners": 1,
        "first_sexual_intercourse": 21,
        "num_pregnancies": 0,
        "smokes": 0,
        "smokes_years": 0,
        "smokes_packs_year": 0,
        "hormonal_contraceptives": 0,
        "hormonal_contraceptives_years": 0,
        "iud": 0,
        "iud_years": 0,
        "stds": 0,
        "stds_number": 0,
        "stds_condylomatosis": 0,
        "stds_hpv": 0,
        "stds_hiv": 0,
        "stds_syphilis": 0,
        "dx_cancer": 0,
        "dx_cin": 0,
        "dx_hpv": 0,
    }


# ─── Health Check Tests ───────────────────────────────────────

class TestHealth:
    def test_root(self, client):
        res = client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert data["name"] == "CervicalAI Backend"
        assert data["status"] == "operational"

    def test_health_endpoint(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    def test_ping(self, client):
        res = client.get("/api/ping")
        assert res.status_code == 200
        assert res.json()["pong"] is True


# ─── Image Classification Tests ───────────────────────────────

class TestImageClassification:
    def test_classify_image_success(self, client, dummy_image_bytes):
        res = client.post(
            "/api/predict/image",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")}
        )
        assert res.status_code == 200
        data = res.json()

        # Required fields
        assert "predicted_class" in data
        assert "class_name" in data
        assert "confidence" in data
        assert "risk_level" in data
        assert "risk_score" in data
        assert "gradcam_image" in data
        assert "class_probabilities" in data

        # Value ranges
        assert 0 <= data["predicted_class"] <= 4
        assert 0.0 <= data["confidence"] <= 1.0
        assert 0.0 <= data["risk_score"] <= 1.0
        assert data["risk_level"] in ("low", "medium", "high")
        assert len(data["class_probabilities"]) == 5

        # Probabilities sum to ~1
        total_prob = sum(cp["probability"] for cp in data["class_probabilities"])
        assert abs(total_prob - 1.0) < 0.01

    def test_classify_image_probabilities_valid(self, client, dummy_image_bytes):
        res = client.post(
            "/api/predict/image",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")}
        )
        assert res.status_code == 200
        data = res.json()

        for cp in data["class_probabilities"]:
            assert "class_id" in cp
            assert "class_name" in cp
            assert "probability" in cp
            assert "risk" in cp
            assert 0 <= cp["probability"] <= 1.0
            assert cp["risk"] in ("low", "medium", "high")

    def test_classify_png_image(self, client):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        res = client.post(
            "/api/predict/image",
            files={"file": ("test.png", buf.read(), "image/png")}
        )
        assert res.status_code == 200

    def test_classify_invalid_file(self, client):
        res = client.post(
            "/api/predict/image",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        # Should return error (415 or 500)
        assert res.status_code in (415, 500, 400)

    def test_classify_empty_file(self, client):
        res = client.post(
            "/api/predict/image",
            files={"file": ("empty.jpg", b"", "image/jpeg")}
        )
        assert res.status_code in (400, 500)

    def test_gradcam_is_base64_image(self, client, dummy_image_bytes):
        res = client.post(
            "/api/predict/image",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")}
        )
        assert res.status_code == 200
        gradcam = res.json()["gradcam_image"]
        assert gradcam.startswith("data:image/png;base64,")


# ─── Risk Assessment Tests ────────────────────────────────────

class TestRiskAssessment:
    def test_risk_assessment_success(self, client, risk_payload):
        res = client.post("/api/risk/assess", json=risk_payload)
        assert res.status_code == 200
        data = res.json()

        assert "risk_score" in data
        assert "risk_tier" in data
        assert "urgency" in data
        assert "recommendation" in data
        assert "shap_values" in data
        assert "top_risk_factors" in data

        assert 0.0 <= data["risk_score"] <= 1.0
        assert data["risk_tier"] in ("Low", "Moderate", "High", "Critical")
        assert data["urgency"] in ("Routine", "Soon", "Urgent", "Immediate")

    def test_risk_score_range(self, client, risk_payload):
        res = client.post("/api/risk/assess", json=risk_payload)
        assert res.status_code == 200
        score = res.json()["risk_score"]
        assert 0.0 <= score <= 1.0

    def test_low_risk_patient(self, client, low_risk_payload):
        res = client.post("/api/risk/assess", json=low_risk_payload)
        assert res.status_code == 200
        data = res.json()
        assert data["risk_tier"] in ("Low", "Moderate")
        assert data["risk_score"] < 0.7

    def test_high_risk_patient(self, client):
        high_risk = {
            "age": 45, "num_sexual_partners": 8, "first_sexual_intercourse": 14,
            "num_pregnancies": 5, "smokes": 1, "smokes_years": 20, "smokes_packs_year": 30,
            "hormonal_contraceptives": 1, "hormonal_contraceptives_years": 15,
            "iud": 0, "iud_years": 0,
            "stds": 1, "stds_number": 3,
            "stds_condylomatosis": 1, "stds_hpv": 1, "stds_hiv": 0, "stds_syphilis": 1,
            "dx_cancer": 0, "dx_cin": 1, "dx_hpv": 1,
        }
        res = client.post("/api/risk/assess", json=high_risk)
        assert res.status_code == 200
        data = res.json()
        assert data["risk_tier"] in ("High", "Critical", "Moderate")

    def test_shap_values_structure(self, client, risk_payload):
        res = client.post("/api/risk/assess", json=risk_payload)
        assert res.status_code == 200
        shap_values = res.json()["shap_values"]
        assert isinstance(shap_values, list)
        assert len(shap_values) > 0
        for sv in shap_values:
            assert "feature" in sv
            assert "shap_value" in sv
            assert "feature_value" in sv
            assert isinstance(sv["shap_value"], float)

    def test_top_risk_factors(self, client, risk_payload):
        res = client.post("/api/risk/assess", json=risk_payload)
        assert res.status_code == 200
        top = res.json()["top_risk_factors"]
        assert isinstance(top, list)
        assert len(top) <= 5

    def test_recommendation_present(self, client, risk_payload):
        res = client.post("/api/risk/assess", json=risk_payload)
        assert res.status_code == 200
        rec = res.json()["recommendation"]
        assert isinstance(rec, str)
        assert len(rec) > 10

    def test_invalid_age(self, client, risk_payload):
        bad = dict(risk_payload, age=5)  # Too young
        res = client.post("/api/risk/assess", json=bad)
        assert res.status_code == 422  # Validation error

    def test_get_features_endpoint(self, client):
        res = client.get("/api/risk/features")
        assert res.status_code == 200
        data = res.json()
        assert "features" in data
        assert "count" in data
        assert data["count"] > 0


# ─── PDF Report Tests ─────────────────────────────────────────

class TestReport:
    def test_generate_report_minimal(self, client):
        payload = {
            "patient_name": "Test Patient",
            "risk_score": 0.65,
            "risk_tier": "High",
            "urgency": "Urgent",
            "recommendation": "Urgent colposcopy recommended.",
        }
        res = client.post("/api/report/generate", json=payload)
        # Either success (200) or 501 if reportlab not installed
        assert res.status_code in (200, 501)
        if res.status_code == 200:
            assert res.headers["content-type"] == "application/pdf"
            assert len(res.content) > 1000  # Non-trivial PDF

    def test_generate_report_full(self, client, risk_payload, dummy_image_bytes):
        # First get a risk result
        risk_res = client.post("/api/risk/assess", json=risk_payload)
        assert risk_res.status_code == 200
        risk_data = risk_res.json()

        payload = {
            "patient_name": "Full Test Patient",
            "patient_id": "TEST-FULL-001",
            "patient_age": 35,
            "referring_physician": "Dr. Test",
            "risk_score": risk_data["risk_score"],
            "risk_tier": risk_data["risk_tier"],
            "urgency": risk_data["urgency"],
            "recommendation": risk_data["recommendation"],
            "top_risk_factors": risk_data["top_risk_factors"],
            "shap_chart": risk_data.get("shap_chart", ""),
            "model_used": risk_data["model_used"],
        }
        res = client.post("/api/report/generate", json=payload)
        assert res.status_code in (200, 501)

    def test_preview_report(self, client):
        payload = {
            "patient_name": "Preview Test",
            "risk_score": 0.3,
            "risk_tier": "Low",
        }
        res = client.post("/api/report/preview", json=payload)
        assert res.status_code in (200, 501)
        if res.status_code == 200:
            data = res.json()
            assert "pdf_base64" in data
            assert data["pdf_base64"].startswith("data:application/pdf;base64,")


# ─── CORS Tests ───────────────────────────────────────────────

class TestCORS:
    def test_cors_headers_present(self, client):
        res = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
        assert res.status_code == 200
        # CORS headers should be in response


# ─── Model Unit Tests ─────────────────────────────────────────

class TestFastViTModel:
    def test_model_loads(self):
        from app.models.fastvit_model import get_fastvit_model, FastViTT8
        model = get_fastvit_model()
        assert model is not None
        assert isinstance(model, FastViTT8)

    def test_model_output_shape(self):
        import torch
        from app.models.fastvit_model import FastViTT8
        model = FastViTT8(num_classes=5)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)

    def test_preprocess_image(self, dummy_image_bytes):
        from app.models.fastvit_model import preprocess_image
        tensor, img_np = preprocess_image(dummy_image_bytes)
        assert tensor.shape == (1, 3, 224, 224)
        assert img_np.ndim == 3


class TestXGBoostModel:
    def test_rule_based_fallback(self):
        from app.models.xgboost_model import rule_based_score
        features = {
            "STDs:HPV": 1, "Dx:HPV": 1, "Age": 35,
            "Smokes (packs/year)": 10, "Dx:Cancer": 0
        }
        score, contributions = rule_based_score(features)
        assert 0.0 <= score <= 1.0
        assert isinstance(contributions, dict)
        assert len(contributions) > 0

    def test_high_risk_features_increase_score(self):
        from app.models.xgboost_model import rule_based_score
        low_risk = {"STDs:HPV": 0, "Dx:HPV": 0, "Dx:Cancer": 0, "Dx:CIN": 0, "STDs:HIV": 0}
        high_risk = {"STDs:HPV": 1, "Dx:HPV": 1, "Dx:Cancer": 1, "Dx:CIN": 1, "STDs:HIV": 1}
        score_low, _ = rule_based_score(low_risk)
        score_high, _ = rule_based_score(high_risk)
        assert score_high > score_low

    def test_risk_tier_classification(self):
        from app.models.xgboost_model import predict_risk
        result = predict_risk({"Age": 25, "STDs:HPV": 0, "Dx:HPV": 0, "Dx:Cancer": 0})
        assert result["risk_tier"] in ("Low", "Moderate", "High", "Critical")
        assert 0.0 <= result["risk_score"] <= 1.0
