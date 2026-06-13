"""
API Router: PDF Report Generation
POST /api/report/generate

Generates a comprehensive clinical PDF report combining:
- Patient demographics
- XGBoost risk assessment + SHAP chart
- FastViT image classification + GradCAM
- Clinical recommendations
"""

import io
import os
import base64
import logging
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

logger = logging.getLogger("CervicalAI.Router.Report")

router = APIRouter()

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage, PageBreak, KeepTogether
    )
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.pdfgen import canvas as rl_canvas
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    logger.warning("ReportLab not installed — PDF generation disabled.")


# ─── Color Palette (matching frontend dark theme) ───────────────
DARK_BG = colors.HexColor("#050a14")
NAVY = colors.HexColor("#0a1220")
NAVY2 = colors.HexColor("#0f1a2e")
CYAN = colors.HexColor("#00d4ff")
CYAN_DIM = colors.HexColor("#003a44")
AMBER = colors.HexColor("#ffaa00")
RED = colors.HexColor("#ff2d55")
GREEN = colors.HexColor("#00e676")
PURPLE = colors.HexColor("#b388ff")
TEXT = colors.HexColor("#e8f0fe")
TEXT2 = colors.HexColor("#8fa8d0")
TEXT3 = colors.HexColor("#445a7a")
BORDER = colors.HexColor("#162040")

RISK_COLORS = {
    "Low": colors.HexColor("#00e676"),
    "Moderate": colors.HexColor("#ffaa00"),
    "High": colors.HexColor("#ff6b35"),
    "Critical": colors.HexColor("#ff2d55"),
}
RISK_BG = {
    "Low": colors.HexColor("#002a18"),
    "Moderate": colors.HexColor("#2a1a00"),
    "High": colors.HexColor("#2a0a00"),
    "Critical": colors.HexColor("#2a0010"),
}
IMAGE_RISK_COLORS = {
    "low": colors.HexColor("#00e676"),
    "medium": colors.HexColor("#ffaa00"),
    "high": colors.HexColor("#ff2d55"),
}


class ReportRequest(BaseModel):
    # Patient info
    patient_name: Optional[str] = Field("Anonymous Patient", description="Patient full name")
    patient_id: Optional[str] = Field(None, description="Patient ID")
    patient_age: Optional[float] = Field(None, description="Patient age")
    referring_physician: Optional[str] = Field("N/A", description="Referring physician name")
    report_date: Optional[str] = Field(None, description="Report date (ISO format)")

    # Stage 1 — XGBoost results
    risk_score: Optional[float] = Field(None, description="XGBoost risk score 0–1")
    risk_tier: Optional[str] = Field(None, description="Low/Moderate/High/Critical")
    urgency: Optional[str] = Field(None)
    recommendation: Optional[str] = Field(None)
    top_risk_factors: Optional[List[Dict]] = Field(None)
    shap_chart: Optional[str] = Field(None, description="Base64 SHAP chart PNG")
    model_used: Optional[str] = Field("XGBoost + SHAP")
    input_features: Optional[Dict] = Field(None)

    # Stage 2 — FastViT results
    image_class_name: Optional[str] = Field(None)
    image_risk_level: Optional[str] = Field(None)
    image_confidence: Optional[float] = Field(None)
    image_description: Optional[str] = Field(None)
    class_probabilities: Optional[List[Dict]] = Field(None)
    gradcam_image: Optional[str] = Field(None, description="Base64 GradCAM PNG")

    # Combined
    combined_risk: Optional[float] = Field(None)

    class Config:
        json_schema_extra = {
            "example": {
                "patient_name": "Jane Doe",
                "patient_id": "PT-20481",
                "patient_age": 35,
                "risk_score": 0.72,
                "risk_tier": "High",
                "image_class_name": "Koilocytotic",
                "image_confidence": 0.89,
            }
        }


# ─── PDF Page Template ─────────────────────────────────────────

def _make_header_footer(canvas_obj, doc, patient_name, report_id):
    """Draw header and footer on every page with explicit logo safety handling."""
    w, h = A4
    canvas_obj.saveState()

    # Header background
    canvas_obj.setFillColor(NAVY)
    canvas_obj.rect(0, h - 28*mm, w, 28*mm, fill=1, stroke=0)

    # Header accent line
    canvas_obj.setFillColor(CYAN)
    canvas_obj.rect(0, h - 29*mm, w, 1*mm, fill=1, stroke=0)

    # --- Integrated Logo Positioning Setup ---
    logo_path = "logo.jpg"
    text_x_position = 14 * mm  # Default left alignment position

    if os.path.exists(logo_path):
        try:
            # Place branding logo cleanly within the top-left layout matrix
            canvas_obj.drawImage(logo_path, 14 * mm, h - 23 * mm, width=18 * mm, height=18 * mm, 
                                 preserveAspectRatio=True, mask='auto')
            text_x_position = 36 * mm  # Push text forward to remove clipping risks
        except Exception as e:
            logger.warning(f"Header branding image failed to render: {e}")

    # Main Branding Title Header
    canvas_obj.setFillColor(CYAN)
    canvas_obj.setFont("Times-Bold", 15)
    canvas_obj.drawString(text_x_position, h - 11 * mm, "CERVICAL CANCER RISK ASSESSMENT")

    # Institutional Metadata
    canvas_obj.setFillColor(TEXT)
    canvas_obj.setFont("Times-Bold", 9.5)
    canvas_obj.drawString(text_x_position, h - 16 * mm, "SCHOOL OF ENGINEERING (SOE) | ACADEMIC PROJECT")

    # Running Timestamp Output
    canvas_obj.setFillColor(TEXT2)
    canvas_obj.setFont("Times-Roman", 8.5)
    canvas_obj.drawString(text_x_position, h - 21 * mm, f"Authenticated Report: {datetime.now().strftime('%B %d, %Y | %H:%M')}")

    # Right side header contextual tracking tags
    canvas_obj.setFillColor(TEXT2)
    canvas_obj.setFont("Times-Roman", 8.5)
    canvas_obj.drawRightString(w - 14*mm, h - 14*mm, f"Patient: {patient_name}")
    canvas_obj.drawRightString(w - 14*mm, h - 19*mm, f"Report ID: {report_id}")

    # Footer
    canvas_obj.setFillColor(NAVY)
    canvas_obj.rect(0, 0, w, 14*mm, fill=1, stroke=0)
    canvas_obj.setFillColor(CYAN)
    canvas_obj.rect(0, 14*mm, w, 0.5*mm, fill=1, stroke=0)

    canvas_obj.setFillColor(TEXT3)
    canvas_obj.setFont("Times-Roman", 8)
    canvas_obj.drawString(14*mm, 8*mm,
        "⚠️ CLINICAL AI ASSISTANCE TOOL — This report is generated by AI and must be reviewed by a qualified healthcare professional.")
    canvas_obj.drawRightString(w - 14*mm, 8*mm, f"Page {doc.page}")

    canvas_obj.restoreState()


def _base64_to_image(b64_str: str, max_w: float, max_h: float):
    """Convert base64 image string to ReportLab Image flowable."""
    if not b64_str:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        buf = io.BytesIO(img_bytes)
        img = RLImage(buf, width=max_w, height=max_h, kind="proportional")
        return img
    except Exception as e:
        logger.warning(f"Image conversion failed: {e}")
        return None


def _section_header(title: str, color=colors.black) -> List:
    """Return flowables for a styled section header."""
    return [
        Spacer(1, 6*mm),
        HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=2*mm),
        Paragraph(f'<font color="#{color.hexval()[2:]}">{title}</font>',
                  ParagraphStyle("SecHead", fontName="Times-Bold", fontSize=14,
                                 textColor=color, spaceAfter=3*mm)),
    ]


def _risk_badge(risk_tier: str, risk_score: float) -> Table:
    """Generate a styled risk indicator table."""
    rc = RISK_COLORS.get(risk_tier, AMBER)
    rb = RISK_BG.get(risk_tier, NAVY2)
    pct = int(risk_score * 100)

    data = [[
        Paragraph(f'<font color="#{rc.hexval()[2:]}" size="22"><b>{pct}%</b></font>',
                  ParagraphStyle("RScore", fontName="Times-Bold", fontSize=22,
                                 textColor=rc, alignment=TA_CENTER)),
        Paragraph(f'<font color="#{rc.hexval()[2:]}" size="18"><b>{risk_tier.upper()} RISK</b></font>',
                  ParagraphStyle("RTier", fontName="Times-Bold", fontSize=18,
                                 textColor=rc, alignment=TA_CENTER)),
    ]]
    t = Table(data, colWidths=[50*mm, 100*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), rb),
        ("ROUNDEDCORNERS", [8]),
        ("BOX", (0, 0), (-1, -1), 1.5, rc),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


# ─── Main Report Builder ──────────────────────────────────────

def build_pdf(req: ReportRequest) -> bytes:
    """Construct and return the PDF bytes."""
    if not HAS_REPORTLAB:
        raise RuntimeError("ReportLab is not installed. Cannot generate PDF.")

    buf = io.BytesIO()
    report_id = f"CAI-{uuid.uuid4().hex[:8].upper()}"
    patient_name = req.patient_name or "Anonymous Patient"

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=14*mm, rightMargin=14*mm,
        topMargin=32*mm, bottomMargin=20*mm,
        title=f"CervicalAI Report — {patient_name}",
        author="CervicalAI v2.0",
        subject="Cervical Cancer Risk Assessment"
    )

    styles = getSampleStyleSheet()
    
    # Text styles used directly on the transparent/white canvas base
    body_style = ParagraphStyle("Body", fontName="Times-Roman", fontSize=11,
                                textColor=colors.black, leading=15, spaceAfter=3*mm)
    label_style = ParagraphStyle("Label", fontName="Times-Bold", fontSize=10.5,
                                  textColor=colors.black, spaceAfter=2*mm)
    value_style = ParagraphStyle("Value", fontName="Times-Roman", fontSize=11,
                                  textColor=colors.black, spaceAfter=4*mm)
    mono_style = ParagraphStyle("Mono", fontName="Times-Roman", fontSize=10,
                                 textColor=colors.black, spaceAfter=2*mm)

    # Contextual styles preserved specifically for high visibility inside navy tables
    label_style_table = ParagraphStyle("LabelTable", fontName="Times-Bold", fontSize=10.5,
                                        textColor=TEXT2, spaceAfter=2*mm)
    value_style_table = ParagraphStyle("ValueTable", fontName="Times-Roman", fontSize=11,
                                        textColor=TEXT, spaceAfter=4*mm)
    body_style_table = ParagraphStyle("BodyTable", fontName="Times-Roman", fontSize=11,
                                       textColor=TEXT, leading=15, spaceAfter=3*mm)

    story = []

    # ── COVER / PATIENT SUMMARY ──────────────────────────────
    story.append(Paragraph("CERVICAL CANCER RISK ASSESSMENT REPORT",
                            ParagraphStyle("Title", fontName="Times-Bold", fontSize=20,
                                           textColor=colors.black, alignment=TA_CENTER, spaceAfter=4*mm)))
    story.append(Paragraph("Explainable Two-Stage AI Detection Framework",
                            ParagraphStyle("Sub", fontName="Times-Roman", fontSize=12,
                                           textColor=colors.black, alignment=TA_CENTER, spaceAfter=6*mm)))

    # Enhanced Patient info table incorporating missing structural properties
    report_date = req.report_date or datetime.now().strftime("%B %d, %Y")
    pid = req.patient_id or report_id

    patient_data = [
        [Paragraph("<b>Patient Name</b>", label_style_table), Paragraph(patient_name, value_style_table),
         Paragraph("<b>Patient ID</b>", label_style_table), Paragraph(pid, value_style_table)],
        [Paragraph("<b>Age</b>", label_style_table), Paragraph(str(req.patient_age or "N/A"), value_style_table),
         Paragraph("<b>Report Date</b>", label_style_table), Paragraph(report_date, value_style_table)],
        [Paragraph("<b>Analysis Mode</b>", label_style_table), Paragraph("Dual-Stage AI", value_style_table),
         Paragraph("<b>Supervision</b>", label_style_table), Paragraph("SOE Institutional", value_style_table)],
        [Paragraph("<b>Referring Physician</b>", label_style_table), Paragraph(req.referring_physician or "N/A", value_style_table),
         Paragraph("<b>Report ID</b>", label_style_table), Paragraph(report_id, value_style_table)],
    ]
    pt = Table(patient_data, colWidths=[38*mm, 50*mm, 38*mm, 50*mm])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("BOX", (0, 0), (-1, -1), 1, BORDER),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(pt)

    # ── EXECUTIVE SUMMARY ────────────────────────────────────
    story += _section_header("EXECUTIVE SUMMARY", colors.black)

    if req.risk_tier and req.risk_score is not None:
        story.append(_risk_badge(req.risk_tier, req.risk_score))
        story.append(Spacer(1, 3*mm))

    if req.recommendation:
        story.append(Paragraph("<b>Clinical Recommendation:</b>", label_style))
        story.append(Paragraph(req.recommendation, body_style))

    if req.image_class_name:
        img_rc = IMAGE_RISK_COLORS.get(req.image_risk_level or "medium", AMBER)
        story.append(Paragraph(
            f'<b>Cytological Finding:</b> <font color="#{img_rc.hexval()[2:]}">{req.image_class_name}</font>'
            f' (Confidence: {req.image_confidence * 100:.1f}%)' if req.image_confidence else f'<b>Cytological Finding:</b> {req.image_class_name}',
            body_style
        ))

    if req.combined_risk is not None:
        story.append(Paragraph(
            f'<b>Combined AI Risk Index:</b> {req.combined_risk * 100:.1f}%',
            body_style
        ))

    # ── STAGE 1: XGBOOST RISK ASSESSMENT ─────────────────────
    story += _section_header("STAGE 1 — XGBOOST CLINICAL RISK ASSESSMENT", AMBER)

    story.append(Paragraph(
        f"<b>Model:</b> {req.model_used or 'XGBoost + SHAP'} | "
        f"<b>Dataset:</b> Kaggle Cervical Cancer Risk Factors (Ranzeet013)",
        mono_style
    ))

    if req.top_risk_factors:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("<b>Top Risk Factors (SHAP Impact):</b>", label_style))
        rf_data = [["Rank", "Risk Factor", "SHAP Value", "Patient Value"]]
        for i, rf in enumerate(req.top_risk_factors[:8], 1):
            sv = rf.get("shap_value", 0)
            fv = rf.get("feature_value", 0)
            sv_color = "#ff2d55" if sv > 0 else "#00d4ff"
            rf_data.append([
                str(i),
                rf.get("display_name", rf.get("feature", "N/A")),
                Paragraph(f'<font color="{sv_color}">{sv:+.4f}</font>',
                          ParagraphStyle("rv", fontName="Times-Roman", fontSize=10, textColor=TEXT)),
                str(round(fv, 2)),
            ])
        rf_table = Table(rf_data, colWidths=[10*mm, 65*mm, 35*mm, 30*mm])
        rf_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), CYAN_DIM),
            ("TEXTCOLOR", (0, 0), (-1, 0), CYAN),
            ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10.5),
            ("BACKGROUND", (0, 1), (-1, -1), NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [NAVY, NAVY2]),
            ("TEXTCOLOR", (0, 1), (-1, -1), TEXT),
            ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("BOX", (0, 0), (-1, -1), 1, BORDER),
            ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ]))
        story.append(rf_table)

    if req.shap_chart:
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph("<b>SHAP Feature Importance Waterfall Chart:</b>", label_style))
        shap_img = _base64_to_image(req.shap_chart, 168*mm, 80*mm)
        if shap_img:
            story.append(shap_img)

    # ── STAGE 2: FASTVIT IMAGE CLASSIFICATION ─────────────────
    story += _section_header("STAGE 2 — FASTVIT CYTOLOGICAL IMAGE CLASSIFICATION", PURPLE)

    story.append(Paragraph(
        "<b>Model:</b> FastViT-T8 | <b>Dataset:</b> SipakMed (Kaggle) | "
        "<b>Classes:</b> Dyskeratotic, Koilocytotic, Metaplastic, Parabasal, Superficial-Intermediate",
        mono_style
    ))

    if req.image_class_name:
        img_rc = IMAGE_RISK_COLORS.get(req.image_risk_level or "medium", AMBER)
        img_data = [
            [Paragraph("<b>Predicted Cell Type</b>", label_style_table),
             Paragraph(f'<font color="#{img_rc.hexval()[2:]}">{req.image_class_name}</font>',
                       ParagraphStyle("IClass", fontName="Times-Bold", fontSize=13, textColor=img_rc))],
            [Paragraph("<b>Risk Level</b>", label_style_table),
             Paragraph(str(req.image_risk_level or "N/A").upper(), value_style_table)],
            [Paragraph("<b>Confidence</b>", label_style_table),
             Paragraph(f"{req.image_confidence * 100:.1f}%" if req.image_confidence else "N/A", value_style_table)],
            [Paragraph("<b>Description</b>", label_style_table),
             Paragraph(req.image_description or "N/A", body_style_table)],
        ]
        img_table = Table(img_data, colWidths=[45*mm, 125*mm])
        img_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), NAVY),
            ("BOX", (0, 0), (-1, -1), 1, BORDER),
            ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(img_table)

    if req.class_probabilities:
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph("<b>Class Probability Distribution:</b>", label_style))
        cp_data = [["Cell Type", "Probability (%)", "Risk Level"]]
        for cp in req.class_probabilities:
            prob_pct = f"{cp.get('probability', 0) * 100:.2f}%"
            risk = cp.get("risk", "N/A")
            rc = IMAGE_RISK_COLORS.get(risk, TEXT2)
            is_pred = cp.get("class_name") == req.image_class_name
            cp_data.append([
                Paragraph(f'<b>{cp.get("class_name", "N/A")}</b>' if is_pred else cp.get("class_name", "N/A"),
                          ParagraphStyle("CPn", fontName="Times-Bold" if is_pred else "Times-Roman", fontSize=10, textColor=CYAN if is_pred else TEXT)),
                prob_pct,
                Paragraph(f'<font color="#{rc.hexval()[2:]}">{risk.upper()}</font>',
                          ParagraphStyle("CPr", fontName="Times-Roman", fontSize=10, textColor=rc)),
            ])
        cp_table = Table(cp_data, colWidths=[70*mm, 55*mm, 45*mm])
        cp_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), CYAN_DIM),
            ("TEXTCOLOR", (0, 0), (-1, 0), CYAN),
            ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10.5),
            ("BACKGROUND", (0, 1), (-1, -1), NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [NAVY, NAVY2]),
            ("TEXTCOLOR", (0, 1), (-1, -1), TEXT),
            ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("BOX", (0, 0), (-1, -1), 1, BORDER),
            ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(cp_table)

    # GradCAM
    if req.gradcam_image:
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph("<b>GradCAM Saliency Map (Regions of Interest):</b>", label_style))
        story.append(Paragraph(
            "Highlighted regions show which areas of the pap smear image most influenced the model's classification decision.",
            ParagraphStyle("GcDesc", fontName="Times-Roman", fontSize=10, textColor=colors.black, spaceAfter=2*mm)
        ))
        gc_img = _base64_to_image(req.gradcam_image, 100*mm, 80*mm)
        if gc_img:
            story.append(gc_img)

    # ── CLINICAL METHODOLOGY ────────────────────────────────
    story += _section_header("CLINICAL METHODOLOGY & MODEL INFORMATION", colors.black)

    methodology_text = """
    <b>Two-Stage Detection Architecture:</b><br/>
    <b>Stage 1 — XGBoost Clinical Risk Stratification:</b> An eXtreme Gradient Boosting classifier trained on 
    the Kaggle Cervical Cancer Risk Factors dataset (Ranzeet013, 858 patients, 36 features). The model 
    predicts biopsy-confirmed cancer risk using demographic, reproductive, and STD-related features. 
    SHAP (SHapley Additive exPlanations) TreeExplainer provides feature-level attribution for each prediction, 
    ensuring clinical interpretability per GDPR/HIPAA AI transparency requirements.<br/><br/>
    <b>Stage 2 — FastViT Cytological Classification:</b> A FastViT-T8 architecture (6M parameters) trained on 
    the SipakMed dataset (Kaggle, 4,049 pap smear cell images, 5 classes). The model employs depthwise separable 
    convolutions and token mixing for efficient vision processing. Gradient-weighted Class Activation Mapping 
    (GradCAM) generates spatial saliency maps highlighting diagnostically relevant regions in the cytological image.<br/><br/>
    <b>Combined Risk Scoring:</b> The final risk index is a weighted combination of Stage 1 clinical risk 
    (60% weight) and Stage 2 image classification risk (40% weight), calibrated on the validation cohort.
    """
    story.append(Paragraph(methodology_text,
                            ParagraphStyle("Meth", fontName="Times-Roman", fontSize=10.5, textColor=colors.black,
                                           leading=14, spaceAfter=3*mm)))

    # ── DISCLAIMER ───────────────────────────────────────────
    story += _section_header("IMPORTANT DISCLAIMER", RED)
    disclaimer = """
    <b>THIS REPORT IS GENERATED BY AN ARTIFICIAL INTELLIGENCE SYSTEM AND IS INTENDED SOLELY AS A 
    CLINICAL DECISION SUPPORT TOOL.</b> It does not constitute a medical diagnosis and must be reviewed, 
    validated, and confirmed by a qualified and licensed healthcare professional before any clinical decisions 
    are made. The AI models, while trained on peer-reviewed datasets, may produce errors and are subject to 
    the limitations of the training data. Cervical cancer diagnosis requires comprehensive clinical evaluation 
    including physical examination, complete medical history, colposcopy, and histopathological biopsy confirmation. 
    In case of urgent symptoms, patients should seek immediate medical attention.
    """
    story.append(Paragraph(disclaimer,
                            ParagraphStyle("Disc", fontName="Times-Roman", fontSize=10, textColor=colors.black,
                                           leading=13, borderPadding=5)))

    # Build PDF
    def make_page(canvas_obj, doc_obj):
        _make_header_footer(canvas_obj, doc_obj, patient_name, report_id)

    doc.build(story, onFirstPage=make_page, onLaterPages=make_page)
    buf.seek(0)
    return buf.getvalue()


# ─── Router Endpoints ─────────────────────────────────────────

@router.post("/generate", summary="Generate comprehensive PDF report")
async def generate_report(request: ReportRequest):
    """
    Generates a full clinical PDF report combining Stage 1 (XGBoost) and Stage 2 (FastViT) results.
    Returns the PDF as a downloadable file stream.
    """
    if not HAS_REPORTLAB:
        raise HTTPException(
            status_code=501,
            detail="PDF generation requires ReportLab. Install with: pip install reportlab"
        )

    try:
        pdf_bytes = build_pdf(request)
        patient_safe = (request.patient_name or "report").replace(" ", "_").replace("/", "-")
        filename = f"CervicalAI_{patient_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        logger.error(f"PDF generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


@router.post("/preview", summary="Get PDF as base64 for inline preview")
async def preview_report(request: ReportRequest):
    """Returns the PDF as a base64-encoded string for inline browser preview."""
    if not HAS_REPORTLAB:
        raise HTTPException(status_code=501, detail="ReportLab not installed.")
    try:
        pdf_bytes = build_pdf(request)
        b64 = base64.b64encode(pdf_bytes).decode()
        return JSONResponse({"pdf_base64": f"data:application/pdf;base64,{b64}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))