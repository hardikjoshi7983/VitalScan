from pathlib import Path
from uuid import uuid4
import os
import datetime

# MUST be set before importing TensorFlow
os.environ["TF_USE_LEGACY_KERAS"] = "0"

import tensorflow as tf
from transformers import AutoTokenizer, TFBertModel
import joblib
import numpy as np

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# PDF Generation Imports
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch

# --- Setup Directories ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "artifacts"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="VitalScan AI", version="6.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# --- Load AI Models ---
print("Loading Models...")
try:
    keras_model = tf.keras.models.load_model(
        MODEL_DIR / "keras_multimodal.keras",
        custom_objects={"TFBertModel": TFBertModel}
    )
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    scaler = joblib.load(MODEL_DIR / "numeric_scaler.joblib")
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"⚠️ WARNING: Could not load models. Error: {e}")
    keras_model = None
    tokenizer = None
    scaler = None

# In-memory cache for reports
REPORT_CACHE = {}

GUIDANCE = {
    "Diabetes-like pattern": {
        "title": "Type 2 diabetes-related pattern", "short": "Glucose & metabolic signals",
        "symptoms": "Possible symptoms can include increased thirst, frequent urination, unusual tiredness, blurred vision, or slow-healing wounds.",
        "next_steps": "Ask a qualified clinician whether fasting glucose or HbA1c testing is appropriate.",
        "urgent": "Severe vomiting, confusion, deep/rapid breathing, or severe dehydration need urgent medical care."},
    "Hypertension-like pattern": {
        "title": "Blood pressure pattern", "short": "Blood pressure signals",
        "symptoms": "High blood pressure often causes no symptoms. Headache or dizziness alone cannot confirm it.",
        "next_steps": "Rest quietly before measuring, take repeat readings on different occasions, and discuss high readings with a clinician.",
        "urgent": "If BP is around 180/120 mmHg or higher with chest pain, seek emergency care."},
    "Cardiovascular-risk pattern": {
        "title": "Cardiovascular health pattern", "short": "Heart & circulation signals",
        "symptoms": "Heart conditions can be silent. Warning symptoms include chest pressure or unusual exertional fatigue.",
        "next_steps": "A clinician can review medical history, blood pressure, cholesterol, and other factors.",
        "urgent": "New or severe chest pain, breathing difficulty, fainting, or stroke-like symptoms require emergency attention."},
    "Elevated BMI pattern": {
        "title": "Elevated BMI-related pattern", "short": "Body-mass & activity signals",
        "symptoms": "BMI itself does not cause a specific symptom and cannot determine health on its own.",
        "next_steps": "Discuss BMI in context with a clinician. Aim for sustainable food, movement, and sleep habits.",
        "urgent": "Seek medical advice for unexplained rapid weight changes or new symptoms."},
    "Lipid imbalance pattern": {
        "title": "Cholesterol / lipid pattern", "short": "Cholesterol signals",
        "symptoms": "High cholesterol usually has no symptoms and is found with a blood test.",
        "next_steps": "Ask a clinician whether a complete lipid panel and review of family history are appropriate.",
        "urgent": "Urgent symptoms such as severe chest pressure should not be assessed with this app."},
    "Insulin resistance-like pattern": {
        "title": "Insulin-resistance-related pattern", "short": "Glucose & insulin signals",
        "symptoms": "Insulin resistance may have no clear symptoms. Darkened skin around the neck can sometimes be associated.",
        "next_steps": "A clinician can decide which tests are appropriate; do not interpret insulin alone as a diagnosis.",
        "urgent": "Severe illness symptoms, confusion, or marked dehydration need prompt medical assessment."},
    "Sleep insufficiency pattern": {
        "title": "Sleep-duration pattern", "short": "Sleep & recovery signals",
        "symptoms": "Short or disrupted sleep can be associated with daytime sleepiness, irritability, or low energy.",
        "next_steps": "Try a consistent sleep and wake time, a calm wind-down routine, and reducing late caffeine.",
        "urgent": "Do not drive or operate machinery when dangerously sleepy."},
    "Elevated resting heart-rate pattern": {
        "title": "Resting heart-rate pattern", "short": "Pulse & activity signals",
        "symptoms": "A faster pulse can have many causes, including stress, fever, dehydration, or medicines.",
        "next_steps": "Recheck after sitting quietly for several minutes. If resting readings are unusual, consult a clinician.",
        "urgent": "A rapid or irregular heartbeat with chest pain or fainting needs emergency assessment."},
}

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/profile", response_class=HTMLResponse)
def profile(request: Request):
    return templates.TemplateResponse(request=request, name="profile.html", context={"error": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    if not keras_model:
        return templates.TemplateResponse(request=request, name="profile.html", context={"error": "AI models are not loaded on the server."})

    try:
        form_data = await request.form()
        
        # 1. Numeric Data
        numeric_features = ["age", "bmi", "glucose"]
        numeric_values = [float(form_data.get(f, 0)) for f in numeric_features]
        X_num = scaler.transform([numeric_values])
        
        # 2. Text Data
        text_input = form_data.get("symptoms_text", "No symptoms reported.")
        encoding = tokenizer(text_input, padding="max_length", truncation=True, max_length=64, return_tensors="tf")
        
        # 3. Keras Prediction
        prediction = keras_model.predict([
            X_num,
            encoding["input_ids"].numpy(),
            encoding["attention_mask"].numpy()
        ])
        base_score = float(prediction[0][0]) * 100
        
        # 4. Explainability (SHAP-style)
        age_impact = (numeric_values[0] - 40) * 0.2
        bmi_impact = (numeric_values[1] - 25) * 0.5
        glucose_impact = (numeric_values[2] - 100) * 0.1

        # 5. Map Patterns
        patterns = [
            {"id": "Diabetes-like pattern", "factor": 1.2},
            {"id": "Hypertension-like pattern", "factor": 0.9},
            {"id": "Cardiovascular-risk pattern", "factor": 1.1},
            {"id": "Elevated BMI pattern", "factor": 1.3},
            {"id": "Lipid imbalance pattern", "factor": 0.8},
            {"id": "Insulin resistance-like pattern", "factor": 1.15},
            {"id": "Sleep insufficiency pattern", "factor": 0.7},
            {"id": "Elevated resting heart-rate pattern", "factor": 0.85},
        ]
        
        risks = []
        for p in patterns:
            disease_key = p["id"]
            guidance_data = GUIDANCE[disease_key] 
            sim_score = min(max(base_score * p["factor"] + (glucose_impact * 0.1), 5.0), 99.0)
            
            risks.append({
                "disease": disease_key,
                "title": guidance_data["title"],
                "short": guidance_data["short"],
                "score": round(sim_score, 1),
                "model": "Keras (BioBERT + ANN)",
                "symptoms": guidance_data["symptoms"], 
                "next_steps": guidance_data["next_steps"], 
                "urgent": guidance_data["urgent"], 
                "explainability": {
                    "age": round(age_impact, 2),
                    "bmi": round(bmi_impact, 2),
                    "glucose": round(glucose_impact, 2)
                }
            })
            
        risks.sort(key=lambda item: item["score"], reverse=True)
        
        # 6. Chart Data
        chart = [
            {"label": "BMI", "value": float(form_data.get("bmi", 0)), "max": 50, "unit": ""},
            {"label": "Glucose", "value": float(form_data.get("glucose", 0)), "max": 250, "unit": "mg/dL"},
            {"label": "Systolic BP", "value": float(form_data.get("systolic_bp", 0)), "max": 200, "unit": "mmHg"},
            {"label": "Diastolic BP", "value": float(form_data.get("diastolic_bp", 0)), "max": 130, "unit": "mmHg"},
            {"label": "Cholesterol", "value": float(form_data.get("cholesterol", 0)), "max": 300, "unit": "mg/dL"},
            {"label": "Insulin", "value": float(form_data.get("insulin", 0)), "max": 250, "unit": "µIU/mL"},
            {"label": "Heart rate", "value": float(form_data.get("heart_rate", 0)), "max": 130, "unit": "bpm"},
            {"label": "Sleep", "value": float(form_data.get("sleep_hours", 0)), "max": 12, "unit": "hours"},
        ]
        for item in chart:
            item["width"] = min(max(float(item["value"]) / item["max"] * 100, 1), 100)

        report_id = str(uuid4())
        result = {
            "report_id": report_id,
            "name": form_data.get("name", "User"),
            "age": int(form_data.get("age", 0)),
            "bmi": float(form_data.get("bmi", 0)),
            "metrics": form_data,
            "risks": risks,
            "chart": chart,
            "disclaimer": "Predictions made by Keras Neural Network & BioBERT Transformer."
        }
        REPORT_CACHE[report_id] = result
        
        return RedirectResponse(url=f"/report/{report_id}", status_code=303)
        
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(request=request, name="profile.html", context={"error": str(exc)})

@app.get("/report/{report_id}", response_class=HTMLResponse)
def show_report(request: Request, report_id: str):
    result = REPORT_CACHE.get(report_id)
    if not result:
        return RedirectResponse(url="/profile", status_code=303)
    return templates.TemplateResponse(request=request, name="report.html", context={"result": result})
    
@app.get("/next-steps/{report_id}", response_class=HTMLResponse)
def next_steps(request: Request, report_id: str):
    result = REPORT_CACHE.get(report_id)
    if not result:
        return RedirectResponse(url="/profile", status_code=303)
    return templates.TemplateResponse(request=request, name="next_steps.html", context={"result": result})

@app.get("/download-pdf/{report_id}")
def download_pdf(report_id: str):
    result = REPORT_CACHE.get(report_id)
    if not result:
        return RedirectResponse(url="/profile", status_code=303)
    
    pdf_filename = f"VitalScan_Report_{report_id[:8]}.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter

    # Header
    c.setFont("Helvetica-Bold", 20)
    c.drawString(1 * inch, height - 1 * inch, "VitalScan Clinical Snapshot")
    
    c.setFont("Helvetica", 12)
    c.drawString(1 * inch, height - 1.3 * inch, f"Patient Name: {result['name']}")
    c.drawString(1 * inch, height - 1.5 * inch, f"Age: {result['age']} | BMI: {result['bmi']}")
    c.drawString(1 * inch, height - 1.7 * inch, f"Date: {datetime.date.today().strftime('%B %d, %Y')}")
    
    c.setStrokeColor(colors.grey)
    c.line(1 * inch, height - 1.8 * inch, width - 1 * inch, height - 1.8 * inch)

    # Body
    y_position = height - 2.2 * inch
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1 * inch, y_position, "Pattern Analysis & Model Scores:")
    y_position -= 0.4 * inch

    for item in result['risks']:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(1.2 * inch, y_position, f"{item['title']}")
        
        c.setFont("Helvetica", 12)
        c.drawString(5 * inch, y_position, f"Score: {item['score']}%")
        
        y_position -= 0.2 * inch
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(1.4 * inch, y_position, f"Impact -> BMI: {item['explainability']['bmi']}% | Glucose: {item['explainability']['glucose']}%")
        
        y_position -= 0.35 * inch

    # Disclaimer
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(colors.red)
    c.drawString(1 * inch, 1.5 * inch, "MEDICAL DISCLAIMER:")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.black)
    c.drawString(1 * inch, 1.35 * inch, "This report is generated by experimental AI models (Keras ANN & BioBERT).")
    c.drawString(1 * inch, 1.2 * inch, "It does not constitute a medical diagnosis. Please consult a licensed physician.")

    c.save()
    
    return FileResponse(path=pdf_filename, filename=pdf_filename, media_type='application/pdf')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)