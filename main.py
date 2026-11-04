from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

# Corrected imports
from models.pipeline import generate_health_scores
from report_generator import create_prescription_pdf

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientData(BaseModel):
    name: str
    bmi: float
    heart_rate: int
    cholesterol: float
    sleep_hours: float

@app.post("/generate-report")
def create_report(data: PatientData):
    # 1. Extract features in the exact order the model was trained on
    
    patient_features = [
        data.bmi, 
        data.heart_rate, 
        data.cholesterol, 
        data.sleep_hours
    ]
    
    # 2. Get predictions from the Keras model pipeline
    scores = generate_health_scores(patient_features)
    
    # 3. Generate PDF
    safe_name = "".join(x for x in data.name if x.isalnum() or x in " -_").strip()
    pdf_filename = f"{safe_name.replace(' ', '_')}_report.pdf"
    
    create_prescription_pdf(data.name, scores, pdf_filename)
    
    # 4. Return the PDF for download
    return FileResponse(
        path=pdf_filename, 
        filename=pdf_filename, 
        media_type='application/pdf'
    )

@app.get("/")
def read_root():
    return {"message": "Health Analytics API is running. Use POST /generate-report to get a PDF."}