**VitalScan**

**An AI-powered health risk analysis platform that combines numerical health indicators with symptom-text analysis using a multimodal ANN and BioBERT Transformer model.**

HealthInsight AI is a FastAPI-based web application designed to let users enter selected health measurements and symptom descriptions, then view educational health-pattern scores, possible symptoms, suggested next steps, and a health report. The current training workflow uses a synthetic demonstration dataset.

> **Important:** HealthInsight AI is an experimental educational project. Its outputs are not medical diagnoses, validated clinical predictions, or a substitute for professional medical advice. Do not use it for emergencies or treatment decisions.

---

## Features

- **Health Profile Assessment** — accepts numeric health inputs and a free-text symptom description.
- **Multimodal Machine Learning** — combines scaled numeric features with tokenized symptom text.
- **ANN + BioBERT Transformer** — uses a Keras neural-network branch for numeric data and a BioBERT transformer branch for text features.
- **Eight Health Patterns** — produces educational scores for diabetes-like, blood-pressure, cardiovascular-risk, elevated-BMI, lipid-imbalance, insulin-resistance-like, sleep-insufficiency, and resting-heart-rate patterns.
- **Risk Explanations** — includes simple age, BMI, and glucose impact values in the report data.
- **Health Snapshot Data** — prepares chart values for BMI, glucose, blood pressure, cholesterol, insulin, heart rate, and sleep.
- **Symptoms and Next-Step Guidance** — provides informational guidance associated with each health pattern.
- **PDF Report Generation** — includes a ReportLab-based PDF report generator.
- **Responsive Web Interface** — uses FastAPI, Jinja2 templates, HTML, and static assets.
- **Animated UI Experience** — the supplied project README describes loading transitions, animated progress, floating elements, hover effects, and responsive layouts.

---

## System Architecture

### High-Level Architecture

```text
┌───────────────────────────────────────────────────────────────┐
│                     CLIENT / WEB BROWSER                      │
│                                                               │
│             HTML Templates + CSS + Static Assets              │
│        index.html | profile.html | report.html                │
│                         next_steps.html                       │
└───────────────────────────────┬───────────────────────────────┘
                                │ HTTP GET / POST
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                     FASTAPI APPLICATION                       │
│                           app.py                              │
│                                                               │
│  Routes:                                                      │
│    GET  /                    Landing page                     │
│    GET  /profile             Assessment form                   │
│    POST /predict             Process profile and predict       │
│    GET  /report/{report_id}  Display assessment report         │
│    GET  /next-steps/{id}     Display next-step guidance         │
│                                                               │
│  Loads model, tokenizer, scaler and prepares report data       │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING LAYER                     │
│                                                               │
│  Numeric inputs ──▶ StandardScaler ──▶ ANN branch              │
│                                                               │
│  Symptom text ────▶ BioBERT Tokenizer ──▶ BioBERT branch       │
│                                                               │
│                   Feature Concatenation                       │
│                           │                                   │
│                           ▼                                   │
│                  Dense + Dropout Layers                        │
│                           │                                   │
│                           ▼                                   │
│                   Sigmoid Prediction                          │
│                                                               │
│  Saved artifacts:                                             │
│    keras_multimodal.keras                                     │
│    numeric_scaler.joblib                                      │
└───────────────────────────────────────────────────────────────┘
```

### Request Lifecycle

```text
User                 FastAPI Application              ML Model
 │                           │                            │
 │  Open landing page        │                            │
 │──────────────────────────▶│                            │
 │  View index.html          │                            │
 │◀──────────────────────────│                            │
 │                           │                            │
 │  Open /profile            │                            │
 │──────────────────────────▶│                            │
 │  View assessment form     │                            │
 │◀──────────────────────────│                            │
 │                           │                            │
 │  Submit health profile    │                            │
 │──────────────────────────▶│                            │
 │                           │  Scale numeric inputs      │
 │                           │  Tokenize symptom text     │
 │                           │───────────────────────────▶│
 │                           │                            │
 │                           │  Prediction scores         │
 │                           │◀───────────────────────────│
 │                           │  Prepare risk patterns     │
 │                           │  and chart data            │
 │                           │                            │
 │  Redirect to report       │                            │
 │◀──────────────────────────│                            │
 │                           │                            │
 │  Open next steps          │                            │
 │──────────────────────────▶│                            │
 │  View guidance            │                            │
 │◀──────────────────────────│                            │
```

### Data Flow

```text
User Health Profile
       │
       ├── Numeric Features
       │      ├── Age
       │      ├── BMI
       │      └── Glucose
       │
       └── Symptom Description
                    │
                    ▼
             Preprocessing
              │         │
              ▼         ▼
        StandardScaler  BioBERT Tokenizer
              │         │
              ▼         ▼
          Numeric ANN  Transformer
              │         │
              └────┬────┘
                   ▼
            Feature Concatenation
                   │
                   ▼
             Keras Prediction
                   │
                   ▼
          Eight Health Patterns
                   │
          ┌────────┼───────────┐
          ▼        ▼           ▼
       Scores   Symptoms   Next Steps
          │        │           │
          └────────┼───────────┘
                   ▼
             Report Templates
```

---

## Project Structure

The following structure reflects the files and paths referenced by the supplied Python code. The template and static directories must be present in the project for the web application to render correctly.

```text
healthinsight-ai/
│
├── app.py
├── main.py
├── report_generator.py
├── train_multimodel.py
├── requirements.txt
├── schema.sql
├── .env.example
├── .gitignore
├── README.md
│
├── artifacts/
│   ├── keras_multimodal.keras
│   └── numeric_scaler.joblib
│
├── templates/
│   ├── index.html
│   ├── profile.html
│   ├── report.html
│   └── next_steps.html
│
└── static/
    ├── css/
    ├── js/
    └── images/
```

`artifacts/` is generated by model training and is ignored by Git in the supplied `.gitignore`. The exact contents of the `static/` subfolders depend on the frontend files in your project.

---

## Machine Learning Pipeline

### Model Overview

HealthInsight AI uses a multimodal model: it processes structured numeric information and unstructured symptom text through separate branches, then combines their learned representations.

**1. Numeric branch — ANN**

The current training script uses these numeric features:

- Age
- BMI
- Glucose

The numeric values are standardized using `StandardScaler`. The ANN branch contains Dense layers, Batch Normalization, and Dropout.

**2. Text branch — BioBERT Transformer**

- Pre-trained model: `dmis-lab/biobert-base-cased-v1.2`
- Symptom descriptions are tokenized.
- Maximum token sequence length: 64.
- The BioBERT model generates a text representation.
- The transformer is frozen in the supplied training script.

**3. Fusion and prediction**

The numeric and text representations are concatenated and passed through Dense and Dropout layers. A sigmoid output produces the model's binary prediction score.

### Training Dataset

The supplied `train_multimodel.py` script creates a synthetic dataset containing 200 samples.

| Feature | Description |
|---------|-------------|
| `age` | Generated age values |
| `bmi` | Generated BMI values |
| `glucose` | Generated glucose values |
| `symptoms_text` | Alternating example symptom descriptions |
| `target` | Random binary target label |

The script uses an 80/20 train-test split, trains for 2 epochs with a batch size of 32, and saves the model and scaler in `artifacts/`.

> The current synthetic target labels are randomly generated. Therefore, model scores and accuracy must not be interpreted as evidence of clinical performance. A larger, representative, properly labelled and ethically sourced dataset, rigorous validation, and clinical review would be required for real-world use.

### Pipeline Flow

```text
Synthetic Data Generation
          │
          ▼
      Preprocessing
       │        │
       ▼        ▼
 Numeric Data  Symptom Text
       │        │
       ▼        ▼
 StandardScaler BioBERT Tokenizer
       │        │
       ▼        ▼
    ANN Branch  BioBERT Branch
       │        │
       └────┬───┘
            ▼
     Feature Concatenation
            │
            ▼
       Dense Layers
            │
            ▼
     Sigmoid Output Layer
            │
            ▼
       Model Evaluation
            │
            ▼
     Save Model + Scaler
```

### Train the Model

Run the training script from the project root:

```bash
python train_multimodel.py
```

The script creates the `artifacts/` directory if it does not exist and saves:

```text
artifacts/
├── keras_multimodal.keras
└── numeric_scaler.joblib
```

The BioBERT tokenizer/model files are downloaded from the Hugging Face model repository when first required. An internet connection may be needed for this initial download.

---

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| Backend | Python, FastAPI, Uvicorn |
| Machine Learning | TensorFlow / Keras, scikit-learn |
| Deep Learning | Artificial Neural Network (ANN) |
| Transformer | Hugging Face Transformers, BioBERT |
| Data Processing | NumPy, pandas |
| Model Serialization | joblib |
| Frontend | HTML5, CSS3, Jinja2 |
| PDF Reports | ReportLab |
| Database Setup | MySQL schema supplied (`schema.sql`) |

---

## Installation

### Prerequisites

- Python 3.10+ recommended
- pip
- Git
- Internet connection for downloading Python packages and the BioBERT model
- MySQL Server only if you plan to use the supplied database schema or add database integration

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/healthinsight-ai.git
cd healthinsight-ai
```

Replace `your-username` with your GitHub username and use your actual repository name.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow and Transformers can require substantial disk space and memory. Use a Python version compatible with the TensorFlow release available for your operating system.

### 4. Configure Environment Variables

Copy the example environment file:

**Windows**

```bash
copy .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

Example configuration:

```env
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=healthinsight
```

Update the values for your local MySQL installation. Never commit `.env` or real database credentials.

**Database note:** `schema.sql` creates the `healthinsight` database and an `assessment_audit` table. The supplied `app.py` does not currently establish a MySQL connection or write assessment records, so database integration is not active in the shown application flow.

### 5. Train the Model

```bash
python train_multimodel.py
```

Make sure the model and scaler files are created inside `artifacts/` before starting `app.py`.

---

## Running the Application

Start the main web application using Uvicorn:

```bash
uvicorn app:app --reload
```

Open the application in your browser:

```text
http://127.0.0.1:8000
```

FastAPI interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

Stop the development server with `CTRL + C`.

---

## Application Workflow

1. **Landing Page (`/`)** — introduces HealthInsight AI and its features.
2. **Health Profile (`/profile`)** — lets the user enter profile measurements and symptom text.
3. **Prediction (`POST /predict`)** — preprocesses numeric and text inputs, runs the multimodal model, and prepares health-pattern scores.
4. **Report (`/report/{report_id}`)** — displays the generated assessment result while the report remains in the in-memory report cache.
5. **Next Steps (`/next-steps/{report_id}`)** — displays the informational guidance associated with the assessment.

The report cache is stored in application memory. Reports are not persistent and may be lost when the application restarts.

---

## API Endpoints

The main `app.py` web application defines these routes:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Render the landing page |
| GET | `/profile` | Render the health assessment form |
| POST | `/predict` | Process submitted profile data and run prediction |
| GET | `/report/{report_id}` | Display a generated report |
| GET | `/next-steps/{report_id}` | Display next-step guidance |

### Additional API File

The repository also contains `main.py`, which defines a separate FastAPI application with:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | API status message |
| POST | `/generate-report` | Generate and return a PDF report |

The `main.py` endpoint imports `generate_health_scores` from `models.pipeline`. That module is not included in the supplied files. The additional API therefore requires that module and its model pipeline to be present and configured before it can run. It is separate from the `app.py` application.

---

## PDF Report Generation

`report_generator.py` uses ReportLab to create a PDF containing:

- Patient name
- Report date
- Model prediction patterns
- Model scores and remaining percentages
- Context text for supported patterns
- An experimental-model and medical-diagnosis disclaimer

The report generator is called by the separate `/generate-report` endpoint in `main.py`.

---

## Database Schema

The supplied `schema.sql` initializes a MySQL database named `healthinsight` and creates an `assessment_audit` table.

```sql
CREATE DATABASE IF NOT EXISTS healthinsight;
USE healthinsight;

CREATE TABLE IF NOT EXISTS assessment_audit (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(80) NOT NULL,
    note VARCHAR(255) NOT NULL
);
```

This is a database schema setup only; the supplied `app.py` does not yet connect to this database.

---

## Important Limitations

- The training data is synthetic and contains randomly generated target labels.
- The current model is a demonstration of a multimodal ML workflow, not a clinically validated prediction system.
- The eight pattern scores are derived from one base model score and manually defined multipliers in `app.py`; they are not eight independently trained disease classifiers.
- Some assessment values are displayed in chart data, but the current model's numeric input branch uses only age, BMI, and glucose.
- The app stores reports in an in-memory dictionary, not in persistent storage.
- MySQL configuration and schema files are supplied, but database operations are not implemented in the shown `app.py`.
- The separate `main.py` API depends on `models.pipeline`, which was not included in the supplied files.
- The PDF report generator expects a score structure containing `model_score` and `remaining` fields; ensure the separate pipeline returns that structure.

---

## Responsible Use

HealthInsight AI is for learning, demonstration, and portfolio purposes only.

- It does not diagnose, prevent, or treat any disease.
- A model score is not a probability of having a disease unless independently calibrated and validated for that interpretation.
- Consult a qualified healthcare professional about health concerns and test results.
- Seek emergency medical help for severe or urgent symptoms rather than relying on this application.
- Do not upload identifiable or sensitive health information to a public deployment without appropriate security, privacy, consent, and legal safeguards.

---

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make and test your changes.
4. Commit your changes:

   ```bash
   git commit -m "Add your feature"
   ```

5. Push your branch:

   ```bash
   git push origin feature/your-feature-name
   ```

6. Open a Pull Request describing your changes.

---

