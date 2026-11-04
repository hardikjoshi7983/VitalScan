FEATURES = [
    "age", "sex", "bmi", "systolic_bp", "diastolic_bp", "glucose",
    "insulin", "cholesterol", "heart_rate", "smoking", "activity_days",
    "family_history", "sleep_hours"
]

NUMERIC = ["age", "bmi", "systolic_bp", "diastolic_bp", "glucose", "insulin",
           "cholesterol", "heart_rate", "activity_days", "sleep_hours"]
CATEGORICAL = ["sex", "smoking", "family_history"]

def validate_patient(form):
    name = (form.get("name") or "").strip()
    if not name or len(name) > 80:
        raise ValueError("Please enter a name (maximum 80 characters).")
    values = {
        "age": int(form["age"]),
        "sex": form["sex"],
        "bmi": float(form["bmi"]),
        "systolic_bp": float(form["systolic_bp"]),
        "diastolic_bp": float(form["diastolic_bp"]),
        "glucose": float(form["glucose"]),
        "insulin": float(form["insulin"]),
        "cholesterol": float(form["cholesterol"]),
        "heart_rate": float(form["heart_rate"]),
        "smoking": form["smoking"],
        "activity_days": int(form["activity_days"]),
        "family_history": form["family_history"],
        "sleep_hours": float(form["sleep_hours"])
    }
    bounds = {
        "age": (18, 100), "bmi": (10, 70), "systolic_bp": (60, 260),
        "diastolic_bp": (30, 160), "glucose": (30, 600), "insulin": (0, 1000),
        "cholesterol": (50, 600), "heart_rate": (25, 240),
        "activity_days": (0, 7), "sleep_hours": (0, 24)
    }
    for key, (low, high) in bounds.items():
        if not low <= values[key] <= high:
            raise ValueError(f"{key.replace('_', ' ').title()} must be between {low} and {high}.")
    if values["sex"] not in {"Female", "Male", "Other"}:
        raise ValueError("Choose a valid sex option.")
    for key in ("smoking", "family_history"):
        if values[key] not in {"No", "Yes"}:
            raise ValueError("Choose valid Yes/No options.")
    display = {
        "name": name, "bmi": values["bmi"], "systolic_bp": values["systolic_bp"],
        "diastolic_bp": values["diastolic_bp"], "glucose": values["glucose"],
        "insulin": values["insulin"], "cholesterol": values["cholesterol"],
        "heart_rate": values["heart_rate"], "activity_days": values["activity_days"],
        "sleep_hours": values["sleep_hours"]
    }
    return values, display
