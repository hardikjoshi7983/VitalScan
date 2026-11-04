"""Train candidate classifiers on synthetic data. This is a learning/demo project, not a clinical model."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from src.features import FEATURES, NUMERIC, CATEGORICAL

DISEASES = [
    "Diabetes-like pattern", "Hypertension-like pattern", "Cardiovascular-risk pattern",
    "Elevated BMI pattern", "Lipid imbalance pattern", "Insulin resistance-like pattern",
    "Sleep insufficiency pattern", "Elevated resting heart-rate pattern"
]

def make_synthetic_data(n=6500, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 86, n)
    sex = rng.choice(["Female", "Male", "Other"], n, p=[.49, .49, .02])
    bmi = np.clip(rng.normal(26, 5.2, n), 15, 55)
    sbp = np.clip(rng.normal(122 + .28*(age-45), 16, n), 80, 220)
    dbp = np.clip(rng.normal(79 + .10*(age-45), 11, n), 45, 140)
    glucose = np.clip(rng.normal(98 + 1.05*(bmi-25), 22, n), 55, 260)
    insulin = np.clip(rng.lognormal(3.2 + .018*(bmi-25), .5, n), 2, 250)
    cholesterol = np.clip(rng.normal(190 + .55*(age-40), 35, n), 90, 360)
    heart_rate = np.clip(rng.normal(74, 12, n), 42, 130)
    smoking = rng.choice(["No", "Yes"], n, p=[.78, .22])
    activity_days = rng.integers(0, 8, n)
    family_history = rng.choice(["No", "Yes"], n, p=[.7, .3])
    sleep_hours = np.clip(rng.normal(7, 1.2, n), 3, 12)
    df = pd.DataFrame({
        "age": age, "sex": sex, "bmi": bmi, "systolic_bp": sbp, "diastolic_bp": dbp,
        "glucose": glucose, "insulin": insulin, "cholesterol": cholesterol,
        "heart_rate": heart_rate, "smoking": smoking, "activity_days": activity_days,
        "family_history": family_history, "sleep_hours": sleep_hours
    })
    # Transparent synthetic label rules with noise; they do NOT represent real clinical criteria.
    logit_d = -4.0 + .035*(age-40) + .075*(bmi-25) + .035*(glucose-100) + .55*(family_history=="Yes") + .003*(insulin-20)
    logit_h = -3.0 + .035*(age-40) + .035*(sbp-120) + .018*(dbp-80) + .45*(family_history=="Yes") + .3*(smoking=="Yes")
    logit_c = -3.4 + .03*(age-40) + .018*(cholesterol-190) + .45*(smoking=="Yes") + .035*(sbp-120) - .16*activity_days
    # Additional demo-only patterns, created from synthetic formulas, not clinical criteria.
    logit_bmi = -1.3 + .19*(bmi-27) - .18*activity_days + .018*(age-40)
    logit_lipid = -2.0 + .024*(cholesterol-190) + .015*(age-40) + .28*(smoking=="Yes") + .12*(family_history=="Yes")
    logit_insulin = -2.6 + .055*(glucose-100) + .025*(insulin-20) + .08*(bmi-25) + .35*(family_history=="Yes")
    logit_sleep = 1.0*(sleep_hours < 6) + .5*(sleep_hours > 9) + .01*(age-40) - 2.0
    logit_hr = -2.4 + .055*(heart_rate-75) + .018*(age-40) + .25*(smoking=="Yes") - .08*activity_days
    sigmoid = lambda z: 1/(1+np.exp(-np.clip(z, -20, 20)))
    for name, logit in zip(DISEASES, [logit_d, logit_h, logit_c, logit_bmi, logit_lipid, logit_insulin, logit_sleep, logit_hr]):
        p = sigmoid(logit)
        # Add stochastic label noise to avoid perfectly deterministic labels.
        df[name] = rng.binomial(1, np.clip(p, .01, .98))
    # Inject a small amount of missingness to exercise preprocessing.
    for col in ["bmi", "insulin", "cholesterol", "sleep_hours"]:
        mask = rng.random(n) < .025
        df.loc[mask, col] = np.nan
    return df

def build_preprocessor():
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric_pipe, NUMERIC), ("cat", categorical_pipe, CATEGORICAL)])

def ensure_models(model_dir="artifacts", force=False):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file, meta_file = model_dir/"models.joblib", model_dir/"metadata.joblib"
    if model_file.exists() and meta_file.exists() and not force:
        try:
            existing = joblib.load(model_file)
            if set(existing.keys()) == set(DISEASES):
                return
        except Exception:
            pass
    df = make_synthetic_data()
    X = df[FEATURES]
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=180, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=140, learning_rate=.08, max_leaf_nodes=20, random_state=42),
        "ANN (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=220, early_stopping=True, random_state=42)
    }
    bundles, metadata = {}, {}
    for disease in DISEASES:
        y = df[disease].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.22, random_state=42, stratify=y)
        best = None
        for model_name, estimator in candidates.items():
            prep = build_preprocessor()
            train_x = prep.fit_transform(X_train)
            test_x = prep.transform(X_test)
            model = clone(estimator)
            model.fit(train_x, y_train)
            probs = model.predict_proba(test_x)[:, 1]
            auc = float(roc_auc_score(y_test, probs))
            if best is None or auc > best["auc"]:
                best = {"model": model, "preprocessor": prep, "auc": auc, "name": model_name}
        bundles[disease] = best
        metadata[disease] = {"name": best["name"], "auc": round(best["auc"], 3)}
    joblib.dump(bundles, model_file)
    joblib.dump(metadata, meta_file)

if __name__ == "__main__":
    ensure_models(force=True)
    print("Trained candidate models and saved the best per target in artifacts/.")
