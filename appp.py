import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

BASE = Path(__file__).resolve().parent

# ---------------- PATHS ----------------
PATHS = {
    "Striker": {
        "model": BASE / "striker_model.pkl",
        "features": BASE / "striker_all_features.json",
        "means": BASE / "striker_feature_means.json",
        "label": "SPI (Striker Performance Index)",
    },
    "Midfielder": {
        "model": BASE / "midfielder_model.pkl",
        "features": BASE / "midfielder_all_features.json",
        "means": BASE / "midfielder_feature_means.json",
        "label": "MPI (Midfielder Performance Index)",
    },
    "Defender": {
        "model": BASE / "defender_model.pkl",
        "features": BASE / "defender_features.json",
        "means": None,  # optional later
        "label": "DPI (Defender Performance Index)",
    },
}

# ---------------- FALLBACK TOP-10 LISTS ----------------
TOP10_FALLBACK = {
    "Striker": [
        "ForcePlateJumpPower(W)",
        "RPE_Avg",
        "Average Total Shots (Before 2024/25)",
        "Muscle Strength Asymmetry (%)",
        "Total Distance(m) (per match)",
        "PlayerLoad AU (GPS-derived)",
        "Passing Accuracy (%) (Before 2024/25)",
        "RPE_Training",
        "Average Shots on Target (Before 2024/25)",
        "Average Big Chances Missed (Before 2024/25)",
    ],
    "Midfielder": [
        "Avg_ASR_2425",
        "Average Successful Dribbles",
        "Acceleration Count (≥3 m/s²)",
        "Average Key Passes",
        "ChangeOfDirectionAngle(degree)",
        "Average Total Shots",
        "Stride Length(m)",
        "Average Big Chances Created",
        "Average Shots on Target",
        "RPE_Match",
    ],
    # If you want a custom Top-10 for defenders, add here.
    "Defender": [],
}

# ---------------- LOADERS ----------------
@st.cache_resource
def load_model(path: Path):
    """
    Robust model loader:
    - Try joblib first (recommended for sklearn).
    - Fallback to pickle if needed.
    """
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

@st.cache_data
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_means(role: str):
    mpath = PATHS[role]["means"]
    if mpath is None or not mpath.exists():
        return {}
    return load_json(mpath)

# ---------------- IMPORTANCE (NO EXTRA FILE NEEDED) ----------------
def get_top10_with_importance(model, feature_names):
    """
    Returns a DataFrame with Top-10 features and importance values.
    Works for sklearn models that have .feature_importances_
    (RandomForest, ExtraTrees, sklearn XGBRegressor wrapper, etc.)
    """
    if hasattr(model, "feature_importances_"):
        try:
            imp = np.array(model.feature_importances_, dtype=float)
            if len(imp) == len(feature_names):
                df = pd.DataFrame({"Feature": feature_names, "Importance": imp})
                df = df.sort_values("Importance", ascending=False).head(10).reset_index(drop=True)
                return df
        except Exception:
            return None
    return None

# ---------------- UI ----------------
st.set_page_config(page_title="Football Player Performance Predictor", layout="wide")
st.title("⚽ Football Player Performance Predictor (Web GUI)")
st.caption("Predict 24/25 performance indices for defenders, midfielders and strikers.")

# Diagnostics (helps confirm correct Python + libraries)
st.sidebar.header("Diagnostics")
st.sidebar.write("Python:", sys.version)

try:
    import sklearn
    st.sidebar.write("scikit-learn:", sklearn.__version__)
except Exception as e:
    st.sidebar.error(f"scikit-learn not available: {e}")

try:
    import xgboost
    st.sidebar.write("xgboost:", xgboost.__version__)
except Exception as e:
    st.sidebar.warning(f"xgboost not available: {e}")

# ---------------- ROLE SELECTION ----------------
role = st.selectbox("Select Position", list(PATHS.keys()))
conf = PATHS[role]

# ---------------- FILE CHECKS ----------------
if not conf["model"].exists():
    st.error(f"❌ Model file not found: {conf['model']}")
    st.stop()

if not conf["features"].exists():
    st.error(f"❌ Feature file not found: {conf['features']}")
    st.stop()

# ---------------- LOAD ASSETS ----------------
all_features = load_json(conf["features"])
means = get_means(role)

try:
    model = load_model(conf["model"])
except Exception as e:
    st.error(f"❌ Failed to load model for {role}: {e}")
    st.stop()

st.success(f"✅ Loaded model for {role}")

# ---------------- TOP-10 SELECTION ----------------
top10_df = get_top10_with_importance(model, all_features)

if top10_df is not None:
    top10 = top10_df["Feature"].tolist()
else:
    top10 = TOP10_FALLBACK.get(role, [])
    if not top10:
        top10 = all_features[:10]

# Keep only valid features (avoid typos)
top10 = [f for f in top10 if f in all_features]

# ---------------- SHOW TOP-10 BUTTON ----------------
if st.button("Show Model Top 10 Features"):
    st.subheader(f"Top 10 Features ({role})")
    if top10_df is not None:
        st.dataframe(top10_df, use_container_width=True)
    else:
        st.info("Importance values not available for this model. Showing feature names only.")
        st.dataframe(pd.DataFrame({"Feature": top10}), use_container_width=True)

st.divider()
st.subheader("Top 10 input features")

# ---------------- DEFAULT INPUT VECTOR (FULL) ----------------
default_inputs = {}
for feat in all_features:
    mv = means.get(feat, 0.0)
    default_inputs[feat] = float(mv) if isinstance(mv, (int, float)) else 0.0

# ---------------- INPUT FORM (ONLY TOP-10 SHOWN) ----------------
MIN_REQUIRED_INPUTS = 5

with st.form("predict_form"):
    cols = st.columns(2)
    user_inputs = dict(default_inputs)  # start with defaults

    for i, feat in enumerate(top10):
        default_val = default_inputs.get(feat, 0.0)
        with cols[i % 2]:
            user_inputs[feat] = float(
                st.number_input(feat, value=float(default_val), format="%.6f")
            )

    submitted = st.form_submit_button("Predict")

# ---------------- VALIDATION + PREDICTION ----------------
if submitted:
    # Count how many Top-10 inputs were changed (vs defaults)
    changed = 0
    for feat in top10:
        if float(user_inputs[feat]) != float(default_inputs.get(feat, 0.0)):
            changed += 1

    if changed < MIN_REQUIRED_INPUTS:
        st.warning(
            f"⚠️ Please enter at least {MIN_REQUIRED_INPUTS} inputs (changed from defaults). "
            f"You changed {changed}."
        )
        st.stop()

    X = pd.DataFrame([user_inputs], columns=all_features)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    try:
        pred = float(model.predict(X)[0])
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    st.subheader("Prediction")
    st.success(f"✅ Predicted {conf['label']}: **{pred:.4f}**")

    with st.expander("Show inputs used (Top 10 + first 10 of full vector)"):
        st.write("Top 10 inputs entered:")
        st.dataframe(X[top10], use_container_width=True)
        st.write("First 10 features of full input vector:")
        st.dataframe(X.iloc[:, :10], use_container_width=True)
