import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Machine Failure Demo", page_icon="⚙️", layout="wide")

MODEL_PATH = Path(__file__).parent / "model.pkl"
N_FEATURES = 20

st.title("⚙️ Machine Failure Prediction — Demo")
st.caption("A small Streamlit interface for the Logistic Regression artifact from the supplied notebook.")

st.warning(
    "Important: this notebook creates a synthetic dataset with 20 generic features. "
    "This app demonstrates model loading and prediction flow; it is not a validated "
    "real-world machine-maintenance system."
)

@st.cache_resource
def load_artifact(path: str):
    # Only load pickle/joblib files from a source you trust: loading them can execute code.
    artifact = joblib.load(path)
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
    else:
        # Also allow a file that stores the estimator directly.
        model = artifact

    if not hasattr(model, "predict"):
        raise TypeError("The saved artifact does not contain an object with a predict() method.")
    return model

if not MODEL_PATH.exists():
    st.error("Could not find model.pkl. Put model.pkl in the same folder as app.py.")
    st.stop()

try:
    model = load_artifact(str(MODEL_PATH))
except Exception as exc:
    st.error("The model could not be loaded.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.info(
        "This can happen if the file is damaged or if the scikit-learn/joblib versions "
        "differ from the environment that created it. Check the notebook environment and requirements."
    )
    st.stop()

expected = getattr(model, "n_features_in_", None)
if expected is not None and expected != N_FEATURES:
    st.error(f"This app expects {N_FEATURES} inputs, but the model expects {expected}.")
    st.stop()

st.write(
    "Enter values for the model's 20 generic input features. "
    "The notebook does not define real sensor names or engineering units."
)

with st.form("prediction_form"):
    values = []
    cols = st.columns(4)
    for i in range(N_FEATURES):
        with cols[i % 4]:
            value = st.number_input(
                f"Feature {i}",
                value=0.0,
                step=0.1,
                format="%.4f",
                key=f"feature_{i}",
            )
            values.append(value)
    submitted = st.form_submit_button("Predict")

if submitted:
    # The notebook trained with a pandas DataFrame whose columns are integers 0..19.
    input_df = pd.DataFrame([values], columns=list(range(N_FEATURES)))
    try:
        prediction = int(model.predict(input_df)[0])
        st.subheader("Prediction")
        if prediction == 1:
            st.error("Model prediction: Failure (class 1)")
        else:
            st.success("Model prediction: No failure (class 0)")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(getattr(model, "classes_", range(len(probabilities))))
            probability_map = {int(label): float(prob) for label, prob in zip(classes, probabilities)}
            failure_probability = probability_map.get(1)
            if failure_probability is not None:
                st.metric("Estimated failure probability", f"{failure_probability:.1%}")
                st.progress(min(max(failure_probability, 0.0), 1.0))
        with st.expander("Show submitted input"):
            st.dataframe(input_df, use_container_width=True)
    except Exception as exc:
        st.error("Prediction failed.")
        st.code(f"{type(exc).__name__}: {exc}")
