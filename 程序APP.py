import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap  # Ensure the shap library is imported
import matplotlib.pyplot as plt

# Load the saved XGBoost model
model = joblib.load('XGB.pkl')

# Updated feature range definition, assuming all protein values are normalized to [-10, 10]
feature_ranges = {
    "Plasma GDF15": {"type": "numerical"},
    "Age": {"type": "numerical"},
    "Systolic Blood Pressure": {"type": "numerical"},
    "Plasma MMP12": {"type": "numerical"},
    "Plasma NTproBNP": {"type": "numerical"},
    "Non Cancer Illness Count": {"type": "numerical"},
    "Sex": {"type": "categorical"},
    "Plasma AGER": {"type": "numerical"},
    "Plasma PRSS8": {"type": "numerical"},
    "Plasma PSPN": {"type": "numerical"},
    "CHOL RATIO": {"type": "numerical"},
    "Plasma WFDC2": {"type": "numerical"},
    "Plasma LPA": {"type": "numerical"},
    "Plasma CXCL17": {"type": "numerical"},
    "Long Standing Illness Disability": {"type": "categorical"},
    "Number of treatments medications taken": {"type": "numerical"},
    "Plasma GAST": {"type": "numerical"},
    "Plasma RGMA": {"type": "numerical"},
    "Plasma EPHA4": {"type": "numerical"},
}

# Streamlit interface
st.title("Prediction Model with SHAP Visualization")

# Dynamically generate input fields
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        # Numerical input box, no range limit
        value = st.number_input(
            label=f"{feature}",
            value=0.0,  # Default value
        )
    elif properties["type"] == "categorical":
        # Categorical select box
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=[0, 1],  # Categories 0 and 1, can be changed as per the actual case
        )
    feature_values.append(value)

# Convert to model input format
features = np.array([feature_values])

# Prediction and SHAP visualization
if st.button("Predict"):
    # Model prediction
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Extract predicted class probability
    probability = predicted_proba[predicted_class] * 100

    # Display prediction result using Matplotlib
    text = f"Based on feature values, predicted possibility of AKI is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # Print SHAP values shape to ensure it is 2D
    print(f"SHAP values shape: {shap_values.shape}")
    
    # Generate SHAP force plot for Class 1 (positive class)
    shap_fig = shap.force_plot(
        explainer.expected_value[1],  # Use the expected value for Class 1 (positive class)
        shap_values[1],  # SHAP values for Class 1
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )

    # Save and display SHAP plot for Class 1
    plt.savefig("shap_force_plot_class_1.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot_class_1.png")

