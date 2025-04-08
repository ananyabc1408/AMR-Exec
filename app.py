import streamlit as st
import joblib
import pandas as pd
import os

# Define the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Background Image
image_path = os.path.join(current_dir, "background.jpg")
# Load the model and label encoders
model = joblib.load('ensemble_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Remedies for low resistance
low_resistance_remedies = [
    "Increase the dosage of the antibiotic.",
    "Combine the antibiotic with another drug to enhance its effectiveness.",
    "Ensure proper sanitation and hygiene practices.",
    "Improve food safety practices to prevent contamination.",
    "Implement rotational use of antibiotics to reduce resistance development."
]

# Remedies for medium resistance
medium_resistance_remedies = [
    "Consider using a higher-generation antibiotic.",
    "Conduct susceptibility testing to identify more effective antibiotics.",
    "Monitor patient closely and adjust treatment as needed.",
    "Enhance infection control measures to prevent spread.",
    "Educate healthcare providers on judicious antibiotic use."
]


# Title of the app
st.title("Microbial Resistance Prediction")
st.image(image_path, use_container_width=True)
# Input fields for user
pathogen = st.selectbox("Select Pathogen", label_encoders['pathogen'].classes_)
antibiotic = st.selectbox("Select Antibiotic", label_encoders['antibiotic'].classes_)

# Predict resistance level
if st.button("Predict Resistance Level"):
    data = {
        'pathogen': [pathogen],
        'antibiotic': [antibiotic]
    }
    df = pd.DataFrame(data)
    df['pathogen'] = label_encoders['pathogen'].transform(df['pathogen'])
    df['antibiotic'] = label_encoders['antibiotic'].transform(df['antibiotic'])

    prediction = model.predict(df)
    resistance_level = ['Low', 'Medium', 'High'][prediction[0]]

    st.write(f"Predicted Resistance Level: **{resistance_level}**")

    if resistance_level == "Low":
        st.write("### Remedies for Low Resistance:")
        for remedy in low_resistance_remedies:
            st.write(f"- {remedy}")
    elif resistance_level == "Medium":
        st.write("### Remedies for Medium Resistance:")
        for remedy in medium_resistance_remedies:
            st.write(f"- {remedy}")

    # Streamlit app
