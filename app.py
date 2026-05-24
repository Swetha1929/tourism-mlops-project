import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Package Prediction", layout="centered")

st.title("Tourism Package Prediction App")
st.write("Predict whether a customer is likely to purchase the Wellness Tourism Package.")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Swetha1929/tourism-package-model",
        filename="best_model.pkl",
        repo_type="model"
    )
    model = joblib.load(model_path)
    return model

model = load_model()

st.header("Enter Customer Details")

TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business", "Others"])
Gender = st.selectbox("Gender", ["Male", "Female"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
DurationOfPitch = st.number_input("Duration of Pitch", min_value=0.0, value=10.0)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=0, value=3)
NumberOfFollowups = st.number_input("Number of Followups", min_value=0.0, value=3.0)
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1.0, max_value=5.0, value=3.0)
NumberOfTrips = st.number_input("Number of Trips", min_value=0.0, value=2.0)
Passport = st.selectbox("Passport", [0, 1])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
OwnCar = st.selectbox("Own Car", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0.0, value=0.0)
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=20000.0)

input_df = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"The customer is likely to purchase the package. Probability: {probability:.2f}")
    else:
        st.warning(f"The customer is unlikely to purchase the package. Probability: {probability:.2f}")
