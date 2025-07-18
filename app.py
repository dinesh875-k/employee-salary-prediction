import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("salary_model.pkl")

# Page config
st.set_page_config(page_title="Employee Salary Prediction", page_icon="💰", layout="centered")

# Main title
st.markdown("<h1 style='text-align: center;'>💸💰 Employee Salary Prediction App </h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Predict estimated salary based on Age, Experience, and Gender</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar inputs
with st.sidebar:
    st.header("🧾 Input Employee Details")
    age = st.slider("🎂 Age", min_value=18, max_value=65, value=30)
    experience = st.slider("💼 Years of Experience", min_value=0, max_value=40, value=5)
    gender = st.selectbox("👤 Gender", ["Male", "Female"])

# Convert gender to numeric format
gender_encoded = 1 if gender == "Male" else 0

# Prepare input DataFrame
input_df = pd.DataFrame({
    'Experience_Years': [experience],
    'Age': [age],
    'Gender': [gender_encoded]
})

# Display user inputs in a nice format
st.subheader("📋 Input Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Age", f"{age} years")
col2.metric("Experience", f"{experience} yrs")
col3.metric("Gender", gender)

# Show raw input DataFrame (optional)
with st.expander("📊 See input dataframe"):
    st.dataframe(input_df, use_container_width=True)

# Predict salary
if st.button("🚀 Predict Salary"):
    prediction = model.predict(input_df)[0]
    final_salary = max(0, prediction)

    st.markdown("---")
    st.markdown("## 🧮 Prediction Result")

    st.success(f"💰 Estimated Salary: ₹{final_salary:,.2f}/-")

    if final_salary < 100000:
        st.error("💔 Salary is too low... 😢 Keep pushing forward!")
        st.info("📉 This seems to be a low salary range.")
    elif final_salary < 500000:
        st.warning("📈 Moderate salary range.")
    else:
        st.balloons()
        st.success("🎉 High salary detected! Great!")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>Made by  Dinesh Kolasani❤️ </div>", unsafe_allow_html=True)
