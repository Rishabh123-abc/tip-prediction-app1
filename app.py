import streamlit as st
import joblib
import pandas as pd
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    text-align: center;
    color: #1E3A8A;
}

.stButton>button {
    background-color: #2563EB;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #1D4ED8;
}

.stNumberInput input {
    border-radius: 10px;
}

.prediction-box {
    background-color: #DCFCE7;
    padding: 15px;
    border-radius: 10px;
    font-size: 24px;
    font-weight: bold;
    color: #166534;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

st.title("Tip Prediction App")

model = joblib.load("linear_regression.pkl")

total_bill = st.number_input("Enter Total Bill")

if st.button("Predict Tip"):
    data = pd.DataFrame({
        "total_bill": [total_bill]
    })

    prediction = model.predict(data)

    predicted_tip = float(prediction.flatten()[0])

    st.markdown(
    f"""
    <div class="prediction-box">
        Predicted Tip: ₹ {predicted_tip:.2f}
    </div>
    """,
    unsafe_allow_html=True
)