import streamlit as st
import pandas as pd
import joblib
import os

# ---------------- Page Configuration ---------------- #

st.set_page_config(
    page_title="Vehicle Resale Value Prediction",
    page_icon="🚗",
    layout="wide"
)

# ---------------- Custom CSS ---------------- #

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

h1,h2,h3,label{
color:white !important;
}

.block-container{
padding-top:2rem;
}

.card{
background:rgba(255,255,255,0.08);
padding:20px;
border-radius:20px;
box-shadow:0px 0px 20px rgba(0,255,255,0.25);
margin-bottom:20px;
}

.stButton>button{
background:linear-gradient(90deg,#00c6ff,#0072ff);
color:white;
font-size:20px;
font-weight:bold;
border:none;
border-radius:12px;
padding:12px;
width:100%;
}

.stButton>button:hover{
background:linear-gradient(90deg,#00ff99,#00b09b);
}

.result{
background:#00c853;
padding:25px;
border-radius:15px;
font-size:30px;
text-align:center;
font-weight:bold;
color:white;
}

.footer{
text-align:center;
margin-top:40px;
font-size:16px;
color:white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Title ---------------- #

st.markdown(
"""
<h1 style='text-align:center;color:#00E5FF;'>
🚗 Vehicle Resale Value Prediction
</h1>
""",
unsafe_allow_html=True
)

st.write("Predict the resale value of a vehicle using Machine Learning.")

# ---------------- Sidebar ---------------- #

st.sidebar.title("📋 Project Information")

st.sidebar.success("Minor Project")

st.sidebar.info("""

### Features

✔ Brand

✔ Make Year

✔ Mileage

✔ Engine Capacity

✔ Fuel Type

✔ Transmission

✔ Owners

✔ Service History

✔ Insurance

✔ Accident Records

""")

# ---------------- Load Model ---------------- #

model = None

if os.path.exists("model.pkl"):
    model = joblib.load("model.pkl")

# ---------------- Input Form ---------------- #

st.markdown("<div class='card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    brand = st.selectbox(
        "Vehicle Brand",
        [
            "Toyota",
            "Honda",
            "Ford",
            "BMW",
            "Audi",
            "Hyundai",
            "Kia",
            "Nissan",
            "Mercedes"
        ]
    )

    make_year = st.number_input(
        "Manufacturing Year",
        2000,
        2026,
        2020
    )

    mileage = st.number_input(
        "Mileage (km/l)",
        5.0,
        40.0,
        18.0
    )

    engine = st.number_input(
        "Engine Capacity (CC)",
        800,
        5000,
        1500
    )

    fuel = st.selectbox(
        "Fuel Type",
        [
            "Petrol",
            "Diesel",
            "Electric",
            "Hybrid"
        ]
    )

with col2:

    owners = st.number_input(
        "Owner Count",
        1,
        5,
        1
    )

    transmission = st.selectbox(
        "Transmission",
        [
            "Manual",
            "Automatic"
        ]
    )

    color = st.selectbox(
        "Vehicle Color",
        [
            "White",
            "Black",
            "Silver",
            "Blue",
            "Red",
            "Grey"
        ]
    )

    service = st.selectbox(
        "Service History",
        [
            "Poor",
            "Average",
            "Good",
            "Excellent"
        ]
    )

    accidents = st.number_input(
        "Accidents Reported",
        0,
        10,
        0
    )

    insurance = st.selectbox(
        "Insurance Valid",
        [
            "Yes",
            "No"
        ]
    )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Encoding ---------------- #

brand_map = {
    "Toyota":0,
    "Honda":1,
    "Ford":2,
    "BMW":3,
    "Audi":4,
    "Hyundai":5,
    "Kia":6,
    "Nissan":7,
    "Mercedes":8
}

fuel_map = {
    "Petrol":0,
    "Diesel":1,
    "Electric":2,
    "Hybrid":3
}

transmission_map = {
    "Manual":0,
    "Automatic":1
}

color_map = {
    "White":0,
    "Black":1,
    "Silver":2,
    "Blue":3,
    "Red":4,
    "Grey":5
}

service_map = {
    "Poor":0,
    "Average":1,
    "Good":2,
    "Excellent":3
}

insurance_map = {
    "No":0,
    "Yes":1
}

# ---------------- Prediction ---------------- #

if st.button("🚀 Predict Vehicle Price"):

    if model is None:

        st.error("model.pkl not found.")

    else:

        input_df = pd.DataFrame([[
            make_year,
            mileage,
            engine,
            fuel_map[fuel],
            owners,
            brand_map[brand],
            transmission_map[transmission],
            color_map[color],
            service_map[service],
            accidents,
            insurance_map[insurance]
        ]],
        columns=[
            "make_year",
            "mileage_kmpl",
            "engine_cc",
            "fuel_type",
            "owner_count",
            "brand",
            "transmission",
            "color",
            "service_history",
            "accidents_reported",
            "insurance_valid"
        ])

        prediction = model.predict(input_df)[0]

        st.markdown(
            f"""
            <div class="result">
            💰 Estimated Resale Value<br><br>
            ${prediction:,.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------- Footer ---------------- #

st.markdown("""
<div class="footer">
Developed using ❤️ Streamlit | Vehicle Resale Value Prediction
</div>
""", unsafe_allow_html=True)