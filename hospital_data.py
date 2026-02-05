
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Hospital Disease Prediction",
    page_icon="🏥",
    layout="centered"
)

# -------------------------------
# Background + GREEN Text + BUBBLES
# -------------------------------
def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1586773860418-d37222d8fce3");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            overflow: hidden;
        }

        /* GREEN TEXT */
        h1, h2, h3, h4, h5, h6, p, label, div {
            color: #2ECC71 !important;
            font-weight: bold;
        }

        /* Button */
        .stButton>button {
            background-color: #27AE60;
            color: white !important;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-size: 16px;
        }

        /* Input boxes */
        .stNumberInput input {
            background-color: rgba(255,255,255,0.9);
            color: black;
        }

        /* BUBBLES */
        .bubble {
            position: fixed;
            bottom: -100px;
            background: rgba(46, 204, 113, 0.25);
            border-radius: 50%;
            animation: rise 18s infinite ease-in;
            z-index: -1;
        }

        .bubble:nth-child(1) { left: 10%; width: 40px; height: 40px; animation-duration: 15s; }
        .bubble:nth-child(2) { left: 30%; width: 60px; height: 60px; animation-duration: 20s; }
        .bubble:nth-child(3) { left: 55%; width: 30px; height: 30px; animation-duration: 14s; }
        .bubble:nth-child(4) { left: 75%; width: 80px; height: 80px; animation-duration: 24s; }
        .bubble:nth-child(5) { left: 90%; width: 50px; height: 50px; animation-duration: 18s; }

        @keyframes rise {
            0% { transform: translateY(0); opacity: 0.6; }
            100% { transform: translateY(-1200px); opacity: 0; }
        }
        </style>

        <div class="bubble"></div>
        <div class="bubble"></div>
        <div class="bubble"></div>
        <div class="bubble"></div>
        <div class="bubble"></div>
        """,
        unsafe_allow_html=True
    )

set_bg()

# -------------------------------
# Title
# -------------------------------
st.title("🏥 Hospital Disease Prediction System 🩺")

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("hospital_data.csv")

st.subheader("📊 Dataset Preview 📋")
st.write(df.head())

# -------------------------------
# Features & Target
# -------------------------------
X = df[["Age", "Fever", "BP", "Sugar"]]
y = df["Disease"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

st.subheader("✅ Model Accuracy 🎯")
st.success(f"📈 Accuracy: {accuracy * 100:.2f} %")

# -------------------------------
# User Input Section
# -------------------------------
st.subheader("🧑‍⚕️ Enter Patient Details 🩻")

age = st.number_input("🎂 Age", 0, 120, 25)
fever = st.number_input("🌡️ Fever (°F)", 90, 110, 98)
bp = st.number_input("💓 Blood Pressure", 60, 200, 120)
sugar = st.number_input("🧪 Sugar Level", 50, 300, 110)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Disease 🧠"):
    patient_data = [[age, fever, bp, sugar]]
    prediction = model.predict(patient_data)

    if prediction[0] == 1:
        st.error("🚨 Disease Detected ❗ Please consult a doctor 🏥")
    else:
        st.success("💚 No Disease Detected 🎉 Stay Healthy!")
