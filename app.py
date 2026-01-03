import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="Spotify Hit Prediction",
    page_icon="🎵",
    layout="centered"
)

# Title
st.title("🎵 Spotify Song Hit Prediction")
st.write("Predict whether a song will be a **HIT** or **NOT A HIT** based on audio features.")

# Load model and scaler
model = joblib.load(open("model.pkl", "rb"))
scaler = joblib.load(open("scaler.pkl", "rb"))

st.subheader("🎧 Enter Song Features")

# Input fields
danceability = st.number_input("Danceability", 0.0, 1.0, 0.5)
energy = st.number_input("Energy", 0.0, 1.0, 0.5)
key = st.number_input("Key (0–11)", 0, 11, 5)
loudness = st.number_input("Loudness (dB)", -60.0, 0.0, -10.0)
mode = st.selectbox("Mode", [0, 1])
speechiness = st.number_input("Speechiness", 0.0, 1.0, 0.1)
acousticness = st.number_input("Acousticness", 0.0, 1.0, 0.5)
instrumentalness = st.number_input("Instrumentalness", 0.0, 1.0, 0.0)
liveness = st.number_input("Liveness", 0.0, 1.0, 0.1)
valence = st.number_input("Valence", 0.0, 1.0, 0.5)
tempo = st.number_input("Tempo (BPM)", 0.0, 250.0, 120.0)
duration_ms = st.number_input("Duration (ms)", 30000, 600000, 180000)
time_signature = st.number_input("Time Signature", 1, 7, 4)
chorus_hit = st.number_input("Chorus Hit (seconds)", 0.0, 60.0, 30.0)
sections = st.number_input("Number of Sections", 0, 50, 10)

# Predict button
if st.button("🔮 Predict"):

    input_data = np.array([[
        danceability, energy, key, loudness, mode,
        speechiness, acousticness, instrumentalness,
        liveness, valence, tempo, duration_ms,
        time_signature, chorus_hit, sections
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.success("HIT")
    else:
        st.error("NOT HIT")
