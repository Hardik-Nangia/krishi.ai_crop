"""
Smart Crop Advisory System - Streamlit App
File: streamlit_smart_crop_advisory.py

Description:
A self-contained Streamlit prototype for a multilingual smart crop advisory system aimed at
small and marginal farmers. This app is intentionally modular and uses local fallbacks so it
will run without external APIs. Sections include:
 - Multilingual UI (English + Hindi sample)
 - Location input and (optional) weather fetch placeholder (OpenWeatherMap stub)
 - Soil health recommendations (pH, NPK heuristics)
 - Pest/disease detection stub via image upload (optional TensorFlow model if provided)
 - Market price tracking via CSV upload or manual input
 - Voice support: basic TTS (pyttsx3) and optional audio -> text using SpeechRecognition
 - Feedback collection saved locally to data/feedback.json

How to run:
1. Install dependencies:
   pip install streamlit pandas numpy pillow scikit-learn
   Optional (for image ML and audio features): tensorflow speechrecognition pydub pyttsx3
2. Run:
   streamlit run streamlit_smart_crop_advisory.py

Notes:
 - This is a prototype. Replace the weather and market-data placeholders with real API calls
   (examples in comments) when you have API keys (OpenWeatherMap / Agmarknet / local mandi API).
 - For pest/disease detection, train a classifier and place model at ./models/pest_model.h5
 - Voice transcription depends on system audio drivers and installed packages.

"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from PIL import Image
import io
import random

# Optional imports: wrap in try/except so app runs without them
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# -------------------------- Utility functions --------------------------

DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.json")

def save_feedback(entry: dict):
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            arr = json.load(f)
    else:
        arr = []
    arr.append(entry)
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)

# Simple translation dictionary (extendable)
TRANSLATIONS = {
    "en": {
        "title": "Smart Crop Advisory",
        "intro": "Personalized, location-aware crop and soil advice for small farmers.",
        "location": "Location (city / lat,lon)",
        "soil_input": "Soil parameters (enter measured values)",
        "ph": "pH",
        "nitrogen": "Nitrogen (kg/ha)",
        "phosphorus": "Phosphorus (kg/ha)",
        "potassium": "Potassium (kg/ha)",
        "analyze_soil": "Analyze Soil",
        "recommendations": "Recommendations",
        "weather": "Weather (placeholder)",
        "pest_detect": "Pest/Disease Detection (image)",
        "upload_image": "Upload crop leaf image",
        "detect": "Detect",
        "market": "Market Price Tracking",
        "upload_market_csv": "Upload market CSV (columns: crop, mandi, price, date)",
        "voice_input": "Voice Input / Output",
        "feedback": "Feedback",
        "submit_feedback": "Submit Feedback"
    },
    "hi": {
        "title": "स्मार्ट फ़सल परामर्श",
        "intro": "छोटे किसानों के लिए व्यक्तिगत, स्थान-आधारित फ़सल और मिट्टी सलाह।",
        "location": "स्थान (शहर / अक्षांश,देशांतर)",
        "soil_input": "मिट्टी के पैरामीटर (मापे गए मान दर्ज करें)",
        "ph": "पीएच",
        "nitrogen": "नाइट्रोजन (kg/ha)",
        "phosphorus": "फॉस्फोरस (kg/ha)",
        "potassium": "पोटैशियम (kg/ha)",
        "analyze_soil": "मिट्टी विश्लेषण",
        "recommendations": "सिफारिशें",
        "weather": "मौसम (नमूना)",
        "pest_detect": "कीट/रोग पहचान (छवि)",
        "upload_image": "फसल के पत्ती की छवि अपलोड करें",
        "detect": "पहचानें",
        "market": "बाजार मूल्य ट्रैकिंग",
        "upload_market_csv": "बाज़ार CSV अपलोड करें (स्तम्भ: crop, mandi, price, date)",
        "voice_input": "आवाज़ इनपुट / आउटपुट",
        "feedback": "प्रतिक्रिया",
        "submit_feedback": "प्रतिक्रिया भेजें"
    }
}

# -------------------------- Soil Recommendation Logic --------------------------

def soil_recommendation(ph, n, p, k, crop=None):
    """Return textual soil health and fertilizer recommendations based on simple heuristics."""
    recs = []
    # pH guidance
    if ph is None:
        recs.append("pH not provided — please test soil pH for accurate recommendations.")
    else:
        if ph < 5.5:
            recs.append("Acidic soil: apply agricultural lime (calculate dose based on buffer pH). Consider 1–2 t/ha as starting guidance and consult local extension.")
        elif ph <= 7.5:
            recs.append("pH is in favorable range for most crops. Maintain organic matter and avoid over-application of acidic fertilizers.")
        else:
            recs.append("Alkaline soil: gypsum and organic matter can help; consider acidifying fertilizers like ammonium sulfate if needed.")

    # NPK guidance (these are illustrative thresholds, replace with local recommendations)
    if n is not None:
        if n < 100:
            recs.append("Nitrogen low: consider applying urea or DAP-based N sources. Split applications for long-duration crops.")
        elif n > 250:
            recs.append("Nitrogen high: reduce N application; consider legume rotation to utilize excess nitrogen biologically.")
        else:
            recs.append("Nitrogen adequate — follow crop-specific schedule.")
    if p is not None:
        if p < 40:
            recs.append("Phosphorus low: apply single super phosphate (SSP) or DAP as basal dose.")
        else:
            recs.append("Phosphorus adequate.")
    if k is not None:
        if k < 100:
            recs.append("Potassium low: apply muriate of potash (MOP) or sulphate of potash depending on crop tolerance.")
        else:
            recs.append("Potassium adequate.")

    # Crop-specific hints (simple)
    if crop:
        crop = crop.lower()
        if "rice" in crop:
            recs.append("Rice: maintain puddled soil for transplanting; nitrogen split application at tillering and panicle initiation.")
        if "wheat" in crop:
            recs.append("Wheat: apply phosphorus at sowing and split nitrogen at tillering and pre-flowering.")

    return recs

# -------------------------- Weather placeholder --------------------------

def fetch_weather_placeholder(lat=None, lon=None):
    """Return a simulated weather forecast. Replace this function to call a real API (OpenWeatherMap, etc.)."""
    choices = [
        {"desc": "Dry, sunny", "temp_c": 34, "rain_prob": 5},
        {"desc": "Partly cloudy", "temp_c": 28, "rain_prob": 20},
        {"desc": "Heavy rain expected", "temp_c": 24, "rain_prob": 85},
        {"desc": "Overnight frost possible (cold)", "temp_c": 2, "rain_prob": 0}
    ]
    return random.choice(choices)

# -------------------------- Pest detection stub --------------------------

def load_pest_model(path="models/pest_model.h5"):
    if TF_AVAILABLE and os.path.exists(path):
        try:
            model = tf.keras.models.load_model(path)
            return model
        except Exception as e:
            st.warning(f"Failed to load model: {e}")
            return None
    return None


def detect_pest_from_image(img_bytes, model=None):
    # If model provided, use it. Otherwise fallback to simple heuristics / random suggestions.
    if model is not None and TF_AVAILABLE:
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
            arr = np.array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            preds = model.predict(arr)
            # This assumes model outputs probabilities for classes; adjust as per your model.
            idx = np.argmax(preds[0])
            # Placeholder class names - replace with real mapping
            class_names = ["Healthy", "Blast", "Brown Spot", "Rust"]
            label = class_names[idx] if idx < len(class_names) else f"class_{idx}"
            confidence = float(np.max(preds[0]))
            return {"label": label, "confidence": confidence}
        except Exception as e:
            return {"label": "unknown", "confidence": 0.0, "error": str(e)}
    # Fallback
    labels = ["Healthy", "Possible fungal infection (suggest lab test)", "Possible pest damage (aphids/borer)"]
    choice = random.choice(labels)
    return {"label": choice, "confidence": round(random.uniform(0.6, 0.95), 2)}

# -------------------------- Market price helpers --------------------------

def summarize_market(csv_df: pd.DataFrame):
    # Expect columns: crop, mandi, price, date
    if csv_df.empty:
        return None
    csv_df['date'] = pd.to_datetime(csv_df['date'], errors='coerce')
    latest = csv_df.sort_values('date').groupby('crop').tail(1)
    return latest[['crop', 'mandi', 'price', 'date']]

# -------------------------- Streamlit App Layout --------------------------

st.set_page_config(page_title="Smart Crop Advisory", layout="wide")

# Sidebar - language and user info
with st.sidebar:
    lang = st.selectbox("Language / भाषा", options=["en", "hi"], format_func=lambda x: "English" if x=="en" else "Hindi (हिंदी)")
    t = TRANSLATIONS[lang]
    st.title(t['title'])
    st.write(t['intro'])
    user_name = st.text_input("Your name / आपका नाम", value="")
    location = st.text_input(t['location'], value="Enter city or lat,lon")
    st.markdown("---")
    st.header(t['voice_input'])
    if TTS_AVAILABLE:
        st.write("Text-to-speech available (pyttsx3 detected).")
    else:
        st.write("Text-to-speech not available — install pyttsx3 for local TTS.")
    if SR_AVAILABLE:
        st.write("Audio transcription available (SpeechRecognition installed).")
    else:
        st.write("Audio transcription not available — install SpeechRecognition + Pydub for audio uploads.")

# Main
col1, col2 = st.columns([2,1])

with col1:
    st.header(t['soil_input'])
    ph = st.number_input(t['ph'], min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    n = st.number_input(t['nitrogen'], min_value=0.0, value=120.0, step=1.0)
    p = st.number_input(t['phosphorus'], min_value=0.0, value=40.0, step=1.0)
    k = st.number_input(t['potassium'], min_value=0.0, value=150.0, step=1.0)
    crop_guess = st.text_input("Crop (optional)", value="Wheat")
    if st.button(t['analyze_soil']):
        recs = soil_recommendation(ph, n, p, k, crop_guess)
        st.subheader(t['recommendations'])
        for r in recs:
            st.write("- " + r)

    st.markdown("---")
    st.header(t['pest_detect'])
    uploaded_img = st.file_uploader(t['upload_image'], type=['png','jpg','jpeg'])
    pest_model = load_pest_model()
    if uploaded_img is not None:
        img_bytes = uploaded_img.read()
        img = Image.open(io.BytesIO(img_bytes))
        st.image(img, caption="Uploaded image", use_column_width=True)
        if st.button(t['detect']):
            with st.spinner("Detecting..."):
                result = detect_pest_from_image(img_bytes, model=pest_model)
                st.success(f"Result: {result.get('label')} (confidence: {result.get('confidence')})")
                if result.get('label') != 'Healthy':
                    st.info("Suggested action: isolate affected plants, consult extension officer, and if possible send sample to lab.")

    st.markdown("---")
    st.header(t['market'])
    market_csv = st.file_uploader(t['upload_market_csv'], type=['csv'])
    if market_csv is not None:
        try:
            df_market = pd.read_csv(market_csv)
            st.write("Preview:")
            st.dataframe(df_market.head())
            summary = summarize_market(df_market)
            if summary is not None:
                st.subheader("Latest prices by crop")
                st.table(summary)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    else:
        st.info("You can upload a CSV of recent mandi prices, or paste prices in the form below.")
        crop_manual = st.text_input("Crop for manual price entry")
        mandi_manual = st.text_input("Mandi / Market")
        price_manual = st.number_input("Price (₹/quintal)", min_value=0.0, value=0.0)
        if st.button("Save price"):
            row = {'crop': crop_manual, 'mandi': mandi_manual, 'price': price_manual, 'date': datetime.now().isoformat()}
            # save to local file
            pfile = os.path.join(DATA_DIR, "market_manual.csv")
            df = pd.DataFrame([row])
            if os.path.exists(pfile):
                df.to_csv(pfile, mode='a', header=False, index=False)
            else:
                df.to_csv(pfile, index=False)
            st.success("Saved price locally.")

with col2:
    st.header(t['weather'])
    st.write("Weather data — replace placeholder with real API responses if you have an API key.")
    if st.button("Get weather (placeholder)"):
        w = fetch_weather_placeholder()
        st.write(f"Forecast: {w['desc']}")
        st.write(f"Temperature: {w['temp_c']} °C")
        st.write(f"Chance of rain: {w['rain_prob']} %")
        if w['rain_prob'] > 60:
            st.warning("Heavy rain expected — consider delaying fertilizer/pesticide spray and ensure drainage.")

    st.markdown("---")
    st.header(t['voice_input'])
    st.write("Type text and play as audio (local TTS), or upload an audio file for transcription (if supported).")
    text_for_tts = st.text_area("Text to speak", value=f"Hello {user_name or 'Farmer'}, here is your advisory summary.")
    if st.button("Play TTS"):
        if TTS_AVAILABLE:
            try:
                engine = pyttsx3.init()
                engine.say(text_for_tts)
                engine.runAndWait()
                st.success("Played via local TTS (pyttsx3).")
            except Exception as e:
                st.error(f"TTS failed: {e}")
        else:
            st.info("Install pyttsx3 to enable local TTS. Or use platform TTS utilities.")

    audio_file = st.file_uploader("Upload voice note (wav/mp3) for transcription", type=['wav','mp3','m4a'])
    if audio_file is not None:
        if SR_AVAILABLE:
            try:
                r = sr.Recognizer()
                with sr.AudioFile(io.BytesIO(audio_file.read())) as source:
                    audio = r.record(source)
                text = r.recognize_google(audio, language='en-IN')
                st.write("Transcription:")
                st.write(text)
            except Exception as e:
                st.error(f"Transcription failed: {e}")
        else:
            st.info("SpeechRecognition not available — install SpeechRecognition and Pydub to enable transcription.")

    st.markdown("---")
    st.header(t['feedback'])
    feedback_text = st.text_area("Share your feedback or a report (in any language)")
    if st.button(t['submit_feedback']):
        entry = {
            'name': user_name,
            'location': location,
            'text': feedback_text,
            'timestamp': datetime.now().isoformat()
        }
        save_feedback(entry)
        st.success("Thanks — your feedback has been saved locally and will help improve the system.")

# -------------------------- Footer / developer notes --------------------------
st.markdown("---")
st.write("**Developer notes:** This app is a modular prototype. To productionize: connect to local agricultural databases for soil recommendations, integrate OpenWeather/Open-Meteo for weather, Agmarknet for mandi prices, and deploy a trained pest-disease classifier. Add user authentication and encrypted storage before collecting identifiable data.")

# Show path to feedback
if os.path.exists(FEEDBACK_FILE):
    st.info(f"Feedback stored at {FEEDBACK_FILE}")
else:
    st.info("No feedback saved yet.")
