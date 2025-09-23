import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from PIL import Image

# Optional imports (graceful fallback)
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
try:
    import speech_recognition as sr
except ImportError:
    sr = None

# --------------------------------------------------
# Multilingual dictionary (English, Hindi, Punjabi)
# --------------------------------------------------
translations = {
    "en": {
        "soil_health": "Soil Health Analysis",
        "weather": "Weather Forecast & Advice",
        "pest": "Pest/Disease Detection",
        "market": "Market Price Tracking",
        "feedback": "Feedback",
        "submit": "Submit",
        "fertilizer_suggestion": "Suggested Fertilizer Guidance:",
        "upload_image": "Upload an image of crop/pest:",
        "detected_pest": "Detected Pest/Disease:",
        "no_pest": "No pest detected or model not available.",
        "market_input": "Enter crop and price data:",
        "feedback_prompt": "Please provide your feedback:",
        "thank_you": "Thank you for your feedback!"
    },
    "hi": {
        "soil_health": "मृदा स्वास्थ्य विश्लेषण",
        "weather": "मौसम पूर्वानुमान और सलाह",
        "pest": "कीट/रोग पहचान",
        "market": "बाजार मूल्य ट्रैकिंग",
        "feedback": "प्रतिपुष्टि",
        "submit": "जमा करें",
        "fertilizer_suggestion": "अनुशंसित उर्वरक मार्गदर्शन:",
        "upload_image": "फसल/कीट की छवि अपलोड करें:",
        "detected_pest": "पहचाना गया कीट/रोग:",
        "no_pest": "कोई कीट नहीं पहचाना गया या मॉडल उपलब्ध नहीं है।",
        "market_input": "फसल और मूल्य डेटा दर्ज करें:",
        "feedback_prompt": "कृपया अपनी प्रतिपुष्टि दें:",
        "thank_you": "आपकी प्रतिपुष्टि के लिए धन्यवाद!"
    },
    "pa": {
        "soil_health": "ਮਿੱਟੀ ਦੀ ਸਿਹਤ ਵਿਸ਼ਲੇਸ਼ਣ",
        "weather": "ਮੌਸਮ ਪੂਰਬ-ਅਨੁਮਾਨ ਅਤੇ ਸਲਾਹ",
        "pest": "ਕੀੜਾ/ਰੋਗ ਪਛਾਣ",
        "market": "ਬਾਜ਼ਾਰ ਕੀਮਤ ਟ੍ਰੈਕਿੰਗ",
        "feedback": "ਫੀਡਬੈਕ",
        "submit": "ਜਮ੍ਹਾ ਕਰੋ",
        "fertilizer_suggestion": "ਸਿਫ਼ਾਰਸ਼ੀ ਖਾਦ ਮਾਰਗਦਰਸ਼ਨ:",
        "upload_image": "ਫਸਲ/ਕੀੜੇ ਦੀ ਤਸਵੀਰ ਅਪਲੋਡ ਕਰੋ:",
        "detected_pest": "ਪਛਾਣਿਆ ਕੀੜਾ/ਰੋਗ:",
        "no_pest": "ਕੋਈ ਕੀੜਾ ਨਹੀਂ ਪਛਾਣਿਆ ਗਿਆ ਜਾਂ ਮਾਡਲ ਉਪਲਬਧ ਨਹੀਂ ਹੈ।",
        "market_input": "ਫਸਲ ਅਤੇ ਕੀਮਤ ਡਾਟਾ ਦਰਜ ਕਰੋ:",
        "feedback_prompt": "ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਫੀਡਬੈਕ ਦਿਓ:",
        "thank_you": "ਤੁਹਾਡੇ ਫੀਡਬੈਕ ਲਈ ਧੰਨਵਾਦ!"
    }
}

# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def get_text(lang, key):
    return translations.get(lang, translations["en"]).get(key, key)

# --------------------------------------------------
# App Layout
# --------------------------------------------------
st.set_page_config(page_title="Krishi.AI", layout="wide")
st.title("🌾 Krishi.AI - Your Personal Crop Advisor")

# Language selection
lang = st.sidebar.selectbox("Select Language / भाषा चुनें / ਭਾਸ਼ਾ ਚੁਣੋ", ("en", "hi", "pa"))

# Tabs
soil_tab, weather_tab, pest_tab, market_tab, feedback_tab = st.tabs([
    get_text(lang, "soil_health"),
    get_text(lang, "weather"),
    get_text(lang, "pest"),
    get_text(lang, "market"),
    get_text(lang, "feedback")
])

# --------------------------------------------------
# Soil Health Tab
# --------------------------------------------------
with soil_tab:
    st.subheader(get_text(lang, "soil_health"))
    ph = st.slider("Soil pH", 0.0, 14.0, 7.0)
    nitrogen = st.number_input("Nitrogen level (kg/ha)", 0, 500, 50)
    phosphorus = st.number_input("Phosphorus level (kg/ha)", 0, 500, 40)
    potassium = st.number_input("Potassium level (kg/ha)", 0, 500, 30)

    if st.button(get_text(lang, "submit"), key="soil_submit"):
        advice = []
        if ph < 6:
            advice.append("Soil is acidic, consider adding lime.")
        elif ph > 8:
            advice.append("Soil is alkaline, add organic matter.")

        if nitrogen < 50:
            advice.append("Add Nitrogen fertilizer (e.g., Urea).")
        if phosphorus < 40:
            advice.append("Add Phosphorus fertilizer (e.g., DAP).")
        if potassium < 30:
            advice.append("Add Potassium fertilizer (e.g., MOP).")

        st.write(get_text(lang, "fertilizer_suggestion"))
        for a in advice:
            st.write("-", a)

# --------------------------------------------------
# Weather Tab (Placeholder)
# --------------------------------------------------
with weather_tab:
    st.subheader(get_text(lang, "weather"))
    st.info("Weather API integration needed (e.g., OpenWeatherMap). Currently showing demo data.")
    fake_forecast = {"Temperature": "30°C", "Rainfall": "10mm", "Condition": "Cloudy"}
    st.json(fake_forecast)

    st.write("Advice: As rain is expected, avoid irrigation today.")

# --------------------------------------------------
# Pest Detection Tab (Image Upload)
# --------------------------------------------------
with pest_tab:
    st.subheader(get_text(lang, "pest"))
    img_file = st.file_uploader(get_text(lang, "upload_image"), type=["jpg", "png", "jpeg"])

    if img_file:
        img = Image.open(img_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write(get_text(lang, "detected_pest"))
        # Placeholder logic
        st.warning(get_text(lang, "no_pest"))

# --------------------------------------------------
# Market Price Tracking Tab
# --------------------------------------------------
with market_tab:
    st.subheader(get_text(lang, "market"))
    crop = st.text_input("Crop Name")
    price = st.number_input("Price per quintal (₹)", 0, 10000, 2000)
    if st.button(get_text(lang, "submit"), key="market_submit"):
        st.success(f"{crop} current price recorded: ₹{price}/quintal")

# --------------------------------------------------
# Feedback Tab
# --------------------------------------------------
with feedback_tab:
    st.subheader(get_text(lang, "feedback"))
    feedback = st.text_area(get_text(lang, "feedback_prompt"))
    if st.button(get_text(lang, "submit"), key="feedback_submit"):
        os.makedirs("data", exist_ok=True)
        feedback_path = "data/feedback.json"
        feedback_data = []
        if os.path.exists(feedback_path):
            with open(feedback_path, "r") as f:
                feedback_data = json.load(f)
        feedback_data.append({"feedback": feedback})
        with open(feedback_path, "w") as f:
            json.dump(feedback_data, f)
        st.success(get_text(lang, "thank_you"))