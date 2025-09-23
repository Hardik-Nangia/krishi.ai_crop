"""
Krishi.AI - Enhanced Streamlit Smart Crop Advisory (app.py)

Features:
- Multilingual UI: English, Hindi, Punjabi
- Top heading: Krishi.AI - Your Personal Crop Advisor
- Soil analysis with charts
- Weather placeholder with API hook comments
- Image upload for pest/disease detection (model hook included)
- Market price tracking (CSV upload + manual)
- Feedback collection saved locally
- Text-to-Speech (pyttsx3 offline + gTTS online fallback)
- Speech-to-Text (SpeechRecognition via uploaded audio files)
- Simple, clean UI with CSS and layout improvements

How to run:
1. Save this file as app.py
2. Install dependencies:
   pip install streamlit pandas numpy pillow matplotlib gTTS pydub pyttsx3 SpeechRecognition
   (Some packages are optional; the app will gracefully fall back if missing.)
3. Run:
   streamlit run app.py

Notes:
- gTTS requires internet for TTS. pydub + simpleaudio or ffplay may be needed for playback on some systems.
- For pest-detection, provide a TF/Keras model at models/pest_model.h5 and a class_map.json mapping.

"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import io
import tempfile
import time
from datetime import datetime
from PIL import Image

# Optional/extended libs
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    from pydub import AudioSegment
    from pydub.playback import play as pydub_play
except Exception:
    AudioSegment = None
    pydub_play = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    tf = None
    TF_AVAILABLE = False

# ----------------------- App Config & Styling -----------------------
st.set_page_config(page_title="Krishi.AI", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for nicer look
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 100%); }
    .big-title {font-size:36px; font-weight:700; color:#0b6b3a;}
    .subtitle {font-size:14px; color:#2b6f4a}
    .card {background: white; padding: 16px; border-radius:10px; box-shadow: 0 2px 8px rgba(14,30,37,0.06);}
    .footer {color: #666; font-size:12px}
    </style>
    """,
    unsafe_allow_html=True,
)

# Top header
st.markdown('<div class="big-title">🌾 Krishi.AI - Your Personal Crop Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Actionable, localized advisory for small & marginal farmers — now multimodal.</div>', unsafe_allow_html=True)
st.markdown('---')

# ----------------------- Translations -----------------------
TRANSLATIONS = {
    'en': {
        'title': 'Krishi.AI - Smart Crop Advisor',
        'intro': 'Personalized, location-aware crop and soil advice for small farmers.',
        'soil_tab': 'Soil & Nutrients',
        'weather_tab': 'Weather',
        'pest_tab': 'Pest / Disease',
        'market_tab': 'Market Prices',
        'feedback_tab': 'Feedback',
        'ph': 'Soil pH',
        'nitrogen': 'Nitrogen (kg/ha)',
        'phosphorus': 'Phosphorus (kg/ha)',
        'potassium': 'Potassium (kg/ha)',
        'analyze': 'Analyze Soil',
        'tts': 'Text-to-Speech Engine',
        'play_advice': 'Play Advice Audio',
        'stt': 'Upload Voice Note (wav/mp3) for Transcription',
        'upload_image': 'Upload crop leaf image',
        'detect': 'Detect Pest/Disease',
        'upload_market_csv': 'Upload market CSV (crop,mandi,price,date)',
        'save_price': 'Save Manual Price',
        'feedback_prompt': 'Share feedback or a field report',
        'submit_feedback': 'Submit Feedback',
    },
    'hi': {
        'title': 'Krishi.AI - स्मार्ट फसल सलाहकार',
        'intro': 'छोटे किसानों के लिए व्यक्तिगत, स्थान-संवेदी फसल और मिट्टी सलाह।',
        'soil_tab': 'मिट्टी और पोषक तत्व',
        'weather_tab': 'मौसम',
        'pest_tab': 'कीट / रोग',
        'market_tab': 'बाज़ार मूल्य',
        'feedback_tab': 'प्रतिक्रिया',
        'ph': 'मिट्टी का pH',
        'nitrogen': 'नाइट्रोजन (kg/ha)',
        'phosphorus': 'फॉस्फोरस (kg/ha)',
        'potassium': 'पोटैशियम (kg/ha)',
        'analyze': 'मिट्टी विश्लेषण',
        'tts': 'टेक्स्ट-टू-स्पीच इंजन',
        'play_advice': 'सलाह ऑडियो चलाएँ',
        'stt': 'ट्रांसक्रिप्शन के लिए वॉइस नोट अपलोड करें (wav/mp3)',
        'upload_image': 'फसल की पत्ती की छवि अपलोड करें',
        'detect': 'कीट/रोग पहचानें',
        'upload_market_csv': 'बाज़ार CSV अपलोड करें (crop,mandi,price,date)',
        'save_price': 'मैन्युअल कीमत सहेजें',
        'feedback_prompt': 'अपनी प्रतिक्रिया या खेत की रिपोर्ट साझा करें',
        'submit_feedback': 'प्रतिक्रिया भेजें',
    },
    'pa': {
        'title': 'Krishi.AI - ਸਮਾਰਟ ਫ਼ਸਲ ਸਲਾਹਕਾਰ',
        'intro': 'ਛੋਟੇ ਕਿਸਾਨਾਂ ਲਈ ਨਿੱਜੀ, ਸਥਾਨਕ-ਅਧਾਰਤ ਫਸਲ ਅਤੇ ਮਿੱਟੀ ਸਲਾਹ।',
        'soil_tab': 'ਮਿੱਟੀ ਅਤੇ ਪੋਸ਼ਕ ਤੱਤ',
        'weather_tab': 'ਮੌਸਮ',
        'pest_tab': 'ਕੀਟ / ਰੋਗ',
        'market_tab': 'ਬਾਜ਼ਾਰ ਕੀਮਤਾਂ',
        'feedback_tab': 'ਫੀਡਬੈਕ',
        'ph': 'ਮਿੱਟੀ pH',
        'nitrogen': 'ਨਾਈਟ੍ਰੋਜਨ (kg/ha)',
        'phosphorus': 'ਫਾਸਫੋਰਸ (kg/ha)',
        'potassium': 'ਪੋਟਾਸ਼ੀਅਮ (kg/ha)',
        'analyze': 'ਮਿੱਟੀ ਵਿਸ਼ਲੇਸ਼ਣ',
        'tts': 'ਟੈਕਸਟ-ਟੂ-ਸਪੀਚ ਇੰਜਣ',
        'play_advice': 'ਸਲਾਹ ਆਡੀਓ ਚਲਾਓ',
        'stt': 'ਟ੍ਰਾਂਸਕ੍ਰਿਪਸ਼ਨ ਲਈ ਵੌਇਸ ਨੋਟ ਅਪਲੋਡ ਕਰੋ (wav/mp3)',
        'upload_image': 'ਫਸਲ ਦੇ ਪੱਤੇ ਦੀ ਤਸਵੀਰ ਅਪਲੋਡ ਕਰੋ',
        'detect': 'ਕੀਟ/ਰੋਗ ਪਛਾਣੋ',
        'upload_market_csv': 'ਬਾਜ਼ਾਰ CSV ਅਪਲੋਡ ਕਰੋ (crop,mandi,price,date)',
        'save_price': 'ਮੈਨੁਅਲ ਕੀਮਤ ਸੇਵ ਕਰੋ',
        'feedback_prompt': 'ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਫੀਡਬੈਕ ਭੇਜੋ',
        'submit_feedback': 'ਫੀਡਬੈਕ ਭੇਜੋ',
    }
}

# ----------------------- Helpers -----------------------

def tr(key):
    return TRANSLATIONS.get(LANG, TRANSLATIONS['en']).get(key, key)


def ensure_data_dir():
    os.makedirs('data', exist_ok=True)


# ----------------------- Sidebar Controls -----------------------
with st.sidebar:
    LANG = st.selectbox('Language / भाषा / ਭਾਸ਼ਾ', options=['en', 'hi', 'pa'], format_func=lambda x: 'English' if x=='en' else ('Hindi (हिंदी)' if x=='hi' else 'Punjabi (ਪੰਜਾਬੀ)'))
    st.write('<br>', unsafe_allow_html=True)
    st.subheader('Multimodal')
    enable_tts_pyttsx3 = st.checkbox('Enable pyttsx3 (offline TTS)', value=True if pyttsx3 else False)
    enable_gtts = st.checkbox('Enable gTTS (online TTS)', value=True if gTTS else False)
    enable_stt = st.checkbox('Enable Speech-to-Text (upload audio)', value=True if sr else False)
    st.markdown('---')
    st.write('Developer notes: connect OpenWeather / Agmarknet APIs for real data.')

# Update tr function to use chosen language
TRANSLATIONS = TRANSLATIONS  # keep reference

# ----------------------- Layout Tabs -----------------------
soil_tab, weather_tab, pest_tab, market_tab, feedback_tab = st.tabs([
    tr('soil_tab'), tr('weather_tab'), tr('pest_tab'), tr('market_tab'), tr('feedback_tab')
])

# ----------------------- Soil Tab -----------------------
with soil_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header(tr('soil_tab'))
    col1, col2 = st.columns([2,1])
    with col1:
        pH = st.slider(tr('ph'), 0.0, 14.0, 6.5)
        nitrogen = st.number_input(tr('nitrogen'), min_value=0, max_value=1000, value=120)
        phosphorus = st.number_input(tr('phosphorus'), min_value=0, max_value=1000, value=40)
        potassium = st.number_input(tr('potassium'), min_value=0, max_value=1000, value=150)
        crop_name = st.text_input('Crop (optional)', value='Wheat')
        if st.button(tr('analyze')):
            advice = []
            if pH < 5.5:
                advice.append('Acidic soil — consider liming (apply agricultural lime).')
            elif pH > 7.8:
                advice.append('Alkaline soil — consider organic matter and gypsum where appropriate.')
            else:
                advice.append('pH in good range for many crops; maintain organic matter.')

            # NPK heuristics
            if nitrogen < 100:
                advice.append('Nitrogen low: consider split applications of N (e.g., urea).')
            else:
                advice.append('Nitrogen adequate.')
            if phosphorus < 40:
                advice.append('Phosphorus low: apply basal P (SSP/DAP).')
            else:
                advice.append('Phosphorus adequate.')
            if potassium < 100:
                advice.append('Potassium low: apply MOP or SOP depending on crop.')
            else:
                advice.append('Potassium adequate.')

            st.subheader('Recommendations')
            for r in advice:
                st.write('- ' + r)

            # Create nutrient bar chart
            df_n = pd.DataFrame({'NPK': ['N', 'P', 'K'], 'Value': [nitrogen, phosphorus, potassium]})
            st.bar_chart(data=df_n.set_index('NPK'))

            # Prepare textual summary for TTS
            tts_text = f"Advice for {crop_name}: " + ' '.join(advice)
            st.session_state['latest_advice'] = tts_text

    with col2:
        st.markdown('**Soil quick facts**')
        st.metric('pH', f'{pH:.1f}')
        st.metric('Nitrogen', f'{nitrogen} kg/ha')
        st.metric('Phosphorus', f'{phosphorus} kg/ha')
        st.metric('Potassium', f'{potassium} kg/ha')

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Weather Tab -----------------------
with weather_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header(tr('weather_tab'))
    st.info('Weather placeholder. Replace with real API (OpenWeatherMap / Open-Meteo).')
    col1, col2 = st.columns(2)
    with col1:
        # Simulated summary
        weather_summary = {'Date': datetime.now().strftime('%Y-%m-%d'), 'Temp_C': 30, 'RainChance_%': 20, 'Condition': 'Partly Cloudy'}
        st.json(weather_summary)
    with col2:
        if weather_summary['RainChance_%'] > 60:
            st.warning('Heavy rain expected soon — delay spraying and ensure drainage.')
        else:
            st.success('No heavy rain expected in near forecast window.')

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Pest Detection Tab -----------------------
with pest_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header(tr('pest_tab'))
    st.write('Upload a clear image of the affected leaf or plant. For production, connect a trained model.')
    uploaded = st.file_uploader(tr('upload_image'), type=['jpg', 'jpeg', 'png'])
    model = None
    class_map = None
    # Attempt to load model and class map if present
    if TF_AVAILABLE:
        try:
            model_path = 'models/pest_model.h5'
            map_path = 'models/class_map.json'
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
            if os.path.exists(map_path):
                with open(map_path, 'r') as f:
                    class_map = json.load(f)
        except Exception as e:
            st.warning(f'Could not load model: {e}')

    if uploaded is not None:
        image = Image.open(uploaded).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        if st.button(tr('detect')):
            with st.spinner('Analyzing...'):
                if model is not None:
                    try:
                        img = image.resize((224,224))
                        arr = np.array(img)/255.0
                        arr = np.expand_dims(arr, axis=0)
                        preds = model.predict(arr)
                        idx = int(np.argmax(preds[0]))
                        label = class_map.get(str(idx), f'class_{idx}') if class_map else f'class_{idx}'
                        conf = float(np.max(preds[0]))
                        st.success(f'Detected: {label} (confidence {conf:.2f})')
                        st.session_state['latest_advice'] = f"Detected {label} with confidence {conf:.2f}. Please isolate and consult extension services."
                    except Exception as e:
                        st.error(f'Model inference failed: {e}')
                else:
                    st.info('No model available in models/pest_model.h5 — this is a demo result.')
                    st.warning('Possible fungal infection. Isolate affected plants and consult local agri-officer.')
                    st.session_state['latest_advice'] = 'Possible fungal infection. Isolate affected plants and consult local extension.'

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Market Tab -----------------------
with market_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header(tr('market_tab'))
    uploaded_csv = st.file_uploader(tr('upload_market_csv'), type=['csv'])
    if uploaded_csv:
        try:
            df_market = pd.read_csv(uploaded_csv)
            st.write('Preview:')
            st.dataframe(df_market.head())
            # show latest prices per crop
            if 'date' in df_market.columns:
                df_market['date'] = pd.to_datetime(df_market['date'], errors='coerce')
            latest = df_market.sort_values('date').groupby('crop').tail(1) if 'date' in df_market.columns else df_market
            st.subheader('Latest prices')
            st.table(latest[['crop','mandi','price']].reset_index(drop=True))
        except Exception as e:
            st.error(f'Failed to parse CSV: {e}')

    st.markdown('**Manual entry**')
    c1, c2, c3 = st.columns([2,2,1])
    with c1:
        crop = st.text_input('Crop name')
    with c2:
        mandi = st.text_input('Mandi / Market')
    with c3:
        price = st.number_input('Price (₹/quintal)', min_value=0.0, value=0.0)
    if st.button(tr('save_price')):
        ensure_data_dir()
        pfile = os.path.join('data','market_manual.csv')
        row = {'crop': crop, 'mandi': mandi, 'price': price, 'date': datetime.now().isoformat()}
        dfrow = pd.DataFrame([row])
        if os.path.exists(pfile):
            dfrow.to_csv(pfile, mode='a', header=False, index=False)
        else:
            dfrow.to_csv(pfile, index=False)
        st.success('Saved price locally.')

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Feedback Tab (with STT & TTS) -----------------------
with feedback_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header(tr('feedback_tab'))
    feedback_text = st.text_area(tr('feedback_prompt'))

    # Speech-to-Text via uploaded audio
    if enable_stt:
        audio_file = st.file_uploader(tr('stt'), type=['wav','mp3','m4a'])
        if audio_file is not None:
            if sr is None:
                st.info('SpeechRecognition not installed — enable SR to transcribe audio.')
            else:
                try:
                    recognizer = sr.Recognizer()
                    audio_bytes = audio_file.read()
                    with tempfile.NamedTemporaryFile(delete=False) as tmpf:
                        tmpf.write(audio_bytes)
                        tmp_filename = tmpf.name
                    with sr.AudioFile(tmp_filename) as source:
                        audio_data = recognizer.record(source)
                        text = recognizer.recognize_google(audio_data, language='en-IN')
                        st.success('Transcription:')
                        st.write(text)
                        feedback_text = (feedback_text + '\n' + text) if feedback_text else text
                except Exception as e:
                    st.error(f'Audio transcription failed: {e}')

    # Play the latest advice with TTS
    if 'latest_advice' in st.session_state:
        st.subheader('Latest generated advisory (click to play)')
        st.write(st.session_state['latest_advice'])
        tts_choice = None
        if enable_tts_pyttsx3 and pyttsx3:
            tts_choice = 'pyttsx3'
        elif enable_gtts and gTTS:
            tts_choice = 'gTTS'

        if tts_choice:
            if st.button(tr('play_advice')):
                text_to_speak = st.session_state['latest_advice']
                if tts_choice == 'pyttsx3' and pyttsx3:
                    try:
                        engine = pyttsx3.init()
                        engine.say(text_to_speak)
                        engine.runAndWait()
                        st.success('Played via pyttsx3 (offline).')
                    except Exception as e:
                        st.error(f'pyttsx3 playback failed: {e}')
                elif tts_choice == 'gTTS' and gTTS:
                    try:
                        tts = gTTS(text_to_speak, lang='en')
                        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                        tts.save(tmpfile.name)
                        # play using pydub if available
                        if AudioSegment and pydub_play:
                            audio_seg = AudioSegment.from_file(tmpfile.name, format='mp3')
                            pydub_play(audio_seg)
                        else:
                            st.info(f'Audio saved to {tmpfile.name} - download and play locally.')
                    except Exception as e:
                        st.error(f'gTTS failed: {e}')
        else:
            st.info('No TTS engine enabled or installed. Enable pyttsx3 or gTTS in sidebar.')

    # Submit feedback
    if st.button(tr('submit_feedback')):
        ensure_data_dir()
        fpath = os.path.join('data','feedback.json')
        arr = []
        if os.path.exists(fpath):
            try:
                with open(fpath,'r',encoding='utf-8') as f:
                    arr = json.load(f)
            except Exception:
                arr = []
        entry = {'text': feedback_text, 'timestamp': datetime.now().isoformat(), 'language': LANG}
        arr.append(entry)
        with open(fpath,'w',encoding='utf-8') as f:
            json.dump(arr, f, ensure_ascii=False, indent=2)
        st.success('Thank you — feedback saved.')

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- Footer -----------------------
st.markdown('---')
st.markdown('<div class="footer">Built as a prototype. To productionize: add authentication, encrypted storage, real weather and market APIs (OpenWeather, Agmarknet), and deploy a trained pest-disease model.</div>', unsafe_allow_html=True)
