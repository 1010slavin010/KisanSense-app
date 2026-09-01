
"""
KisanSense — farmer-friendly app (Streamlit)

Run locally:
    streamlit run app.py

Deploy:
    Push this folder to GitHub, then deploy on https://share.streamlit.io
    pointing at app.py. Add ANTHROPIC_API_KEY (or OPENAI_API_KEY) under the
    app's Settings -> Secrets for full chatbot answers — see README.md.
"""

import math
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from telemetry import get_latest_readings, get_hub_status, moisture_alerts, battery_alerts
from vision import analyze_image

st.set_page_config(page_title="KisanSense", page_icon="🌱", layout="wide")

BASE_DIR = Path(__file__).parent

# ==========================================================================
# Language / translations
# ==========================================================================
LANGUAGES = {
    "en": "English",
    "ta": "தமிழ்",
    "kn": "ಕನ್ನಡ",
    "tcy": "ತುಳು",
    "ml": "മലയാളം",
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"

STRINGS = {
    "nav_home": {"en": "🏠 Home", "ta": "🏠 முகப்பு", "kn": "🏠 ಮುಖಪುಟ", "tcy": "🏠 ಮುಖಪುರ್", "ml": "🏠 ഹോം"},
    "nav_myfarm": {"en": "🌾 My Farm", "ta": "🌾 என் பண்ணை", "kn": "🌾 ನನ್ನ ಜಮೀನು", "tcy": "🌾 ಎನ್ನ ಕುರ್ಲೆ", "ml": "🌾 എന്റെ കൃഷിയിടം"},
    "nav_crophealth": {"en": "🌿 Crop Health", "ta": "🌿 பயிர் ஆரோக்கியம்", "kn": "🌿 ಬೆಳೆ ಆರೋಗ್ಯ", "tcy": "🌿 ಬೆಳೆದ್ ಆರೋಗ್ಯೊ", "ml": "🌿 വിള ആരോഗ്യം"},
    "nav_irrigation": {"en": "💧 Irrigation", "ta": "💧 நீர்ப்பாசனம்", "kn": "💧 ನೀರಾವರಿ", "tcy": "💧 ನೀರ್ ಬುಡುನಿ", "ml": "💧 ജലസേചനം"},
    "nav_alerts": {"en": "⚠️ Alerts", "ta": "⚠️ எச்சரிக்கைகள்", "kn": "⚠️ ಎಚ್ಚರಿಕೆಗಳು", "tcy": "⚠️ ಎಚ್ಚರಿಕೆಲು", "ml": "⚠️ മുന്നറിയിപ്പുകൾ"},
    "nav_analytics": {"en": "📈 Analytics", "ta": "📈 பகுப்பாய்வு", "kn": "📈 ವಿಶ್ಲೇಷಣೆ", "tcy": "📈 ವಿಶ್ಲೇಷಣೆ", "ml": "📈 അനലിറ്റിക്സ്"},
    "nav_settings": {"en": "⚙️ Settings", "ta": "⚙️ அமைப்புகள்", "kn": "⚙️ ಸೆಟ್ಟಿಂಗ್‌ಗಳು", "tcy": "⚙️ ಸೆಟ್ಟಿಂಗ್ಸ್", "ml": "⚙️ ക്രമീകരണങ്ങൾ"},

    "welcome_home": {
        "en": "Namaste! Farmer, welcome back to KisanSense.",
        "ta": "வணக்கம்! விவசாயி, KisanSense-க்கு மீண்டும் வரவேற்கிறோம்.",
        "kn": "ನಮಸ್ತೆ! ರೈತರೇ, KisanSense ಗೆ ಮತ್ತೆ ಸ್ವಾಗತ.",
        "tcy": "ನಮಸ್ತೆ! ಕೃಷಿಕೆರೆ, KisanSense ಗ್ ಮತ್ತ್ ಸ್ವಾಗತ.",
        "ml": "നമസ്തേ! കർഷകാ, KisanSense-ലേക്ക് വീണ്ടും സ്വാഗതം.",
    },
    "status_line": {
        "en": "Your farm is being monitored in real time.",
        "ta": "உங்கள் பண்ணை நேரடியாகக் கண்காணிக்கப்படுகிறது.",
        "kn": "ನಿಮ್ಮ ಜಮೀನನ್ನು ನೈಜ ಸಮಯದಲ್ಲಿ ಮೇಲ್ವಿಚಾರಣೆ ಮಾಡಲಾಗುತ್ತಿದೆ.",
        "tcy": "ಈ ಪೊರ್ತುಡು ಒರಿ ಕುರ್ಲೆದ ಮೇಲ್ವಿಚಾರಣೆ ಆವೊಂದುಂಡು.",
        "ml": "നിങ്ങളുടെ കൃഷിയിടം തത്സമയം നിരീക്ഷിക്കപ്പെടുന്നു.",
    },
    "insights_title": {"en": "Insights", "ta": "நுண்ணறிவு", "kn": "ಒಳನೋಟಗಳು", "tcy": "ಒಳಪುದೆಲ್", "ml": "വിവരങ്ങൾ"},

    "quick_irrigate": {"en": "💧 Should I irrigate now?", "ta": "💧 இப்போது நீர்ப்பாசனம் செய்யலாமா?", "kn": "💧 ಈಗ ನೀರಾವರಿ ಮಾಡಬೇಕೆ?", "tcy": "💧 ಈಗ ನೀರ್ ಬುಡೊಡುಗಾ?", "ml": "💧 ഇപ്പോൾ നനയ്ക്കണോ?"},
    "quick_crop": {"en": "🌱 Is my crop healthy?", "ta": "🌱 என் பயிர் ஆரோக்கியமாக உள்ளதா?", "kn": "🌱 ನನ್ನ ಬೆಳೆ ಆರೋಗ್ಯವಾಗಿದೆಯೇ?", "tcy": "🌱 ಎನ್ನ ಬೆಳೆ ಆರೋಗ್ಯೊಡುಂಡಾ?", "ml": "🌱 എന്റെ വിള ആരോഗ്യമുള്ളതാണോ?"},
    "quick_pest": {"en": "🐛 How do I control pests?", "ta": "🐛 பூச்சிகளை எவ்வாறு கட்டுப்படுத்துவது?", "kn": "🐛 ಕೀಟಗಳನ್ನು ಹೇಗೆ ನಿಯಂತ್ರಿಸುವುದು?", "tcy": "🐛 ಕೀಟೊಲೆನ್ ಇಂಚ ನಿಯಂತ್ರಣ ಮಲ್ಪುನಿ?", "ml": "🐛 കീടങ്ങളെ എങ്ങനെ നിയന്ത്രിക്കാം?"},
    "quick_temp": {"en": "🌡 Is the temperature dangerous?", "ta": "🌡 வெப்பநிலை ஆபத்தானதா?", "kn": "🌡 ತಾಪಮಾನ ಅಪಾಯಕಾರಿಯೇ?", "tcy": "🌡 ಬಿಸಿ ಅಪಾಯೊಡುಂಡಾ?", "ml": "🌡 താപനില അപകടകരമാണോ?"},
    "quick_rain": {"en": "🌧 Is rain expected?", "ta": "🌧 மழை பெய்யுமா?", "kn": "🌧 ಮಳೆ ನಿರೀಕ್ಷಿಸಲಾಗಿದೆಯೇ?", "tcy": "🌧 ಮಳೆ ಬರುಗಾ?", "ml": "🌧 മഴ പ്രതീക്ഷിക്കുന്നുണ്ടോ?"},

    "reco_title": {"en": "🌾 Today's Recommendation", "ta": "🌾 இன்றைய பரிந்துரை", "kn": "🌾 ಇಂದಿನ ಶಿಫಾರಸು", "tcy": "🌾 ಇನಿತ್ತಿನ ಸಲಹೆ", "ml": "🌾 ഇന്നത്തെ ശുപാർശ"},
    "reco_irrig_title": {"en": "Irrigation recommended", "ta": "நீர்ப்பாசனம் பரிந்துரைக்கப்படுகிறது", "kn": "ನೀರಾವರಿ ಶಿಫಾರಸು ಮಾಡಲಾಗಿದೆ", "tcy": "ನೀರ್ ಬುಡುನಿ ಸಲಹೆ ಕೊರ್ಪುನಿ", "ml": "ജലസേചനം ശുപാർശ ചെയ്യുന്നു"},
    "reco_irrig_action": {"en": "Check irrigation for this field in the morning.", "ta": "காலையில் இந்த வயலுக்கு நீர்ப்பாசனத்தை சரிபார்க்கவும்.", "kn": "ಬೆಳಿಗ್ಗೆ ಈ ಹೊಲದ ನೀರಾವರಿಯನ್ನು ಪರಿಶೀಲಿಸಿ.", "tcy": "ಬಿರೆ ಈ ಗದ್ದೆದ ನೀರ್ ಬುಡುನಿನ್ ಪರಿಶೀಲನೆ ಮಲ್ಪುಲೆ.", "ml": "രാവിലെ ഈ പാടത്തെ ജലസേചനം പരിശോധിക്കുക."},
    "reco_heat_title": {"en": "Watch for heat stress", "ta": "வெப்ப அழுத்தத்தை கவனிக்கவும்", "kn": "ಶಾಖದ ಒತ್ತಡವನ್ನು ಗಮನಿಸಿ", "tcy": "ಬಿಸಿದ ಒತ್ತಡೊಗು ಗಮನ ಕೊರ್ಲೆ", "ml": "ചൂട് സമ്മർദ്ദം ശ്രദ്ധിക്കുക"},
    "reco_heat_action": {"en": "Consider shade netting or extra watering during peak heat hours.", "ta": "அதிக வெப்ப நேரங்களில் நிழல் வலை அல்லது கூடுதல் நீர்ப்பாசனத்தை பரிசீலிக்கவும்.", "kn": "ಗರಿಷ್ಠ ಬಿಸಿಲಿನ ಸಮಯದಲ್ಲಿ ನೆರಳು ಜಾಲರಿ ಅಥವಾ ಹೆಚ್ಚುವರಿ ನೀರಾವರಿ ಪರಿಗಣಿಸಿ.", "tcy": "಼ಜಾಸ್ತಿ ಬಿಸಿದ ಪೊರ್ತುಡು ನೆರೊಳಿ ಜಾಲ್ ಇಲ್ಲಡ ಹೆಚ್ಚ ನೀರ್ ಬುಡುನಿ ಪರಿಗಣನೆ ಮಲ್ಪುಲೆ.", "ml": "ഉച്ചവെയിലിൽ ഷെയ്ഡ് നെറ്റ് അല്ലെങ്കിൽ കൂടുതൽ നന പരിഗണിക്കുക."},
    "reco_none_title": {"en": "No action needed", "ta": "எந்த நடவடிக்கையும் தேவையில்லை", "kn": "ಯಾವುದೇ ಕ್ರಮ ಅಗತ್ಯವಿಲ್ಲ", "tcy": "ಯಾವುದೇ ಕ್ರಮ ಬೇಡಂದ್", "ml": "നടപടിയൊന്നും ആവശ്യമില്ല"},
    "reco_none_action": {"en": "Looks good — continue normal monitoring.", "ta": "நன்றாக உள்ளது — வழக்கமான கண்காணிப்பைத் தொடரவும்.", "kn": "ಚೆನ್ನಾಗಿದೆ — ಸಾಮಾನ್ಯ ಮೇಲ್ವಿಚಾರಣೆಯನ್ನು ಮುಂದುವರಿಸಿ.", "tcy": "಼ಲಾಯಿಕ್ ಉಂಡು — ಸಾಮಾನ್ಯ ಮೇಲ್ವಿಚಾರಣೆ ಮುಂದುವರಪುಲೆ.", "ml": "കുഴപ്പമില്ല — സാധാരണ നിരീക്ഷണം തുടരുക."},
    "reco_soil_temp": {
        "en": "Soil moisture {m:.0f}%, Temp {t:.0f}°C",
        "ta": "மண் ஈரப்பதம் {m:.0f}%, வெப்பநிலை {t:.0f}°C",
        "kn": "ಮಣ್ಣಿನ ತೇವಾಂಶ {m:.0f}%, ತಾಪಮಾನ {t:.0f}°C",
        "tcy": "ಮಣ್ಣ್‌ದ ತೇವ {m:.0f}%, ಬಿಸಿ {t:.0f}°C",
        "ml": "മണ്ണിലെ ഈർപ്പം {m:.0f}%, താപനില {t:.0f}°C",
    },
    "reco_action_label": {"en": "Recommended action:", "ta": "பரிந்துரைக்கப்படும் நடவடிக்கை:", "kn": "ಶಿಫಾರಸು ಮಾಡಿದ ಕ್ರಮ:", "tcy": "ಸಲಹೆ ಮಲ್ತಿನ ಕ್ರಮ:", "ml": "ശുപാർശ ചെയ്യുന്ന നടപടി:"},

    "alerts_title": {"en": "⚠ Farm Alerts", "ta": "⚠ பண்ணை எச்சரிக்கைகள்", "kn": "⚠ ಜಮೀನು ಎಚ್ಚರಿಕೆಗಳು", "tcy": "⚠ ಕುರ್ಲೆದ ಎಚ್ಚರಿಕೆಲು", "ml": "⚠ ഫാം മുന്നറിയിപ്പുകൾ"},
    "all_clear": {"en": "ALL CLEAR", "ta": "அனைத்தும் சரி", "kn": "ಎಲ್ಲಾ ಸರಿ", "tcy": "ಎಲ್ಲ ಸರಿ", "ml": "എല്ലാം ശരി"},
    "no_critical": {"en": "No critical alerts. All systems operational.", "ta": "முக்கியமான எச்சரிக்கைகள் இல்லை. அனைத்து அமைப்புகளும் இயங்குகின்றன.", "kn": "ಯಾವುದೇ ನಿರ್ಣಾಯಕ ಎಚ್ಚರಿಕೆಗಳಿಲ್ಲ. ಎಲ್ಲಾ ವ್ಯವಸ್ಥೆಗಳು ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತಿವೆ.", "tcy": "ಎಚ್ಚರಿಕೆದ ವಿಷಯೊಂತೂ ಇಜ್ಜಿ. ಎಲ್ಲ ವ್ಯವಸ್ಥೆ ಸರಿಯಾದುಂಡು.", "ml": "ഗുരുതരമായ മുന്നറിയിപ്പുകളില്ല. എല്ലാ സംവിധാനങ്ങളും പ്രവർത്തിക്കുന്നു."},
    "needs_attention": {"en": "NEEDS ATTENTION", "ta": "கவனம் தேவை", "kn": "ಗಮನ ಬೇಕು", "tcy": "ಗಮನ ಬೇಕ್", "ml": "ശ്രദ്ധ ആവശ്യമാണ്"},

    "live_dashboard": {"en": "Live Data Dashboard", "ta": "நேரடி தரவு டாஷ்போர்டு", "kn": "ಲೈವ್ ಡೇಟಾ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್", "tcy": "ಲೈವ್ ಡೇಟಾ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್", "ml": "തത്സമയ ഡാറ്റ ഡാഷ്‌ബോർഡ്"},
    "soil_moisture": {"en": "💧 Soil moisture", "ta": "💧 மண் ஈரப்பதம்", "kn": "💧 ಮಣ್ಣಿನ ತೇವಾಂಶ", "tcy": "💧 ಮಣ್ಣ್‌ದ ತೇವ", "ml": "💧 മണ്ണിലെ ഈർപ്പം"},
    "air_temp": {"en": "🌡 Air temperature", "ta": "🌡 காற்று வெப்பநிலை", "kn": "🌡 ಗಾಳಿಯ ಉಷ್ಣಾಂಶ", "tcy": "🌡 ಗಾಳಿದ ಬಿಸಿ", "ml": "🌡 വായുവിന്റെ താപനില"},
    "light_levels": {"en": "☀️ Light levels", "ta": "☀️ ஒளி அளவுகள்", "kn": "☀️ ಬೆಳಕಿನ ಮಟ್ಟ", "tcy": "☀️ ಬೊಲ್ಪುದ ಮಟ್ಟ", "ml": "☀️ പ്രകാശ നിലവാരം"},

    "ask_title": {"en": "💬 Ask KisanSense", "ta": "💬 KisanSense-இடம் கேளுங்கள்", "kn": "💬 KisanSense ಗೆ ಕೇಳಿ", "tcy": "💬 KisanSense ಡ್ ಕೇಣ್ಲೆ", "ml": "💬 KisanSense-നോട് ചോദിക്കൂ"},
    "ask_caption": {"en": "Namaste! 👋 Ask about irrigation, crop health, pests, weather, or anything about your farm.", "ta": "வணக்கம்! 👋 நீர்ப்பாசனம், பயிர் ஆரோக்கியம், பூச்சிகள், வானிலை அல்லது உங்கள் பண்ணை பற்றி எதுவும் கேளுங்கள்.", "kn": "ನಮಸ್ತೆ! 👋 ನೀರಾವರಿ, ಬೆಳೆ ಆರೋಗ್ಯ, ಕೀಟಗಳು, ಹವಾಮಾನ ಅಥವಾ ನಿಮ್ಮ ಜಮೀನಿನ ಬಗ್ಗೆ ಏನಾದರೂ ಕೇಳಿ.", "tcy": "ನಮಸ್ತೆ! 👋 ನೀರ್ ಬುಡುನಿ, ಬೆಳೆದ ಆರೋಗ್ಯೊ, ಕೀಟೊಲು, ಹವಾಮಾನ ಇಂಚಿಪ್ಪುನ ಎಚ್ಚಿನ್ ಬೋಡಾಂಡಲ ಕೇಣ್ಲೆ.", "ml": "നമസ്തേ! 👋 ജലസേചനം, വിള ആരോഗ്യം, കീടങ്ങൾ, കാലാവസ്ഥ അല്ലെങ്കിൽ നിങ്ങളുടെ കൃഷിയിടത്തെക്കുറിച്ച് എന്തും ചോദിക്കൂ."},
    "ask_placeholder": {"en": "Ask KisanSense...", "ta": "KisanSense-இடம் கேளுங்கள்...", "kn": "KisanSense ಗೆ ಕೇಳಿ...", "tcy": "KisanSense ಡ್ ಕೇಣ್ಲೆ...", "ml": "KisanSense-നോട് ചോദിക്കൂ..."},
    "offline_caption": {"en": "ℹ️ Offline mode — simple answers from live farm data. Add an API key secret for fuller answers.", "ta": "ℹ️ ஆஃப்லைன் முறை — நேரடி பண்ணை தரவிலிருந்து எளிய பதில்கள். முழுமையான பதில்களுக்கு API key சேர்க்கவும்.", "kn": "ℹ️ ಆಫ್‌ಲೈನ್ ಮೋಡ್ — ಲೈವ್ ಜಮೀನು ಡೇಟಾದಿಂದ ಸರಳ ಉತ್ತರಗಳು. ಪೂರ್ಣ ಉತ್ತರಗಳಿಗಾಗಿ API key ಸೇರಿಸಿ.", "tcy": "ℹ️ ಆಫ್‌ಲೈನ್ ಮೋಡ್ — ಲೈವ್ ಕುರ್ಲೆದ ಡೇಟಾಡ್ದ್ ಸರಳ ಉತ್ತರೊಲು. ಪೂರ್ಣ ಉತ್ತರೊಗು API key ಸೇರಿಸ್ಲೆ.", "ml": "ℹ️ ഓഫ്‌ലൈൻ മോഡ് — തത്സമയ കൃഷിയിടം ഡാറ്റയിൽ നിന്നുള്ള ലളിതമായ ഉത്തരങ്ങൾ. കൂടുതൽ പൂർണ്ണമായ ഉത്തരങ്ങൾക്ക് API key ചേർക്കുക."},

    "myfarm_welcome": {"en": "My Farm", "ta": "என் பண்ணை", "kn": "ನನ್ನ ಜಮೀನು", "tcy": "ಎನ್ನ ಕುರ್ಲೆ", "ml": "എന്റെ കൃഷിയിടം"},
    "myfarm_caption": {"en": "Status of every sensor node and the main hub.", "ta": "ஒவ்வொரு சென்சார் நோட் மற்றும் முதன்மை மையத்தின் நிலை.", "kn": "ಪ್ರತಿ ಸೆನ್ಸಾರ್ ನೋಡ್ ಮತ್ತು ಮುಖ್ಯ ಹಬ್‌ನ ಸ್ಥಿತಿ.", "tcy": "ಪ್ರತಿಯೊಂಜಿ ಸೆನ್ಸಾರ್ ನೋಡ್ ಬೊಕ್ಕ ಮುಖ್ಯ ಹಬ್‌ದ ಸ್ಥಿತಿ.", "ml": "ഓരോ സെൻസർ നോഡിന്റെയും പ്രധാന ഹബിന്റെയും അവസ്ഥ."},
    "main_hub": {"en": "Main hub", "ta": "முதன்மை மையம்", "kn": "ಮುಖ್ಯ ಹಬ್", "tcy": "ಮುಖ್ಯ ಹಬ್", "ml": "പ്രധാന ഹബ്"},

    "crophealth_welcome": {"en": "Crop Health", "ta": "பயிர் ஆரோக்கியம்", "kn": "ಬೆಳೆ ಆರೋಗ್ಯ", "tcy": "ಬೆಳೆದ ಆರೋಗ್ಯೊ", "ml": "വിള ആരോഗ്യം"},
    "crophealth_caption": {"en": "Upload a field photo to check canopy health. This uses a lightweight placeholder heuristic until the real edge-AI model is trained and flashed to the hub.", "ta": "விதானத்தின் ஆரோக்கியத்தை சரிபார்க்க வயல் புகைப்படத்தை பதிவேற்றவும். உண்மையான edge-AI மாதிரி பயிற்சி பெறும் வரை இது ஒரு எளிய தற்காலிக முறையைப் பயன்படுத்துகிறது.", "kn": "ಮೇಲಾವರಣ ಆರೋಗ್ಯ ಪರಿಶೀಲಿಸಲು ಹೊಲದ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ. ನೈಜ edge-AI ಮಾದರಿಯನ್ನು ತರಬೇತಿಗೊಳಿಸಿ ಹಬ್‌ಗೆ ಫ್ಲಾಶ್ ಮಾಡುವವರೆಗೆ ಇದು ಹಗುರವಾದ ತಾತ್ಕಾಲಿಕ ವಿಧಾನವನ್ನು ಬಳಸುತ್ತದೆ.", "tcy": "ಮೇಲಾವರಣದ ಆರೋಗ್ಯ ಪರಿಶೀಲನೆಗ್ ಗದ್ದೆದ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಲ್ಪುಲೆ. ನಿಜದ edge-AI ಮಾದರಿ ತರಬೇತಿಯಾಪುನ ಒರೆಗ್ ಇಂದ ಹಗುರೊದ ತಾತ್ಕಾಲಿಕ ವಿಧಾನ ಉಪಯೋಗ ಮಲ್ಪುಂಡು.", "ml": "മേലാപ്പിന്റെ ആരോഗ്യം പരിശോധിക്കാൻ ഒരു കൃഷിയിട ഫോട്ടോ അപ്‌ലോഡ് ചെയ്യുക. യഥാർത്ഥ edge-AI മോഡൽ പരിശീലിപ്പിച്ച് ഹബിലേക്ക് ഫ്ലാഷ് ചെയ്യുന്നതുവരെ ഇത് ഒരു ലളിതമായ താൽക്കാലിക രീതി ഉപയോഗിക്കുന്നു."},
    "upload_photo": {"en": "Upload a photo from the field", "ta": "வயலிலிருந்து புகைப்படத்தை பதிவேற்றவும்", "kn": "ಹೊಲದಿಂದ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ", "tcy": "ಗದ್ದೆಡ್ದ್ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಲ್ಪುಲೆ", "ml": "കൃഷിയിടത്തിൽ നിന്ന് ഒരു ഫോട്ടോ അപ്‌ലോഡ് ചെയ്യുക"},
    "uploaded_photo": {"en": "Uploaded photo", "ta": "பதிவேற்றப்பட்ட புகைப்படம்", "kn": "ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಫೋಟೋ", "tcy": "ಅಪ್‌ಲೋಡ್ ಮಲ್ತಿನ ಫೋಟೋ", "ml": "അപ്‌ലോഡ് ചെയ്ത ഫോട്ടോ"},
    "verdict": {"en": "Verdict", "ta": "முடிவு", "kn": "ತೀರ್ಪು", "tcy": "ತೀರ್ಪ್", "ml": "വിധി"},
    "green_cover": {"en": "Green cover", "ta": "பசுமை மறைப்பு", "kn": "ಹಸಿರು ಹೊದಿಕೆ", "tcy": "ಪಚ್ಚೆ ಮುಚ್ಚಾವುನಿ", "ml": "ഹരിത ആവരണം"},
    "last_hub_inference": {"en": "Last hub inference:", "ta": "கடைசி மைய முடிவு:", "kn": "ಕೊನೆಯ ಹಬ್ ಅನುಮಾನ:", "tcy": "ಕಡೆತ್ತ ಹಬ್ ಅನುಮಾನ:", "ml": "അവസാന ഹബ് നിഗമനം:"},

    "irrigation_welcome": {"en": "Irrigation", "ta": "நீர்ப்பாசனம்", "kn": "ನೀರಾವರಿ", "tcy": "ನೀರ್ ಬುಡುನಿ", "ml": "ജലസേചനം"},
    "irrigation_caption": {"en": "Per-node moisture and watering guidance.", "ta": "ஒவ்வொரு நோட் ஈரப்பதம் மற்றும் நீர்ப்பாசன வழிகாட்டுதல்.", "kn": "ಪ್ರತಿ-ನೋಡ್ ತೇವಾಂಶ ಮತ್ತು ನೀರಾವರಿ ಮಾರ್ಗದರ್ಶನ.", "tcy": "ಪ್ರತಿಯೊಂಜಿ ನೋಡ್‌ದ ತೇವ ಬೊಕ್ಕ ನೀರ್ ಬುಡುನಿದ ಮಾರ್ಗದರ್ಶನ.", "ml": "ഓരോ നോഡിന്റെയും ഈർപ്പവും നന മാർഗ്ഗനിർദ്ദേശവും."},
    "irrigate_soon": {"en": "Irrigate soon", "ta": "விரைவில் நீர்ப்பாசனம் செய்யவும்", "kn": "ಬೇಗ ನೀರಾವರಿ ಮಾಡಿ", "tcy": "ಬೇಗ ನೀರ್ ಬುಡ್ಲೆ", "ml": "ഉടൻ നനയ്ക്കുക"},
    "reduce_watering": {"en": "Reduce watering", "ta": "நீர்ப்பாசனத்தை குறைக்கவும்", "kn": "ನೀರಾವರಿ ಕಡಿಮೆ ಮಾಡಿ", "tcy": "ನೀರ್ ಕಮ್ಮಿ ಮಲ್ಪುಲೆ", "ml": "നന കുറയ്ക്കുക"},
    "no_action": {"en": "No action needed", "ta": "நடவடிக்கை தேவையில்லை", "kn": "ಯಾವುದೇ ಕ್ರಮ ಅಗತ್ಯವಿಲ್ಲ", "tcy": "ಕ್ರಮ ಬೋಡಂದ್", "ml": "നടപടി വേണ്ട"},

    "alerts_welcome": {"en": "Farm Alerts", "ta": "பண்ணை எச்சரிக்கைகள்", "kn": "ಜಮೀನು ಎಚ್ಚರಿಕೆಗಳು", "tcy": "ಕುರ್ಲೆದ ಎಚ್ಚರಿಕೆಲು", "ml": "ഫാം മുന്നറിയിപ്പുകൾ"},

    "analytics_welcome": {"en": "Farm Analytics", "ta": "பண்ணை பகுப்பாய்வு", "kn": "ಜಮೀನು ವಿಶ್ಲೇಷಣೆ", "tcy": "ಕುರ್ಲೆದ ವಿಶ್ಲೇಷಣೆ", "ml": "ഫാം അനലിറ്റിക്സ്"},
    "analytics_caption": {"en": "Historical trends and detailed sensor data.", "ta": "வரலாற்று போக்குகள் மற்றும் விரிவான சென்சார் தரவு.", "kn": "ಐತಿಹಾಸಿಕ ಪ್ರವೃತ್ತಿಗಳು ಮತ್ತು ವಿವರವಾದ ಸೆನ್ಸಾರ್ ಡೇಟಾ.", "tcy": "ಇತಿಹಾಸೊದ ಪ್ರವೃತ್ತಿಲು ಬೊಕ್ಕ ವಿವರೊದ ಸೆನ್ಸಾರ್ ಡೇಟಾ.", "ml": "ചരിത്രപരമായ പ്രവണതകളും വിശദമായ സെൻസർ ഡാറ്റയും."},
    "trends_title": {"en": "Trends", "ta": "போக்குகள்", "kn": "ಪ್ರವೃತ್ತಿಗಳು", "tcy": "ಪ್ರವೃತ್ತಿಲು", "ml": "ട്രെൻഡുകൾ"},
    "raw_readings_title": {"en": "Raw node readings", "ta": "மூல நோட் அளவீடுகள்", "kn": "ಕಚ್ಚಾ ನೋಡ್ ವಾಚನಗಳು", "tcy": "ಕಚ್ಚಾ ನೋಡ್ ರೀಡಿಂಗ್ಸ್", "ml": "അസംസ്‌കൃത നോഡ് റീഡിംഗുകൾ"},
    "auto_refresh": {"en": "Auto-refresh every 5s", "ta": "ஒவ்வொரு 5 வினாடிக்கும் தானாக புதுப்பிக்கவும்", "kn": "ಪ್ರತಿ 5 ಸೆಕೆಂಡಿಗೆ ಸ್ವಯಂ-ರಿಫ್ರೆಶ್", "tcy": "ಪ್ರತಿ 5 ಸೆಕೆಂಡ್‌ಗ್ ಸ್ವಯಂ-ರಿಫ್ರೆಶ್", "ml": "ഓരോ 5 സെക്കൻഡിലും ഓട്ടോ-റിഫ്രഷ്"},

    "settings_welcome": {"en": "Settings", "ta": "அமைப்புகள்", "kn": "ಸೆಟ್ಟಿಂಗ್‌ಗಳು", "tcy": "ಸೆಟ್ಟಿಂಗ್ಸ್", "ml": "ക്രമീകരണങ്ങൾ"},
    "chatbot_section": {"en": "Chatbot", "ta": "அரட்டைப்பெட்டி", "kn": "ಚಾಟ್‌ಬಾಟ್", "tcy": "ಚಾಟ್‌ಬಾಟ್", "ml": "ചാറ്റ്ബോട്ട്"},
    "anthropic_key_line": {"en": "Anthropic key configured:", "ta": "Anthropic key கட்டமைக்கப்பட்டதா:", "kn": "Anthropic key ಕಾನ್ಫಿಗರ್ ಮಾಡಲಾಗಿದೆಯೇ:", "tcy": "Anthropic key ಕಾನ್ಫಿಗರ್ ಆದುಂಡಾ:", "ml": "Anthropic key ക്രമീകരിച്ചിട്ടുണ്ടോ:"},
    "openai_key_line": {"en": "OpenAI key configured:", "ta": "OpenAI key கட்டமைக்கப்பட்டதா:", "kn": "OpenAI key ಕಾನ್ಫಿಗರ್ ಮಾಡಲಾಗಿದೆಯೇ:", "tcy": "OpenAI key ಕಾನ್ಫಿಗರ್ ಆದುಂಡಾ:", "ml": "OpenAI key ക്രമീകരിച്ചിട്ടുണ്ടോ:"},
    "secrets_caption": {"en": "Add ANTHROPIC_API_KEY or OPENAI_API_KEY under Settings → Secrets on Streamlit Cloud.", "ta": "Streamlit Cloud-ல் Settings → Secrets-இல் ANTHROPIC_API_KEY அல்லது OPENAI_API_KEY சேர்க்கவும்.", "kn": "Streamlit Cloud ನಲ್ಲಿ Settings → Secrets ಅಡಿಯಲ್ಲಿ ANTHROPIC_API_KEY ಅಥವಾ OPENAI_API_KEY ಸೇರಿಸಿ.", "tcy": "Streamlit Cloud ದ Settings → Secrets ಡ್ ANTHROPIC_API_KEY ಇಂಚಿಪ್ಪುನ OPENAI_API_KEY ಸೇರಿಸ್ಲೆ.", "ml": "Streamlit Cloud-ൽ Settings → Secrets-ൽ ANTHROPIC_API_KEY അല്ലെങ്കിൽ OPENAI_API_KEY ചേർക്കുക."},
    "data_source_section": {"en": "Data source", "ta": "தரவு மூலம்", "kn": "ಡೇಟಾ ಮೂಲ", "tcy": "ಡೇಟಾ ಮೂಲ", "ml": "ഡാറ്റ സ്രോതസ്സ്"},
    "sensor_readings_line": {"en": "Sensor readings: **Simulated** (telemetry.py) — no hardware connected yet.", "ta": "சென்சார் அளவீடுகள்: **உருவகப்படுத்தப்பட்டது** (telemetry.py) — இன்னும் வன்பொருள் இணைக்கப்படவில்லை.", "kn": "ಸೆನ್ಸಾರ್ ವಾಚನಗಳು: **ಸಿಮ್ಯುಲೇಟೆಡ್** (telemetry.py) — ಇನ್ನೂ ಹಾರ್ಡ್‌ವೇರ್ ಸಂಪರ್ಕಿಸಲಾಗಿಲ್ಲ.", "tcy": "ಸೆನ್ಸಾರ್ ರೀಡಿಂಗ್ಸ್: **ಸಿಮ್ಯುಲೇಟೆಡ್** (telemetry.py) — ಇನ್ನೂಂಚಿ ಹಾರ್ಡ್‌ವೇರ್ ಜೋಡುನಂದ್.", "ml": "സെൻസർ റീഡിംഗുകൾ: **സിമുലേറ്റഡ്** (telemetry.py) — ഹാർഡ്‌വെയർ ഇതുവരെ ബന്ധിപ്പിച്ചിട്ടില്ല."},
    "appearance_section": {"en": "Appearance", "ta": "தோற்றம்", "kn": "ಗೋಚರತೆ", "tcy": "ಕಾಣ್ಣುನಿ", "ml": "രൂപഭാവം"},
    "current_theme_line": {"en": "Current theme: **{theme}** (toggle in the sidebar)", "ta": "தற்போதைய தீம்: **{theme}** (சைட்பாரில் மாற்றவும்)", "kn": "ಪ್ರಸ್ತುತ ಥೀಮ್: **{theme}** (ಸೈಡ್‌ಬಾರ್‌ನಲ್ಲಿ ಬದಲಿಸಿ)", "tcy": "ಈಗಿನ ಥೀಮ್: **{theme}** (ಸೈಡ್‌ಬಾರ್‌ಡ್ ಬದಲಾಪುಲೆ)", "ml": "നിലവിലെ തീം: **{theme}** (സൈഡ്ബാറിൽ മാറ്റുക)"},
    "clear_chat_btn": {"en": "Clear chat history", "ta": "அரட்டை வரலாற்றை அழிக்கவும்", "kn": "ಚಾಟ್ ಇತಿಹಾಸ ತೆರವುಗೊಳಿಸಿ", "tcy": "ಚಾಟ್ ಇತಿಹಾಸ ಅಳಿಪುಲೆ", "ml": "ചാറ്റ് ചരിത്രം മായ്ക്കുക"},

    "language_label": {"en": "🌐 Language", "ta": "🌐 மொழி", "kn": "🌐 ಭಾಷೆ", "tcy": "🌐 ಭಾಷೆ", "ml": "🌐 ഭാഷ"},
}


def tr(key: str, **kwargs) -> str:
    """Look up a UI string in the current language, falling back to English."""
    entry = STRINGS.get(key, {})
    text = entry.get(st.session_state.lang, entry.get("en", key))
    return text.format(**kwargs) if kwargs else text


# ==========================================================================
# Theme (Dark / Light)
# ==========================================================================
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

THEMES = {
    "Dark": {
        "app_bg": "linear-gradient(160deg, #10181C 0%, #131E22 60%, #0F1720 100%)",
        "sidebar_bg": "#0D1417",
        "text_primary": "#EAF3EE",
        "text_secondary": "#9FB3AC",
        "text_muted": "#748883",
        "card_bg": "rgba(255,255,255,0.045)",
        "card_border": "rgba(255,255,255,0.08)",
        "accent": "#4FD1A5",
        "accent_soft": "rgba(79,209,165,0.15)",
        "pill_bg": "rgba(255,255,255,0.06)",
        "pill_border": "rgba(255,255,255,0.1)",
        "good": "#4FD1A5",
        "warn": "#F0B94D",
        "bad": "#F0705E",
        "good_bg": "rgba(79,209,165,0.12)",
        "warn_bg": "rgba(240,185,77,0.12)",
        "bad_bg": "rgba(240,112,94,0.12)",
        "chart_line": "#4FD1A5",
    },
    "Light": {
        "app_bg": "linear-gradient(160deg, #F5FAF6 0%, #EFF6F0 60%, #F7F9F4 100%)",
        "sidebar_bg": "#FFFFFF",
        "text_primary": "#1E3324",
        "text_secondary": "#4B6B57",
        "text_muted": "#7C927F",
        "card_bg": "rgba(255,255,255,0.85)",
        "card_border": "rgba(30,51,36,0.08)",
        "accent": "#2F9E6E",
        "accent_soft": "rgba(47,158,110,0.12)",
        "pill_bg": "rgba(47,158,110,0.07)",
        "pill_border": "rgba(47,158,110,0.18)",
        "good": "#2F9E6E",
        "warn": "#B9821F",
        "bad": "#C24A38",
        "good_bg": "rgba(47,158,110,0.1)",
        "warn_bg": "rgba(185,130,31,0.1)",
        "bad_bg": "rgba(194,74,56,0.1)",
        "chart_line": "#2F9E6E",
    },
}

T = THEMES[st.session_state.theme]

st.markdown(
    f"""
    <style>
    .stApp {{ background: {T['app_bg']}; }}
    section[data-testid="stSidebar"] {{ background: {T['sidebar_bg']}; border-right: 1px solid {T['card_border']}; }}
    h1, h2, h3, p, span, label, .stMarkdown {{ color: {T['text_primary']}; }}

    .ks-header-title {{ font-size: 1.5rem; font-weight: 800; color: {T['text_primary']}; margin-bottom: 0; }}
    .ks-welcome {{ font-size: 1.6rem; font-weight: 800; color: {T['text_primary']}; margin: 0.4rem 0 0.2rem 0; }}
    .ks-status-line {{ font-size: 0.95rem; color: {T['text_muted']}; margin-bottom: 0.6rem; }}
    .ks-section-title {{ font-size: 1.05rem; font-weight: 700; color: {T['text_secondary']}; margin: 1.3rem 0 0.5rem 0; letter-spacing: 0.3px; text-transform: uppercase; }}

    .ks-card {{
        background: {T['card_bg']};
        border: 1px solid {T['card_border']};
        border-radius: 16px;
        padding: 18px 20px;
    }}
    .ks-card-title {{ font-size: 0.95rem; font-weight: 700; color: {T['text_secondary']}; margin-bottom: 8px; }}
    .ks-value {{ font-size: 2.1rem; font-weight: 800; color: {T['text_primary']}; }}
    .ks-status-good {{ color: {T['good']}; font-weight: 700; }}
    .ks-status-warn {{ color: {T['warn']}; font-weight: 700; }}
    .ks-status-bad  {{ color: {T['bad']}; font-weight: 700; }}

    .ks-badge-good {{ background: {T['good_bg']}; color: {T['good']}; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 0.85rem; display: inline-block; }}
    .ks-badge-warn {{ background: {T['warn_bg']}; color: {T['warn']}; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 0.85rem; display: inline-block; }}
    .ks-badge-bad  {{ background: {T['bad_bg']};  color: {T['bad']};  padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 0.85rem; display: inline-block; }}

    .ks-alert-row {{ color: {T['text_secondary']}; font-size: 0.9rem; margin-top: 8px; }}

    /* Pill-style quick-question buttons */
    div.stButton > button {{
        background: {T['pill_bg']};
        border: 1px solid {T['pill_border']};
        color: {T['text_primary']};
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 6px 14px;
    }}
    div.stButton > button:hover {{
        border-color: {T['accent']};
        color: {T['accent']};
    }}

    /* Sidebar nav radio -> look like a vertical list of nav items */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        padding: 8px 10px;
        border-radius: 10px;
        margin-bottom: 2px;
        color: {T['text_secondary']};
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
        background: {T['accent_soft']};
        color: {T['accent']};
    }}

    /* Top-right language switcher */
    .ks-lang-select div[data-baseweb="select"] {{
        border-radius: 999px;
    }}
    .ks-lang-select label {{
        font-size: 0.75rem;
        color: {T['text_muted']};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================================
# Top bar: title (left) + language switcher (top-right)
# ==========================================================================
top_left, top_right = st.columns([5, 1])
with top_right:
    st.markdown('<div class="ks-lang-select">', unsafe_allow_html=True)
    selected_label = st.selectbox(
        tr("language_label"),
        options=list(LANGUAGES.values()),
        index=list(LANGUAGES.keys()).index(st.session_state.lang),
        key="lang_selector",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)
    selected_code = [k for k, v in LANGUAGES.items() if v == selected_label][0]
    if selected_code != st.session_state.lang:
        st.session_state.lang = selected_code
        st.rerun()

# ==========================================================================
# Navigation
# ==========================================================================
NAV_PAGES = {
    "HOME": tr("nav_home"),
    "MY FARM": tr("nav_myfarm"),
    "CROP HEALTH": tr("nav_crophealth"),
    "IRRIGATION": tr("nav_irrigation"),
    "ALERTS": tr("nav_alerts"),
    "ANALYTICS": tr("nav_analytics"),
    "SETTINGS": tr("nav_settings"),
}

st.sidebar.markdown('<div class="ks-header-title">🌱 KisanSense</div>', unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
page_label = st.sidebar.radio(
    "Navigate", list(NAV_PAGES.values()), index=0, label_visibility="collapsed"
)
page = [k for k, v in NAV_PAGES.items() if v == page_label][0]

st.sidebar.markdown("---")
theme_choice = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True,
                                 index=0 if st.session_state.theme == "Dark" else 1)
if theme_choice != st.session_state.theme:
    st.session_state.theme = theme_choice
    st.rerun()

st.sidebar.caption(
    "Sensor values are simulated for now (no hardware connected yet). "
    "Swap `telemetry.py` for the real LoRa/Wi-Fi feed once the hub is built."
)

# ==========================================================================
# Live data (shared across pages)
# ==========================================================================


def get_crop_health(avg_moisture: float, avg_temp: float, hub_status):
    """Simple crop-health rollup until a trained model feeds this directly."""
    if avg_moisture < 20 or avg_temp > 38:
        return "Poor", "Crops may be under stress — check the field soon."
    if avg_moisture < 30 or avg_temp > 34:
        return "Fair", "Crops are okay but conditions are trending unfavorable."
    return "Good", "Your crops are currently healthy."


readings = get_latest_readings()
hub = get_hub_status()
avg_moisture = sum(r.soil_moisture_pct for r in readings) / len(readings)
avg_temp = sum(r.temperature_c for r in readings) / len(readings)
crop_health, crop_health_note = get_crop_health(avg_moisture, avg_temp, hub)
# Simple simulated light-level reading (no light sensor in telemetry.py yet)
light_level = max(0, min(100, 55 + 40 * math.sin(time.time() / 25)))

if "history" not in st.session_state:
    st.session_state.history = []
st.session_state.history.append(
    {"t": time.time(), "Soil moisture": avg_moisture, "Air temperature": avg_temp, "Light level": light_level}
)
st.session_state.history = st.session_state.history[-120:]


def moisture_status(value: float) -> str:
    if value < 30:
        return "Low"
    if value > 70:
        return "High"
    return "Good"


def temp_status(value: float) -> str:
    if value < 20:
        return "Cool"
    if value > 38:
        return "Heat Stress"
    if value > 32:
        return "High"
    return "Normal"


STATUS_CLASS = {
    "Good": "ks-status-good", "Normal": "ks-status-good",
    "Low": "ks-status-warn", "High": "ks-status-warn", "Cool": "ks-status-warn", "Fair": "ks-status-warn",
    "Heat Stress": "ks-status-bad", "Poor": "ks-status-bad",
}


def render_card(icon_label: str, value: str, status: str):
    css_class = STATUS_CLASS.get(status, "ks-status-good")
    st.markdown(
        f"""
        <div class="ks-card">
            <div class="ks-card-title">{icon_label}</div>
            <div class="ks-value">{value}</div>
            <div class="{css_class}">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mini_chart(label: str, icon: str, column: str, value_fmt: str):
    df = pd.DataFrame(st.session_state.history)
    latest = df[column].iloc[-1]
    st.markdown(
        f"""<div class="ks-card"><div class="ks-card-title">{icon} {label}</div>
        <div class="ks-value" style="font-size:1.6rem;">{value_fmt.format(latest)}</div></div>""",
        unsafe_allow_html=True,
    )
    chart_df = df.set_index(pd.to_datetime(df["t"], unit="s"))[[column]]
    st.line_chart(chart_df, height=110, use_container_width=True)


# ==========================================================================
# Chatbot logic (shared — used on HOME)
# ==========================================================================


def get_api_key(name: str):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


ANTHROPIC_KEY = get_api_key("ANTHROPIC_API_KEY")
OPENAI_KEY = get_api_key("OPENAI_API_KEY")

live_context = "\n".join(
    f"- {r.node_id}: moisture {r.soil_moisture_pct:.0f}%, temp {r.temperature_c:.1f}C, "
    f"humidity {r.humidity_pct:.0f}%, battery {r.battery_pct:.0f}% "
    f"({'charging' if r.solar_charging else 'idle'})"
    for r in readings
)
LANGUAGE_NAMES_FOR_PROMPT = {
    "en": "English",
    "ta": "Tamil",
    "kn": "Kannada",
    "tcy": "Tulu",
    "ml": "Malayalam",
}
SYSTEM_PROMPT = f"""You are the KisanSense farming assistant, talking directly to a farmer
who may not be technical. Answer in simple, plain language about irrigation, crop health,
pests, disease, fertilizer, and weather-related farm decisions. Use the live field data
below when relevant. Keep answers short and actionable. Respond in
{LANGUAGE_NAMES_FOR_PROMPT.get(st.session_state.lang, "English")}, regardless of the
language used in this system prompt.

Current field data:
{live_context}
Crop health rollup: {crop_health} — {crop_health_note}
"""


def fallback_answer(question: str) -> str:
    q = question.lower()
    if "irrigat" in q or "water" in q:
        if avg_moisture < 30:
            return f"Soil moisture is {avg_moisture:.0f}%, which is low — irrigation is recommended today."
        return f"Soil moisture is {avg_moisture:.0f}%, which is healthy — no irrigation needed right now."
    if "crop" in q or "health" in q:
        return f"Crop health looks **{crop_health}**. {crop_health_note}"
    if "temp" in q or "heat" in q:
        return f"Current average temperature is {avg_temp:.1f}°C ({temp_status(avg_temp)})."
    if "pest" in q or "bug" in q or "insect" in q:
        return (
            "For general pest control: inspect leaves regularly, use neem-based sprays "
            "for common pests, and remove visibly infected plants early. For a specific "
            "pest, add an API key so I can give more tailored advice."
        )
    if "rain" in q or "weather" in q:
        return "I don't have a live weather feed connected yet — check your local forecast before deciding on irrigation."
    return (
        "I can answer from live farm data (soil moisture, temperature, crop health) "
        "in this simple mode, or give fuller answers if an ANTHROPIC_API_KEY / "
        "OPENAI_API_KEY secret is configured."
    )


def stream_anthropic():
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    history_msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages]
    with client.messages.stream(
        model="claude-sonnet-4-6", max_tokens=800, system=SYSTEM_PROMPT, messages=history_msgs
    ) as stream:
        for text in stream.text_stream:
            yield text


def stream_openai():
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_KEY)
    history_msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages]
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history_msgs,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def handle_question(question: str):
    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if ANTHROPIC_KEY:
            reply = st.write_stream(stream_anthropic())
        elif OPENAI_KEY:
            reply = st.write_stream(stream_openai())
        else:
            reply = fallback_answer(question)
            st.markdown(reply)
    st.session_state.chat_messages.append({"role": "assistant", "content": reply})


if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

QUICK_QUESTIONS = [
    ("quick_irrigate", "Should I irrigate now?"),
    ("quick_crop", "Is my crop healthy?"),
    ("quick_pest", "How do I control pests?"),
    ("quick_temp", "Is the temperature dangerous?"),
    ("quick_rain", "Is rain expected?"),
]

# ==========================================================================
# HOME
# ==========================================================================
if page == "HOME":
    st.markdown(f'<div class="ks-welcome">{tr("welcome_home")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ks-status-line">{tr("status_line")}</div>', unsafe_allow_html=True)

    # --- Insights: quick-question pills ------------------------------------
    st.markdown(f'<div class="ks-section-title">{tr("insights_title")}</div>', unsafe_allow_html=True)
    qcols = st.columns(len(QUICK_QUESTIONS))
    for i, (qcol, (qkey, qtext_en)) in enumerate(zip(qcols, QUICK_QUESTIONS)):
        if qcol.button(tr(qkey), use_container_width=True, key=f"quick_{i}"):
            # Send the underlying English intent to the chatbot for reliable matching,
            # while the button itself is shown in the chosen language.
            handle_question(qtext_en)

    # --- Today's Recommendation + Farm Alerts, side by side ----------------
    col_reco, col_alert = st.columns(2)

    if avg_moisture < 30:
        reco_title, reco_action, reco_status = tr("reco_irrig_title"), tr("reco_irrig_action"), "warn"
    elif avg_temp > 36:
        reco_title, reco_action, reco_status = tr("reco_heat_title"), tr("reco_heat_action"), "warn"
    else:
        reco_title, reco_action, reco_status = tr("reco_none_title"), tr("reco_none_action"), "good"

    reco_icon = "✅" if reco_status == "good" else "🟡"
    with col_reco:
        st.markdown(
            f"""
            <div class="ks-card">
                <div class="ks-card-title">{tr("reco_title")}</div>
                <div style="font-weight:700; margin-top:4px;">{reco_icon} {reco_title}</div>
                <div class="ks-alert-row">{tr("reco_soil_temp", m=avg_moisture, t=avg_temp)}</div>
                <div class="ks-alert-row"><b>{tr("reco_action_label")}</b> {reco_action}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    m_alerts = moisture_alerts(readings)
    b_alerts = battery_alerts(readings)
    all_alerts = m_alerts + b_alerts + (["Possible crop stress detected"] if crop_health == "Poor" else [])
    with col_alert:
        if not all_alerts:
            st.markdown(
                f"""
                <div class="ks-card">
                    <div class="ks-card-title">{tr("alerts_title")}</div>
                    <span class="ks-badge-good">{tr("all_clear")}</span>
                    <div class="ks-alert-row">{tr("no_critical")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            rows = "".join(f'<div class="ks-alert-row">🔴 {a}</div>' for a in all_alerts)
            st.markdown(
                f"""
                <div class="ks-card">
                    <div class="ks-card-title">{tr("alerts_title")}</div>
                    <span class="ks-badge-warn">{tr("needs_attention")}</span>
                    {rows}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- Live Data Dashboard: mini sparklines --------------------------------
    st.markdown(f'<div class="ks-section-title">{tr("live_dashboard")}</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        mini_chart(tr("soil_moisture").split(" ", 1)[1], "💧", "Soil moisture", "{:.0f}%")
    with c2:
        mini_chart(tr("air_temp").split(" ", 1)[1], "🌡", "Air temperature", "{:.0f}°C")
    with c3:
        mini_chart(tr("light_levels").split(" ", 1)[1], "☀️", "Light level", "{:.0f}%")

    # --- Ask KisanSense --------------------------------------------------------
    st.markdown(f'<div class="ks-section-title">{tr("ask_title")}</div>', unsafe_allow_html=True)
    if not st.session_state.chat_messages:
        st.caption(tr("ask_caption"))

    for msg in st.session_state.chat_messages[-6:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    typed_question = st.chat_input(tr("ask_placeholder"))
    if typed_question:
        handle_question(typed_question)

    if not ANTHROPIC_KEY and not OPENAI_KEY:
        st.caption(tr("offline_caption"))

# ==========================================================================
# MY FARM — node & power status
# ==========================================================================
elif page == "MY FARM":
    st.markdown(f'<div class="ks-welcome">{tr("myfarm_welcome")}</div>', unsafe_allow_html=True)
    st.caption(tr("myfarm_caption"))

    table = pd.DataFrame(
        [
            {
                "Node": r.node_id,
                "Moisture %": round(r.soil_moisture_pct, 1),
                "Temp °C": round(r.temperature_c, 1),
                "Humidity %": round(r.humidity_pct, 1),
                "Battery %": round(r.battery_pct, 0),
                "Solar": "Charging" if r.solar_charging else "Idle",
                "LoRa RSSI (dBm)": r.lora_rssi_dbm,
            }
            for r in readings
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown(f'<div class="ks-section-title">{tr("main_hub")}</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        render_card("🔋 Hub battery", f"{hub.battery_pct:.0f}%", "Good" if hub.battery_pct > 30 else "Low")
    with c2:
        render_card("☀️ Solar", "Charging" if hub.solar_charging else "Idle", "Good" if hub.solar_charging else "Fair")
    with c3:
        render_card("📷 Camera", "Online" if hub.camera_online else "Offline", "Good" if hub.camera_online else "Poor")

# ==========================================================================
# CROP HEALTH — camera / edge-AI panel
# ==========================================================================
elif page == "CROP HEALTH":
    st.markdown(f'<div class="ks-welcome">{tr("crophealth_welcome")}</div>', unsafe_allow_html=True)
    st.caption(tr("crophealth_caption"))

    uploaded = st.file_uploader(tr("upload_photo"), type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption=tr("uploaded_photo"), use_container_width=True)
        with col2:
            result = analyze_image(image)
            st.metric(tr("verdict"), result.verdict)
            st.progress(result.green_ratio, text=f"{tr('green_cover')}: {result.green_ratio*100:.0f}%")
            st.info(result.note)
    else:
        st.markdown(f"**{tr('last_hub_inference')}** {hub.last_inference} ({hub.last_inference_confidence*100:.0f}% confidence)")

# ==========================================================================
# IRRIGATION
# ==========================================================================
elif page == "IRRIGATION":
    st.markdown(f'<div class="ks-welcome">{tr("irrigation_welcome")}</div>', unsafe_allow_html=True)
    st.caption(tr("irrigation_caption"))

    for r in readings:
        status = moisture_status(r.soil_moisture_pct)
        action = tr("irrigate_soon") if status == "Low" else (tr("reduce_watering") if status == "High" else tr("no_action"))
        st.markdown(f"**{r.node_id}** — {r.soil_moisture_pct:.0f}% ({status}) — {action}")
        st.progress(min(r.soil_moisture_pct / 100, 1.0))

# ==========================================================================
# ALERTS
# ==========================================================================
elif page == "ALERTS":
    st.markdown(f'<div class="ks-welcome">{tr("alerts_welcome")}</div>', unsafe_allow_html=True)
    m_alerts = moisture_alerts(readings)
    b_alerts = battery_alerts(readings)
    if not m_alerts and not b_alerts:
        st.markdown(f'<span class="ks-badge-good">{tr("all_clear")}</span>', unsafe_allow_html=True)
        st.caption(tr("no_critical"))
    for a in m_alerts:
        st.markdown(f'<div class="ks-card" style="margin-bottom:8px;">🟡 {a}</div>', unsafe_allow_html=True)
    for a in b_alerts:
        st.markdown(f'<div class="ks-card" style="margin-bottom:8px;">🔴 {a}</div>', unsafe_allow_html=True)

# ==========================================================================
# ANALYTICS — all the charts/history live here now
# ==========================================================================
elif page == "ANALYTICS":
    st.markdown(f'<div class="ks-welcome">{tr("analytics_welcome")}</div>', unsafe_allow_html=True)
    st.caption(tr("analytics_caption"))

    df = pd.DataFrame(st.session_state.history).set_index("t")
    df.index = pd.to_datetime(df.index, unit="s")
    st.markdown(f'<div class="ks-section-title">{tr("trends_title")}</div>', unsafe_allow_html=True)
    st.line_chart(df)

    st.markdown(f'<div class="ks-section-title">{tr("raw_readings_title")}</div>', unsafe_allow_html=True)
    table = pd.DataFrame(
        [
            {
                "Node": r.node_id,
                "Moisture %": round(r.soil_moisture_pct, 1),
                "Temp °C": round(r.temperature_c, 1),
                "Humidity %": round(r.humidity_pct, 1),
                "Battery %": round(r.battery_pct, 0),
                "LoRa RSSI (dBm)": r.lora_rssi_dbm,
            }
            for r in readings
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    if st.checkbox(tr("auto_refresh")):
        time.sleep(5)
        st.rerun()

# ==========================================================================
# SETTINGS
# ==========================================================================
elif page == "SETTINGS":
    st.markdown(f'<div class="ks-welcome">{tr("settings_welcome")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ks-section-title">{tr("chatbot_section")}</div>', unsafe_allow_html=True)
    st.write(f"{tr('anthropic_key_line')} {'✅' if ANTHROPIC_KEY else '❌'}")
    st.write(f"{tr('openai_key_line')} {'✅' if OPENAI_KEY else '❌'}")
    st.caption(tr("secrets_caption"))

    st.markdown(f'<div class="ks-section-title">{tr("data_source_section")}</div>', unsafe_allow_html=True)
    st.write(tr("sensor_readings_line"))

    st.markdown(f'<div class="ks-section-title">{tr("appearance_section")}</div>', unsafe_allow_html=True)
    st.write(tr("current_theme_line", theme=st.session_state.theme))

    if st.button(tr("clear_chat_btn")):
        st.session_state.chat_messages = []
        st.rerun()
