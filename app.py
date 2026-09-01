"""
KisanSense — farmer-friendly app (Streamlit)

Run locally:
    streamlit run app.py

Deploy:
    Push this folder to GitHub, then deploy on https://share.streamlit.io
    pointing at app.py. Add ANTHROPIC_API_KEY (or OPENAI_API_KEY) under the
    app's Settings -> Secrets for full chatbot answers — see README.md.
"""

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
# Global style — light background, green brand color, rounded cards
# ==========================================================================
st.markdown(
    """
    <style>
    /* ---- Earthy green/brown glassmorphic theme ("Apple glass" look) ---- */
    .stApp {
        background: radial-gradient(circle at 15% 0%, #DCEBD8 0%, #EDE6D8 45%, #E4D9C6 100%) fixed;
    }
    h1, h2, h3 { color: #2F4F33; }

    .ks-header-title {
        font-size: 2.3rem; font-weight: 800; color: #2F4F33; margin-bottom: 0;
        letter-spacing: 0.2px;
    }
    .ks-header-sub { font-size: 1.1rem; color: #6B4F32; margin-top: 0; font-weight: 600; }
    .ks-welcome { font-size: 1.4rem; font-weight: 700; color: #2F4F33; margin-top: 1rem; }
    .ks-status-line { font-size: 1rem; color: #7A6A55; margin-bottom: 1rem; }

    /* Frosted-glass card */
    .ks-card {
        background: rgba(255, 255, 255, 0.45);
        backdrop-filter: blur(18px) saturate(160%);
        -webkit-backdrop-filter: blur(18px) saturate(160%);
        border-radius: 22px;
        padding: 26px 18px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(58, 46, 27, 0.14), inset 0 1px 0 rgba(255,255,255,0.6);
        border: 1px solid rgba(255, 255, 255, 0.55);
    }
    .ks-card .ks-icon-label { font-size: 1.05rem; font-weight: 700; color: #4B7A4E; }
    .ks-card .ks-value { font-size: 2.7rem; font-weight: 800; color: #2F4F33; margin: 6px 0 2px 0; }
    .ks-card .ks-status { font-size: 1.05rem; font-weight: 700; }
    .ks-status-good { color: #3E7D3E; }
    .ks-status-warn { color: #A6742C; }
    .ks-status-bad  { color: #A13B2A; }

    .ks-section-title { font-size: 1.4rem; font-weight: 800; color: #2F4F33; margin-top: 1.6rem; }
    .ks-section-sub { color: #7A6A55; margin-top: -0.4rem; margin-bottom: 0.8rem; }

    /* Frosted-glass recommendation card, brown accent edge */
    .ks-reco-card {
        background: rgba(255, 255, 255, 0.42);
        backdrop-filter: blur(16px) saturate(150%);
        -webkit-backdrop-filter: blur(16px) saturate(150%);
        border-left: 6px solid #8A6E4B;
        border-radius: 18px;
        padding: 18px 20px;
        margin-top: 0.5rem;
        box-shadow: 0 6px 24px rgba(58, 46, 27, 0.12);
        border-top: 1px solid rgba(255,255,255,0.55);
        border-right: 1px solid rgba(255,255,255,0.55);
        border-bottom: 1px solid rgba(255,255,255,0.55);
    }

    .ks-alert-good {
        background: rgba(233, 245, 230, 0.6); backdrop-filter: blur(10px);
        border-radius: 14px; padding:12px 16px; color:#2F4F33; font-weight:600;
        border: 1px solid rgba(255,255,255,0.5);
    }
    .ks-alert-warn {
        background: rgba(250, 238, 214, 0.65); backdrop-filter: blur(10px);
        border-radius: 14px; padding:12px 16px; color:#7A5A22; font-weight:600; margin-bottom:6px;
        border: 1px solid rgba(255,255,255,0.5);
    }
    .ks-alert-bad {
        background: rgba(249, 226, 216, 0.65); backdrop-filter: blur(10px);
        border-radius: 14px; padding:12px 16px; color:#8C3A26; font-weight:600; margin-bottom:6px;
        border: 1px solid rgba(255,255,255,0.5);
    }

    /* Sidebar + buttons pick up the same glass/earthy feel */
    section[data-testid="stSidebar"] {
        background: rgba(232, 224, 208, 0.55);
        backdrop-filter: blur(14px);
    }
    div.stButton > button {
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.6);
        color: #2F4F33;
        border-radius: 14px;
        font-weight: 600;
    }
    div.stButton > button:hover {
        border: 1px solid #8A6E4B;
        color: #6B4F32;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================================
# Navigation
# ==========================================================================
NAV_PAGES = ["HOME", "MY FARM", "CROP HEALTH", "IRRIGATION", "ALERTS", "ANALYTICS", "SETTINGS"]

st.sidebar.markdown("### 🌱 KisanSense")
page = st.sidebar.radio("Navigate", NAV_PAGES, index=0, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Sensor values are simulated for now (no hardware connected yet). "
    "Swap `telemetry.py` for the real LoRa/Wi-Fi feed once the hub is built."
)

# ==========================================================================
# Live data (shared across pages)
# ==========================================================================


def get_soil_moisture():
    """Farm-wide soil moisture — average across all sensor nodes."""
    readings = get_latest_readings()
    return sum(r.soil_moisture_pct for r in readings) / len(readings), readings


def get_temperature():
    readings = get_latest_readings()
    return sum(r.temperature_c for r in readings) / len(readings), readings


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

if "history" not in st.session_state:
    st.session_state.history = []
st.session_state.history.append(
    {"t": time.time(), **{r.node_id: r.soil_moisture_pct for r in readings}}
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


def render_card(icon_label: str, value: str, status: str, status_text: str):
    css_class = STATUS_CLASS.get(status, "ks-status-good")
    st.markdown(
        f"""
        <div class="ks-card">
            <div class="ks-icon-label">{icon_label}</div>
            <div class="ks-value">{value}</div>
            <div class="ks-status {css_class}">{status}</div>
            <div style="color:#8A7A63; font-size:0.9rem; margin-top:4px;">{status_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
SYSTEM_PROMPT = f"""You are the KisanSense farming assistant, talking directly to a farmer
who may not be technical. Answer in simple, plain language about irrigation, crop health,
pests, disease, fertilizer, and weather-related farm decisions. Use the live field data
below when relevant. Keep answers short and actionable.

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
    "💧 Should I irrigate now?",
    "🌱 Is my crop healthy?",
    "🐛 How do I control pests?",
    "🌡 Is the temperature dangerous?",
    "🌧 Is rain expected?",
]

# ==========================================================================
# HOME
# ==========================================================================
if page == "HOME":
    st.markdown('<div class="ks-header-title">🌱 KisanSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="ks-header-sub">Smart Farming Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="ks-welcome">Good Morning, Farmer 🌱</div>', unsafe_allow_html=True)
    st.markdown('<div class="ks-status-line">Your farm is being monitored in real time.</div>', unsafe_allow_html=True)

    # --- Main sensor cards -------------------------------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        m_status = moisture_status(avg_moisture)
        render_card("💧 Soil Moisture", f"{avg_moisture:.0f}%", m_status, "")
    with col2:
        t_status = temp_status(avg_temp)
        render_card("🌡 Temperature", f"{avg_temp:.0f}°C", t_status, "")
    with col3:
        render_card("🌱 Crop Health", crop_health, crop_health, crop_health_note)

    # --- Chatbot -------------------------------------------------------------
    st.markdown('<div class="ks-section-title">🤖 Ask KisanSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="ks-section-sub">Your AI Farming Assistant</div>', unsafe_allow_html=True)

    if not st.session_state.chat_messages:
        st.info(
            "Namaste! 👋 I'm your KisanSense farming assistant. Ask me about irrigation, "
            "crop health, pests, diseases, fertilizer, weather, or anything about your farm."
        )

    qcols = st.columns(len(QUICK_QUESTIONS))
    for qcol, qtext in zip(qcols, QUICK_QUESTIONS):
        if qcol.button(qtext, use_container_width=True):
            handle_question(qtext.split(" ", 1)[1])

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    typed_question = st.chat_input("Ask about your farm...")
    if typed_question:
        handle_question(typed_question)

    if not ANTHROPIC_KEY and not OPENAI_KEY:
        st.caption(
            "ℹ️ Offline mode — simple answers from live farm data. Add an API key "
            "secret for fuller conversational answers."
        )

    # --- Today's recommendation ----------------------------------------------
    st.markdown('<div class="ks-section-title">🌾 Today\'s Recommendation</div>', unsafe_allow_html=True)
    if avg_moisture < 30:
        reco_title = "Irrigation recommended"
        reco_action = "Check irrigation for this field in the morning."
    elif avg_temp > 36:
        reco_title = "Watch for heat stress"
        reco_action = "Consider shade netting or extra watering during peak heat hours."
    else:
        reco_title = "No action needed"
        reco_action = "Conditions look good — continue normal monitoring."

    st.markdown(
        f"""
        <div class="ks-reco-card">
            <div style="font-weight:800; color:#2F4F33; font-size:1.1rem;">{reco_title}</div>
            <div style="color:#5C4A38; margin-top:4px;">
                Soil moisture is currently {avg_moisture:.0f}% and the temperature is {avg_temp:.0f}°C.
            </div>
            <div style="margin-top:8px; font-weight:700; color:#6B4F32;">Recommended action:</div>
            <div style="color:#5C4A38;">{reco_action}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("View Recommendation"):
        st.write(
            f"- Farm-wide soil moisture: **{avg_moisture:.0f}%** ({moisture_status(avg_moisture)})\n"
            f"- Farm-wide temperature: **{avg_temp:.0f}°C** ({temp_status(avg_temp)})\n"
            f"- Crop health rollup: **{crop_health}**\n\n{reco_action}"
        )

    # --- Alerts ----------------------------------------------------------------
    st.markdown('<div class="ks-section-title">⚠ Farm Alerts</div>', unsafe_allow_html=True)
    m_alerts = moisture_alerts(readings)
    b_alerts = battery_alerts(readings)
    if not m_alerts and not b_alerts and crop_health == "Good":
        st.markdown('<div class="ks-alert-good">🟢 No critical alerts</div>', unsafe_allow_html=True)
    else:
        for a in m_alerts:
            st.markdown(f'<div class="ks-alert-warn">🟡 {a}</div>', unsafe_allow_html=True)
        for a in b_alerts:
            st.markdown(f'<div class="ks-alert-bad">🔴 {a}</div>', unsafe_allow_html=True)
        if crop_health == "Poor":
            st.markdown('<div class="ks-alert-bad">🔴 Possible crop stress detected</div>', unsafe_allow_html=True)

# ==========================================================================
# MY FARM — node & power status
# ==========================================================================
elif page == "MY FARM":
    st.title("My Farm")
    st.caption("Status of every sensor node and the main hub.")

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

    st.markdown("### Main hub")
    c1, c2, c3 = st.columns(3)
    c1.metric("Hub battery", f"{hub.battery_pct:.0f}%")
    c2.metric("Solar", "Charging" if hub.solar_charging else "Idle")
    c3.metric("Camera", "Online" if hub.camera_online else "Offline")

# ==========================================================================
# CROP HEALTH — camera / edge-AI panel
# ==========================================================================
elif page == "CROP HEALTH":
    st.title("Crop Health")
    st.caption(
        "Upload a field photo to check canopy health. This uses a lightweight "
        "placeholder heuristic until the real edge-AI model is trained and "
        "flashed to the hub."
    )

    uploaded = st.file_uploader("Upload a photo from the field", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded photo", use_container_width=True)
        with col2:
            result = analyze_image(image)
            st.metric("Verdict", result.verdict)
            st.progress(result.green_ratio, text=f"Green cover: {result.green_ratio*100:.0f}%")
            st.info(result.note)
    else:
        st.markdown(f"**Last hub inference:** {hub.last_inference} ({hub.last_inference_confidence*100:.0f}% confidence)")

# ==========================================================================
# IRRIGATION
# ==========================================================================
elif page == "IRRIGATION":
    st.title("Irrigation")
    st.caption("Per-node moisture and watering guidance.")

    for r in readings:
        status = moisture_status(r.soil_moisture_pct)
        action = "Irrigate soon" if status == "Low" else ("Reduce watering" if status == "High" else "No action needed")
        st.markdown(f"**{r.node_id}** — {r.soil_moisture_pct:.0f}% ({status}) — {action}")
        st.progress(min(r.soil_moisture_pct / 100, 1.0))

# ==========================================================================
# ALERTS
# ==========================================================================
elif page == "ALERTS":
    st.title("Farm Alerts")
    m_alerts = moisture_alerts(readings)
    b_alerts = battery_alerts(readings)
    if not m_alerts and not b_alerts:
        st.markdown('<div class="ks-alert-good">🟢 No critical alerts</div>', unsafe_allow_html=True)
    for a in m_alerts:
        st.markdown(f'<div class="ks-alert-warn">🟡 {a}</div>', unsafe_allow_html=True)
    for a in b_alerts:
        st.markdown(f'<div class="ks-alert-bad">🔴 {a}</div>', unsafe_allow_html=True)

# ==========================================================================
# ANALYTICS — all the charts/history live here now
# ==========================================================================
elif page == "ANALYTICS":
    st.title("Farm Analytics")
    st.caption("Historical trends and detailed sensor data.")

    df = pd.DataFrame(st.session_state.history).set_index("t")
    df.index = pd.to_datetime(df.index, unit="s")
    st.markdown("### Soil moisture over time")
    st.line_chart(df)

    st.markdown("### Raw node readings")
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

    if st.checkbox("Auto-refresh every 5s"):
        time.sleep(5)
        st.rerun()

# ==========================================================================
# SETTINGS
# ==========================================================================
elif page == "SETTINGS":
    st.title("Settings")
    st.markdown("### Chatbot")
    st.write(f"Anthropic key configured: {'✅' if ANTHROPIC_KEY else '❌'}")
    st.write(f"OpenAI key configured: {'✅' if OPENAI_KEY else '❌'}")
    st.caption("Add ANTHROPIC_API_KEY or OPENAI_API_KEY under Settings → Secrets on Streamlit Cloud.")

    st.markdown("### Data source")
    st.write("Sensor readings: **Simulated** (telemetry.py) — no hardware connected yet.")

    if st.button("Clear chat history"):
        st.session_state.chat_messages = []
        st.rerun()
