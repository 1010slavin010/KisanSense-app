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
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================================
# Navigation
# ==========================================================================
NAV_PAGES = {
    "HOME": "🏠 Home",
    "MY FARM": "🌾 My Farm",
    "CROP HEALTH": "🌿 Crop Health",
    "IRRIGATION": "💧 Irrigation",
    "ALERTS": "⚠️ Alerts",
    "ANALYTICS": "📈 Analytics",
    "SETTINGS": "⚙️ Settings",
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
    st.markdown('<div class="ks-welcome">Namaste! Farmer, welcome back to KisanSense.</div>', unsafe_allow_html=True)
    st.markdown('<div class="ks-status-line">Your farm is being monitored in real time.</div>', unsafe_allow_html=True)

    # --- Insights: quick-question pills ------------------------------------
    st.markdown('<div class="ks-section-title">Insights</div>', unsafe_allow_html=True)
    qcols = st.columns(len(QUICK_QUESTIONS))
    for i, (qcol, qtext) in enumerate(zip(qcols, QUICK_QUESTIONS)):
        if qcol.button(qtext, use_container_width=True, key=f"quick_{i}"):
            handle_question(qtext.split(" ", 1)[1])

    # --- Today's Recommendation + Farm Alerts, side by side ----------------
    col_reco, col_alert = st.columns(2)

    if avg_moisture < 30:
        reco_title, reco_action, reco_status = "Irrigation recommended", "Check irrigation for this field in the morning.", "warn"
    elif avg_temp > 36:
        reco_title, reco_action, reco_status = "Watch for heat stress", "Consider shade netting or extra watering during peak heat hours.", "warn"
    else:
        reco_title, reco_action, reco_status = "No action needed", "Looks good — continue normal monitoring.", "good"

    reco_icon = "✅" if reco_status == "good" else "🟡"
    with col_reco:
        st.markdown(
            f"""
            <div class="ks-card">
                <div class="ks-card-title">🌾 Today's Recommendation</div>
                <div style="font-weight:700; margin-top:4px;">{reco_icon} {reco_title}</div>
                <div class="ks-alert-row">Soil moisture {avg_moisture:.0f}%, Temp {avg_temp:.0f}°C</div>
                <div class="ks-alert-row"><b>Recommended action:</b> {reco_action}</div>
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
                """
                <div class="ks-card">
                    <div class="ks-card-title">⚠ Farm Alerts</div>
                    <span class="ks-badge-good">ALL CLEAR</span>
                    <div class="ks-alert-row">No critical alerts. All systems operational.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            rows = "".join(f'<div class="ks-alert-row">🔴 {a}</div>' for a in all_alerts)
            st.markdown(
                f"""
                <div class="ks-card">
                    <div class="ks-card-title">⚠ Farm Alerts</div>
                    <span class="ks-badge-warn">NEEDS ATTENTION</span>
                    {rows}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- Live Data Dashboard: mini sparklines --------------------------------
    st.markdown('<div class="ks-section-title">Live Data Dashboard</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        mini_chart("Soil moisture", "💧", "Soil moisture", "{:.0f}%")
    with c2:
        mini_chart("Air temperature", "🌡", "Air temperature", "{:.0f}°C")
    with c3:
        mini_chart("Light levels", "☀️", "Light level", "{:.0f}%")

    # --- Ask KisanSense --------------------------------------------------------
    st.markdown('<div class="ks-section-title">💬 Ask KisanSense</div>', unsafe_allow_html=True)
    if not st.session_state.chat_messages:
        st.caption("Namaste! 👋 Ask about irrigation, crop health, pests, weather, or anything about your farm.")

    for msg in st.session_state.chat_messages[-6:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    typed_question = st.chat_input("Ask KisanSense...")
    if typed_question:
        handle_question(typed_question)

    if not ANTHROPIC_KEY and not OPENAI_KEY:
        st.caption("ℹ️ Offline mode — simple answers from live farm data. Add an API key secret for fuller answers.")

# ==========================================================================
# MY FARM — node & power status
# ==========================================================================
elif page == "MY FARM":
    st.markdown('<div class="ks-welcome">My Farm</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="ks-section-title">Main hub</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="ks-welcome">Crop Health</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="ks-welcome">Irrigation</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="ks-welcome">Farm Alerts</div>', unsafe_allow_html=True)
    m_alerts = moisture_alerts(readings)
    b_alerts = battery_alerts(readings)
    if not m_alerts and not b_alerts:
        st.markdown('<span class="ks-badge-good">ALL CLEAR</span>', unsafe_allow_html=True)
        st.caption("No critical alerts. All systems operational.")
    for a in m_alerts:
        st.markdown(f'<div class="ks-card" style="margin-bottom:8px;">🟡 {a}</div>', unsafe_allow_html=True)
    for a in b_alerts:
        st.markdown(f'<div class="ks-card" style="margin-bottom:8px;">🔴 {a}</div>', unsafe_allow_html=True)

# ==========================================================================
# ANALYTICS — all the charts/history live here now
# ==========================================================================
elif page == "ANALYTICS":
    st.markdown('<div class="ks-welcome">Farm Analytics</div>', unsafe_allow_html=True)
    st.caption("Historical trends and detailed sensor data.")

    df = pd.DataFrame(st.session_state.history).set_index("t")
    df.index = pd.to_datetime(df.index, unit="s")
    st.markdown('<div class="ks-section-title">Trends</div>', unsafe_allow_html=True)
    st.line_chart(df)

    st.markdown('<div class="ks-section-title">Raw node readings</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="ks-welcome">Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="ks-section-title">Chatbot</div>', unsafe_allow_html=True)
    st.write(f"Anthropic key configured: {'✅' if ANTHROPIC_KEY else '❌'}")
    st.write(f"OpenAI key configured: {'✅' if OPENAI_KEY else '❌'}")
    st.caption("Add ANTHROPIC_API_KEY or OPENAI_API_KEY under Settings → Secrets on Streamlit Cloud.")

    st.markdown('<div class="ks-section-title">Data source</div>', unsafe_allow_html=True)
    st.write("Sensor readings: **Simulated** (telemetry.py) — no hardware connected yet.")

    st.markdown('<div class="ks-section-title">Appearance</div>', unsafe_allow_html=True)
    st.write(f"Current theme: **{st.session_state.theme}** (toggle in the sidebar)")

    if st.button("Clear chat history"):
        st.session_state.chat_messages = []
        st.rerun()
