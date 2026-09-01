"""
KisanSense — live monitoring app (Streamlit)

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

st.set_page_config(page_title="KisanSense", page_icon="🌾", layout="wide")

BASE_DIR = Path(__file__).parent

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("🌾 KisanSense")
page = st.sidebar.radio(
    "Navigate", ["Dashboard", "Nodes & Power", "Camera / AI Hub", "Chatbot"]
)
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto-refresh every 5s", value=(page == "Dashboard"))
st.sidebar.caption(
    "Sensor values are simulated for now (no hardware connected yet). Swap "
    "`telemetry.py` for the real LoRa/Wi-Fi feed once the hub is built."
)

if "history" not in st.session_state:
    st.session_state.history = []  # list of (timestamp, {node_id: reading})

readings = get_latest_readings()
hub = get_hub_status()

st.session_state.history.append(
    {
        "t": time.time(),
        **{r.node_id: r.soil_moisture_pct for r in readings},
    }
)
st.session_state.history = st.session_state.history[-120:]  # keep ~10 min at 5s

# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------
if page == "Dashboard":
    st.title("Field Dashboard")

    alerts = moisture_alerts(readings) + battery_alerts(readings)
    if alerts:
        for a in alerts:
            st.warning(a, icon="⚠️")
    else:
        st.success("All nodes nominal — no alerts.", icon="✅")

    cols = st.columns(len(readings))
    for col, r in zip(cols, readings):
        with col:
            st.metric(r.node_id, f"{r.soil_moisture_pct:.0f}% moisture")
            st.caption(f"🌡️ {r.temperature_c:.1f}°C  •  💧 {r.humidity_pct:.0f}% RH")
            st.progress(r.battery_pct / 100, text=f"Battery {r.battery_pct:.0f}%")

    st.markdown("### Soil moisture — last readings")
    df = pd.DataFrame(st.session_state.history).set_index("t")
    df.index = pd.to_datetime(df.index, unit="s")
    st.line_chart(df)

# --------------------------------------------------------------------------
# Nodes & Power
# --------------------------------------------------------------------------
elif page == "Nodes & Power":
    st.title("Nodes & Power Status")

    st.markdown("### Sensor nodes")
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

    st.caption(
        "Reference power design: sensor nodes on 3.2V 6Ah LiFePO4 + 5W solar; "
        "main hub on 3.2V 10Ah LiFePO4 + 10W solar, regulated down to 3.3V for "
        "the ESP32-S3."
    )

# --------------------------------------------------------------------------
# Camera / AI Hub
# --------------------------------------------------------------------------
elif page == "Camera / AI Hub":
    st.title("Camera / AI Hub")
    st.caption(
        "Upload a field photo to run it through the hub's canopy-health check. "
        "This is a lightweight placeholder heuristic — swap `vision.py` for the "
        "real edge-AI model's output once it's trained and flashed to the "
        "ESP32-S3 hub."
    )

    uploaded = st.file_uploader("Upload a photo from the field", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded photo", use_container_width=True)
        with col2:
            result = analyze_image(image)
            st.metric("Verdict", result.verdict)
            st.progress(result.green_ratio, text=f"Green cover: {result.green_ratio*100:.0f}%")
            st.info(result.note)
    else:
        st.markdown(f"**Last hub inference:** {hub.last_inference} "
                    f"({hub.last_inference_confidence*100:.0f}% confidence)")

# --------------------------------------------------------------------------
# Chatbot
# --------------------------------------------------------------------------
elif page == "Chatbot":
    st.title("💬 KisanSense Assistant")
    st.caption("Ask about current field conditions, node/battery status, or the hardware design.")

    def get_api_key(name: str):
        try:
            if name in st.secrets:
                return st.secrets[name]
        except Exception:
            pass
        return os.environ.get(name)

    anthropic_key = get_api_key("ANTHROPIC_API_KEY")
    openai_key = get_api_key("OPENAI_API_KEY")

    live_context = "\n".join(
        f"- {r.node_id}: moisture {r.soil_moisture_pct:.0f}%, temp {r.temperature_c:.1f}C, "
        f"humidity {r.humidity_pct:.0f}%, battery {r.battery_pct:.0f}% "
        f"({'charging' if r.solar_charging else 'idle'}), LoRa RSSI {r.lora_rssi_dbm}dBm"
        for r in readings
    )
    hub_context = (
        f"Hub: battery {hub.battery_pct:.0f}% "
        f"({'charging' if hub.solar_charging else 'idle'}), "
        f"camera {'online' if hub.camera_online else 'offline'}, "
        f"last inference '{hub.last_inference}' at {hub.last_inference_confidence*100:.0f}% confidence."
    )

    SYSTEM_PROMPT = f"""You are the KisanSense farm assistant. You can see the current
(simulated) sensor readings below and should use them when giving irrigation or
maintenance advice. Be concise and practical. If asked about the hardware design
(boards, LoRa, battery choices), answer from general knowledge of the project: ESP32-S3
sensor nodes + ESP32-S3 camera/AI hub over SX1278 LoRa, LiFePO4 batteries (6Ah nodes /
10Ah hub) with 5W/10W solar panels.

Current field data:
{live_context}
{hub_context}
"""

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    def fallback_answer(question: str) -> str:
        q = question.lower()
        if "water" in q or "irrigat" in q or "moisture" in q:
            low = [r for r in readings if r.soil_moisture_pct < 25]
            if low:
                return "Irrigate now:\n" + "\n".join(
                    f"- {r.node_id}: {r.soil_moisture_pct:.0f}% moisture" for r in low
                )
            return "All nodes are above the 25% moisture threshold — no irrigation needed right now."
        if "battery" in q or "power" in q or "solar" in q:
            lines = [f"- {r.node_id}: {r.battery_pct:.0f}% ({'charging' if r.solar_charging else 'idle'})" for r in readings]
            lines.append(f"- Hub: {hub.battery_pct:.0f}% ({'charging' if hub.solar_charging else 'idle'})")
            return "Current power status:\n" + "\n".join(lines)
        if "camera" in q or "hub" in q:
            return hub_context
        return (
            "I can answer from live sensor data (ask about moisture, battery, or the "
            "camera/hub) with a simple built-in mode, or give fuller answers if you "
            "add an ANTHROPIC_API_KEY / OPENAI_API_KEY secret."
        )

    def stream_anthropic():
        import anthropic

        client = anthropic.Anthropic(api_key=anthropic_key)
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages]
        with client.messages.stream(
            model="claude-sonnet-4-6", max_tokens=1000, system=SYSTEM_PROMPT, messages=history
        ) as stream:
            for text in stream.text_stream:
                yield text

    def stream_openai():
        from openai import OpenAI

        client = OpenAI(api_key=openai_key)
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages]
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    question = st.chat_input("Ask KisanSense...")
    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            if anthropic_key:
                full_reply = st.write_stream(stream_anthropic())
            elif openai_key:
                full_reply = st.write_stream(stream_openai())
            else:
                full_reply = fallback_answer(question)
                st.markdown(full_reply)
        st.session_state.chat_messages.append({"role": "assistant", "content": full_reply})

    if not anthropic_key and not openai_key:
        st.info(
            "Offline mode: answers come from simple rules over live sensor data. "
            "Add an API key secret for full conversational answers.",
            icon="ℹ️",
        )

if auto_refresh:
    time.sleep(5)
    st.rerun()
