# KisanSense App

A working Streamlit app for the KisanSense features: a live field dashboard, node/power
status, a camera panel that runs a canopy-health check, and a chatbot that can see current
sensor readings and answer questions.

## Files
- `app.py` — the app (4 tabs: Dashboard, Nodes & Power, Camera / AI Hub, Chatbot)
- `telemetry.py` — sensor data layer. **Simulated for now** (no hardware yet) — swap
  `get_latest_readings()` / `get_hub_status()` for real calls once the hub is sending
  data over LoRa/Wi-Fi.
- `vision.py` — lightweight greenness-ratio check for uploaded field photos, standing in
  for the real edge-AI model until one is trained and flashed to the ESP32-S3 hub.
- `requirements.txt`, `.gitignore`, `.streamlit/secrets.toml.example` — setup files.

## Run locally
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
The dashboard and chatbot work immediately with simulated data — no API key required.
For full conversational chatbot answers instead of the built-in rule-based fallback:
```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and add a real key
```

## Push to GitHub
```bash
cd kisansense2
git init
git add .
git commit -m "KisanSense monitoring app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Deploy on Streamlit Community Cloud
1. <https://share.streamlit.io> → sign in with GitHub → **New app**
2. Pick your repo/branch, set main file to `app.py`, click **Deploy**
3. In **Settings → Secrets**, paste `ANTHROPIC_API_KEY = "sk-ant-..."` for full chatbot answers

## Recommendations
- **Wire up real data next**: the natural next step is having the hub push readings to a
  small endpoint this app reads from — easiest path is the hub POSTing JSON to a free
  service (e.g. a lightweight FastAPI backend, or even writing rows to a Google Sheet /
  Firebase RTDB) and swapping `telemetry.py` to read from there instead of `random`.
- **Camera model**: `vision.py`'s greenness check is a placeholder so the tab is usable
  today. Once you have a trained model (even a small TFLite model on the ESP32-S3), have
  it output a JSON verdict and swap that in — the UI (`app.py`) won't need to change.
- **Auto-refresh**: the Dashboard polls every 5s by default; on Streamlit Community Cloud
  this keeps the app "awake" during a demo, but for a real deployment you'd want the
  refresh interval tied to how often nodes actually report (LoRa duty cycle), not a fixed
  timer.
- **Alerts**: moisture/battery alert thresholds are hardcoded (25% moisture, 20% battery)
  in `telemetry.py` — worth making these configurable per node once you know real crop
  water needs.
- **Costs**: Anthropic/OpenAI chat calls are pay-per-token; the built-in offline fallback
  (rule-based over live sensor values) is free and enough for demos before you wire up a
  key.
