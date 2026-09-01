"""
telemetry.py — simulated sensor feed for KisanSense.

Swap this module for a real LoRa/Wi-Fi/MQTT feed once hardware (ESP32 nodes +
hub) is wired up. Every function here keeps the exact same signature/shape so
the rest of the app (app.py) does not need to change when real hardware
arrives.
"""

import math
import random
import time
from dataclasses import dataclass


@dataclass
class NodeReading:
    node_id: str
    soil_moisture_pct: float
    temperature_c: float
    humidity_pct: float
    battery_pct: float
    solar_charging: bool
    lora_rssi_dbm: int


@dataclass
class HubStatus:
    battery_pct: float
    solar_charging: bool
    camera_online: bool
    last_inference: str
    last_inference_confidence: float


_NODE_IDS = ["Node-01", "Node-02", "Node-03", "Node-04"]


def _wobble(base: float, amplitude: float, speed: float, phase: float = 0.0) -> float:
    return base + amplitude * math.sin(time.time() / speed + phase)


def get_latest_readings():
    readings = []
    for i, node_id in enumerate(_NODE_IDS):
        moisture = max(5, min(95, _wobble(45, 20, 40, phase=i) + random.uniform(-2, 2)))
        temp = max(10, min(45, _wobble(29, 6, 50, phase=i * 1.3) + random.uniform(-0.5, 0.5)))
        humidity = max(10, min(100, _wobble(60, 15, 35, phase=i * 0.7) + random.uniform(-2, 2)))
        battery = max(5, min(100, 80 - i * 6 + random.uniform(-3, 3)))
        readings.append(
            NodeReading(
                node_id=node_id,
                soil_moisture_pct=moisture,
                temperature_c=temp,
                humidity_pct=humidity,
                battery_pct=battery,
                solar_charging=random.random() > 0.4,
                lora_rssi_dbm=int(-60 - i * 8 + random.uniform(-5, 5)),
            )
        )
    return readings


def get_hub_status():
    return HubStatus(
        battery_pct=max(10, min(100, _wobble(75, 10, 60))),
        solar_charging=random.random() > 0.3,
        camera_online=True,
        last_inference="Healthy canopy",
        last_inference_confidence=0.87,
    )


def moisture_alerts(readings):
    alerts = []
    for r in readings:
        if r.soil_moisture_pct < 20:
            alerts.append(f"{r.node_id}: soil moisture critically low ({r.soil_moisture_pct:.0f}%)")
        elif r.soil_moisture_pct > 85:
            alerts.append(f"{r.node_id}: soil moisture very high ({r.soil_moisture_pct:.0f}%) — check for waterlogging")
    return alerts


def battery_alerts(readings):
    alerts = []
    for r in readings:
        if r.battery_pct < 20:
            alerts.append(f"{r.node_id}: battery low ({r.battery_pct:.0f}%)")
    return alerts


def get_weather():
    """Simulated current weather + short forecast until a real weather API is wired up."""
    rain_chance = max(0, min(100, _wobble(35, 30, 45)))
    return {
        "temp_c": round(_wobble(29, 5, 55), 1),
        "humidity_pct": round(_wobble(68, 12, 40), 0),
        "rain_chance_pct": round(rain_chance, 0),
        "wind_kmh": round(max(2, _wobble(10, 6, 30)), 1),
        "condition": "Rain likely later" if rain_chance > 55 else ("Partly cloudy" if rain_chance > 25 else "Clear"),
    }
