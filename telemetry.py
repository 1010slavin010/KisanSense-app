"""
Telemetry layer for KisanSense.

Right now this generates simulated readings for each sensor node and the
main hub, so the app is usable before the hardware is built. Swap
`get_latest_readings()` / `get_hub_status()` for real calls once the hub is
sending data (e.g. the hub posts JSON to a small REST endpoint or writes to
a CSV/SQLite file that this module reads instead of `random`).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field


@dataclass
class NodeReading:
    node_id: str
    soil_moisture_pct: float
    temperature_c: float
    humidity_pct: float
    battery_pct: float
    solar_charging: bool
    lora_rssi_dbm: int
    last_seen: float = field(default_factory=time.time)


NODE_IDS = ["Node-1 (North field)", "Node-2 (South field)", "Node-3 (Greenhouse)"]

# Persistent-ish in-memory state so values drift instead of jumping randomly
# every rerun. Streamlit reruns the script on every interaction, so this
# lives in a module-level dict keyed by node id.
_STATE: dict[str, NodeReading] = {}


def _drift(value: float, low: float, high: float, step: float) -> float:
    value += random.uniform(-step, step)
    return max(low, min(high, value))


def _init_node(node_id: str) -> NodeReading:
    return NodeReading(
        node_id=node_id,
        soil_moisture_pct=random.uniform(35, 55),
        temperature_c=random.uniform(24, 30),
        humidity_pct=random.uniform(45, 65),
        battery_pct=random.uniform(70, 100),
        solar_charging=True,
        lora_rssi_dbm=random.randint(-95, -60),
    )


def get_latest_readings() -> list[NodeReading]:
    """Return one simulated (and slowly drifting) reading per sensor node."""
    for node_id in NODE_IDS:
        if node_id not in _STATE:
            _STATE[node_id] = _init_node(node_id)
        r = _STATE[node_id]
        r.soil_moisture_pct = _drift(r.soil_moisture_pct, 5, 90, 2.5)
        r.temperature_c = _drift(r.temperature_c, 15, 42, 0.6)
        r.humidity_pct = _drift(r.humidity_pct, 20, 95, 2.0)
        # battery slowly drains, tops back up when "solar_charging"
        drain = -0.15 if not r.solar_charging else 0.1
        r.battery_pct = _drift(r.battery_pct + drain, 0, 100, 0.5)
        r.solar_charging = random.random() > 0.15
        r.lora_rssi_dbm = int(_drift(r.lora_rssi_dbm, -110, -40, 4))
        r.last_seen = time.time()
    return list(_STATE.values())


@dataclass
class HubStatus:
    battery_pct: float
    solar_charging: bool
    camera_online: bool
    last_inference: str
    last_inference_confidence: float


_HUB_STATE = HubStatus(
    battery_pct=88.0,
    solar_charging=True,
    camera_online=True,
    last_inference="Healthy crop canopy",
    last_inference_confidence=0.91,
)


def get_hub_status() -> HubStatus:
    _HUB_STATE.battery_pct = _drift(_HUB_STATE.battery_pct, 0, 100, 0.4)
    _HUB_STATE.solar_charging = random.random() > 0.15
    return _HUB_STATE


def moisture_alerts(readings: list[NodeReading], threshold: float = 25.0) -> list[str]:
    return [
        f"{r.node_id}: soil moisture at {r.soil_moisture_pct:.1f}% — below {threshold:.0f}% threshold, consider irrigating"
        for r in readings
        if r.soil_moisture_pct < threshold
    ]


def battery_alerts(readings: list[NodeReading], threshold: float = 20.0) -> list[str]:
    return [
        f"{r.node_id}: battery at {r.battery_pct:.0f}% and not currently charging"
        for r in readings
        if r.battery_pct < threshold and not r.solar_charging
    ]
