"""Sensor Group abstraction and registry for PhotonShield AI.

Groups sensors into functional domains:
- "optical": photodiode_1, photodiode_2, etc.
- "environment": temperature, humidity, pressure
- "motion": accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
- "distance": distance
- "quality": per-channel quality/validity channels

Allows dynamic channel registration without modifying pipeline code.
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional


DEFAULT_SENSOR_GROUPS: Dict[str, List[str]] = {
    "optical": ["photodiode_1", "photodiode_2"],
    "environment": ["temperature", "humidity", "pressure"],
    "motion": ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
    "distance": ["distance"],
}


class SensorGroupRegistry:
    """Registry managing sensor channel to group mappings."""

    def __init__(self, custom_groups: Optional[Dict[str, List[str]]] = None):
        self.groups: Dict[str, List[str]] = {}
        initial = custom_groups or DEFAULT_SENSOR_GROUPS
        for grp_name, channels in initial.items():
            self.groups[grp_name] = list(channels)

    def register_channel(self, group_name: str, channel_name: str) -> None:
        """Add a channel to a sensor group."""
        if group_name not in self.groups:
            self.groups[group_name] = []
        if channel_name not in self.groups[group_name]:
            self.groups[group_name].append(channel_name)

    def get_group_for_channel(self, channel_name: str) -> Optional[str]:
        """Find which group a channel belongs to."""
        for grp_name, channels in self.groups.items():
            if channel_name in channels:
                return grp_name
        return None

    def get_channels_in_group(self, group_name: str, available_channels: Set[str]) -> List[str]:
        """Return channels in group that exist in current dataset/stream."""
        grp_channels = self.groups.get(group_name, [])
        return [ch for ch in grp_channels if ch in available_channels]

    def get_available_groups(self, available_channels: Set[str]) -> List[str]:
        """Return group names that have at least one channel in available_channels."""
        active = []
        for grp_name, channels in self.groups.items():
            if any(ch in available_channels for ch in channels):
                active.append(grp_name)
        return active
