"""Tests for sensor_groups.py"""
from module_03_sensor_fusion.sensor_groups import SensorGroupRegistry, DEFAULT_SENSOR_GROUPS


def test_registry_defaults():
    reg = SensorGroupRegistry()
    assert "optical" in reg.groups
    assert "photodiode_1" in reg.groups["optical"]
    assert "motion" in reg.groups
    assert "accel_x" in reg.groups["motion"]


def test_register_custom_channel():
    reg = SensorGroupRegistry()
    reg.register_channel("custom_group", "sensor_x")
    assert "custom_group" in reg.groups
    assert "sensor_x" in reg.groups["custom_group"]


def test_get_group_for_channel():
    reg = SensorGroupRegistry()
    assert reg.get_group_for_channel("temperature") == "environment"
    assert reg.get_group_for_channel("distance") == "distance"
    assert reg.get_group_for_channel("non_existent") is None


def test_get_channels_in_group():
    reg = SensorGroupRegistry()
    available = {"photodiode_1", "temperature"}
    opt_chs = reg.get_channels_in_group("optical", available)
    assert opt_chs == ["photodiode_1"]
