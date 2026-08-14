"""Embedded Hardware Runtime Specifications for Arduino Uno Q and Edge Targets.

Defines memory, compute, and clock constraints for micro-edge deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class HardwareProfile:
    """Hardware profile specification."""
    name: str
    flash_bytes: int          # Total available non-volatile flash storage
    sram_bytes: int           # Total fast SRAM memory for activations/stack
    clock_frequency_hz: int   # CPU clock frequency in Hz
    max_macs_budget: int      # Suggested real-time MAC budget per frame
    fpu_support: bool         # Hardware floating-point unit present


# Hardware Target Profiles
ARDUINO_UNO_Q_PROFILE = HardwareProfile(
    name="Arduino Uno Q (Target MCU)",
    flash_bytes=512 * 1024,        # 512 KB Flash
    sram_bytes=64 * 1024,          # 64 KB SRAM
    clock_frequency_hz=64_000_000, # 64 MHz
    max_macs_budget=150_000,       # 150k MACs for ~50ms frame budget
    fpu_support=False,             # Integer/Quantized optimized
)

STM32_H7_PROFILE = HardwareProfile(
    name="STM32H7 High-Performance Edge MCU",
    flash_bytes=2 * 1024 * 1024,   # 2 MB Flash
    sram_bytes=1024 * 1024,        # 1 MB SRAM
    clock_frequency_hz=480_000_000, # 480 MHz
    max_macs_budget=2_000_000,
    fpu_support=True,
)

GENERIC_CORTEX_M4_PROFILE = HardwareProfile(
    name="Generic ARM Cortex-M4 Microcontroller",
    flash_bytes=256 * 1024,        # 256 KB Flash
    sram_bytes=32 * 1024,          # 32 KB SRAM
    clock_frequency_hz=48_000_000, # 48 MHz
    max_macs_budget=80_000,
    fpu_support=False,
)
