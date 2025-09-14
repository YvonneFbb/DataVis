#!/usr/bin/env python3

# Quick test to debug the border detection logic

import sys
import os
sys.path.append('/Users/yinwhe/Desktop/Fbb/DataVis/charsis')

from src.config import BORDER_REMOVAL_CONFIG

print("=== BORDER_REMOVAL_CONFIG ===")
for key, value in BORDER_REMOVAL_CONFIG.items():
    print(f"{key}: {value}")

print(f"\n=== Test gradient check logic ===")
max_coverage = 0.9
drop_magnitude = 0.2750
spike_gradient_threshold = BORDER_REMOVAL_CONFIG.get('spike_gradient_threshold', 0.4)
gradient_threshold = max_coverage * spike_gradient_threshold

print(f"max_coverage: {max_coverage}")
print(f"drop_magnitude: {drop_magnitude}")
print(f"spike_gradient_threshold: {spike_gradient_threshold}")
print(f"gradient_threshold: {gradient_threshold}")
print(f"gradient_ok: {drop_magnitude >= gradient_threshold}")

if drop_magnitude >= gradient_threshold:
    print("❌ This should be FALSE but evaluates to TRUE")
else:
    print("✅ This correctly evaluates to FALSE")