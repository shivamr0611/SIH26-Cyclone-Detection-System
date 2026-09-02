#!/usr/bin/env python3
"""
test_false_positives.py - False Positive Detection & Validation Test Suite for CycloneAI.

Asserts detected=False on non-cyclone images:
1. Random noise image
2. Solid gray / blank image
3. Synthetic nighttime city lights image (scattered bright dots on dark background)
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.preprocessor import preprocess_tir, validate_satellite_image
from detection.detector import detect_cyclone


def test_random_noise_image():
    """Random noise should fail cyclone detection gate (low/dispersed concentration)."""
    np.random.seed(42)
    img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    raw_bytes = buf.tobytes()

    pre = preprocess_tir(raw_bytes)
    det = detect_cyclone(pre["processed_image_b64"])

    print("\n[1] Random Noise Test:")
    print(f"    Detected:             {det['cyclone_detected']}")
    print(f"    Reason:               {det['not_detected_reason']}")
    print(f"    Cloud Coverage:       {pre['cloud_coverage_pct']}%")
    print(f"    Vortex Concentration: {det['vortex_concentration_score']}")

    assert not det["cyclone_detected"], "Failed: Random noise was falsely flagged as a cyclone!"


def test_solid_gray_image():
    """Solid uniform image should fail detection gate due to zero convective features."""
    img = np.full((256, 256), 180, dtype=np.uint8)  # uniform warm ocean / gray
    _, buf = cv2.imencode(".png", img)
    raw_bytes = buf.tobytes()

    pre = preprocess_tir(raw_bytes)
    det = detect_cyclone(pre["processed_image_b64"])

    print("\n[2] Solid Gray Test:")
    print(f"    Detected:             {det['cyclone_detected']}")
    print(f"    Reason:               {det['not_detected_reason']}")
    print(f"    Cloud Coverage:       {pre['cloud_coverage_pct']}%")
    print(f"    CDO Radius:           {det['cdo_radius_km']} km")

    assert not det["cyclone_detected"], "Failed: Solid gray image was falsely flagged as a cyclone!"


def test_synthetic_city_lights_image():
    """City lights (scattered bright point-like dots on dark background) should not detect a cyclone."""
    np.random.seed(123)
    img = np.zeros((256, 256), dtype=np.uint8)  # dark background

    # Scatter 80 small bright isolated points (simulating nighttime city lights)
    for _ in range(80):
        rx, ry = np.random.randint(10, 246), np.random.randint(10, 246)
        r_size = np.random.randint(1, 3)
        cv2.circle(img, (rx, ry), r_size, int(np.random.randint(210, 255)), -1)

    _, buf = cv2.imencode(".png", img)
    raw_bytes = buf.tobytes()

    # Check validator or detector gate
    is_valid, val_err = validate_satellite_image(raw_bytes)
    pre = preprocess_tir(raw_bytes)
    det = detect_cyclone(pre["processed_image_b64"])

    print("\n[3] Synthetic City Lights Test:")
    print(f"    Validator Result:     is_valid={is_valid}, error='{val_err}'")
    print(f"    Detected:             {det['cyclone_detected']}")
    print(f"    Reason:               {det['not_detected_reason']}")
    print(f"    Vortex Concentration: {det['vortex_concentration_score']}")

    # Either validator caught city lights OR detector gate marked detected=False
    assert not det["cyclone_detected"] or not is_valid, "Failed: City lights image was falsely flagged as a cyclone!"


def test_real_cyclone_positive_control():
    """Positive control: Organized circular convective vortex with cold core must be detected."""
    img = np.ones((512, 512), dtype=np.uint8) * 200  # warm ocean background
    cx, cy = 256, 256
    y, x = np.ogrid[:512, :512]
    # Concentrated cold convective core (radius 100px = 400km CDO)
    mask = (x - cx)**2 + (y - cy)**2 <= 100**2
    img[mask] = 40  # cold cloud top

    _, buf = cv2.imencode(".png", img)
    raw_bytes = buf.tobytes()

    pre = preprocess_tir(raw_bytes)
    det = detect_cyclone(pre["processed_image_b64"])

    print("\n[4] Positive Control (Organized Cyclone Vortex):")
    print(f"    Detected:             {det['cyclone_detected']}")
    print(f"    CDO Radius:           {det['cdo_radius_km']} km")
    print(f"    Vortex Concentration: {det['vortex_concentration_score']}")

    assert det["cyclone_detected"], "Failed: Legitimate cyclone vortex was missed by detector!"


if __name__ == "__main__":
    print("=== Running CycloneAI False Positive & Gate Tests ===")
    test_random_noise_image()
    test_solid_gray_image()
    test_synthetic_city_lights_image()
    test_real_cyclone_positive_control()
    print("\n[ALL TESTS PASSED] False positive rejection gates verified successfully!")
