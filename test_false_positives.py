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
    """Random noise should be rejected by validator or fail detection gate."""
    np.random.seed(42)
    img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    raw_bytes = buf.tobytes()

    is_valid, val_err = validate_satellite_image(raw_bytes)
    print("\n[1] Random Noise Test:")
    print(f"    Validator Result:     is_valid={is_valid}, error='{val_err}'")

    if is_valid:
        pre = preprocess_tir(raw_bytes)
        det = detect_cyclone(pre["processed_image_b64"])
        print(f"    Detected:             {det['cyclone_detected']}")
        print(f"    Reason:               {det['not_detected_reason']}")
        assert not det["cyclone_detected"], "Failed: Random noise was flagged as a cyclone!"
    else:
        assert not is_valid, "Random noise was correctly rejected by validator."


def test_solid_gray_image():
    """Solid uniform image should fail validator due to zero contrast."""
    img = np.full((256, 256), 180, dtype=np.uint8)  # uniform warm ocean / gray
    _, buf = cv2.imencode(".png", img)
    raw_bytes = buf.tobytes()

    is_valid, val_err = validate_satellite_image(raw_bytes)
    print("\n[2] Solid Gray Test:")
    print(f"    Validator Result:     is_valid={is_valid}, error='{val_err}'")

    assert not is_valid, "Failed: Solid uniform image should be rejected by validator!"


def test_synthetic_city_lights_image():
    """
    Nighttime city lights (e.g. NOAA Day-Night band over India) IS a valid satellite scan,
    so it MUST PASS validation, but MUST FAIL cyclone detection due to insufficient convective core.
    """
    np.random.seed(123)
    img = np.zeros((300, 300, 3), dtype=np.uint8)  # dark background

    # Scatter 80 small bright isolated points (simulating nighttime city lights)
    for _ in range(80):
        rx, ry = np.random.randint(10, 290), np.random.randint(10, 290)
        r_size = np.random.randint(1, 3)
        cv2.circle(img, (rx, ry), r_size, (200, 240, 255), -1)

    _, buf = cv2.imencode(".png", img)
    raw_bytes = buf.tobytes()

    # Step 1: Validator MUST accept nighttime satellite scan
    is_valid, val_err = validate_satellite_image(raw_bytes)
    print("\n[3] Synthetic City Lights (Nighttime Visible Satellite) Test:")
    print(f"    Validator Result:     is_valid={is_valid} (Must be True for real satellite scans)")

    assert is_valid, f"Failed: Night visible satellite image was wrongly rejected by validator: {val_err}"

    # Step 2: Detector MUST return detected=False
    pre = preprocess_tir(raw_bytes)
    det = detect_cyclone(pre["processed_image_b64"])
    print(f"    Cloud Coverage:       {pre['cloud_coverage_pct']}%")
    print(f"    Detected:             {det['cyclone_detected']}")
    print(f"    Reason:               {det['not_detected_reason']}")

    assert not det["cyclone_detected"], "Failed: City lights scan was falsely flagged as a cyclone!"


def test_satellite_with_colored_overlays():
    """
    Satellite imagery with colored overlay borders/lat-lon lines MUST PASS validation
    and PASS cyclone detection when an organized vortex is present.
    """
    img = np.ones((512, 512, 3), dtype=np.uint8) * 180  # warm ocean background
    cx, cy = 256, 256
    y, x = np.ogrid[:512, :512]
    mask = (x - cx)**2 + (y - cy)**2 <= 80**2
    img[mask] = (40, 40, 40)  # cold convective cloud core

    # Draw colored state borders and coastline markers
    cv2.line(img, (0, 150), (512, 150), (0, 0, 255), 2)  # red border line
    cv2.line(img, (200, 0), (200, 512), (0, 255, 0), 2)  # green grid line

    _, buf = cv2.imencode(".png", img)
    raw_bytes = buf.tobytes()

    # Step 1: Validator MUST accept satellite image with border overlays
    is_valid, val_err = validate_satellite_image(raw_bytes)
    print("\n[4] Satellite TIR with Border Overlays Test:")
    print(f"    Validator Result:     is_valid={is_valid}")
    assert is_valid, f"Failed: Overlay-annotated satellite image was wrongly rejected: {val_err}"

    # Step 2: Detector MUST identify the cyclone vortex
    pre = preprocess_tir(raw_bytes)
    det = detect_cyclone(pre["processed_image_b64"])
    print(f"    Detected:             {det['cyclone_detected']}")
    print(f"    CDO Radius:           {det['cdo_radius_km']} km")
    print(f"    Vortex Concentration: {det['vortex_concentration_score']}")

    assert det["cyclone_detected"], "Failed: Cyclone with border overlays was not detected!"


def test_natural_color_photo_rejection():
    """Natural color daylight photograph must be rejected by validator with detailed reason."""
    photo = np.random.randint(140, 240, (300, 300, 3), dtype=np.uint8)
    photo[:, :, 0] = np.random.randint(40, 100, (300, 300))
    photo[:, :, 1] = np.random.randint(150, 255, (300, 300))
    photo[:, :, 2] = np.random.randint(180, 255, (300, 300))

    _, buf = cv2.imencode(".png", photo)
    raw_bytes = buf.tobytes()

    is_valid, val_err = validate_satellite_image(raw_bytes)
    print("\n[5] Natural Color Photo Test:")
    print(f"    Validator Result:     is_valid={is_valid}")
    print(f"    Rejection Reason:     {val_err}")

    assert not is_valid, "Failed: Natural color photo should be rejected by validator!"
    assert "broad color saturation" in val_err or "insufficient dark background" in val_err


if __name__ == "__main__":
    print("=== Running CycloneAI Validation & False Positive Suite ===")
    test_random_noise_image()
    test_solid_gray_image()
    test_synthetic_city_lights_image()
    test_satellite_with_colored_overlays()
    test_natural_color_photo_rejection()
    print("\n[ALL TESTS PASSED] Satellite validation and false positive gates verified successfully!")
