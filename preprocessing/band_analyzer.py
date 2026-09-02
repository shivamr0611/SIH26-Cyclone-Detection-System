#!/usr/bin/env python3
"""
band_analyzer.py - Spiral Banding & Vorticity Analysis for CycloneAI.

Implements:
1. analyze_spiral_bands: Transforms satellite imagery to log-polar space around the cyclone
   center, detects linear spiral arms via Canny + HoughLines, measures dominant band curvature,
   evaluates Dvorak banding score, and computes rotation direction from gradient orientation statistics.
2. full_detection: Orchestrates end-to-end pipeline:
   preprocess_tir -> detect_cyclone -> analyze_spiral_bands.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from detection.detector import detect_cyclone
from preprocessing.preprocessor import _decode_to_grayscale, preprocess_tir

# Configure logger
logger = logging.getLogger("cycloneai.band_analyzer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _apply_log_polar(
    image: np.ndarray,
    center: Tuple[float, float],
    scale_factor_m: float = 40.0,
) -> np.ndarray:
    """
    Applies log-polar transformation centered on the cyclone circulation center.

    Under the log-polar transformation (log(r), theta), logarithmic spiral cloud bands
    (r = a * exp(b * theta)) map to straight line segments.
    """
    h, w = image.shape[:2]
    max_radius = np.hypot(max(center[0], w - center[0]), max(center[1], h - center[1]))

    # Support both OpenCV legacy logPolar and modern warpPolar
    if hasattr(cv2, "logPolar"):
        flags = cv2.WARP_FILL_OUTLIERS | cv2.INTER_LINEAR
        log_polar = cv2.logPolar(image, center, scale_factor_m, flags)
    else:
        flags = cv2.WARP_POLAR_LOG | cv2.WARP_FILL_OUTLIERS | cv2.INTER_LINEAR
        dsize = (w, h)
        log_polar = cv2.warpPolar(image, dsize, center, max_radius, flags)

    return log_polar


def _detect_rotation_direction(
    gray: np.ndarray,
    center: Tuple[float, float],
) -> str:
    """
    Estimate cyclonic vs anticyclonic rotation direction from radial/tangential gradient statistics.

    In the Northern Hemisphere (NIO/Bay of Bengal), cyclones rotate counter-clockwise (cyclonic).
    Spiral cloud bands produce a distinct asymmetric tangential gradient phase distribution.
    """
    h, w = gray.shape[:2]
    cx, cy = center
    y_coords, x_coords = np.ogrid[:h, :w]

    dx = x_coords - cx
    dy = y_coords - cy
    radius = np.sqrt(dx**2 + dy**2)
    radius_safe = np.maximum(radius, 1.0)

    # Unit vectors in radial (R) and counter-clockwise tangential (T) directions
    rx, ry = dx / radius_safe, dy / radius_safe
    tx, ty = -dy / radius_safe, dx / radius_safe

    # Compute Sobel image gradients
    gx = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)

    grad_mag = np.sqrt(gx**2 + gy**2)

    # Focus on active convective bands outside the immediate eye
    band_mask = (radius > 12.0) & (radius < min(w, h) * 0.45)
    if not np.any(band_mask):
        return "unclear"

    mag_thresh = np.percentile(grad_mag[band_mask], 65)
    active_mask = band_mask & (grad_mag > mag_thresh)

    if np.count_nonzero(active_mask) < 30:
        return "unclear"

    gr = rx[active_mask] * gx[active_mask] + ry[active_mask] * gy[active_mask]
    gt = tx[active_mask] * gx[active_mask] + ty[active_mask] * gy[active_mask]

    # Angle of gradient relative to radial vector
    alpha = np.arctan2(gt, gr)
    sin_metric = float(np.mean(np.sin(2.0 * alpha)))

    if sin_metric > 0.12:
        return "cyclonic"
    elif sin_metric < -0.12:
        return "anticyclonic"
    else:
        return "unclear"


def analyze_spiral_bands(
    image_bytes: bytes,
    center: Union[Tuple[int, int], Tuple[float, float], List[int]],
) -> Dict[str, Any]:
    """
    Analyzes spiral cloud bands from satellite imagery around a center point.

    Pipeline:
      1. Converts image bytes to grayscale.
      2. Transforms to Log-Polar space centered at `center` with M=40.0.
      3. Applies Canny edge detection (thresholds 50, 150).
      4. Detects lines using HoughLines (rho=1, theta=1 deg, threshold=30).
      5. Filters lines in spiral orientation angle range (20 deg to 70 deg).
      6. Estimates dominant band angle and Dvorak banding score (0.0 to 1.0).
      7. Determines cyclonic rotation direction from gradient orientation.

    Args:
        image_bytes: Raw bytes of the satellite image.
        center: Circulation center coordinates (x, y).

    Returns:
        dict containing:
          - band_count (int): Number of distinct spiral band segments.
          - dominant_band_angle_deg (float): Average spiral angle in degrees.
          - rotation_direction (str): "cyclonic" | "anticyclonic" | "unclear".
          - banding_score (float): 0.0 to 1.0 normalized Dvorak banding feature.
    """
    logger.debug("Analyzing spiral bands centered at %s...", center)

    # 1. Grayscale decoding
    gray = _decode_to_grayscale(image_bytes)
    cx, cy = float(center[0]), float(center[1])

    # 2. Log-Polar transform with M = 40.0
    log_polar = _apply_log_polar(gray, (cx, cy), scale_factor_m=40.0)

    # 3. Canny edge detection on log-polar representation
    blurred_lp = cv2.GaussianBlur(log_polar, (3, 3), 0)
    edges = cv2.Canny(blurred_lp, threshold1=50, threshold2=150)

    # 4. HoughLines to identify linear spiral trajectories
    lines = cv2.HoughLines(edges, rho=1.0, theta=np.pi / 180.0, threshold=30)

    spiral_angles: List[float] = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle_deg = float(np.rad2deg(theta) % 180.0)

            # 5. Filter by spiral angle range: 20 deg to 70 deg (and complementary 110-160 deg)
            if 20.0 <= angle_deg <= 70.0:
                spiral_angles.append(angle_deg)

    # 6. Compute band statistics
    band_count = len(spiral_angles)
    if band_count > 0:
        dominant_band_angle_deg = round(float(np.mean(spiral_angles)), 2)
        # Normalized score: 0 to 1 based on band presence (saturation at 10 bands)
        banding_score = round(float(min(1.0, max(0.1, band_count / 10.0))), 2)
    else:
        dominant_band_angle_deg = 0.0
        banding_score = 0.0

    # 7. Rotation direction
    rotation_direction = _detect_rotation_direction(gray, (cx, cy))

    logger.debug(
        "Spiral analysis complete: bands=%d, dominant_angle=%.2f deg, rotation=%s, score=%.2f",
        band_count,
        dominant_band_angle_deg,
        rotation_direction,
        banding_score,
    )

    return {
        "band_count": band_count,
        "dominant_band_angle_deg": dominant_band_angle_deg,
        "rotation_direction": rotation_direction,
        "banding_score": banding_score,
    }


def full_detection(image_bytes: bytes) -> Dict[str, Any]:
    """
    Executes the comprehensive detection and feature extraction pipeline:
      preprocessor.preprocess_tir() -> detect_cyclone() -> analyze_spiral_bands().

    Args:
        image_bytes: Raw bytes of satellite image (TIR).

    Returns:
        Unified dictionary with all preprocessed metrics, eye & circulation centers,
        and spiral banding / vorticity attributes.
    """
    logger.info("Executing full detection pipeline...")

    # Step 1: Preprocess TIR
    prep_result = preprocess_tir(image_bytes)

    # Step 2: Detect cyclone & eye
    det_result = detect_cyclone(prep_result["processed_image_b64"])

    # Determine center for spiral analysis
    if det_result["has_clear_eye"] and det_result["eye_center"]:
        center = det_result["eye_center"]
    else:
        center = det_result["circulation_center"]

    # Step 3: Analyze spiral bands
    band_result = analyze_spiral_bands(image_bytes, center)

    # Merge unified output
    merged = {
        "cyclone_detected": det_result["cyclone_detected"],
        "has_clear_eye": det_result["has_clear_eye"],
        "confidence": det_result["confidence"],
        "eye_center": det_result["eye_center"],
        "eye_diameter_km": det_result["eye_diameter_km"],
        "cdo_radius_km": det_result["cdo_radius_km"],
        "circulation_center": det_result["circulation_center"],
        "band_count": band_result["band_count"],
        "dominant_band_angle_deg": band_result["dominant_band_angle_deg"],
        "rotation_direction": band_result["rotation_direction"],
        "banding_score": band_result["banding_score"],
        "cloud_coverage_pct": prep_result["cloud_coverage_pct"],
        "cold_core_density": prep_result["cold_core_density"],
        "mean_brightness": prep_result["mean_brightness"],
        "processed_image_b64": prep_result["processed_image_b64"],
        "histogram": prep_result["histogram"],
        "detection_details": det_result,
        "spiral_details": band_result,
    }

    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Testing Spiral Band Analyzer & Full Detection Pipeline ===")

    # 1. Create a synthetic cyclone vortex with spiral arms
    h, w = 256, 256
    cx, cy = 128, 128
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    theta = np.arctan2(y - cy, x - cx)

    # Log-spiral intensity modulation
    b = 0.5
    spiral_pattern = np.sin(np.log(np.maximum(r, 1.0)) * 5.0 + b * theta * 4.0)
    img_vortex = np.clip(160 + spiral_pattern * 60 - r * 0.3, 0, 255).astype(np.uint8)
    # Clear warm eye at center
    cv2.circle(img_vortex, (cx, cy), 18, 25, -1)
    img_vortex = cv2.GaussianBlur(img_vortex, (5, 5), 0)

    _, buf = cv2.imencode(".png", img_vortex)
    synthetic_bytes = buf.tobytes()

    # 2. Run analyze_spiral_bands directly
    print("\n--- 1. Testing analyze_spiral_bands ---")
    band_out = analyze_spiral_bands(synthetic_bytes, (cx, cy))
    for k, v in band_out.items():
        print(f"   - {k}: {v}")

    # 3. Run full_detection
    print("\n--- 2. Testing full_detection ---")
    full_out = full_detection(synthetic_bytes)
    for k, v in full_out.items():
        if k not in ("processed_image_b64", "histogram", "detection_details", "spiral_details"):
            print(f"   - {k}: {v}")
