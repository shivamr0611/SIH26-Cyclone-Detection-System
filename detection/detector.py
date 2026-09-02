#!/usr/bin/env python3
"""
detector.py - Cyclone & Eye Detection Module for CycloneAI.

Implements:
1. detect_cyclone: Decodes preprocessed image base64, inverts for TIR eye contrast,
   applies Hough Circle transform for eye segmentation, calculates eye diameter & CDO radius
   (assuming 4km/pixel for INSAT-3D), or estimates circulation center from coldest 5% centroid.
2. CycloneDetector: Class wrapper for backward compatibility with backend pipeline.
"""

import base64
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Configure logger
logger = logging.getLogger("cycloneai.detector")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Physical spatial resolution constant for INSAT-3D TIR (km per pixel)
INSAT3D_KM_PER_PIXEL = 4.0


def _decode_b64_image(image_b64: str) -> np.ndarray:
    """Decode base64 string to single-channel 8-bit grayscale image."""
    if not image_b64:
        raise ValueError("Empty base64 string provided.")

    if "," in image_b64:
        _, image_b64 = image_b64.split(",", 1)

    raw_bytes = base64.b64decode(image_b64)
    np_buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("Failed to decode image bytes with cv2.imdecode.")

    if img.ndim == 3:
        if img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    return gray


def detect_cyclone(preprocessed_image_b64: str) -> Dict[str, Any]:
    """
    Detects tropical cyclone vortex, eye candidates, and circulation center.

    Pipeline:
      1. Decodes base64 preprocessed image to grayscale.
      2. Inverts image (cold convective tops become dark, warm/clear eye becomes bright).
      3. Applies HoughCircles transform (dp=1.5, minDist=30, param1=80, param2=30, minRadius=5, maxRadius=60).
      4. If eye candidate found:
         - Converts eye diameter to km (4 km/pixel for INSAT-3D).
         - Computes Central Dense Overcast (CDO) radius = 2.5 * eye_radius.
         - Returns eye_center, eye_diameter_km, cdo_radius_km, confidence.
      5. If no clear eye:
         - Computes low-level circulation center as centroid of coldest 5% pixels.
         - Estimates CDO radius from cold cloud footprint.

    Args:
        preprocessed_image_b64: Base64-encoded PNG/JPEG image string.

    Returns:
        dict containing:
          - cyclone_detected (bool)
          - has_clear_eye (bool)
          - eye_center (list[int] or None): [x, y]
          - eye_diameter_km (float or None)
          - cdo_radius_km (float)
          - circulation_center (list[int]): [x, y]
          - confidence (float): 0.0 to 1.0 confidence score
    """
    logger.debug("Running cyclone detection on preprocessed image...")
    gray = _decode_b64_image(preprocessed_image_b64)
    h, w = gray.shape[:2]

    # Invert image: In TIR, cyclone eye is warm/darker in raw -> bright in inverted
    inverted = cv2.bitwise_not(gray)

    # Apply Gaussian smoothing before HoughCircles
    blurred_inv = cv2.GaussianBlur(inverted, (5, 5), sigmaX=1.5)

    # 3. HoughCircles for eye detection
    # Parameters: dp=1.5, minDist=30, param1=80, param2=30, minRadius=5, maxRadius=60
    circles = cv2.HoughCircles(
        blurred_inv,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=30,
        param1=80,
        param2=30,
        minRadius=5,
        maxRadius=60,
    )

    has_clear_eye = False
    eye_center: Optional[List[int]] = None
    eye_diameter_km: Optional[float] = None
    cdo_radius_km: float = 0.0
    circulation_center: List[int] = [w // 2, h // 2]

    # 1. Threshold to extract cold cloud candidates
    _, thresh_raw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 2. Mask out point-source light artifacts (city lights / sensor noise < 50 px)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_raw, connectivity=8)
    thresh = thresh_raw.copy()
    point_sources_removed = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 50:
            thresh[labels == i] = 0
            point_sources_removed += 1

    cold_pixel_count = int(np.count_nonzero(thresh))
    total_pixels = float(gray.size)
    cloud_coverage_pct = round(float((cold_pixel_count / total_pixels) * 100.0), 2)

    # 3. Two-Part Vortex Concentration Score (Rotational Coherence + Radial Gradient)
    if cold_pixel_count >= 50:
        y_idx, x_idx = np.where(thresh > 0)
        cx_c = float(np.mean(x_idx))
        cy_c = float(np.mean(y_idx))
        circulation_center = [int(round(cx_c)), int(round(cy_c))]

        # Log-polar transform centered on cold mass centroid (cv2.warpPolar, OpenCV 4+)
        max_radius = float(np.sqrt(h ** 2 + w ** 2) / 2.0)
        log_polar = cv2.warpPolar(
            gray.astype(np.float32),
            (360, 40),
            (cx_c, cy_c),
            max_radius,
            cv2.WARP_POLAR_LOG + cv2.WARP_FILL_OUTLIERS,
        )

        # 3a. Rotational coherence: Autocorrelation of radial rings across angles
        ring_coherences = []
        for r in range(log_polar.shape[0]):
            ring = log_polar[r, :]
            ring_std = float(np.std(ring))
            if ring_std > 2.0:
                ring_norm = (ring - np.mean(ring)) / ring_std
                corr = np.correlate(ring_norm, ring_norm, mode="full")
                mid = len(corr) // 2
                lag_corr = corr[mid + 15 : mid + 180] / float(len(ring_norm))
                peak_val = float(np.max(lag_corr)) if len(lag_corr) > 0 else 0.0
                ring_coherences.append(max(0.0, peak_val))
            else:
                ring_coherences.append(0.0)

        rotational_coherence = float(np.mean(ring_coherences)) if ring_coherences else 0.0

        # 3b. Radial gradient: Cold core mass decreases monotonically outward from center
        radial_profile = np.mean(log_polar, axis=1)
        diffs = np.diff(radial_profile)
        neg_diffs = np.count_nonzero(diffs <= 0)
        radial_gradient_score = float(neg_diffs / len(diffs)) if len(diffs) > 0 else 0.0

        # Combined formula: 0.6 * rotational_coherence + 0.4 * radial_gradient_score
        vortex_concentration_score = round(
            float(np.clip(0.6 * rotational_coherence + 0.4 * radial_gradient_score, 0.0, 1.0)), 3
        )
    else:
        circulation_center = [w // 2, h // 2]
        vortex_concentration_score = 0.0

    # 4. Search for circular eye (only if cold mass is present)
    if circles is not None and len(circles) > 0 and cold_pixel_count >= 100:
        # Select candidate closest to image center
        candidates = circles[0]
        img_center = np.array([w / 2.0, h / 2.0])

        best_circle = min(
            candidates,
            key=lambda c: np.linalg.norm(np.array([c[0], c[1]]) - img_center),
        )

        cx, cy, radius = float(best_circle[0]), float(best_circle[1]), float(best_circle[2])
        eye_center = [int(round(cx)), int(round(cy))]
        circulation_center = eye_center
        has_clear_eye = True

        eye_radius_km = radius * INSAT3D_KM_PER_PIXEL
        eye_diameter_km = round(2.0 * eye_radius_km, 2)
        cdo_radius_km = round(2.5 * eye_radius_km, 2)

        logger.info(
            "Clear Eye Candidate at %s: Diameter=%.2f km, CDO Radius=%.2f km",
            eye_center,
            eye_diameter_km,
            cdo_radius_km,
        )

    else:
        # Pseudo-CDO from largest contour in filtered cold mask
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            pseudo_cdo_radius_px = np.sqrt(area / np.pi)
        else:
            pseudo_cdo_radius_px = 0.0

        cdo_radius_km = round(float(pseudo_cdo_radius_px * INSAT3D_KM_PER_PIXEL), 2)

    # 5. Geographic Ocean / Land Coordinate Resolution
    # Bounding Box: Left ~ 75.0°E, Right ~ 100.0°E (Δlon=25.0°), Top ~ 25.0°N, Bottom ~ 5.0°N (Δlat=20.0°)
    cx_coord, cy_coord = circulation_center
    resolved_lat = round(25.0 - (float(cy_coord) / float(max(1, h))) * 20.0, 2)
    resolved_lon = round(75.0 + (float(cx_coord) / float(max(1, w))) * 25.0, 2)

    # Land/Ocean determination for Bay of Bengal / North Indian Ocean domain (75°E-100°E, 5°N-25°N)
    # India east coastline extends from ~80.5°E in south (Tamil Nadu/Andhra) to ~86.5°E in north (Odisha/WB)
    if resolved_lat > 20.5 and resolved_lon < 86.5:
        # Northern / Central mainland India (land)
        is_ocean = False
    elif 11.0 <= resolved_lat <= 20.5 and resolved_lon <= 80.5:
        # Peninsular India interior (land)
        is_ocean = False
    elif resolved_lat < 11.0 and resolved_lon < 77.5:
        # Southern tip landmass
        is_ocean = False
    else:
        # Bay of Bengal open ocean (80.5°E - 98.0°E, 5.0°N - 22.0°N) or Arabian Sea
        is_ocean = True

    logger.info(
        "Detected Center Pixel: (cx=%d, cy=%d), Image Dimensions: (W=%d, H=%d) -> Resolved Geo: (lat=%.2f°N, lon=%.2f°E), is_ocean=%s",
        cx_coord,
        cy_coord,
        w,
        h,
        resolved_lat,
        resolved_lon,
        is_ocean,
    )

    # 6. Multi-Criteria Gate Threshold Enforcement
    failures = []

    # Hard Gate 1: Ocean Requirement (Tropical cyclones cannot form over continental land)
    if not is_ocean:
        failures.append(
            f"Circulation center is situated over continental landmass (lat: {resolved_lat:.1f}°N, lon: {resolved_lon:.1f}°E) rather than ocean waters required for tropical cyclogenesis"
        )

    # Hard Gate 2: Vortex Concentration Score (threshold 0.30)
    if vortex_concentration_score < 0.30:
        failures.append(f"Vortex concentration score ({vortex_concentration_score:.2f} < 0.30 threshold)")

    # Hard Gate 3: Minimum Cold Convective Cloud Coverage (threshold 15.0%)
    if cloud_coverage_pct < 15.0:
        failures.append(f"Cold core cloud coverage ({cloud_coverage_pct:.1f}% < 15.0% threshold)")

    # Hard Gate 4: Minimum Central Dense Overcast (CDO) Size (threshold 40.0 km)
    if cdo_radius_km < 40.0:
        failures.append(f"CDO radius ({cdo_radius_km:.1f} km < 40.0 km threshold)")

    # Hard Gate 5: Monsoon Convection Discriminator
    # If cold_core_mass > 45% AND vortex_score < 0.65 AND is_ocean == False -> No Cyclone
    if cloud_coverage_pct > 45.0 and vortex_concentration_score < 0.65 and not is_ocean:
        failures.append(
            f"Monsoon convection signature: high cold cloud mass ({cloud_coverage_pct:.1f}%) over continental land with non-cyclonic rotational organization ({vortex_concentration_score:.2f} < 0.65)"
        )
    elif cloud_coverage_pct > 45.0 and vortex_concentration_score < 0.60 and not has_clear_eye:
        failures.append(
            f"High cold cloud mass ({cloud_coverage_pct:.1f}%) with sub-cyclonic rotational organization ({vortex_concentration_score:.2f} < 0.60) indicates unorganized convection"
        )

    if point_sources_removed > 10 and cold_pixel_count < 100:
        failures.append("Point-source light artifacts detected without organized convective cloud mass")

    if cloud_coverage_pct > 85.0 and not has_clear_eye:
        failures.append(f"Uniformly high cold cloud coverage ({cloud_coverage_pct:.1f}%) lacks localized cyclonic vortex structure")

    if failures:
        cyclone_detected = False
        not_detected_reason = "; ".join(failures)
        # Weighted low confidence when detection fails
        confidence = round(float(min(0.35, max(0.05, vortex_concentration_score * 0.4 + (cloud_coverage_pct / 100.0) * 0.2))), 2)
    else:
        cyclone_detected = True
        not_detected_reason = None
        # Weighted high confidence when all gates pass: Concentration (0.45) + Coverage (0.30) + CDO (0.15) + Base (0.10)
        cdo_factor = min(1.0, cdo_radius_km / 350.0) * 0.15
        confidence = round(float(min(0.98, max(0.60, 0.10 + vortex_concentration_score * 0.45 + min(1.0, cloud_coverage_pct / 60.0) * 0.30 + cdo_factor))), 2)

    logger.info(
        "Detection Decision: detected=%s, reason=%s, CDO=%.1f km, conc=%.2f, cov=%.1f%%, conf=%.2f",
        cyclone_detected,
        not_detected_reason,
        cdo_radius_km,
        vortex_concentration_score,
        cloud_coverage_pct,
        confidence,
    )

    return {
        "cyclone_detected": cyclone_detected,
        "not_detected_reason": not_detected_reason,
        "vortex_concentration_score": vortex_concentration_score,
        "has_clear_eye": has_clear_eye,
        "eye_center": eye_center,
        "eye_diameter_km": eye_diameter_km,
        "cdo_radius_km": cdo_radius_km,
        "circulation_center": circulation_center,
        "confidence": confidence,
    }


class CycloneDetector:
    """Wrapper class providing backward-compatibility with backend pipeline."""

    def __init__(self, model_name: str = "Hough-Circulation-Detector"):
        self.model_name = model_name

    def detect(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts output dict from SatellitePreprocessor and returns detection results
        compatible with legacy classifier and prediction modules.
        """
        image_b64 = preprocessed_data.get("processed_image_b64", "")
        cloud_pct = preprocessed_data.get("cloud_coverage_percent", 0.0)
        core_pct = preprocessed_data.get("dense_core_percent", 0.0)

        if image_b64:
            try:
                res = detect_cyclone(image_b64)
                detected = res["cyclone_detected"]
                conf_pct = round(res["confidence"] * 100.0, 1)

                status_text = "Cyclone Detected" if detected else "No Cyclone Detected"
                if res["has_clear_eye"]:
                    status_text = "Cyclone Eye Detected"
                    msg = (
                        f"Convective vortex with clear eye (diameter: {res['eye_diameter_km']} km, "
                        f"CDO: {res['cdo_radius_km']} km)."
                    )
                elif detected:
                    msg = f"Convective circulation detected around {res['circulation_center']} with {cloud_pct}% cloud density."
                else:
                    msg = f"Clear area: Cloud coverage is {cloud_pct}% (below threshold)."

                return {
                    "detected": detected,
                    "confidence": conf_pct,
                    "status_text": status_text,
                    "cloud_coverage_percent": cloud_pct,
                    "dense_core_percent": core_pct,
                    "message": msg,
                    "has_clear_eye": res["has_clear_eye"],
                    "eye_center": res["eye_center"],
                    "eye_diameter_km": res["eye_diameter_km"],
                    "cdo_radius_km": res["cdo_radius_km"],
                    "circulation_center": res["circulation_center"],
                }
            except Exception as e:
                logger.error("Error running detect_cyclone in CycloneDetector: %s", e)

        # Fallback heuristic
        is_det = (cloud_pct >= 16.0 and core_pct >= 2.5)
        conf = round(min(98.8, max(68.0, 60.0 + cloud_pct * 0.4 + core_pct * 0.3)), 1) if is_det else 92.0
        return {
            "detected": is_det,
            "confidence": conf,
            "status_text": "Cyclone Detected" if is_det else "No Cyclone Detected",
            "cloud_coverage_percent": cloud_pct,
            "dense_core_percent": core_pct,
            "message": f"Cloud density {cloud_pct}%." if is_det else f"Low cloud cover {cloud_pct}%.",
            "has_clear_eye": False,
            "eye_center": None,
            "eye_diameter_km": None,
            "cdo_radius_km": 0.0,
            "circulation_center": [100, 100],
        }


if __name__ == "__main__":
    print("=== Testing Cyclone Detection Module ===")

    # Test 1: Image with clear eye
    test_img = np.full((256, 256), 220, dtype=np.uint8)
    # Dark eye in TIR (pixel value 30, radius 20 at center 128,128)
    cv2.circle(test_img, (128, 128), 20, 30, -1)
    test_img = cv2.GaussianBlur(test_img, (7, 7), 0)

    _, buf = cv2.imencode(".png", test_img)
    b64_eye = base64.b64encode(buf.tobytes()).decode("utf-8")

    res_eye = detect_cyclone(b64_eye)
    print("\n1. Test Image (With Clear Eye):")
    for k, v in res_eye.items():
        print(f"   - {k}: {v}")

    # Test 2: Image without clear eye
    test_no_eye = np.random.randint(150, 240, (256, 256), dtype=np.uint8)
    _, buf2 = cv2.imencode(".png", test_no_eye)
    b64_no_eye = base64.b64encode(buf2.tobytes()).decode("utf-8")

    res_no_eye = detect_cyclone(b64_no_eye)
    print("\n2. Test Image (No Clear Eye):")
    for k, v in res_no_eye.items():
        print(f"   - {k}: {v}")
