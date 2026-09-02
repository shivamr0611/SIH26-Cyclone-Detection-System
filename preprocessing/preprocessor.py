#!/usr/bin/env python3
"""
preprocessor.py - OpenCV Satellite Image Preprocessing Pipeline for CycloneAI.

Provides:
1. preprocess_tir: Full OpenCV pipeline for Thermal Infrared (TIR) contrast enhancement,
   Gaussian denoising, adaptive thresholding, morphological closing, and cloud feature extraction.
2. preprocess_multichannel: 3-channel (TIR, WV, VIS) stacker normalized for PyTorch model inference (3, 224, 224).
3. SatellitePreprocessor: Class wrapper for backward compatibility with backend endpoints.
"""

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Configure module logger
logger = logging.getLogger("cycloneai.preprocessor")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def validate_satellite_image(image_bytes: bytes) -> Tuple[bool, Optional[str]]:
    """
    Validates whether the uploaded image is plausibly a meteorological satellite image
    (INSAT-3D/3DR, GOES-R, NOAA, Meteosat, NASA Worldview).

    Permits:
      - Grayscale and single-channel TIR/WV/VIS
      - Satellite images with colored overlay borders, lat/lon grids, and coastline markers
      - False-color enhanced IR composites
      - Nighttime Day-Night band (VIIRS/DNB) satellite scans

    Rejects (via multi-signal score):
      - Natural daylight photographs (people, landscapes, rooms, objects)
      - Complex high-detail photographic scenes
      - Screenshots of colorful street/road maps
      - Blank or completely uniform images
      - Images smaller than 100x100 pixels

    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    if not image_bytes or len(image_bytes) < 32:
        return False, "Uploaded image payload is empty or corrupted."

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_UNCHANGED)

    if image is None:
        return False, "Failed to decode image. Please upload a valid PNG, JPEG, or GeoTIFF file."

    # 1. Minimum Resolution Check (Hard Gate)
    h, w = image.shape[:2]
    if h < 100 or w < 100:
        return False, (
            f"Image resolution ({w}x{h} px) is below minimum requirement (100x100 px). "
            "Satellite meteorological analysis requires at least 100x100 pixels."
        )

    # Convert to grayscale for structural analysis
    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY) if (image.ndim == 3 and image.shape[2] >= 3) else (image if image.ndim == 2 else image[:, :, 0])

    # 2. Blank / Zero-Variance Image Check (Hard Gate)
    std_dev = float(np.std(gray))
    if std_dev < 2.0:
        return False, (
            f"Image is blank or completely uniform (standard deviation {std_dev:.2f} < 2.0). "
            "Satellite imagery must contain discernible cloud or oceanic surface features."
        )

    # 3. High-Frequency Noise / Corrupted Array Check (Hard Gate)
    diff_h = float(np.mean(np.abs(gray[1:, :].astype(np.float32) - gray[:-1, :].astype(np.float32))))
    diff_w = float(np.mean(np.abs(gray[:, 1:].astype(np.float32) - gray[:, :-1].astype(np.float32))))
    if diff_h > 55.0 and diff_w > 55.0:
        return False, (
            f"Image exhibits extreme high-frequency pixel variation typical of unstructured random noise "
            f"or corrupted sensor grain (measured gradient: {diff_h:.1f} > 55.0 threshold)."
        )

    # 4. Multi-Signal Score for Non-Satellite Imagery
    flags = []

    # Signal A: Broad Color Saturation
    high_sat_pct = 0.0
    if image.ndim == 3 and image.shape[2] >= 3:
        hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        high_sat_pct = float(np.mean(sat > 60) * 100.0)
        if high_sat_pct > 40.0:
            flags.append(f"broad color saturation ({high_sat_pct:.1f}% of pixels > 40.0% threshold)")

    # Signal B: Background Darkness / Oceanic Baseline
    dark_pct = float(np.mean(gray < 128) * 100.0)
    if dark_pct < 20.0:
        flags.append(f"insufficient dark background / oceanic baseline ({dark_pct:.1f}% < 20.0% threshold)")

    # Signal C: Photographic Scene Texture Complexity
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 80, 180)
    edge_density = float(np.mean(edges > 0) * 100.0)
    if laplacian_var > 900.0 and edge_density > 25.0:
        flags.append(f"excessive scene texture complexity (Laplacian variance {laplacian_var:.1f}, edge density {edge_density:.1f}%)")

    # Reject only when multiple signals strongly indicate non-satellite source
    if len(flags) >= 2 or high_sat_pct > 65.0:
        reason_list = "; ".join(flags)
        return False, (
            f"Uploaded image does not appear to be a meteorological satellite image ({reason_list}). "
            "Accepted inputs: Single-channel or overlay-annotated Thermal Infrared (TIR), Water Vapor (WV), "
            "Visible (VIS), or false-color satellite scans from MOSDAC (ISRO), NOAA Worldview, NASA Worldview, or EUMETSAT."
        )

    return True, None


def _decode_to_grayscale(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into a single-channel 8-bit grayscale image."""
    if not image_bytes:
        raise ValueError("Image bytes input is empty.")

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError("cv2.imdecode failed to decode the provided image bytes.")

    if image.ndim == 3:
        if image.shape[2] == 4:
            # BGRA -> Gray
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            # BGR -> Gray
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"Unsupported image dimensions: {image.shape}")

    # Ensure uint8
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return gray


def preprocess_tir(image_bytes: bytes) -> Dict[str, Any]:
    """
    Process raw Thermal Infrared (TIR) satellite image bytes using OpenCV.

    Pipeline:
      1. Decode image bytes to grayscale
      2. Apply CLAHE (clipLimit=3.0, tileGridSize=(8,8)) for contrast enhancement
      3. Apply GaussianBlur (5x5 kernel) for high-frequency noise reduction
      4. Apply Adaptive Otsu's Thresholding:
         - Standard TIR (mean >= 50): THRESH_BINARY_INV to segment dark cold tops as white
         - Dark/Night visible (mean < 50): THRESH_BINARY to segment bright structures as white
      5. Apply Morphological Closing (5x5 kernel, 3 iterations) to fill speckle holes
      6. Compute statistics: cloud coverage %, cold core density, mean brightness, histogram
    """
    logger.debug("Starting TIR preprocessing pipeline (%d bytes)...", len(image_bytes))

    # 1. Decode image
    gray = _decode_to_grayscale(image_bytes)

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Gaussian Blur (5x5 kernel)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), sigmaX=0)

    # 4. Otsu's Thresholding (brightness-adaptive)
    mean_b = float(np.mean(gray))
    if mean_b >= 50.0:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Morphological Closing (5x5 structuring element, 3 iterations)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    # 6. Extract metrics
    cloud_coverage_pct = round(float((np.sum(closed_mask > 0) / closed_mask.size) * 100.0), 2)

    # Cold core density: percentage of pixels in coldest 10th percentile of raw image
    cold_threshold = np.percentile(gray, 10)
    cold_core_pixels = np.count_nonzero(gray <= cold_threshold)
    cold_core_density = round(float((cold_core_pixels / gray.size) * 100.0), 2)

    mean_brightness = round(float(np.mean(gray)), 2)

    # 256-bin histogram
    hist_raw = cv2.calcHist([gray], [0], None, [256], [0, 256])
    histogram: List[int] = hist_raw.ravel().astype(int).tolist()

    # Base64 encode processed mask for downstream detector
    success, buffer = cv2.imencode(".png", closed_mask)
    if not success:
        raise RuntimeError("Failed to encode processed image to PNG buffer.")
    processed_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    return {
        "cloud_coverage_pct": cloud_coverage_pct,
        "cold_core_density": cold_core_density,
        "mean_brightness": mean_brightness,
        "processed_image_b64": processed_b64,
        "histogram": histogram,
    }


def _process_single_channel(channel_bytes: Optional[bytes], target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    """Decode, resize to target_size, and normalize single channel to [0.0, 1.0]."""
    if channel_bytes is None:
        return None
    try:
        gray = _decode_to_grayscale(channel_bytes)
        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        return normalized
    except Exception as err:
        logger.warning("Failed to decode channel: %s. Falling back to default.", err)
        return None


def preprocess_multichannel(
    tir_bytes: bytes,
    wv_bytes: Optional[bytes] = None,
    vis_bytes: Optional[bytes] = None,
) -> np.ndarray:
    """
    Prepares multi-channel satellite input (TIR, WV, VIS) for PyTorch neural network inference.

    Steps:
      - Mandatory TIR channel decoded, resized to (224, 224), and normalized to [0, 1]
      - Optional Water Vapor (WV) channel; duplicates TIR if None
      - Optional Visible (VIS) channel; duplicates TIR if None
      - Stacks channels into shape (3, 224, 224) (Channel-first PyTorch format)

    Args:
        tir_bytes: Raw bytes of Thermal Infrared (TIR) channel (Mandatory)
        wv_bytes: Raw bytes of Water Vapor (WV) channel (Optional)
        vis_bytes: Raw bytes of Visible (VIS) channel (Optional)

    Returns:
        np.ndarray: Preprocessed float32 array with shape (3, 224, 224), values in [0.0, 1.0]
    """
    logger.debug("Processing multi-channel input stack...")

    # 1. Process mandatory TIR channel
    tir_norm = _process_single_channel(tir_bytes, target_size=(224, 224))
    if tir_norm is None:
        raise ValueError("Mandatory TIR channel could not be decoded.")

    # 2. Process WV channel (duplicate TIR if missing)
    wv_norm = _process_single_channel(wv_bytes, target_size=(224, 224))
    if wv_norm is None:
        wv_norm = tir_norm.copy()

    # 3. Process VIS channel (duplicate TIR if missing)
    vis_norm = _process_single_channel(vis_bytes, target_size=(224, 224))
    if vis_norm is None:
        vis_norm = tir_norm.copy()

    # 4. Stack into PyTorch channel-first format (3, 224, 224)
    stacked = np.stack([tir_norm, wv_norm, vis_norm], axis=0).astype(np.float32)

    logger.debug("Multi-channel tensor prepared with shape %s, range [%.2f, %.2f]", stacked.shape, float(stacked.min()), float(stacked.max()))
    return stacked


class SatellitePreprocessor:
    """Wrapper class providing backward-compatibility with backend/app.py."""

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size

    def preprocess_image_b64(self, image_b64: str) -> Dict[str, Any]:
        """
        Accepts base64-encoded image string from browser API,
        executes preprocess_tir pipeline, and returns standard metrics.
        """
        if not image_b64:
            return {
                "status": "error",
                "error": "Empty base64 image received",
                "cloud_coverage_percent": 0.0,
                "dense_core_percent": 0.0,
            }

        # Strip data URL prefix if present
        if "," in image_b64:
            _, image_b64 = image_b64.split(",", 1)

        try:
            image_bytes = base64.b64decode(image_b64)
            result = preprocess_tir(image_bytes)

            return {
                "status": "success",
                "cloud_coverage_percent": result["cloud_coverage_pct"],
                "dense_core_percent": result["cold_core_density"],
                "cloud_coverage_pct": result["cloud_coverage_pct"],
                "cold_core_density": result["cold_core_density"],
                "mean_brightness": result["mean_brightness"],
                "processed_image_b64": result["processed_image_b64"],
                "histogram": result["histogram"],
            }
        except Exception as e:
            logger.error("Error in preprocess_image_b64: %s", e)
            return {
                "status": "error",
                "error": str(e),
                "cloud_coverage_percent": 0.0,
                "dense_core_percent": 0.0,
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Running Satellite Preprocessor Verification ===")

    # 1. Generate synthetic grayscale test image (256x256) with dark circular core
    np.random.seed(42)
    synthetic_img = np.random.randint(120, 200, (256, 256), dtype=np.uint8)
    
    # Draw cold circular storm core (low pixel values)
    cv2.circle(synthetic_img, (128, 128), radius=40, color=20, thickness=-1)
    synthetic_img = cv2.GaussianBlur(synthetic_img, (15, 15), 0)

    # Encode synthetic image to PNG bytes
    success, encoded_bytes = cv2.imencode(".png", synthetic_img)
    if not success:
        raise RuntimeError("Failed to encode synthetic test image.")
    test_bytes = encoded_bytes.tobytes()

    # 2. Run preprocess_tir
    print("\n--- 1. Testing preprocess_tir ---")
    tir_result = preprocess_tir(test_bytes)
    print(f"Cloud Coverage:     {tir_result['cloud_coverage_pct']}%")
    print(f"Cold Core Density:  {tir_result['cold_core_density']}%")
    print(f"Mean Brightness:    {tir_result['mean_brightness']}")
    print(f"Histogram bins:     {len(tir_result['histogram'])} bins (Sum: {sum(tir_result['histogram'])})")
    print(f"Base64 string len:  {len(tir_result['processed_image_b64'])} chars")

    # 3. Run preprocess_multichannel
    print("\n--- 2. Testing preprocess_multichannel ---")
    tensor = preprocess_multichannel(test_bytes)
    print(f"Tensor Shape:       {tensor.shape} (Channels, Height, Width)")
    print(f"Tensor Data Type:   {tensor.dtype}")
    print(f"Min value:          {tensor.min():.4f}, Max value: {tensor.max():.4f}")

    # 4. Save processed image to /tmp/test_preprocessed.png
    tmp_dir = Path("/tmp")
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_file = tmp_dir / "test_preprocessed.png"
    except Exception:
        # Fallback to local scratch or temp
        out_file = Path("test_preprocessed.png")

    # Decode base64 image and save
    decoded_processed = base64.b64decode(tir_result["processed_image_b64"])
    with open(out_file, "wb") as f:
        f.write(decoded_processed)
    print(f"\n[SUCCESS] Saved processed image to: {out_file.resolve()}")
