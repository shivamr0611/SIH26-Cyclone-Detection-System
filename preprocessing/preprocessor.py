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


def validate_satellite_image(image_input: Union[bytes, bytearray, np.ndarray]) -> Tuple[bool, Optional[str]]:
    """
    Validates whether an uploaded image is a legitimate meteorological satellite product
    (INSAT-3D/3DR, GOES-R, NOAA, NASA Worldview, MOSDAC, EUMETSAT) versus non-meteorological photo.

    Evaluates 4 signals:
      1. Saturation distribution: percentage of pixels with s > 100
      2. Hue diversity: standard deviation of hue channel
      3. Brightness texture: Laplacian variance for isolated spot detection
      4. Color channel correlation: mean Pearson correlation of (R,G) and (R,B) channels

    Rejection rule: Requires at least 3 failing signals simultaneously to reject.
    """
    if isinstance(image_input, (bytes, bytearray)):
        if len(image_input) < 32:
            return False, "Uploaded image payload is empty or corrupted."
        np_buffer = np.frombuffer(image_input, dtype=np.uint8)
        img_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif image_input.shape[2] == 4:
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
        else:
            img_bgr = image_input
    else:
        return False, "Invalid image input type."

    if img_bgr is None:
        return False, "Failed to decode image. Please upload a valid PNG, JPEG, or GeoTIFF file."

    h_img, w_img = img_bgr.shape[:2]
    if h_img < 100 or w_img < 100:
        return False, f"Image resolution ({w_img}x{h_img} px) is below minimum requirement (100x100 px)."

    # 1. Blank / Zero-Variance Image Check
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    std_dev = float(np.std(gray))
    if std_dev < 2.0:
        return False, f"Image is blank or completely uniform (std {std_dev:.2f} < 2.0)."

    # Multi-Signal Analysis
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    # Signal 1: Saturation distribution (>50% vivid pixels)
    high_sat_ratio = float(np.mean(s > 100))

    # Signal 2: Hue diversity — satellite images have narrow hue range, non-satellite has wide spread
    hue_std = float(np.std(h))

    # Signal 3: Brightness pattern — city lights have sharp isolated spots, satellite clouds are smooth
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Signal 4: Color channel correlation
    # Satellite images: R, G, B channels are highly correlated (grey-ish)
    # Non-satellite: channels decorrelate (city lights, colored objects)
    b_ch, g_ch, r_ch = cv2.split(img_bgr)
    rg_mat = np.corrcoef(r_ch.flatten(), g_ch.flatten())
    rb_mat = np.corrcoef(r_ch.flatten(), b_ch.flatten())
    rg_corr = float(rg_mat[0, 1]) if not np.isnan(rg_mat[0, 1]) else 1.0
    rb_corr = float(rb_mat[0, 1]) if not np.isnan(rb_mat[0, 1]) else 1.0
    channel_correlation = (rg_corr + rb_corr) / 2.0

    # Rejection rules — need 3+ signals to reject
    is_too_colorful = high_sat_ratio > 0.50       # >50% vivid pixels
    is_hue_diverse = hue_std > 60.0               # wild color variety
    is_city_lights = laplacian_var > 800.0        # sharp isolated bright spots
    is_decorrelated = channel_correlation < 0.85  # RGB channels diverge

    rejection_score = int(sum([is_too_colorful, is_hue_diverse, is_city_lights, is_decorrelated]))
    is_valid = rejection_score < 3

    if not is_valid:
        failed_signals = []
        if is_too_colorful: failed_signals.append(f"high saturation ({high_sat_ratio*100:.1f}% > 50%)")
        if is_hue_diverse: failed_signals.append(f"broad hue diversity (std {hue_std:.1f} > 60)")
        if is_city_lights: failed_signals.append(f"high texture complexity (Laplacian var {laplacian_var:.1f} > 800)")
        if is_decorrelated: failed_signals.append(f"decorrelated RGB channels ({channel_correlation:.2f} < 0.85)")
        reason = f"Image failed {rejection_score}/4 satellite checks: " + "; ".join(failed_signals)
        logger.warning("validate_satellite_image REJECTED: %s", reason)
        return False, reason

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

    # Filter isolated point-source light artifacts (< 50 px) before morphology
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 50:
            thresh[labels == i] = 0

    # 5. Morphological Closing (5x5 structuring element, 3 iterations)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    # 6. Extract metrics
    cloud_coverage_pct = round(float((np.sum(closed_mask > 0) / closed_mask.size) * 100.0), 2)

    # Cold core density: percentage of convective cold pixels (dark cloud tops in TIR)
    if mean_b >= 50.0:
        cold_core_pixels = np.count_nonzero((closed_mask > 0) & (gray <= 85))
    else:
        cold_core_pixels = np.count_nonzero(closed_mask > 0)

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
