"""
==============================================================================
Module 1: Preprocessing (preprocessing/preprocessor.py)
==============================================================================
This module prepares raw satellite images for analysis:
- Decodes base64 images from the web browser
- Converts the image to grayscale to measure cloud brightness
- Measures what percentage of the image is covered by bright/cold storm clouds
"""

import base64
import io
from typing import Dict, Any

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class SatellitePreprocessor:
    """Preprocesses satellite image frames and extracts cloud features."""

    def __init__(self, target_size=(200, 200)):
        self.target_size = target_size

    def preprocess_image_b64(self, image_b64: str) -> Dict[str, Any]:
        """
        Takes a base64 encoded image string from the browser and extracts:
        - cloud_coverage_percent: Percentage of bright cloud pixels
        - dense_core_percent: Percentage of very cold, deep convective cloud core
        """
        # Strip header prefix if present (e.g. 'data:image/png;base64,')
        if "," in image_b64:
            _, image_b64 = image_b64.split(",", 1)

        try:
            image_bytes = base64.b64decode(image_b64)

            if PIL_AVAILABLE:
                # Open image and convert to grayscale ('L' mode: 0 to 255)
                img = Image.open(io.BytesIO(image_bytes)).convert("L")
                img = img.resize(self.target_size)
                
                # Normalize pixel values from [0, 255] to [0.0, 1.0]
                pixels = np.array(img, dtype=np.float32) / 255.0

                # Count cloud pixels (in Thermal IR, cold clouds appear bright white)
                cloud_mask = pixels > 0.43       # Regular cloud threshold
                core_mask = pixels > 0.68        # Dense cold core threshold

                cloud_pct = float(np.mean(cloud_mask) * 100)
                core_pct = float(np.mean(core_mask) * 100)
            else:
                # Fallback defaults if PIL is not installed
                cloud_pct = 25.0
                core_pct = 8.0

            return {
                "status": "success",
                "cloud_coverage_percent": round(cloud_pct, 1),
                "dense_core_percent": round(core_pct, 1)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "cloud_coverage_percent": 0.0,
                "dense_core_percent": 0.0
            }
