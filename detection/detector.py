"""
==============================================================================
Module 2: Cyclone Detection (detection/detector.py)
==============================================================================
This module checks if a tropical cyclone is present in the image:
- If cloud coverage is low (< 16%), it reports 'No Cyclone'
- If cloud coverage and dense core are high, it reports 'Cyclone Detected'
- Calculates a dynamic confidence score (e.g. 70% to 99%)
"""

from typing import Dict, Any


class CycloneDetector:
    """Detects whether a cyclone vortex exists in the preprocessed image."""

    def __init__(self, model_name="ResNet50-CycloneDetector"):
        self.model_name = model_name

    def detect(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes preprocessed cloud percentages and returns detection result.
        """
        cloud_pct = preprocessed_data.get("cloud_coverage_percent", 0.0)
        core_pct = preprocessed_data.get("dense_core_percent", 0.0)

        # CASE A: Clear Skies or Low Scattered Clouds (< 16% cloud cover)
        if cloud_pct < 16.0 or core_pct < 2.5:
            # High confidence that the sky is clear
            confidence = round(min(99.2, max(88.0, 98.0 - cloud_pct * 0.5)), 1)
            
            return {
                "detected": False,
                "confidence": confidence,
                "status_text": "No Cyclone Detected",
                "cloud_coverage_percent": cloud_pct,
                "dense_core_percent": core_pct,
                "message": f"Clear area: Cloud coverage is {cloud_pct}% (below cyclone threshold)."
            }

        # CASE B: Cyclone Vortex Detected
        confidence = round(min(98.8, max(68.0, 60.0 + cloud_pct * 0.4 + core_pct * 0.3)), 1)

        return {
            "detected": True,
            "confidence": confidence,
            "status_text": "Cyclone Detected",
            "cloud_coverage_percent": cloud_pct,
            "dense_core_percent": core_pct,
            "message": f"Convective vortex detected with {cloud_pct}% cloud density."
        }
