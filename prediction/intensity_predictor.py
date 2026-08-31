"""
==============================================================================
Module 4: Prediction (prediction/intensity_predictor.py)
==============================================================================
This module predicts future cyclone intensity over 5 time horizons:
- Now
- +12 Hours
- +24 Hours (Peak intensification)
- +48 Hours
- +72 Hours (Dissipation / Landfall weakening)
"""

from typing import List, Dict, Any


class IntensityPredictor:
    """Predicts future wind speeds and central pressures."""

    def predict(self, classification_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates forecast table rows.
        """
        if not classification_result.get("detected", False):
            return []

        current_wind = classification_result.get("wind_speed_kmh", 120)
        current_pressure = classification_result.get("pressure_hpa", 960)
        category = classification_result.get("category", "Category 2")

        return [
            {
                "horizon": "Now",
                "wind": current_wind,
                "pressure": current_pressure,
                "category": category,
                "trend": "flat"
            },
            {
                "horizon": "+12 hr",
                "wind": int(round(current_wind * 1.08)),
                "pressure": current_pressure - 6,
                "category": category,
                "trend": "up"
            },
            {
                "horizon": "+24 hr",
                "wind": int(round(current_wind * 1.14)),
                "pressure": current_pressure - 12,
                "category": category,
                "trend": "up"
            },
            {
                "horizon": "+48 hr",
                "wind": int(round(current_wind * 1.02)),
                "pressure": current_pressure - 3,
                "category": category,
                "trend": "down"
            },
            {
                "horizon": "+72 hr",
                "wind": int(round(current_wind * 0.80)),
                "pressure": current_pressure + 15,
                "category": "Weakening",
                "trend": "down"
            }
        ]
