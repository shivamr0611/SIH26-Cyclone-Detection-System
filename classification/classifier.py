"""
==============================================================================
Module 3: Classification & Intensity (classification/classifier.py)
==============================================================================
This module calculates meteorological metrics for detected cyclones:
- Dvorak T-Number (T1.5 to T7.0 standard hurricane rating)
- Maximum Sustained Wind Speed (km/h)
- Central Sea-Level Pressure (MSLP in hPa)
- Saffir-Simpson / IMD Storm Category (Depression, Cat 1 to Cat 5)
"""

from typing import Dict, Any


class CycloneClassifier:
    """Estimates storm category, wind speed, and pressure."""

    def classify(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates intensity numbers if cyclone was detected.
        """
        if not detection_result.get("detected", False):
            return {
                "detected": False,
                "category": "None",
                "category_color": "#10b981",
                "wind_speed_kmh": 0,
                "pressure_hpa": 1013,
                "dvorak_t_number": "T0.5",
                "risk_level": "NONE",
                "risk_color": "#10b981"
            }

        cloud_pct = detection_result.get("cloud_coverage_percent", 30.0)
        core_pct = detection_result.get("dense_core_percent", 10.0)

        # 1. Dvorak T-Number (T1.5 - T7.2)
        t_number = min(7.2, max(1.5, 1.0 + (cloud_pct * 0.05) + (core_pct * 0.12)))

        # 2. Wind Speed (km/h)
        wind_speed = int(round(30 + (t_number ** 2.1) * 3.5))

        # 3. Central Pressure (hPa)
        pressure = int(round(1013 - (wind_speed * 0.45)))

        # 4. Storm Category based on Wind Speed
        category = "Category 1"
        category_color = "#eab308"

        if wind_speed >= 215:
            category = "Category 5"
            category_color = "#ec4899"
        elif wind_speed >= 165:
            category = "Category 4"
            category_color = "#ef4444"
        elif wind_speed >= 130:
            category = "Category 3"
            category_color = "#f97316"
        elif wind_speed >= 90:
            category = "Category 2"
            category_color = "#f59e0b"
        elif wind_speed < 62:
            category = "Depression"
            category_color = "#3b82f6"

        # 5. Risk Assessment
        if wind_speed >= 160:
            risk_level = "HIGH"
            risk_color = "#ef4444"
        elif wind_speed >= 90:
            risk_level = "MODERATE"
            risk_color = "#f59e0b"
        else:
            risk_level = "LOW"
            risk_color = "#10b981"

        return {
            "detected": True,
            "category": category,
            "category_color": category_color,
            "wind_speed_kmh": wind_speed,
            "pressure_hpa": pressure,
            "dvorak_t_number": f"T{round(t_number, 1)}",
            "risk_level": risk_level,
            "risk_color": risk_color
        }
