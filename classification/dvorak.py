#!/usr/bin/env python3
"""
dvorak.py - Operational Dvorak Technique (Dvorak 1984) Rule-Based Cyclone Classifier.

Estimates tropical cyclone intensity (T-Number), Maximum Sustained Wind Speed (kt / km/h),
Central Sea-Level Pressure (hPa), and official IMD/Saffir-Simpson categories from
satellite features: Central Dense Overcast (CDO) radius, log-polar spiral banding score,
and eye geometry.
"""

import logging
from typing import Any, Dict, Optional

# Module logger
logger = logging.getLogger("cycloneai.dvorak")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def compute_t_number(
    cdo_radius_km: float = 0.0,
    banding_score: float = 0.0,
    has_clear_eye: bool = False,
    eye_diameter_km: Optional[float] = None,
) -> float:
    """
    Computes Dvorak T-Number (T1.0 to T8.0) based on operational Dvorak rules.
    """
    t = 1.0

    if cdo_radius_km > 111.0:
        t += 0.5
    if cdo_radius_km > 222.0:
        t += 0.5
    if cdo_radius_km > 333.0:
        t += 0.5

    if banding_score > 0.3:
        t += 0.5
    if banding_score > 0.6:
        t += 0.5

    if has_clear_eye:
        t += 1.0
        if eye_diameter_km and eye_diameter_km < 20.0:
            t += 0.5

    t = min(t, 8.0)
    t = round(t, 1)

    print(f"[Dvorak] CDO={cdo_radius_km:.1f}km, banding={banding_score:.2f}, eye={has_clear_eye} -> T={t}")
    return t


def t_number_to_category(t: float) -> Dict[str, Any]:
    """
    Maps Dvorak T-number to IMD cyclone category, wind speed, pressure, and Saffir-Simpson equivalent.

    Dvorak - IMD Correlation Table:
      - T 1.0 - 1.5 : Depression (25-35 kt, 1000-1004 hPa)
      - T 2.0 - 2.5 : Deep Depression (28-33 kt, 996-1000 hPa)
      - T 3.0 - 3.5 : Cyclonic Storm (34-47 kt, 985-996 hPa)
      - T 4.0 - 4.5 : Severe Cyclonic Storm (48-63 kt, 970-985 hPa)
      - T 5.0 - 5.5 : Very Severe Cyclonic Storm (64-89 kt, 950-970 hPa)
      - T 6.0 - 6.5 : Extremely Severe Cyclonic Storm (90-119 kt, 920-950 hPa)
      - T 7.0+      : Super Cyclonic Storm (120+ kt, <920 hPa)

    Args:
        t: Dvorak T-Number.

    Returns:
        dict: Meteorological intensity attributes.
    """
    if t < 2.0:
        category = "Depression"
        wind_kt = 30.0
        pressure_hpa = 1002.0
        saffir_simpson = "Tropical Depression"
        category_color = "#3b82f6"
        risk_level = "LOW"
        risk_color = "#10b981"
    elif 2.0 <= t < 3.0:
        category = "Deep Depression"
        wind_kt = 33.0
        pressure_hpa = 998.0
        saffir_simpson = "Tropical Depression"
        category_color = "#06b6d4"
        risk_level = "LOW"
        risk_color = "#10b981"
    elif 3.0 <= t < 4.0:
        category = "Cyclonic Storm"
        wind_kt = 45.0
        pressure_hpa = 990.0
        saffir_simpson = "Tropical Storm"
        category_color = "#eab308"
        risk_level = "MODERATE"
        risk_color = "#f59e0b"
    elif 4.0 <= t < 5.0:
        category = "Severe Cyclonic Storm"
        wind_kt = 55.0
        pressure_hpa = 978.0
        saffir_simpson = "Category 1 Hurricane"
        category_color = "#f97316"
        risk_level = "HIGH"
        risk_color = "#ef4444"
    elif 5.0 <= t < 6.0:
        category = "Very Severe Cyclonic Storm"
        wind_kt = 75.0
        pressure_hpa = 960.0
        saffir_simpson = "Category 2/3 Hurricane"
        category_color = "#ef4444"
        risk_level = "HIGH"
        risk_color = "#ef4444"
    elif 6.0 <= t < 7.0:
        category = "Extremely Severe Cyclonic Storm"
        wind_kt = 105.0
        pressure_hpa = 935.0
        saffir_simpson = "Category 4 Hurricane"
        category_color = "#ec4899"
        risk_level = "EXTREME"
        risk_color = "#9333ea"
    else:
        category = "Super Cyclonic Storm"
        wind_kt = 135.0
        pressure_hpa = 910.0
        saffir_simpson = "Category 5 Hurricane"
        category_color = "#a855f7"
        risk_level = "EXTREME"
        risk_color = "#9333ea"

    wind_speed_kmh = round(wind_kt * 1.852, 1)

    return {
        "category": category,
        "wind_kt": wind_kt,
        "wind_speed_kmh": int(round(wind_speed_kmh)),
        "pressure_hpa": int(round(pressure_hpa)),
        "t_number": f"T{t:.1f}",
        "t_number_val": t,
        "saffir_simpson_equivalent": saffir_simpson,
        "category_color": category_color,
        "risk_level": risk_level,
        "risk_color": risk_color,
    }


def classify(detection_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts geometric & morphological inputs from detection results and returns Dvorak classification.

    Args:
        detection_dict: Dictionary returned by detect_cyclone or full_detection.

    Returns:
        dict: Complete Dvorak meteorological assessment.
    """
    cdo_radius = float(detection_dict.get("cdo_radius_km", 0.0) or 0.0)
    banding = float(detection_dict.get("banding_score", 0.0) or 0.0)
    has_eye = bool(detection_dict.get("has_clear_eye", False))
    eye_dia = detection_dict.get("eye_diameter_km")
    eye_dia_val = float(eye_dia) if eye_dia is not None else None

    t_val = compute_t_number(
        cdo_radius_km=cdo_radius,
        banding_score=banding,
        has_clear_eye=has_eye,
        eye_diameter_km=eye_dia_val,
    )

    result = t_number_to_category(t_val)
    result["cdo_radius_km"] = cdo_radius
    result["banding_score"] = banding
    result["has_clear_eye"] = has_eye
    result["eye_diameter_km"] = eye_dia_val

    return result


if __name__ == "__main__":
    print("=== Testing Dvorak Technique Module ===")

    scenarios = [
        {"desc": "Developing Depression", "cdo_radius_km": 50, "banding_score": 0.1, "has_clear_eye": False},
        {"desc": "Cyclonic Storm with Bands", "cdo_radius_km": 150, "banding_score": 0.45, "has_clear_eye": False},
        {"desc": "Very Severe Cyclone with Eye", "cdo_radius_km": 250, "banding_score": 0.7, "has_clear_eye": True, "eye_diameter_km": 35},
        {"desc": "Super Cyclone (Tight Pin-hole Eye)", "cdo_radius_km": 300, "banding_score": 0.85, "has_clear_eye": True, "eye_diameter_km": 15},
    ]

    for sc in scenarios:
        res = classify(sc)
        print(f"\nScenario: {sc['desc']}")
        print(f"  -> T-Number:      {res['t_number']}")
        print(f"  -> IMD Category:  {res['category']}")
        print(f"  -> Wind Speed:    {res['wind_speed_kmh']} km/h ({res['wind_kt']} kt)")
        print(f"  -> Central MSLP:  {res['pressure_hpa']} hPa")
        print(f"  -> Saffir-Simpson:{res['saffir_simpson_equivalent']}")
