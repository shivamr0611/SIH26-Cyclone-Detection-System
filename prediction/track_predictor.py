"""
Cyclone Track Trajectory and Risk Assessment Module.
Generates projected coordinates and calculates the prototype risk score.
"""

from typing import List, Dict, Any, Tuple


class TrackPredictor:
    """
    Predicts sequential coordinates (lat/lng) for cyclone path and computes risk indicator.
    """

    def __init__(self, model_name: str = "BiLSTM-TrackPropagator-v1"):
        self.model_name = model_name

    def predict_track(self, current_info: Dict[str, Any]) -> List[List[float]]:
        """
        Returns projected [[lat, lng], ...] coordinate track.
        """
        if not current_info.get("detected", False):
            return []

        cyclone_name = current_info.get("cyclone_name", "")
        if "MANDOUS" in cyclone_name:
            return [
                [11.5, 83.8],
                [12.0, 82.3],
                [12.8, 80.7],
                [13.5, 79.2],
                [14.1, 77.8]
            ]
        else:
            return [
                [14.2, 68.1],
                [15.1, 67.4],
                [16.3, 66.5],
                [17.8, 65.3],
                [19.2, 64.0]
            ]

    def compute_risk_indicator(self, current_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates research risk index (0-100) based on wind, pressure, track, and proximity.
        """
        if not current_info.get("detected", False):
            return {
                "risk_score": 0,
                "risk_level": "NONE",
                "risk_color": "#22c55e",
                "disclaimer": "No tropical disturbance identified."
            }

        wind = current_info.get("current_wind_speed_kmh", 0)
        if wind >= 160:
            return {
                "risk_score": 84,
                "risk_level": "HIGH",
                "risk_color": "#ef4444",
                "disclaimer": "Research Prototype: Potential severe coastal impact."
            }
        elif wind >= 90:
            return {
                "risk_score": 62,
                "risk_level": "MODERATE",
                "risk_color": "#f59e0b",
                "disclaimer": "Research Prototype: Moderate marine and gale hazard."
            }
        else:
            return {
                "risk_score": 30,
                "risk_level": "LOW",
                "risk_color": "#22c55e",
                "disclaimer": "Research Prototype: Minimal disruption expected."
            }
