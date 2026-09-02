#!/usr/bin/env python3
"""
CycloneAI - Production REST API Backend (backend/app.py).

Wires together the full cyclone processing pipeline:
- POST /api/analyze  : Multi-channel preprocessing -> Hough/Vorticity Detection -> CNN/Dvorak Classification
- POST /api/predict  : Multi-horizon (+12h to +72h) LSTM & NOAA SST physics intensity forecasting
- GET  /api/explain  : Grad-CAM heatmap explainability visualization
- GET  /api/track    : GeoJSON track forecasting with 24h, 48h, 72h uncertainty cones
- GET  /api/alert    : Coastal district impact & storm surge hazard assessment via Shapely
- GET  /api/live     : Active North Indian Ocean cyclone monitoring feed
- GET  /api/health   : Diagnostic health check of deep learning & meteorological engines
"""

import base64
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root and site-packages are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional fallback search path for local Windows Python packages
_py313_site = r"C:\Users\ASHWITH\AppData\Local\Programs\Python\Python313\Lib\site-packages"
if os.path.exists(_py313_site) and _py313_site not in sys.path:
    sys.path.append(_py313_site)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("cycloneai.api")

# Try importing Flask and Flask-CORS
try:
    from flask import Flask, Response, jsonify, request, send_from_directory
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logger.warning("Flask or Flask-CORS not installed. Install via pip install flask flask-cors.")

# Try importing Shapely
try:
    from shapely.geometry import Point, Polygon, mapping, shape
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    logger.warning("Shapely not installed. Using Haversine geometric fallback.")

# Import CycloneAI Pipeline Modules
from classification.classifier import CycloneClassifier, classify_with_cnn, full_classify, generate_grad_cam
from classification.dvorak import classify, compute_t_number, t_number_to_category
from detection.detector import CycloneDetector, detect_cyclone
from prediction.intensity_predictor import CycloneIntensityLSTM, IntensityPredictor, predict_intensity
from preprocessing.band_analyzer import analyze_spiral_bands, full_detection
from preprocessing.preprocessor import (
    SatellitePreprocessor,
    preprocess_multichannel,
    preprocess_tir,
    validate_satellite_image,
)

# Initialize Flask application
DASHBOARD_FOLDER = PROJECT_ROOT / "dashboard"
DATA_FOLDER = PROJECT_ROOT / "backend" / "data"

if FLASK_AVAILABLE:
    app = Flask(__name__, static_folder=str(DASHBOARD_FOLDER), static_url_path="")
    CORS(app)
else:
    app = None  # type: ignore

# In-memory session cache for Grad-CAM explainability
LAST_ANALYSIS_CACHE: Dict[str, Any] = {
    "heatmap_b64": "",
    "predicted_class": "Cyclonic_Storm",
    "confidence": 0.85,
    "timestamp": None,
    "image_bytes": None,
}


# ==============================================================================
# Middleware: Request Logging & Performance Profiling
# ==============================================================================

if FLASK_AVAILABLE and app:
    @app.before_request
    def start_timer() -> None:
        request._start_time = time.time()

    @app.after_request
    def log_request(response: Any) -> Any:
        duration = 0.0
        if hasattr(request, "_start_time"):
            duration = round((time.time() - request._start_time) * 1000, 2)
        logger.info(
            "%s %s -> %d (%s ms)",
            request.method,
            request.path,
            response.status_code,
            duration,
        )
        return response


# ==============================================================================
# Route 0: Frontend Dashboard Static Hosting & Diagnostics
# ==============================================================================

if FLASK_AVAILABLE and app:
    @app.route("/")
    def serve_dashboard() -> Any:
        """Serves the frontend web dashboard index.html."""
        return send_from_directory(str(DASHBOARD_FOLDER), "index.html")

    @app.route("/api/test", methods=["GET"])
    def test_pipeline() -> Any:
        """Test that different synthetic inputs give different outputs"""
        import cv2
        import numpy as np

        results = []
        for size in [30, 80, 150]:
            img = np.ones((512, 512), dtype=np.uint8) * 200  # warm ocean
            cx, cy = 256, 256
            y, x = np.ogrid[:512, :512]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= size ** 2
            img[mask] = 50  # cold cloud top

            _, buf = cv2.imencode(".png", img)
            image_bytes = buf.tobytes()

            pre = preprocess_tir(image_bytes)
            det = detect_cyclone(pre["processed_image_b64"])
            band = analyze_spiral_bands(image_bytes, (256, 256))
            dvorak = classify({
                "cdo_radius_km": det["cdo_radius_km"],
                "banding_score": band["banding_score"],
                "has_clear_eye": det["has_clear_eye"],
                "eye_diameter_km": det.get("eye_diameter_km"),
            })

            results.append({
                "test_cold_core_radius_px": size,
                "cloud_coverage_pct": pre["cloud_coverage_pct"],
                "cdo_radius_km": det["cdo_radius_km"],
                "t_number": dvorak.get("t_number", dvorak.get("t_number_val")),
                "category": dvorak["category"],
            })

        t_numbers = [r["t_number"] for r in results]
        return jsonify({
            "status": "PASS" if len(set(t_numbers)) > 1 else "FAIL — pipeline is still returning constant T-numbers",
            "results": results,
        })


# ==============================================================================
# Route 1: Health Diagnostic Check
# ==============================================================================

from models.model_loader import get_model_status

if FLASK_AVAILABLE and app:
    @app.route("/api/health", methods=["GET"])
    def health_check() -> Any:
        """Diagnostic health check verifying module availability."""
        model_status = get_model_status()
        status_info = {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "models": model_status,
            "modules": {
                "cnn": model_status["cnn"]["available"],
                "lstm": model_status["lstm"]["available"],
                "dvorak": model_status["dvorak"]["available"],
                "shapely_gis": SHAPELY_AVAILABLE,
            },
        }
        return jsonify(status_info), 200


# ==============================================================================
# Route 2: /api/analyze (Full Multi-Channel Ingestion & Analysis Pipeline)
# ==============================================================================

if FLASK_AVAILABLE and app:
    @app.route("/api/analyze", methods=["POST"])
    def analyze_satellite_image() -> Any:
        """
        POST /api/analyze
        Accepts multipart/form-data:
          - tir (required): Thermal Infrared satellite channel
          - wv  (optional): Water Vapor channel
          - vis (optional): Visible channel
        Also accepts JSON with base64 'image' or 'tir' for backward compatibility.
        """
        current_stage = "request_parsing"
        try:
            tir_bytes: Optional[bytes] = None
            wv_bytes: Optional[bytes] = None
            vis_bytes: Optional[bytes] = None

            # 1. Parse Multipart Form Data
            if request.files:
                if "tir" in request.files and request.files["tir"].filename:
                    tir_bytes = request.files["tir"].read()
                elif "image" in request.files and request.files["image"].filename:
                    tir_bytes = request.files["image"].read()

                if "wv" in request.files and request.files["wv"].filename:
                    wv_bytes = request.files["wv"].read()
                if "vis" in request.files and request.files["vis"].filename:
                    vis_bytes = request.files["vis"].read()

            # 2. Parse JSON base64 input fallback
            if not tir_bytes and request.is_json:
                data = request.get_json() or {}
                b64_str = data.get("tir") or data.get("image", "")
                if b64_str:
                    if "," in b64_str:
                        _, b64_str = b64_str.split(",", 1)
                    tir_bytes = base64.b64decode(b64_str)

                if data.get("wv"):
                    wv_raw = data["wv"]
                    if "," in wv_raw:
                        _, wv_raw = wv_raw.split(",", 1)
                    wv_bytes = base64.b64decode(wv_raw)

                if data.get("vis"):
                    vis_raw = data["vis"]
                    if "," in vis_raw:
                        _, vis_raw = vis_raw.split(",", 1)
                    vis_bytes = base64.b64decode(vis_raw)

            if not tir_bytes:
                return jsonify({
                    "status": "error",
                    "error": "Missing mandatory Thermal Infrared channel (field 'tir')",
                    "stage": "input_validation",
                }), 400

            # Image Validation Stage (BUG 3 Fix)
            current_stage = "image_validation"
            is_valid, validation_err = validate_satellite_image(tir_bytes)
            if not is_valid:
                return jsonify({
                    "status": "error",
                    "error": validation_err,
                    "stage": "image_validation",
                }), 400

            # Stage 1: Preprocess
            current_stage = "preprocessing"
            pre = preprocess_tir(tir_bytes)
            img_array = preprocess_multichannel(tir_bytes, wv_bytes, vis_bytes)

            # Stage 2: Detect — PASS THE PROCESSED IMAGE (Base64)
            current_stage = "detection"
            processed_b64 = pre["processed_image_b64"]
            det = detect_cyclone(processed_b64)

            # Stage 3: Band analysis — PASS THE ACTUAL CENTER
            current_stage = "band_analysis"
            center = det.get("eye_center") or det.get("circulation_center") or [112, 112]
            band = analyze_spiral_bands(tir_bytes, tuple(center))

            # Stage 4: Dvorak — PASS ACTUAL detection + band values
            current_stage = "dvorak"
            dvorak_result = classify({
                "cdo_radius_km": det["cdo_radius_km"],
                "banding_score": band["banding_score"],
                "has_clear_eye": det["has_clear_eye"],
                "eye_diameter_km": det.get("eye_diameter_km"),
            })

            # Stage 5: Predict — PASS ACTUAL dvorak wind/pressure
            current_stage = "intensity_prediction"
            wind_kt = dvorak_result["wind_kt"]
            pressure = dvorak_result["pressure_hpa"]
            forecast = predict_intensity(
                current_wind_kt=wind_kt,
                current_pressure=pressure,
                lat=15.0,
                lon=85.0,
                storm_speed=10.0,
            )

            # Stage 6: CNN (image-based)
            current_stage = "cnn"
            cnn = full_classify(tir_bytes, dvorak_result)

            # Update Session Cache for /api/explain
            LAST_ANALYSIS_CACHE["heatmap_b64"] = cnn.get("grad_cam_b64", "")
            LAST_ANALYSIS_CACHE["predicted_class"] = cnn.get("consensus_category", dvorak_result["category"])
            LAST_ANALYSIS_CACHE["confidence"] = cnn.get("confidence", 0.85)
            LAST_ANALYSIS_CACHE["timestamp"] = datetime.now(timezone.utc).isoformat()
            LAST_ANALYSIS_CACHE["image_bytes"] = tir_bytes

            is_cyclone = bool(det["cyclone_detected"])

            # Format forecast table
            forecast_table = []
            if is_cyclone:
                for horizon in ["now", "+12h", "+24h", "+48h", "+72h"]:
                    if horizon in forecast:
                        f_item = forecast[horizon]
                        trend = "up" if "+12h" in horizon or "+24h" in horizon else ("down" if "+48h" in horizon or "+72h" in horizon else "flat")
                        forecast_table.append({
                            "horizon": horizon.replace("now", "Now").replace("+", "+").replace("h", " hr"),
                            "wind": f_item.get("wind_kmh", f_item.get("wind_speed_kmh", 0)),
                            "pressure": f_item.get("pressure_hpa", 1000),
                            "category": f_item.get("category", ""),
                            "trend": trend,
                        })

            # Assemble response
            response_payload = {
                "status": "success",
                "preprocessing": pre,
                "detection": det,
                "band_analysis": band,
                "dvorak": dvorak_result,
                "cnn": cnn,
                "forecast": forecast if is_cyclone else {},
                "forecast_table": forecast_table,
                "cyclone_detected": is_cyclone,
                "not_detected_reason": det.get("not_detected_reason"),
                "vortex_concentration_score": det.get("vortex_concentration_score", 0.0),
                # Compatibility fields for frontend
                "isCyclone": is_cyclone,
                "confidence": det.get("confidence", cnn.get("confidence", 0.85)),
                "category": dvorak_result["category"] if is_cyclone else "None",
                "categoryColor": dvorak_result.get("category_color", "#f97316") if is_cyclone else "#10b981",
                "windSpeed": dvorak_result.get("wind_speed_kmh", 0) if is_cyclone else 0,
                "wind_kt": dvorak_result["wind_kt"] if is_cyclone else 0.0,
                "pressure": dvorak_result["pressure_hpa"] if is_cyclone else 1012,
                "dvorakRating": dvorak_result.get("t_number", dvorak_result.get("t_number_val", 1.0)) if is_cyclone else "T0.0",
                "riskLevel": dvorak_result.get("risk_level", "LOW") if is_cyclone else "NONE",
                "riskColor": dvorak_result.get("risk_color", "#10b981") if is_cyclone else "#10b981",
                "cloudCoverage": pre["cloud_coverage_pct"],
                "denseCore": pre["cold_core_density"],
                "meanBrightness": pre["mean_brightness"],
                "hasEye": det["has_clear_eye"] if is_cyclone else False,
                "eyeCenter": det["eye_center"] if is_cyclone else None,
                "eyeDiameterKm": det["eye_diameter_km"] if is_cyclone else None,
                "cdoRadiusKm": det["cdo_radius_km"],
                "circulationCenter": det["circulation_center"],
                "bandCount": band["band_count"],
                "dominantBandAngle": band["dominant_band_angle_deg"],
                "rotationDirection": band["rotation_direction"],
                "bandingScore": band["banding_score"],
                "gradCamB64": cnn.get("grad_cam_b64", ""),
                "forecast_72h": forecast if is_cyclone else {},
            }

            return jsonify(response_payload), 200

        except Exception as err:
            logger.error("Analysis pipeline failed at stage '%s': %s", current_stage, err, exc_info=True)
            return jsonify({
                "status": "error",
                "error": str(err),
                "stage": current_stage,
            }), 400


# ==============================================================================
# Route 3: /api/predict (Multi-Horizon 72h LSTM Intensity Forecasting)
# ==============================================================================

if FLASK_AVAILABLE and app:
    @app.route("/api/predict", methods=["POST"])
    def predict_endpoint() -> Any:
        """
        POST /api/predict
        Accepts JSON:
          - wind_kt (float)
          - pressure_hpa (float)
          - lat (float, optional)
          - lon (float, optional)
          - storm_speed (float, optional)
          - history (list of 8 timesteps, optional)
        """
        try:
            data = request.get_json() or {}

            # Handle backward compatibility if image sent to predict
            if "image" in data or "tir" in data:
                return analyze_satellite_image()

            wind_kt = float(data.get("wind_kt", data.get("wind", data.get("wind_speed_kmh", 120) / 1.852)))
            pressure_hpa = float(data.get("pressure_hpa", data.get("pressure", 960)))
            lat = float(data.get("lat", 16.0))
            lon = float(data.get("lon", 85.0))
            storm_speed = float(data.get("storm_speed", 12.0))

            history_list = data.get("history")
            recent_history = np.array(history_list, dtype=np.float32) if history_list else None

            forecast_res = predict_intensity(
                current_wind_kt=wind_kt,
                current_pressure=pressure_hpa,
                lat=lat,
                lon=lon,
                storm_speed=storm_speed,
                recent_history=recent_history,
            )

            return jsonify({
                "status": "success",
                "forecast": forecast_res,
                "table": [
                    {"horizon": "Now", "wind": forecast_res["now"]["wind_speed_kmh"], "pressure": forecast_res["now"]["pressure_hpa"], "category": forecast_res["now"]["category"], "trend": "flat"},
                    {"horizon": "+12 hr", "wind": forecast_res["+12h"]["wind_speed_kmh"], "pressure": forecast_res["+12h"]["pressure_hpa"], "category": forecast_res["+12h"]["category"], "trend": "up"},
                    {"horizon": "+24 hr", "wind": forecast_res["+24h"]["wind_speed_kmh"], "pressure": forecast_res["+24h"]["pressure_hpa"], "category": forecast_res["+24h"]["category"], "trend": "up"},
                    {"horizon": "+48 hr", "wind": forecast_res["+48h"]["wind_speed_kmh"], "pressure": forecast_res["+48h"]["pressure_hpa"], "category": forecast_res["+48h"]["category"], "trend": "down"},
                    {"horizon": "+72 hr", "wind": forecast_res["+72h"]["wind_speed_kmh"], "pressure": forecast_res["+72h"]["pressure_hpa"], "category": forecast_res["+72h"]["category"], "trend": "down"},
                ],
            }), 200

        except Exception as err:
            logger.error("Prediction failed: %s", err, exc_info=True)
            return jsonify({"status": "error", "error": str(err)}), 400


# ==============================================================================
# Route 4: /api/explain (Grad-CAM Heatmap Decision Explanation)
# ==============================================================================

if FLASK_AVAILABLE and app:
    @app.route("/api/explain", methods=["GET"])
    def explain_endpoint() -> Any:
        """
        GET /api/explain
        Query params:
          - image_path (optional): Path to image file
        Returns cached Grad-CAM overlay or generates from image.
        """
        try:
            image_path_param = request.args.get("image_path")

            if image_path_param and Path(image_path_param).exists():
                with open(image_path_param, "rb") as f:
                    raw_bytes = f.read()
                img_tensor = preprocess_multichannel(raw_bytes)
                heatmap_b64 = generate_grad_cam(img_tensor, target_class=2)
                cnn_res = classify_with_cnn(img_tensor)
                return jsonify({
                    "status": "success",
                    "heatmap_b64": heatmap_b64,
                    "predicted_class": cnn_res["predicted_class"],
                    "confidence": cnn_res["confidence"],
                }), 200

            # Fallback to in-memory cached analysis
            if LAST_ANALYSIS_CACHE["heatmap_b64"]:
                return jsonify({
                    "status": "success",
                    "heatmap_b64": LAST_ANALYSIS_CACHE["heatmap_b64"],
                    "predicted_class": LAST_ANALYSIS_CACHE["predicted_class"],
                    "confidence": LAST_ANALYSIS_CACHE["confidence"],
                    "timestamp": LAST_ANALYSIS_CACHE["timestamp"],
                }), 200

            # Generate synthetic demonstration heatmap
            dummy_tensor = np.zeros((3, 224, 224), dtype=np.float32)
            y, x = np.ogrid[:224, :224]
            dummy_tensor[0, (x - 112)**2 + (y - 112)**2 < 45**2] = 0.95
            demo_heatmap = generate_grad_cam(dummy_tensor, target_class=2)

            return jsonify({
                "status": "success",
                "heatmap_b64": demo_heatmap,
                "predicted_class": "Cyclonic_Storm",
                "confidence": 0.82,
                "note": "Demonstration mode: No image analyzed yet in current session.",
            }), 200

        except Exception as err:
            logger.error("Explain endpoint failed: %s", err, exc_info=True)
            return jsonify({"status": "error", "error": str(err)}), 400


# ==============================================================================
# Route 5: /api/track (GeoJSON 72h Analog Track & Uncertainty Cone Generator)
# ==============================================================================

def _generate_circle_polygon(center_lon: float, center_lat: float, radius_km: float, num_points: int = 36) -> List[List[float]]:
    """Generate polygon coordinates for an uncertainty circle."""
    coords = []
    # 1 degree latitude ~= 111.13 km
    # 1 degree longitude ~= 111.13 * cos(lat) km
    lat_deg = radius_km / 111.13
    lon_deg = radius_km / (111.13 * max(0.1, math.cos(math.radians(center_lat))))

    for i in range(num_points):
        angle = math.radians(i * (360.0 / num_points))
        p_lon = round(center_lon + lon_deg * math.cos(angle), 4)
        p_lat = round(center_lat + lat_deg * math.sin(angle), 4)
        coords.append([p_lon, p_lat])
    # Close polygon
    coords.append(coords[0])
    return coords


if FLASK_AVAILABLE and app:
    @app.route("/api/track", methods=["GET"])
    def track_endpoint() -> Any:
        """
        GET /api/track
        Query params:
          - lat (float, default: 15.0)
          - lon (float, default: 85.0)
          - wind_kt (float, default: 65.0)
          - heading_deg (float, default: 315.0 - North-West toward Odisha/Andhra)
          - speed_kmh (float, default: 15.0)
        Returns GeoJSON FeatureCollection with track LineString, Points, and Uncertainty Cones.
        """
        try:
            cur_lat = float(request.args.get("lat", 15.0))
            cur_lon = float(request.args.get("lon", 85.0))
            cur_wind = float(request.args.get("wind_kt", 65.0))
            heading = float(request.args.get("heading_deg", 315.0))
            speed_kmh = float(request.args.get("speed_kmh", 15.0))

            # 72-hour forecast in 6-hour intervals (13 steps: 0, 6, 12, ... 72)
            time_steps = list(range(0, 78, 6))
            features: List[Dict[str, Any]] = []
            line_coords: List[List[float]] = []

            p_lat, p_lon = cur_lat, cur_lon
            cur_heading = heading

            track_points_metadata: List[Dict[str, Any]] = []

            for step in time_steps:
                time_label = f"+{step}h" if step > 0 else "Now"

                # Coriolis beta-drift: tracks in Bay of Bengal bend slightly right (northward/northeastward)
                if step > 0:
                    cur_heading += 0.8
                    dist_km = speed_kmh * 6.0
                    rad = math.radians(cur_heading)
                    delta_lat = (dist_km * math.cos(rad)) / 111.13
                    delta_lon = (dist_km * math.sin(rad)) / (111.13 * max(0.1, math.cos(math.radians(p_lat))))
                    p_lat += delta_lat
                    p_lon += delta_lon

                # Wind intensity modulation along track
                if step <= 24:
                    step_wind = cur_wind * (1.0 + (step / 24.0) * 0.20)
                elif step <= 48:
                    step_wind = cur_wind * (1.20 - ((step - 24) / 24.0) * 0.15)
                else:
                    step_wind = cur_wind * (1.05 - ((step - 48) / 24.0) * 0.35)

                step_wind = round(max(25.0, step_wind), 1)
                cat = "Depression"
                if step_wind >= 64:
                    cat = "Very Severe Cyclonic Storm"
                elif step_wind >= 48:
                    cat = "Severe Cyclonic Storm"
                elif step_wind >= 34:
                    cat = "Cyclonic Storm"
                elif step_wind >= 28:
                    cat = "Deep Depression"

                coord_pt = [round(p_lon, 4), round(p_lat, 4)]
                line_coords.append(coord_pt)

                # Uncertainty radius expands with time: ~30km at 0h, 85km at 24h, 160km at 48h, 240km at 72h
                uncertainty_km = round(25.0 + 3.0 * step, 1)

                point_props = {
                    "time": time_label,
                    "hour": step,
                    "lat": coord_pt[1],
                    "lon": coord_pt[0],
                    "wind_kt": step_wind,
                    "wind_kmh": int(round(step_wind * 1.852)),
                    "category": cat,
                    "uncertainty_radius_km": uncertainty_km,
                }
                track_points_metadata.append(point_props)

                # Point Feature
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": coord_pt},
                    "properties": point_props,
                })

                # Uncertainty Cone Polygons at 24h, 48h, 72h
                if step in (24, 48, 72):
                    cone_poly = _generate_circle_polygon(coord_pt[0], coord_pt[1], uncertainty_km)
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [cone_poly]},
                        "properties": {
                            "type": "uncertainty_cone",
                            "horizon": time_label,
                            "center_lat": coord_pt[1],
                            "center_lon": coord_pt[0],
                            "radius_km": uncertainty_km,
                        },
                    })

            # Primary Track LineString Feature
            features.insert(0, {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": line_coords},
                "properties": {
                    "type": "primary_track",
                    "start_time": "Now",
                    "end_time": "+72h",
                    "total_points": len(line_coords),
                },
            })

            geojson_doc = {
                "type": "FeatureCollection",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "features": features,
            }

            return jsonify(geojson_doc), 200

        except Exception as err:
            logger.error("Track generation failed: %s", err, exc_info=True)
            return jsonify({"status": "error", "error": str(err)}), 400


# ==============================================================================
# Route 6: /api/alert (Coastal Impact & Storm Surge Risk Zone Generator)
# ==============================================================================

def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great circle distance in km between two coordinates."""
    r = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(r * c, 1)


if FLASK_AVAILABLE and app:
    @app.route("/api/alert", methods=["GET"])
    def alert_endpoint() -> Any:
        """
        GET /api/alert
        Query params:
          - lat (float, default: 17.5)
          - lon (float, default: 85.5)
          - wind_kt (float, default: 75.0)
          - radius_km (float, default: 250.0)
        Cross-references buffered hazard polygon with Indian coastal districts.
        """
        try:
            lat = float(request.args.get("lat", 17.5))
            lon = float(request.args.get("lon", 85.5))
            wind_kt = float(request.args.get("wind_kt", 75.0))
            radius_km = float(request.args.get("radius_km", 250.0))

            districts_file = DATA_FOLDER / "india_districts.geojson"
            affected_districts: List[Dict[str, Any]] = []

            if districts_file.exists():
                with open(districts_file, "r", encoding="utf-8") as f:
                    districts_data = json.load(f)

                # Use Shapely if installed for precise polygon intersection
                if SHAPELY_AVAILABLE:
                    # Buffer point in degrees (~111.13 km per degree)
                    buffer_deg = radius_km / 111.13
                    hazard_circle = Point(lon, lat).buffer(buffer_deg)

                    for feat in districts_data.get("features", []):
                        poly = shape(feat["geometry"])
                        props = feat.get("properties", {})
                        d_lat = props.get("lat", poly.centroid.y)
                        d_lon = props.get("lon", poly.centroid.x)
                        dist = _haversine_distance(lat, lon, d_lat, d_lon)

                        if hazard_circle.intersects(poly) or dist <= radius_km:
                            affected_districts.append({
                                "name": props.get("name", "Unknown"),
                                "state": props.get("state", "India"),
                                "distance_km": dist,
                                "population_estimate": props.get("population_estimate", 1500000),
                                "elevation_m": props.get("elevation_m", 5.0),
                            })
                else:
                    # Haversine distance fallback
                    for feat in districts_data.get("features", []):
                        props = feat.get("properties", {})
                        d_lat = props.get("lat", 20.0)
                        d_lon = props.get("lon", 86.0)
                        dist = _haversine_distance(lat, lon, d_lat, d_lon)
                        if dist <= radius_km:
                            affected_districts.append({
                                "name": props.get("name", "Unknown"),
                                "state": props.get("state", "India"),
                                "distance_km": dist,
                                "population_estimate": props.get("population_estimate", 1500000),
                                "elevation_m": props.get("elevation_m", 5.0),
                            })

            # Sort affected districts by proximity
            affected_districts.sort(key=lambda x: x["distance_km"])

            # Alert level determination
            if wind_kt >= 64.0 and any(d["distance_km"] <= 160.0 for d in affected_districts):
                alert_level = "red"
            elif wind_kt >= 34.0 and len(affected_districts) > 0:
                alert_level = "orange"
            else:
                alert_level = "green"

            # Storm surge risk evaluation
            storm_surge_risk = bool(wind_kt >= 55.0 and any(d["distance_km"] <= 180.0 for d in affected_districts))

            # Landfall projection statement
            if affected_districts:
                closest = affected_districts[0]
                hours_to_landfall = max(6, int(round(closest["distance_km"] / 15.0)))
                landfall_estimate = (
                    f"Estimated in {hours_to_landfall}-{hours_to_landfall + 6} hours near "
                    f"{closest['name']} coastline ({closest['state']})"
                )
            else:
                landfall_estimate = "Open ocean track; no immediate coastal landfall threat in 48 hours."

            return jsonify({
                "status": "success",
                "alert_level": alert_level,
                "storm_surge_risk": storm_surge_risk,
                "landfall_estimate": landfall_estimate,
                "affected_districts_count": len(affected_districts),
                "affected_districts": affected_districts,
                "query_parameters": {
                    "storm_lat": lat,
                    "storm_lon": lon,
                    "wind_kt": wind_kt,
                    "radius_km": radius_km,
                },
            }), 200

        except Exception as err:
            logger.error("Alert calculation failed: %s", err, exc_info=True)
            return jsonify({"status": "error", "error": str(err)}), 400


# ==============================================================================
# Route 7: /api/live (Active North Indian Ocean Cyclone Monitoring Feed)
# ==============================================================================

# JTWC / IMD RSS Integration Instructions:
# To connect to live real-time tropical cyclone feeds:
# 1. Parse JTWC RSS feed: https://www.metoc.navy.mil/jtwc/rss/jtwc.rss using feedparser or requests
# 2. Query IMD RSMC New Delhi Cyclone Bulletins at: https://rsmcnewdelhi.imd.gov.in/
# 3. Pull MOSDAC INSAT-3D/3DR TIR GeoTIFFs from: https://mosdac.gov.in/

if FLASK_AVAILABLE and app:
    @app.route("/api/live", methods=["GET"])
    def live_cyclones_endpoint() -> Any:
        """
        GET /api/live
        Returns active monitoring data for North Indian Ocean cyclone systems.
        """
        mock_active_storms = [
            {
                "id": "NIO-2026-01",
                "name": "Severe Cyclone Remal",
                "basin": "Bay of Bengal",
                "lat": 19.4,
                "lon": 87.2,
                "wind_kt": 65.0,
                "wind_speed_kmh": 120,
                "pressure_hpa": 972,
                "category": "Severe Cyclonic Storm",
                "category_color": "#f97316",
                "heading_deg": 325.0,
                "speed_kmh": 16.0,
                "status": "Active / Intensifying",
                "last_updated": datetime.now(timezone.utc).isoformat(),
            },
            {
                "id": "NIO-2026-02",
                "name": "Depression BOB-02",
                "basin": "South Bay of Bengal",
                "lat": 11.8,
                "lon": 86.5,
                "wind_kt": 25.0,
                "wind_speed_kmh": 45,
                "pressure_hpa": 1004,
                "category": "Depression",
                "category_color": "#3b82f6",
                "heading_deg": 310.0,
                "speed_kmh": 12.0,
                "status": "Developing",
                "last_updated": datetime.now(timezone.utc).isoformat(),
            },
            {
                "id": "NIO-2026-03",
                "name": "Deep Depression ARB-01",
                "basin": "Arabian Sea",
                "lat": 14.2,
                "lon": 67.8,
                "wind_kt": 30.0,
                "wind_speed_kmh": 55,
                "pressure_hpa": 998,
                "category": "Deep Depression",
                "category_color": "#06b6d4",
                "heading_deg": 290.0,
                "speed_kmh": 14.0,
                "status": "Tracking West-Northwest",
                "last_updated": datetime.now(timezone.utc).isoformat(),
            },
        ]

        return jsonify({
            "status": "success",
            "source": "NOAA IBTrACS / IMD RSMC New Delhi Feed (Demo)",
            "active_storms_count": len(mock_active_storms),
            "storms": mock_active_storms,
        }), 200


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    if FLASK_AVAILABLE and app:
        port = int(os.environ.get("PORT", 5000))
        print(f"[*] CycloneAI REST API Server running on: http://127.0.0.1:{port}")
        app.run(host="0.0.0.0", port=port, debug=True)
    else:
        print("[!] Flask is not installed. Install requirements via: pip install -r requirements.txt")
