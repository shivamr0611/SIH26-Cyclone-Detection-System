"""
==============================================================================
Backend API Server (backend/app.py)
==============================================================================
Simple Flask web server that:
1. Serves the frontend web dashboard (dashboard/index.html)
2. Provides a single REST API endpoint: POST /api/predict
   - Receives the uploaded image
   - Runs preprocessor -> detector -> classifier -> intensity predictor
   - Returns the JSON result back to the frontend
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

# Add project root to python search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.preprocessor import SatellitePreprocessor
from detection.detector import CycloneDetector
from classification.classifier import CycloneClassifier

# Initialize Flask app to serve the frontend folder
DASHBOARD_FOLDER = PROJECT_ROOT / "dashboard"
app = Flask(__name__, static_folder=str(DASHBOARD_FOLDER), static_url_path="")

# Initialize pipeline modules
preprocessor = SatellitePreprocessor()
detector = CycloneDetector()
classifier = CycloneClassifier()


@app.route("/")
def serve_dashboard():
    """Serves the dashboard HTML page when user opens http://127.0.0.1:5000/"""
    return send_from_directory(str(DASHBOARD_FOLDER), "index.html")


@app.route("/api/predict", methods=["POST"])
def predict_cyclone():
    """
    API endpoint: receives base64 image and returns analysis results.
    """
    try:
        data = request.get_json() or {}
        image_base64 = data.get("image", "")

        # 1. Preprocess: Extract cloud coverage & cold core percentages
        prep_data = preprocessor.preprocess_image_b64(image_base64)

        # 2. Detect: Determine if cyclone exists and calculate confidence
        det_data = detector.detect(prep_data)

        # 3. Classify: Compute category and risk
        class_data = classifier.classify(det_data)

        return jsonify({
            "status": "success",
            "isCyclone": det_data["detected"],
            "confidence": det_data["confidence"],
            "statusTitle": det_data["status_text"],
            "statusDescription": det_data["message"],
            "cloudCoverage": prep_data["cloud_coverage_percent"],
            "category": class_data["category"],
            "categoryColor": class_data["category_color"],
            "riskLevel": class_data["risk_level"],
            "riskColor": class_data["risk_color"]
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] CycloneAI Server running on: http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
