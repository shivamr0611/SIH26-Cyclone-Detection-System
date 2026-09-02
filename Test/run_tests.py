#!/usr/bin/env python3
"""
run_tests.py - Test runner script for CycloneAI Test/ directory images.
Evaluates every image against POST /api/analyze and outputs a formatted summary table.
"""

import base64
import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    import urllib.request
    import urllib.error

TEST_DIR = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:5000/api/analyze"


def main():
    results = []
    image_files = sorted([
        p for p in TEST_DIR.glob("*.*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    if not image_files:
        print(f"No test images found in {TEST_DIR}")
        return

    print(f"Running automated test suite across {len(image_files)} images in {TEST_DIR}...\n")

    for img_path in image_files:
        with open(img_path, "rb") as f:
            file_bytes = f.read()

        b64_data = base64.b64encode(file_bytes).decode("utf-8")
        
        try:
            # Use requests if available, fallback to urllib
            if "requests" in sys.modules:
                resp = requests.post(API_URL, json={"tir": b64_data}, timeout=30)
                status_code = resp.status_code
                try:
                    result = resp.json()
                except Exception:
                    result = {"error": resp.text}
            else:
                req = urllib.request.Request(
                    API_URL,
                    data=json.dumps({"tir": b64_data}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(req, timeout=30) as r:
                        status_code = r.status
                        result = json.loads(r.read().decode("utf-8"))
                except urllib.error.HTTPError as e:
                    status_code = e.code
                    try:
                        result = json.loads(e.read().decode("utf-8"))
                    except Exception:
                        result = {"error": str(e)}
        except Exception as e:
            status_code = 500
            result = {"error": str(e)}

        detected = result.get("cyclone_detected", result.get("detected", result.get("isCyclone", False)))
        conf = result.get("confidence", 0)
        if isinstance(conf, float) and conf <= 1.0:
            conf_str = f"{conf * 100:.1f}%"
        else:
            conf_str = f"{conf}%"

        dvorak_t = result.get("dvorakRating", result.get("dvorak", {}).get("t_number", "N/A"))
        category = result.get("category", result.get("dvorak", {}).get("category", "N/A"))
        reason = result.get("not_detected_reason") or result.get("error") or "Cyclone Signature Confirmed"

        results.append({
            "file": img_path.name,
            "status": status_code,
            "detected": detected,
            "confidence": conf_str,
            "dvorak": str(dvorak_t),
            "category": str(category),
            "reason": str(reason),
        })

    # Print summary table
    print(f"{'File':<34} {'Detected':<10} {'Conf%':<8} {'Dvorak':<8} {'Category':<28} {'Notes'}")
    print("-" * 125)
    for r in results:
        print(f"{r['file']:<34} {str(r['detected']):<10} {r['confidence']:<8} {r['dvorak']:<8} {r['category']:<28} {r['reason']}")

    # Save to JSON
    output_json = TEST_DIR / "test_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_json}")


if __name__ == "__main__":
    main()
