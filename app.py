"""
==============================================================================
Main Entry Point (app.py)
==============================================================================
To start the CycloneAI project, open terminal and run:
    python app.py

Then open your browser at:
    http://127.0.0.1:5000
"""

from backend.app import app

if __name__ == "__main__":
    print("[*] Starting CycloneAI Application...")
    print("[*] Open your browser at: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
