# 🌀 CycloneAI - Tropical Cyclone Detection & Prediction System

A simple AI-based web application to detect and estimate the intensity of tropical cyclones from Thermal Infrared (TIR) satellite imagery.

---

## 📂 Project Structure

```text
SIH 2026/
│
├── dataset/                     # Sample datasets (SRS Structure)
│   ├── satellite/               # Satellite image documentation
│   ├── labels/                  # Sample storm labels (JSON)
│   └── weather/                 # Sample environmental weather data (CSV)
│
├── preprocessing/               # Step 1: Preprocesses images & measures cloud density
│   ├── __init__.py
│   └── preprocessor.py
│
├── detection/                   # Step 2: Binary cyclone detection (Cyclone vs No Cyclone)
│   ├── __init__.py
│   └── detector.py
│
├── classification/              # Step 3: Classifies category, wind speed & pressure
│   ├── __init__.py
│   └── classifier.py
│
├── prediction/                  # Step 4: Forecasts future intensity (+12h to +72h)
│   ├── __init__.py
│   └── intensity_predictor.py
│
├── models/                      # Model registry & benchmark metrics
│   ├── __init__.py
│   └── model_loader.py
│
├── backend/                     # Backend Flask API Server
│   ├── __init__.py
│   └── app.py
│
├── dashboard/                   # Frontend Web Application (HTML / CSS / JS)
│   ├── index.html               # Main dashboard webpage
│   ├── css/
│   │   └── style.css            # Clean light-theme styles
│   └── js/
│       └── app.js               # Image processing & calculation logic
│
├── app.py                       # Main application runner
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 How to Run on Your Machine

### Option 1: Run with Python Server (Recommended)

1. **Install Requirements** (Only needed once):
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Application**:
   ```bash
   python app.py
   ```

3. **Open in Browser**:
   Visit: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

### Option 2: Run Standalone (No Terminal Needed)

Double-click on **`dashboard/index.html`** or open it directly in any web browser (Chrome, Edge, Firefox).

---

## 💡 How It Works

1. **Ingest Satellite Image**:
   - Drag & drop or browse a Thermal IR satellite image.
   - You can get live satellite images using the shortcut buttons for **IMD Satellite** or **Zoom Earth**.

2. **Analyze**:
   - Click **"Analyze Image"**.
   - The app reads pixel brightness to determine **Cloud Coverage %** and **Cold Core Density %**.

3. **View Results**:
   - **No Cyclone**: When cloud cover is low ($< 16\%$), it confirms calm/clear skies with high confidence.
   - **Cyclone Detected**: When a vortex is present, it displays:
     - **Confidence %**
     - **Cyclone Category** (Depression, Cat 1 to Cat 5)
     - **Hazard Risk Level** (Low, Moderate, High)
     - **Cloud Coverage %**
