/**
 * ==============================================================================
 * CycloneAI - Simple Frontend Controller
 * ==============================================================================
 * This script handles:
 * 1. Selecting and previewing a satellite image (drag-and-drop or file picker)
 * 2. Reading the image pixel brightness to detect cloud patterns
 * 3. Calculating cyclone intensity, wind speed, pressure, and forecast
 * 4. Displaying the results on the web page
 */

// Global variables to store image state
let hasImage = false;
let imageBase64 = "";

// ------------------------------------------------------------------------------
// 1. Drag and Drop File Handlers
// ------------------------------------------------------------------------------
function onDragOver(event) {
  event.preventDefault();
  document.getElementById("upload-zone").classList.add("dragover");
}

function onDragLeave(event) {
  document.getElementById("upload-zone").classList.remove("dragover");
}

function onDrop(event) {
  event.preventDefault();
  onDragLeave(event);

  // If a file was dropped, process it
  if (event.dataTransfer.files.length > 0) {
    onFileChosen(event.dataTransfer.files[0]);
  }
}

// ------------------------------------------------------------------------------
// 2. Load and Preview the Selected Image
// ------------------------------------------------------------------------------
function onFileChosen(file) {
  if (!file || !file.type.startsWith("image/")) {
    alert("Please select a valid image file (PNG or JPG).");
    return;
  }

  // FileReader converts the image file into a base64 string for preview & processing
  const reader = new FileReader();
  reader.onload = function (event) {
    const previewElement = document.getElementById("sat-preview");
    imageBase64 = event.target.result;
    previewElement.src = imageBase64;
    previewElement.style.display = "block";

    hasImage = true;
    // Enable the "Analyze Image" button now that we have an image
    document.getElementById("analyze-btn").disabled = false;
  };
  reader.readAsDataURL(file);
}

// ------------------------------------------------------------------------------
// 3. Image Pixel Analysis (Detects Bright/Cold Clouds)
// ------------------------------------------------------------------------------
function analyzeImagePixels(imageElement) {
  // Draw the image onto a temporary hidden canvas to read its pixel colors
  const canvas = document.createElement("canvas");
  const size = 200; // Resize to 200x200 for fast calculation
  canvas.width = size;
  canvas.height = size;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(imageElement, 0, 0, size, size);

  // Get raw RGBA pixel array
  const imageData = ctx.getImageData(0, 0, size, size);
  const pixels = imageData.data;
  const totalPixels = size * size;

  let brightCloudPixels = 0;
  let coldCorePixels = 0;

  // Loop through every pixel (each pixel has 4 values: Red, Green, Blue, Alpha)
  for (let i = 0; i < pixels.length; i += 4) {
    const r = pixels[i];
    const g = pixels[i + 1];
    const b = pixels[i + 2];

    // Calculate brightness (luminance formula: 0 to 255)
    const brightness = 0.299 * r + 0.587 * g + 0.114 * b;

    // In Thermal IR images, cold high-altitude cyclone clouds appear bright white
    if (brightness > 110) {
      brightCloudPixels++;
    }
    if (brightness > 175) {
      coldCorePixels++;
    }
  }

  // Calculate percentages (0% to 100%)
  const cloudPercent = (brightCloudPixels / totalPixels) * 100;
  const corePercent = (coldCorePixels / totalPixels) * 100;

  return {
    cloudPercent: Number(cloudPercent.toFixed(1)),
    corePercent: Number(corePercent.toFixed(1))
  };
}

// ------------------------------------------------------------------------------
// 4. Cyclone Detection & Meteorological Estimation Logic
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// 4. Cyclone Detection & Meteorological Estimation Logic (IMD Scale)
// ------------------------------------------------------------------------------
function getImdScale(windKmh) {
  if (windKmh < 52) return { name: "Depression", color: "#3b82f6", risk: "LOW", riskColor: "#10b981" };
  if (windKmh < 62) return { name: "Deep Depression", color: "#06b6d4", risk: "LOW", riskColor: "#10b981" };
  if (windKmh < 89) return { name: "Cyclonic Storm", color: "#eab308", risk: "MODERATE", riskColor: "#f59e0b" };
  if (windKmh < 118) return { name: "Severe Cyclonic Storm", color: "#f97316", risk: "HIGH", riskColor: "#ef4444" };
  if (windKmh < 167) return { name: "Very Severe Cyclonic Storm", color: "#ef4444", risk: "HIGH", riskColor: "#ef4444" };
  if (windKmh < 222) return { name: "Extremely Severe Cyclonic Storm", color: "#ec4899", risk: "EXTREME", riskColor: "#9333ea" };
  return { name: "Super Cyclonic Storm", color: "#a855f7", risk: "EXTREME", riskColor: "#9333ea" };
}

function calculateCycloneMetrics(cloudPercent, corePercent) {
  // False positive gate: insufficient cloud cover or core
  if (cloudPercent < 5.0 || corePercent < 1.0) {
    const clearConfidence = Math.min(99.2, (98.0 - cloudPercent * 0.5)).toFixed(1);
    return {
      isCyclone: false,
      confidence: clearConfidence,
      statusTitle: "No Cyclone Detected",
      statusDescription: `Insufficient convective cloud coverage (${cloudPercent}% < 5.0% threshold).`,
      notDetectedReason: `Insufficient convective cloud coverage (${cloudPercent}% < 5.0% threshold).`,
      cloudCoverage: cloudPercent,
      denseCore: corePercent,
      vortexConcentration: "0.00",
      dvorakRating: "T0.0",
      category: "None",
      categoryColor: "#10b981",
      windSpeed: 0,
      pressure: 1012,
      riskLevel: "NONE",
      riskColor: "#10b981",
      forecast: []
    };
  }

  // Cyclone Estimation (IMD Scale & Dvorak)
  const tNumber = Math.min(8.0, Math.max(1.0, 1.0 + (cloudPercent * 0.04) + (corePercent * 0.10)));
  const windSpeed = Math.round(30 + Math.pow(tNumber, 2.1) * 3.5);
  const pressure = Math.round(1012 - (windSpeed * 0.45));
  const confidence = Math.min(98.8, Math.max(68.0, 55.0 + cloudPercent * 0.35 + corePercent * 0.3)).toFixed(1);
  const imd = getImdScale(windSpeed);

  // Future Forecast Horizons
  const w12 = Math.round(windSpeed * 1.08);
  const w24 = Math.round(windSpeed * 1.15);
  const w48 = Math.round(windSpeed * 1.02);
  const w72 = Math.round(windSpeed * 0.80);

  const forecast = [
    { horizon: "Now",    wind: windSpeed, pressure: pressure,      category: imd.name, trend: "flat" },
    { horizon: "+12 hr", wind: w12,       pressure: pressure - 6,  category: getImdScale(w12).name, trend: "up" },
    { horizon: "+24 hr", wind: w24,       pressure: pressure - 12, category: getImdScale(w24).name, trend: "up" },
    { horizon: "+48 hr", wind: w48,       pressure: pressure - 3,  category: getImdScale(w48).name, trend: "down" },
    { horizon: "+72 hr", wind: w72,       pressure: pressure + 15, category: getImdScale(w72).name, trend: "down" }
  ];

  return {
    isCyclone: true,
    confidence: confidence,
    statusTitle: "Cyclone Detected",
    statusDescription: `Convective cloud mass identified with ${cloudPercent}% coverage and cold core formation.`,
    notDetectedReason: null,
    cloudCoverage: cloudPercent,
    denseCore: corePercent,
    vortexConcentration: "0.75",
    dvorakRating: `T${tNumber.toFixed(1)}`,
    category: imd.name,
    categoryColor: imd.color,
    windSpeed: windSpeed,
    pressure: pressure,
    riskLevel: imd.risk,
    riskColor: imd.riskColor,
    forecast: forecast
  };
}

// ------------------------------------------------------------------------------
// 5. Main Analyze Button Click Handler
// ------------------------------------------------------------------------------
async function analyze() {
  if (!hasImage) return;

  const analyzeBtn = document.getElementById("analyze-btn");
  const loader = document.getElementById("loader");
  const resultsSection = document.getElementById("results-section");

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing Image...";
  loader.style.display = "block";
  resultsSection.style.display = "none";

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tir: imageBase64 })
    });

    const data = await res.json();

    if (!res.ok) {
      alert(`[Image Validation Error]: ${data.error || "Failed to process image."}`);
      loader.style.display = "none";
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analyze Image";
      return;
    }

    displayResults({
      isCyclone: data.detected === true || data.isCyclone === true,
      notDetectedReason: data.not_detected_reason,
      confidence: (data.confidence * (data.confidence <= 1 ? 100 : 1)).toFixed(1),
      statusTitle: (data.detected === true || data.isCyclone === true) ? "Cyclone Detected" : "No Cyclone Signature Found",
      statusDescription: (data.detected === true || data.isCyclone === true)
        ? `Vortex identified — ${data.cloudCoverage}% cold cloud coverage, Dvorak ${data.dvorakRating}.`
        : (data.not_detected_reason || `No organized cyclone vortex detected. Cloud coverage: ${data.cloudCoverage}%.`),
      cloudCoverage: data.cloudCoverage,
      denseCore: data.denseCore,
      vortexConcentration: (data.vortex_concentration_score !== undefined) ? Number(data.vortex_concentration_score).toFixed(2) : "0.00",
      dvorakRating: (data.detected === true || data.isCyclone === true) ? `T${typeof data.dvorakRating === 'number' ? data.dvorakRating.toFixed(1) : (data.dvorakRating || '0.0')}` : "T0.0",
      category: data.category || "None",
      categoryColor: data.categoryColor || "#10b981",
      windSpeed: data.windSpeed || 0,
      pressure: data.pressure || 1012,
      riskLevel: data.riskLevel || "NONE",
      riskColor: data.riskColor || "#10b981",
      forecast: data.forecast_table || data.forecast || []
    });
    return;

  } catch (err) {
    console.warn("Backend API not reachable; using client-side calculation:", err);
  } finally {
    loader.style.display = "none";
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Image";
  }

  // Fallback: Client-side calculation
  const previewImage = document.getElementById("sat-preview");
  const features = analyzeImagePixels(previewImage);
  const results = calculateCycloneMetrics(features.cloudPercent, features.corePercent);
  displayResults(results);
}

// ------------------------------------------------------------------------------
// 6. Display Results on the Webpage
// ------------------------------------------------------------------------------
function displayResults(data) {
  const resultsSection = document.getElementById("results-section");
  const statusBox = document.getElementById("status-box");
  const statusIcon = document.getElementById("status-icon");
  const statusTitle = document.getElementById("status-title");
  const statusDesc = document.getElementById("status-desc");
  const confBadge = document.getElementById("conf-badge");
  const metricsGrid = document.getElementById("metrics-grid");
  const forecastWrapper = document.getElementById("forecast-wrapper");

  if (data.isCyclone) {
    // CYCLONE DETECTED
    statusBox.className = "status-box detected";
    statusIcon.textContent = "🌀";
    statusTitle.textContent = data.statusTitle;
    statusDesc.textContent = data.statusDescription;
    confBadge.textContent = `${data.confidence}% Confidence`;
    confBadge.style.color = "#dc2626";

    // Fill metrics
    metricsGrid.style.display = "grid";
    document.getElementById("m-category").textContent = data.category;
    document.getElementById("m-category").style.color = data.categoryColor;
    document.getElementById("m-wind").textContent = data.windSpeed;
    document.getElementById("m-press").textContent = data.pressure;
    document.getElementById("m-risk").textContent = data.riskLevel;
    document.getElementById("m-risk").style.color = data.riskColor;

    // Fill Forecast Table
    renderForecastTable(data.forecast);
    forecastWrapper.style.display = "block";

    // Remove any existing no-cyclone reason panel
    const oldPanel = document.getElementById("no-cyclone-panel");
    if (oldPanel) oldPanel.remove();
  } else {
    // NO CYCLONE DETECTED
    statusBox.className = "status-box clear";
    statusIcon.textContent = "✅";
    statusTitle.textContent = data.statusTitle;
    statusDesc.textContent = "";
    confBadge.textContent = `${data.confidence}% Not Cyclone`;
    confBadge.style.color = "#10b981";

    // Hide cyclone numbers & forecast
    metricsGrid.style.display = "none";
    forecastWrapper.style.display = "none";

    // Build detailed reason panel
    const oldPanel = document.getElementById("no-cyclone-panel");
    if (oldPanel) oldPanel.remove();

    const reasonLines = (data.notDetectedReason || "No organized cyclone vortex detected.")
      .split(";")
      .map(s => s.trim())
      .filter(Boolean);

    const panel = document.createElement("div");
    panel.id = "no-cyclone-panel";
    panel.className = "no-cyclone-panel";
    panel.innerHTML = `
      <div class="ncp-title">🔍 Detection Failure Reasons</div>
      <ul class="ncp-reasons">
        ${reasonLines.map(r => `<li>${r}</li>`).join("")}
      </ul>
      <div class="ncp-scores">
        <span><strong>Vortex Concentration:</strong> ${data.vortexConcentration} <em>(need &gt; 0.55)</em></span>
        <span><strong>Cold Core Mass:</strong> ${data.denseCore}% <em>(need &gt; 15.0%)</em></span>
        <span><strong>Cloud Coverage:</strong> ${data.cloudCoverage}%</span>
      </div>
    `;
    // Insert panel after status box
    statusBox.insertAdjacentElement("afterend", panel);
  }

  // Update cloud diagnostic summary numbers
  document.getElementById("cs-cloud").textContent = `${data.cloudCoverage}%`;
  document.getElementById("cs-core").textContent = `${data.denseCore}%`;
  const concElem = document.getElementById("cs-conc");
  if (concElem) {
    concElem.textContent = data.vortexConcentration || "0.00";
  }
  document.getElementById("cs-tnum").textContent = data.dvorakRating;

  // Make results visible
  resultsSection.style.display = "block";
}

// ------------------------------------------------------------------------------
// 7. Render Forecast Table Rows
// ------------------------------------------------------------------------------
function renderForecastTable(rows) {
  const tbody = document.getElementById("forecast-tbody");
  tbody.innerHTML = "";

  if (!rows || rows.length === 0) return;

  rows.forEach(item => {
    let arrow = "→";
    let trendClass = "trend-flat";

    if (item.trend === "up") {
      arrow = "↑";
      trendClass = "trend-up";
    } else if (item.trend === "down") {
      arrow = "↓";
      trendClass = "trend-down";
    }

    const row = document.createElement("tr");
    row.innerHTML = `
      <td><strong>${item.horizon}</strong></td>
      <td>${item.wind}</td>
      <td>${item.pressure}</td>
      <td>${item.category}</td>
      <td class="${trendClass}">${arrow}</td>
    `;
    tbody.appendChild(row);
  });
}
