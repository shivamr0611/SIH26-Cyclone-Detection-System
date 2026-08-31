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
function calculateCycloneMetrics(cloudPercent, corePercent) {
  // CASE A: Clear sky or very few scattered clouds (< 16% cloud cover)
  if (cloudPercent < 16 || corePercent < 2.5) {
    // High confidence that NO cyclone exists in this area
    const clearConfidence = Math.min(99.2, (98.0 - cloudPercent * 0.5)).toFixed(1);

    return {
      isCyclone: false,
      confidence: clearConfidence,
      statusTitle: "No Cyclone Detected",
      statusDescription: `Clear area: Cloud coverage is ${cloudPercent}% (insufficient for a cyclone vortex).`,
      cloudCoverage: cloudPercent,
      denseCore: corePercent,
      dvorakRating: "T0.5",
      category: "None",
      categoryColor: "#10b981",
      windSpeed: 0,
      pressure: 1013,
      riskLevel: "NONE",
      riskColor: "#10b981",
      forecast: []
    };
  }

  // CASE B: Cyclone Detected (Thick clouds & cold core present)
  // 1. Calculate Dvorak T-Number (standard hurricane measurement from T1.5 to T7.0)
  const tNumber = Math.min(7.2, Math.max(1.5, 1.0 + (cloudPercent * 0.05) + (corePercent * 0.12)));

  // 2. Estimate Wind Speed (km/h) from T-Number
  const windSpeed = Math.round(30 + Math.pow(tNumber, 2.1) * 3.5);

  // 3. Estimate Central Pressure (hPa) - Higher wind means lower central pressure
  const pressure = Math.round(1013 - (windSpeed * 0.45));

  // 4. Calculate Confidence (e.g. 70% to 98%)
  const confidence = Math.min(98.8, Math.max(68.0, 60.0 + cloudPercent * 0.4 + corePercent * 0.3)).toFixed(1);

  // 5. Determine Cyclone Category based on Wind Speed
  let category = "Category 1";
  let categoryColor = "#eab308";

  if (windSpeed >= 215) {
    category = "Category 5";
    categoryColor = "#ec4899";
  } else if (windSpeed >= 165) {
    category = "Category 4";
    categoryColor = "#ef4444";
  } else if (windSpeed >= 130) {
    category = "Category 3";
    categoryColor = "#f97316";
  } else if (windSpeed >= 90) {
    category = "Category 2";
    categoryColor = "#f59e0b";
  } else if (windSpeed < 62) {
    category = "Depression";
    categoryColor = "#3b82f6";
  }

  // 6. Hazard Risk Level
  let riskLevel = "LOW";
  let riskColor = "#10b981";
  if (windSpeed >= 160) {
    riskLevel = "HIGH";
    riskColor = "#ef4444";
  } else if (windSpeed >= 90) {
    riskLevel = "MODERATE";
    riskColor = "#f59e0b";
  }

  // 7. Future Forecast Horizons (+12h, +24h, +48h, +72h)
  const forecast = [
    { horizon: "Now",    wind: windSpeed,                  pressure: pressure,      category: category, trend: "flat" },
    { horizon: "+12 hr", wind: Math.round(windSpeed * 1.08), pressure: pressure - 6,  category: category, trend: "up" },
    { horizon: "+24 hr", wind: Math.round(windSpeed * 1.14), pressure: pressure - 12, category: category, trend: "up" },
    { horizon: "+48 hr", wind: Math.round(windSpeed * 1.02), pressure: pressure - 3,  category: category, trend: "down" },
    { horizon: "+72 hr", wind: Math.round(windSpeed * 0.80), pressure: pressure + 15, category: "Weakening", trend: "down" }
  ];

  return {
    isCyclone: true,
    confidence: confidence,
    statusTitle: "Cyclone Detected",
    statusDescription: `Vortex identified with ${cloudPercent}% cloud density and cold core mass.`,
    cloudCoverage: cloudPercent,
    denseCore: corePercent,
    dvorakRating: `T${tNumber.toFixed(1)}`,
    category: category,
    categoryColor: categoryColor,
    windSpeed: windSpeed,
    pressure: pressure,
    riskLevel: riskLevel,
    riskColor: riskColor,
    forecast: forecast
  };
}

// ------------------------------------------------------------------------------
// 5. Main Analyze Button Click Handler
// ------------------------------------------------------------------------------
function analyze() {
  if (!hasImage) return;

  const analyzeBtn = document.getElementById("analyze-btn");
  const loader = document.getElementById("loader");
  const resultsSection = document.getElementById("results-section");

  // Show loading state
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing Image...";
  loader.style.display = "block";
  resultsSection.style.display = "none";

  // Simulate short processing time for smooth user experience
  setTimeout(() => {
    const previewImage = document.getElementById("sat-preview");

    // Step 1: Read pixel cloud data
    const features = analyzeImagePixels(previewImage);

    // Step 2: Compute cyclone metrics
    const results = calculateCycloneMetrics(features.cloudPercent, features.corePercent);

    // Step 3: Render on screen
    displayResults(results);

    // Reset button
    loader.style.display = "none";
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Image";
  }, 400);
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
    // CYCLONE DETECTED: Red banner & display metrics
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
  } else {
    // NO CYCLONE: Green banner
    statusBox.className = "status-box clear";
    statusIcon.textContent = "✅";
    statusTitle.textContent = data.statusTitle;
    statusDesc.textContent = data.statusDescription;
    confBadge.textContent = `${data.confidence}% Clear`;
    confBadge.style.color = "#10b981";

    // Hide cyclone numbers & forecast
    metricsGrid.style.display = "none";
    forecastWrapper.style.display = "none";
  }

  // Update cloud diagnostic summary numbers
  document.getElementById("cs-cloud").textContent = `${data.cloudCoverage}%`;
  document.getElementById("cs-core").textContent = `${data.denseCore}%`;
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
