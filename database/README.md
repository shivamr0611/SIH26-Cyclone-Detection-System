# CycloneAI Training & Satellite Database

This directory houses raw satellite imagery, labeled IMD category datasets, augmented samples, and IBTrACS NIO cyclone tracks.

---

## Directory Structure

```text
database/
├── raw/                                  # Raw satellite imagery
│   ├── insat3d_tir/                      # INSAT-3D Thermal Infrared (TIR-1 10.8µm / TIR-2 12.0µm)
│   ├── insat3d_wv/                       # INSAT-3D Water Vapor channel (6.7–7.1µm)
│   ├── insat3d_vis/                      # INSAT-3D Visible channel (0.65µm)
│   └── goes_r_abi/                       # NOAA GOES-R Advanced Baseline Imager data
├── labeled/                              # IMD-categorized cyclone images
│   ├── Depression/                       # Wind < 28 knots (WMO T1.5)
│   ├── Deep_Depression/                  # Wind 28–33 knots (WMO T2.0)
│   ├── Cyclonic_Storm/                   # Wind 34–47 knots (WMO T2.5–T3.0)
│   ├── Severe_Cyclonic_Storm/            # Wind 48–63 knots (WMO T3.5)
│   └── Very_Severe_Cyclonic_Storm/       # Wind >= 64 knots (WMO T4.0+)
├── augmented/                            # Augmented dataset (flip, rotations, brightness, noise)
├── ibtracs/                              # IBTrACS North Indian Ocean (NIO) best-track data
├── metadata/                             # Metadata mapping (image_labels.json)
└── scripts/                              # Dataset ingestion & processing pipelines
    ├── download_ibtracs.py               # Auto-downloads official NOAA IBTrACS NIO dataset
    ├── label_from_ibtracs.py             # Parses IBTrACS CSV & generates image_labels.json
    ├── augment_dataset.py                # Performs 5x OpenCV/NumPy augmentations
    └── scrape_gee_osint.py               # MODIS/Terra GEE thermal image collector
```

---

## How to Populate the Database

### 1. Download IBTrACS NIO Historical Tracks

Download the official North Indian Ocean (NIO) cyclone archive from NOAA:
- **Automatic Download**:
  ```bash
  python database/scripts/download_ibtracs.py
  ```
- **Manual Download**: [NOAA IBTrACS NIO CSV Archive](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.NI.list.v04r00.csv) → Save to `database/ibtracs/ibtracs_NIO.csv`

---

### 2. Auto-Generate IMD Labels & Metadata

Run `label_from_ibtracs.py` to parse tracks, assign IMD classifications, and output `database/metadata/image_labels.json`:

```bash
# Preview parsing without writing files
python database/scripts/label_from_ibtracs.py --dry-run

# Generate metadata/image_labels.json
python database/scripts/label_from_ibtracs.py
```

#### IMD Classification Rules:
| IMD Category | Wind Speed (kt) | Wind Speed (km/h) |
|---|---|---|
| **Depression** | < 28 kt | < 52 km/h |
| **Deep_Depression** | 28–33 kt | 52–61 km/h |
| **Cyclonic_Storm** | 34–47 kt | 62–88 km/h |
| **Severe_Cyclonic_Storm** | 48–63 kt | 89–117 km/h |
| **Very_Severe_Cyclonic_Storm** | >= 64 kt | >= 118 km/h |

---

### 3. Fetch Satellite Imagery from Google Earth Engine (GEE)

#### Step 3.1: Authenticate Google Earth Engine
```bash
# One-time login to GEE
earthengine authenticate
```

#### Step 3.2: Scrape MODIS/Terra Thermal Imagery
Pulls MODIS Band 31 (10.8µm proxy) clipped to the Bay of Bengal (`lon 60–100, lat 5–25`):
```bash
python database/scripts/scrape_gee_osint.py \
  --storm-name "Cyclone_Fani" \
  --start-date 2019-05-01 \
  --end-date 2019-05-04 \
  --band LST_Day_1km \
  --scale 2000
```
Images will be saved to `database/raw/insat3d_tir/`.

---

### 4. Run Dataset Augmentation

Apply OpenCV/NumPy transformations to expand the labeled dataset:
- Horizontal Flip (`_flip`)
- +15° & -15° Rotation (`_rot15`, `_rot-15`)
- Brightness adjustment (`_bright`)
- Gaussian noise (`_noise`)

```bash
# Preview augmentation counts
python database/scripts/augment_dataset.py --dry-run

# Generate augmented dataset in database/augmented/
python database/scripts/augment_dataset.py
```
