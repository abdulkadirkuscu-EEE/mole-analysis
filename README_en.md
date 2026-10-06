# Skin Lesion (Mole) Analysis System

An image processing system that analyzes photographs of skin lesions (moles)
taken on top of a calibration card. By comparing two images of the same mole
taken at different times, it automatically detects **color and size changes**.

## Overview

- **8-step processing flow**: image processing + color calibration + scale and
  size analysis
- **Calibration card**: 16 color patches + a lesion area in the center
- **CIEDE2000** color-difference metric (closest to human-perceived difference)
- RGB channel calibration via **least squares**
- Scale estimation from **inter-patch distances** (works even if the
  camera-to-card distance changes between captures)
- Both an **HTTP API** (FastAPI) and a **mobile-friendly web UI** (app.html)
- **Two languages**: Turkish and English source files

## Processing Steps

| Step | Description | Output |
|------|-------------|--------|
| 1 | BW (black-and-white) image generation | `bw.png` |
| 2 | Label bounding-box detection | `output.jpg`, `crop.jpg` |
| 3 | Perspective correction | `perspective.jpg`, `color_square.jpg` |
| 4 | Patch color drift detection | (numerical) |
| 5 | RGB channel calibration (least squares) | (a, b, R² coefficients) |
| 6 | Mole color change (ΔE2000) | (color difference and threshold) |
| 7 | Scale factor estimation | `step7_scale.jpg` |
| 8 | Mole size change | `step8_size.jpg`, `comparison.jpg` |

## Folder Layout

```
Cilt_Lezyonu_Analizi/
├── label_pipeline.py       Steps 1-3 functions (Turkish)
├── label_pipeline_en.py    Steps 1-3 functions (English)
├── api_server.py           Steps 4-8 + REST API (Turkish)
├── api_server_en.py        Steps 4-8 + REST API (English)
├── app.html                Mobile-friendly web UI
├── requirements.txt        Python dependencies
├── README.md               Turkish documentation
├── README_en.md            This file
└── ornek_goruntuler/       Sample input and output images
```

## Installation

```bash
# 1) Python 3.9+ is required (3.10 or 3.11 recommended)
python --version

# 2) Install dependencies
pip install -r requirements.txt
```

## Usage

### Method 1: API Server (Recommended)

```bash
python api_server_en.py
```

Then in the browser:
- **Web app**: <http://localhost:8000/app>
- **API docs** (auto-generated Swagger UI): <http://localhost:8000/docs>

In the web app:
1. Enter a mole name on the phone (e.g. `left_arm_1`)
2. Use the "PHOTO" button to capture the mole on the card
3. As you add new captures, you can compare them to older captures

### Method 2: Folder Watch Mode (Steps 1-3 only)

```bash
python label_pipeline_en.py
```

This mode watches the `input_images/` folder. Every image dropped there gets
Steps 1-3 applied automatically, with outputs written to the matching folders.

### Running the Turkish Version

```bash
python api_server.py          # API server with Turkish comments
python label_pipeline.py      # Folder watcher with Turkish comments
```

> **NOTE**: Do not run the English and Turkish servers at the same time —
> they both bind to port 8000. Pick one to run and read the other for
> reference.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/` | Health check |
| GET    | `/app` | Serve the web app |
| GET    | `/moles` | List all moles |
| GET    | `/moles/{name}/captures` | List captures for a mole |
| POST   | `/analyze` | Process a new photo (Steps 1-3) |
| POST   | `/compare` | Compare two captures (Steps 4-8) |
| PUT    | `/moles/{name}` | Rename a mole |
| DELETE | `/moles/{name}` | Delete a mole |
| DELETE | `/moles/{name}/{timestamp}` | Delete a capture |
| GET    | `/image/{name}/{timestamp}/{file}` | Fetch a step image |

## Technical Details

### Calibration Card

The card carries 16 saturated color patches arranged in a U-shape:
- Top row: 4 horizontal patches
- Bottom row: 4 horizontal patches
- Left column: 4 vertical patches
- Right column: 4 vertical patches
- Lesion area in the middle

Each patch is 0.075 × 0.075 in normalized coordinates. Patch centers were
measured by hand from the card template (the `PATCHES` constant in
`label_pipeline_en.py`).

### Patch Detection: Band + Projection Method

1. Find 4 color-patch bands in the saturation (S) channel.
2. Take an intensity (V) projection inside each band.
3. Use valley-peak analysis to split each band into 4 patches.
4. Result: 16 patches total.

This method is **more robust than fixed coordinates** because patch positions
can shift slightly after perspective correction.

### Color Comparison: CIEDE2000

CIE's most refined color-difference formula and the closest to the human
perceptual response.

| ΔE2000 | Meaning |
|--------|---------|
| < 1.0 | Imperceptible to the human eye |
| 1-2 | Noticeable on close inspection |
| 2-10 | Noticeable at a glance |
| > 10 | Clearly different colors |

Color-change threshold: **max(mean_residual + 3σ, 6.0)** — no "changed"
verdict is issued unless there is at least a 6.0 ΔE2000 difference.

### Scale Factor and Size Thresholds

Camera-to-card distance may vary between captures. We solve this with:

```
scale = median(test_patch_distances / reference_patch_distances)
```

Size-change thresholds (Otsu segmentation jitters a few pixels for small
moles, so thresholds stay conservative):
- **Diameter threshold**: minimum ±18%
- **Area threshold**: minimum ±35%

Decision: both diameter and area must exceed their thresholds to qualify
as "size changed".

## Data Storage

```
MOLES/                        (English server) or BENLER/ (Turkish server)
├── left_arm_1/               Mole name
│   ├── 2025-11-01_10-30-00/  Capture timestamp
│   │   ├── input.jpg         Original photo
│   │   ├── bw.png            Step 1 output
│   │   ├── output.jpg        Step 2 output
│   │   ├── crop.jpg          Tightly cropped
│   │   ├── perspective.jpg   Step 3 output
│   │   ├── color_square.jpg  Patch + mole visualization
│   │   ├── data.json         Patch RGB + distance data
│   │   ├── step7_scale.jpg   Post-comparison (Step 7)
│   │   ├── step8_size.jpg    Post-comparison (Step 8)
│   │   ├── comparison.jpg    3-panel comparison visualization
│   │   └── comparison_data.json  Comparison results
│   └── 2025-11-15_14-22-00/  Next capture
└── right_cheek/              A second mole
```

## Sample Images

Please add **sample input photos and processing outputs** to the
`ornek_goruntuler/` folder. The folder is currently empty — before submitting
to the instructor, place your photos taken on the template card and their
processed outputs there.

## Implementation Notes

- **Consistent outputs**: Every capture is saved with the same folder layout,
  so comparisons can fetch the matching files directly.
- **data.json cache**: Patch RGBs and distances are written to data.json
  after each capture. Comparisons read from there first; they fall back to
  dynamic detection only if necessary.
- **Fallbacks**: If dynamic patch detection fails, fixed PATCHES coordinates
  are used. If no BW quadrilateral is found, a dilated mask is tried.
- **Mole detection**: A ±13% ROI around the image center is searched, and
  blobs are scored by circularity + aspect ratio + area.

## Author

Skin Lesion Analysis System - Digital Image Processing Project
