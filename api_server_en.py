"""
============================================================================
SKIN LESION (MOLE) ANALYSIS - FastAPI BACKEND
============================================================================
This server exposes the image processing functions in label_pipeline_en.py as
HTTP API endpoints. When a photo is uploaded, Steps 1-3 run and the results
are saved to disk. To compare two captures, Steps 4-8 are executed.

PROCESSING STEPS:
  Step 1: BW (black-and-white) image generation       } from
  Step 2: Label bounding box detection                } label_pipeline_en.py
  Step 3: Perspective correction                      } functions
  Step 4: Color drift detection                        \\
  Step 5: RGB channel calibration (least squares)       \\ run in
  Step 6: Mole color change detection (CIEDE2000)      // this file
  Step 7: Scale factor estimation                      //
  Step 8: Mole size change detection                   //

API ENDPOINTS:
  GET  /                          - Health check
  GET  /app                       - Serve web app (app.html)
  GET  /moles                     - List all moles
  GET  /moles/{name}/captures     - List captures for a mole
  POST /analyze                   - Process new photo (Steps 1-3)
  POST /compare                   - Compare two captures (Steps 4-8)
  PUT  /moles/{name}              - Rename a mole
  DELETE /moles/{name}            - Delete a mole
  DELETE /moles/{name}/{ts}       - Delete a capture
  GET  /image/{name}/{ts}/{file}  - Fetch a step image

HOW TO RUN:
  python api_server_en.py
  Then in the browser: http://localhost:8000/app
============================================================================
"""

import cv2
import numpy as np
import json
import shutil
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import image processing functions from our internal code module
from label_pipeline_en import (
    generate_bw, get_bbox_from_bw, perspective_correct,
    find_label_frame, detect_patches_dynamic, detect_mole,
    detect_color_squares, PATCHES, PATCH_SIZE,
    rgb_to_lab, delta_e_ciede2000
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Root folder for all user data
# Layout: MOLES/{mole_name}/{timestamp}/file.jpg
BASE_DIR = Path("MOLES")
BASE_DIR.mkdir(exist_ok=True)

# Create the FastAPI app
app = FastAPI(title="Mole Analysis API", version="1.0")

# CORS: allow the HTML app to call us from a different port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_pipeline_step1_3(input_image_path: Path, save_directory: Path):
    """
    Run Steps 1-3 in order and produce every output file:
    BW, output (outside-bbox whitened), cropped, perspective-corrected, and
    the patch+mole visualization.

    Args:
        input_image_path: Path to the uploaded original photo.
        save_directory:   Capture folder where all outputs are written.

    Returns:
        (result_dict, error_message): on success, error_message is None.
    """
    original_image = cv2.imread(str(input_image_path))
    if original_image is None:
        return None, "Could not read image"

    original_height, original_width = original_image.shape[:2]

    # --- STEP 1: BW GENERATION ---
    bw_image_path = save_directory / "bw.png"
    if not generate_bw(input_image_path, bw_image_path):
        return None, "Could not generate BW"

    # Re-read the BW image
    bw_image = cv2.imread(str(bw_image_path), cv2.IMREAD_GRAYSCALE)
    if bw_image is None:
        return None, "Could not read BW"

    # Fix any size mismatch
    bw_height, bw_width = bw_image.shape
    if bw_height != original_height or bw_width != original_width:
        bw_image = cv2.resize(bw_image, (original_width, original_height),
                              interpolation=cv2.INTER_NEAREST)

    # --- STEP 2: BBOX DETECTION ---
    label_bbox = get_bbox_from_bw(bw_image)
    if label_bbox is None:
        return None, "Could not find bbox"

    bbox_x, bbox_y, bbox_width, bbox_height = label_bbox

    # Whiten everything outside the bbox (isolate the label)
    isolated_image = original_image.copy()
    isolated_image[:bbox_y, :] = 255
    isolated_image[bbox_y + bbox_height:, :] = 255
    isolated_image[:, :bbox_x] = 255
    isolated_image[:, bbox_x + bbox_width:] = 255
    output_image_path = save_directory / "output.jpg"
    cv2.imwrite(str(output_image_path), isolated_image)

    # --- TIGHT CROP AROUND THE NON-WHITE AREA ---
    gray_isolated = cv2.cvtColor(isolated_image, cv2.COLOR_BGR2GRAY)
    non_white_mask = gray_isolated < 250
    rows_with_content = np.any(non_white_mask, axis=1)
    cols_with_content = np.any(non_white_mask, axis=0)

    if not np.any(rows_with_content) or not np.any(cols_with_content):
        return None, "Crop failed"

    row_start, row_end = np.where(rows_with_content)[0][[0, -1]]
    col_start, col_end = np.where(cols_with_content)[0][[0, -1]]

    # 5 px safety padding
    crop_padding = 5
    row_start = max(0, row_start - crop_padding)
    row_end = min(original_height, row_end + crop_padding)
    col_start = max(0, col_start - crop_padding)
    col_end = min(original_width, col_end + crop_padding)

    cropped_image = isolated_image[row_start:row_end, col_start:col_end]
    crop_image_path = save_directory / "crop.jpg"
    cv2.imwrite(str(crop_image_path), cropped_image)

    # --- STEP 3: PERSPECTIVE CORRECTION ---
    perspective_image_path = save_directory / "perspective.jpg"
    perspective_ok = perspective_correct(bw_image, original_image, label_bbox,
                                          perspective_image_path)
    if not perspective_ok:
        return None, "Perspective correction failed"

    # --- COLOR PATCH + MOLE VISUALIZATION ---
    color_visualization_path = save_directory / "color_square.jpg"
    detect_color_squares(perspective_image_path, color_visualization_path)

    return {
        "bw":           str(bw_image_path),
        "output":       str(output_image_path),
        "crop":         str(crop_image_path),
        "perspective":  str(perspective_image_path),
        "color_square": str(color_visualization_path),
    }, None


def extract_patch_rgb(perspective_image, patches):
    """
    Extract the mean RGB color from the central circular region of each
    color patch.

    Why a circle: pixels near the rectangular patch borders may be affected
    by perspective/lighting artifacts. The central circular region gives a
    cleaner color sample.

    Args:
        perspective_image: Perspective-corrected BGR image.
        patches:           Output of detect_patches_dynamic: [{'bbox': (x,y,w,h)}].

    Returns:
        [[R, G, B], ...] - mean RGB for each patch.
    """
    image_height, image_width = perspective_image.shape[:2]
    patch_rgb_list = []

    for patch in patches:
        patch_x, patch_y, patch_width, patch_height = patch['bbox']

        # Patch center and sampling radius
        center_x = patch_x + patch_width // 2
        center_y = patch_y + patch_height // 2
        sampling_radius = min(patch_width, patch_height) // 2

        # Mask out everything except the central circle
        circular_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        cv2.circle(circular_mask, (center_x, center_y), sampling_radius, 255, -1)

        # Mean BGR inside the mask (OpenCV uses BGR order)
        mean_bgr = cv2.mean(perspective_image, mask=circular_mask)[:3]

        # Convert BGR -> RGB
        patch_rgb_list.append([
            round(mean_bgr[2], 1),  # R
            round(mean_bgr[1], 1),  # G
            round(mean_bgr[0], 1),  # B
        ])

    return patch_rgb_list


def extract_mole_info(perspective_image):
    """
    Extract clinically meaningful measurements from the mole region:
      - Area (square pixels)
      - Equivalent diameter (eq_diameter): the diameter of a circle with
        the same area
      - Perimeter (pixels)
      - Mean RGB color
      - Otsu contour (for visualization)

    Args:
        perspective_image: Perspective-corrected BGR image.

    Returns:
        Measurement dict, or None if no mole was detected.
    """
    mole_bbox = detect_mole(perspective_image)
    if not mole_bbox:
        return None

    mole_x, mole_y, mole_width, mole_height = mole_bbox

    # Otsu-threshold the ROI to outline the mole
    gray_image = cv2.cvtColor(perspective_image, cv2.COLOR_BGR2GRAY)
    mole_roi_gray = gray_image[mole_y:mole_y + mole_height, mole_x:mole_x + mole_width]
    _, mole_roi_binary = cv2.threshold(mole_roi_gray, 0, 255,
                                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleanup
    morph_kernel = np.ones((3, 3), np.uint8)
    mole_roi_binary = cv2.morphologyEx(mole_roi_binary, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
    mole_roi_binary = cv2.morphologyEx(mole_roi_binary, cv2.MORPH_OPEN, morph_kernel)

    contours, _ = cv2.findContours(mole_roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # The biggest contour is the mole
    largest_contour = max(contours, key=cv2.contourArea)
    mole_area_pixels = cv2.contourArea(largest_contour)
    mole_perimeter_pixels = cv2.arcLength(largest_contour, True)

    # Equivalent diameter: pi*r^2 = A => 2r = 2*sqrt(A/pi) = sqrt(4A/pi)
    equivalent_diameter = float(np.sqrt(4 * mole_area_pixels / np.pi))

    # Build a full-image mask from the contour and compute mean RGB
    image_height, image_width = perspective_image.shape[:2]
    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    # Convert ROI-local coords to absolute coords
    largest_contour_absolute = largest_contour + np.array([mole_x, mole_y])
    cv2.drawContours(full_mask, [largest_contour_absolute], -1, 255, -1)
    mean_bgr = cv2.mean(perspective_image, mask=full_mask)[:3]

    return {
        "area_px":        float(mole_area_pixels),
        "eq_diameter_px": equivalent_diameter,
        "perimeter_px":   float(mole_perimeter_pixels),
        "rgb":            [round(mean_bgr[2], 1), round(mean_bgr[1], 1), round(mean_bgr[0], 1)],
        "contour":        largest_contour_absolute,
    }


def patch_distances(patches):
    """
    Compute pairwise Euclidean distances between all patch centers.
    With 16 patches we get C(16, 2) = 120 distances.

    These distances are used to estimate the inter-capture scale factor
    (Step 7): scale = test_distances / reference_distances.

    Args:
        patches: Output of detect_patches_dynamic.

    Returns:
        List of float: every pairwise distance.
    """
    # Patch center coordinates
    patch_centers = []
    for patch in patches:
        bx, by, bw, bh = patch['bbox']
        patch_centers.append((bx + bw / 2, by + bh / 2))
    patch_centers = np.array(patch_centers)

    # All pairwise distances
    pairwise_distances = []
    num_patches = len(patch_centers)
    for i in range(num_patches):
        for j in range(i + 1, num_patches):
            distance = float(np.sqrt(np.sum((patch_centers[i] - patch_centers[j]) ** 2)))
            pairwise_distances.append(distance)

    return pairwise_distances


# ============================================================================
# COMPARISON: STEPS 4-8 (REFERENCE vs TEST)
# ============================================================================

def run_comparison(reference_directory: Path, test_directory: Path):
    """
    Compare two captures (reference + test) of the same mole.

    Computed metrics:
      Step 4: Patch color drift (raw, pre-calibration)
      Step 5: RGB channel calibration parameters (a*x + b, R^2)
      Step 6: Mole color change (post-calibration ΔE2000)
      Step 7: Scale factor (median of inter-patch distance ratios)
      Step 8: Mole size change (area and diameter percent change)

    Args:
        reference_directory: Reference (older) capture folder.
        test_directory:      Test (newer) capture folder.

    Returns:
        (result_dict, error_message): error_message is None on success.
    """
    reference_perspective_path = reference_directory / "perspective.jpg"
    test_perspective_path = test_directory / "perspective.jpg"

    if not reference_perspective_path.exists() or not test_perspective_path.exists():
        return None, "Perspective images not found"

    image_reference = cv2.imread(str(reference_perspective_path))
    image_test = cv2.imread(str(test_perspective_path))

    # ------------------------------------------------------------------------
    # PATCH RGBS AND DISTANCES: FIRST READ FROM data.json
    # ------------------------------------------------------------------------
    # During /analyze we already ran patch detection and stored results in
    # data.json. Reusing them is faster and more consistent.
    reference_data_path = reference_directory / "data.json"
    test_data_path = test_directory / "data.json"

    rgb_reference = None
    rgb_test = None
    distances_reference = None
    distances_test = None

    if reference_data_path.exists() and test_data_path.exists():
        try:
            with open(reference_data_path) as f:
                reference_data = json.load(f)
            with open(test_data_path) as f:
                test_data = json.load(f)

            if reference_data.get("patch_rgbs") and test_data.get("patch_rgbs"):
                rgb_reference = np.array(reference_data["patch_rgbs"])
                rgb_test = np.array(test_data["patch_rgbs"])
                print(f"  [COMPARE] Loaded from data.json: "
                      f"ref={len(rgb_reference)}, test={len(rgb_test)} patches")

            if reference_data.get("patch_distances") and test_data.get("patch_distances"):
                distances_reference = np.array(reference_data["patch_distances"])
                distances_test = np.array(test_data["patch_distances"])
        except Exception as e:
            print(f"  [WARN] Could not read data.json: {e}")

    # ------------------------------------------------------------------------
    # FALLBACK: if data.json is missing or incomplete, run dynamic detection
    # ------------------------------------------------------------------------
    if (rgb_reference is None or rgb_test is None
            or len(rgb_reference) < 14 or len(rgb_test) < 14):
        print(f"  [COMPARE] Falling back to dynamic detection...")

        frame_bbox_reference = find_label_frame(image_reference)
        frame_bbox_test = find_label_frame(image_test)
        patches_reference = detect_patches_dynamic(image_reference, frame_bbox_reference)
        patches_test = detect_patches_dynamic(image_test, frame_bbox_test)

        if len(patches_reference) < 14 or len(patches_test) < 14:
            return None, (f"Patch detection failed "
                          f"(ref={len(patches_reference)}, test={len(patches_test)})")

        rgb_reference = np.array(extract_patch_rgb(image_reference, patches_reference))
        rgb_test = np.array(extract_patch_rgb(image_test, patches_test))
        distances_reference = np.array(patch_distances(patches_reference))
        distances_test = np.array(patch_distances(patches_test))

    # ------------------------------------------------------------------------
    # STEP 4: COLOR DRIFT - PRE-CALIBRATION
    # ------------------------------------------------------------------------
    # For each patch: Euclidean distance between ref RGB and test RGB
    num_patches = min(len(rgb_reference), len(rgb_test))
    patch_drifts = []
    for i in range(num_patches):
        drift = float(np.sqrt(np.sum((rgb_test[i] - rgb_reference[i]) ** 2)))
        patch_drifts.append(drift)
    mean_drift = float(np.mean(patch_drifts))

    # ------------------------------------------------------------------------
    # STEP 5: RGB CHANNEL CALIBRATION (LEAST SQUARES)
    # ------------------------------------------------------------------------
    # To bring the test capture closer (in color) to the reference, we fit
    # a line y = a*x + b for each of R, G, B.
    # a, b: channel gain and bias. R^2: fit quality.
    calibration_parameters = {}
    for channel_index, channel_name in enumerate(['R', 'G', 'B']):
        # x: test channel, y: reference channel (mean per patch, 16 patches)
        x_values = rgb_test[:num_patches, channel_index]
        y_values = rgb_reference[:num_patches, channel_index]

        # Least squares: A * [a; b] = y, where A = [x | 1]
        design_matrix = np.vstack([x_values, np.ones(len(x_values))]).T
        slope, intercept = np.linalg.lstsq(design_matrix, y_values, rcond=None)[0]

        # R^2
        predicted_y = slope * x_values + intercept
        sum_squared_residuals = np.sum((y_values - predicted_y) ** 2)
        sum_squared_total = np.sum((y_values - np.mean(y_values)) ** 2)
        r_squared = float(1 - sum_squared_residuals / sum_squared_total) if sum_squared_total > 0 else 0

        calibration_parameters[channel_name] = {
            'a': float(slope),
            'b': float(intercept),
            'r2': r_squared,
        }

    # Apply calibration to the patch colors
    rgb_test_calibrated = np.zeros_like(rgb_test[:num_patches])
    for channel_index, channel_name in enumerate(['R', 'G', 'B']):
        rgb_test_calibrated[:, channel_index] = np.clip(
            calibration_parameters[channel_name]['a'] * rgb_test[:num_patches, channel_index]
            + calibration_parameters[channel_name]['b'],
            0, 255
        )

    # Post-calibration patch residuals measured in ΔE2000 (LAB space)
    patch_residuals_de2000 = [
        delta_e_ciede2000(rgb_to_lab(rgb_reference[i]), rgb_to_lab(rgb_test_calibrated[i]))
        for i in range(num_patches)
    ]
    mean_residual = float(np.mean(patch_residuals_de2000))
    std_residual = float(np.std(patch_residuals_de2000))

    # ------------------------------------------------------------------------
    # STEP 6: MOLE COLOR CHANGE
    # ------------------------------------------------------------------------
    mole_info_reference = extract_mole_info(image_reference)
    mole_info_test = extract_mole_info(image_test)

    if not mole_info_reference or not mole_info_test:
        return None, "Mole detection failed"

    mole_rgb_reference = np.array(mole_info_reference['rgb'])
    mole_rgb_test = np.array(mole_info_test['rgb'])

    # Calibrate the test mole color using the Step 5 coefficients
    mole_rgb_test_calibrated = np.array([
        np.clip(calibration_parameters['R']['a'] * mole_rgb_test[0]
                + calibration_parameters['R']['b'], 0, 255),
        np.clip(calibration_parameters['G']['a'] * mole_rgb_test[1]
                + calibration_parameters['G']['b'], 0, 255),
        np.clip(calibration_parameters['B']['a'] * mole_rgb_test[2]
                + calibration_parameters['B']['b'], 0, 255),
    ])

    # ΔE2000: both raw and post-calibration
    delta_e_color_raw = delta_e_ciede2000(rgb_to_lab(mole_rgb_reference),
                                           rgb_to_lab(mole_rgb_test))
    delta_e_color_calibrated = delta_e_ciede2000(rgb_to_lab(mole_rgb_reference),
                                                  rgb_to_lab(mole_rgb_test_calibrated))

    # Color-change threshold: mean patch residual + 3*sigma, minimum 6.0 ΔE2000
    # NOTE: ΔE2000 has a different scale than RGB (<1 imperceptible, 2-10
    # noticeable at a glance). Since moles are dark/brown while patches are
    # saturated, calibration doesn't fit the mole perfectly; we keep a floor
    # of 6.0 ΔE2000.
    color_change_threshold = max(mean_residual + 3 * std_residual, 6.0)
    color_changed = delta_e_color_calibrated > color_change_threshold

    # ------------------------------------------------------------------------
    # STEP 7: SCALE FACTOR (from inter-patch distances)
    # ------------------------------------------------------------------------
    # Camera-to-card distance may differ between captures; we solve for the
    # scale by taking the median ratio of patch distances: scale = d_test / d_ref.
    min_distance_count = min(len(distances_reference), len(distances_test))
    pairwise_scale_factors = distances_test[:min_distance_count] / distances_reference[:min_distance_count]
    scale_factor = float(np.median(pairwise_scale_factors))
    scale_factor_std = float(np.std(pairwise_scale_factors))

    # Calibrate the test mole's area and diameter by the scale factor
    mole_area_calibrated = mole_info_test['area_px'] / (scale_factor ** 2)
    mole_diameter_calibrated = mole_info_test['eq_diameter_px'] / scale_factor

    # ------------------------------------------------------------------------
    # STEP 8: MOLE SIZE CHANGE
    # ------------------------------------------------------------------------
    area_change_percent = ((mole_area_calibrated - mole_info_reference['area_px'])
                           / mole_info_reference['area_px'] * 100)
    diameter_change_percent = ((mole_diameter_calibrated - mole_info_reference['eq_diameter_px'])
                               / mole_info_reference['eq_diameter_px'] * 100)

    # Relative scale error -> threshold adaptation
    # For small moles Otsu segmentation can jitter a few pixels; area scales
    # quadratically with diameter, so we treat diameter as primary and area
    # as secondary.
    relative_scale_error = scale_factor_std / scale_factor
    area_change_threshold = max(2 * (2 * relative_scale_error * 100), 35.0)       # min 35%
    diameter_change_threshold = max(2 * (relative_scale_error * 100), 18.0)       # min 18%

    # Decision: "changed" only when both diameter and area exceed their thresholds
    size_changed = (abs(diameter_change_percent) > diameter_change_threshold
                    and abs(area_change_percent) > area_change_threshold)

    # ------------------------------------------------------------------------
    # COMPARISON VISUAL: REFERENCE - TEST - CALIBRATED TEST (horizontal)
    # ------------------------------------------------------------------------
    # Calibrate the test image channel-by-channel (visually)
    test_image_calibrated = np.zeros_like(image_test, dtype=np.float64)
    # OpenCV uses BGR order: ch[0]=B, ch[1]=G, ch[2]=R
    test_image_calibrated[:, :, 2] = (calibration_parameters['R']['a']
                                       * image_test[:, :, 2].astype(np.float64)
                                       + calibration_parameters['R']['b'])
    test_image_calibrated[:, :, 1] = (calibration_parameters['G']['a']
                                       * image_test[:, :, 1].astype(np.float64)
                                       + calibration_parameters['G']['b'])
    test_image_calibrated[:, :, 0] = (calibration_parameters['B']['a']
                                       * image_test[:, :, 0].astype(np.float64)
                                       + calibration_parameters['B']['b'])
    test_image_calibrated = np.clip(test_image_calibrated, 0, 255).astype(np.uint8)

    # Resize all three to the same height
    target_height = max(image_reference.shape[0], image_test.shape[0])

    def resize_to_target_height(image_to_resize):
        scale = target_height / image_to_resize.shape[0]
        return cv2.resize(image_to_resize,
                          (int(image_to_resize.shape[1] * scale), target_height))

    image_reference_resized = resize_to_target_height(image_reference)
    image_test_resized = resize_to_target_height(image_test)
    image_calibrated_resized = resize_to_target_height(test_image_calibrated)

    # Label each panel
    for image_panel, label_text in [(image_reference_resized,  "REFERENCE"),
                                     (image_test_resized,       "TEST"),
                                     (image_calibrated_resized, "CALIBRATED")]:
        cv2.putText(image_panel, label_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Concatenate horizontally and save
    comparison_image = np.hstack([image_reference_resized,
                                   image_test_resized,
                                   image_calibrated_resized])
    comparison_image_path = test_directory / "comparison.jpg"
    cv2.imwrite(str(comparison_image_path), comparison_image)

    # ------------------------------------------------------------------------
    # STEP 7 VISUAL: scale and area comparison
    # ------------------------------------------------------------------------
    step7_reference = image_reference.copy()
    step7_test = image_test.copy()

    # Draw mole contours in red
    if 'contour' in mole_info_reference:
        cv2.drawContours(step7_reference, [mole_info_reference['contour']], -1, (0, 0, 255), 2)
    if 'contour' in mole_info_test:
        cv2.drawContours(step7_test, [mole_info_test['contour']], -1, (0, 0, 255), 2)

    # Labels and scale text
    cv2.putText(step7_reference, "REFERENCE", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(step7_test, "TEST", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(step7_test, f"Scale: {scale_factor:.3f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Area annotations
    cv2.putText(step7_reference,
                f"Area: {mole_info_reference['area_px']:.0f}px2",
                (10, step7_reference.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(step7_test,
                f"Area: {mole_info_test['area_px']:.0f}px2 -> calibrated: "
                f"{mole_area_calibrated:.0f}px2 ({area_change_percent:+.1f}%)",
                (10, step7_test.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    # Side-by-side canvas (dark gray background)
    step7_canvas_height = max(step7_reference.shape[0], step7_test.shape[0])
    step7_canvas = np.zeros((step7_canvas_height,
                              step7_reference.shape[1] + step7_test.shape[1] + 20, 3),
                             dtype=np.uint8)
    step7_canvas[:, :] = 40
    step7_canvas[:step7_reference.shape[0], :step7_reference.shape[1]] = step7_reference
    test_panel_offset_x = step7_reference.shape[1] + 20
    step7_canvas[:step7_test.shape[0],
                 test_panel_offset_x:test_panel_offset_x + step7_test.shape[1]] = step7_test
    step7_image_path = test_directory / "step7_scale.jpg"
    cv2.imwrite(str(step7_image_path), step7_canvas)

    # ------------------------------------------------------------------------
    # STEP 8 VISUAL: size comparison (with area + diameter decision)
    # ------------------------------------------------------------------------
    step8_reference = image_reference.copy()
    step8_test = image_test.copy()

    if 'contour' in mole_info_reference:
        cv2.drawContours(step8_reference, [mole_info_reference['contour']], -1, (0, 0, 255), 2)
    if 'contour' in mole_info_test:
        cv2.drawContours(step8_test, [mole_info_test['contour']], -1, (0, 0, 255), 2)

    cv2.putText(step8_reference, "REFERENCE", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(step8_reference,
                f"Area={mole_info_reference['area_px']:.0f}px2  "
                f"Diam={mole_info_reference['eq_diameter_px']:.1f}px",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(step8_test, "TEST (CALIBRATED)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(step8_test,
                f"Area={mole_area_calibrated:.0f}px2  Diam={mole_diameter_calibrated:.1f}px",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    # Decision text
    size_decision_text = "SIZE UNCHANGED" if not size_changed else "SIZE CHANGED!"
    size_decision_color = (0, 200, 0) if not size_changed else (0, 0, 255)

    step8_canvas_height = max(step8_reference.shape[0], step8_test.shape[0])
    step8_canvas = np.zeros((step8_canvas_height + 40,
                              step8_reference.shape[1] + step8_test.shape[1] + 20, 3),
                             dtype=np.uint8)
    step8_canvas[:, :] = 40
    step8_canvas[:step8_reference.shape[0], :step8_reference.shape[1]] = step8_reference
    test_panel_offset_x_8 = step8_reference.shape[1] + 20
    step8_canvas[:step8_test.shape[0],
                 test_panel_offset_x_8:test_panel_offset_x_8 + step8_test.shape[1]] = step8_test

    # Bottom info / decision banner
    cv2.putText(step8_canvas,
                f"Area: {area_change_percent:+.1f}% | "
                f"Diam: {diameter_change_percent:+.1f}% | "
                f"Threshold: +/-{area_change_threshold:.1f}% | {size_decision_text}",
                (10, step8_canvas_height + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, size_decision_color, 2)
    step8_image_path = test_directory / "step8_size.jpg"
    cv2.imwrite(str(step8_image_path), step8_canvas)

    # ------------------------------------------------------------------------
    # BUILD THE RESULT DICT
    # ------------------------------------------------------------------------
    return {
        "step4_mean_drift": round(mean_drift, 1),
        "step5_calibration": calibration_parameters,
        "step5_mean_residual": round(mean_residual, 1),
        "patch_rgbs_ref": rgb_reference[:num_patches].tolist(),
        "patch_rgbs_test": rgb_test[:num_patches].tolist(),
        "patch_de2000": [
            round(delta_e_ciede2000(rgb_to_lab(rgb_reference[i]),
                                     rgb_to_lab(rgb_test[i])), 1)
            for i in range(num_patches)
        ],
        "step6_color": {
            "ref_rgb": mole_info_reference['rgb'],
            "test_rgb": mole_info_test['rgb'],
            "test_calibrated_rgb": [round(x, 1) for x in mole_rgb_test_calibrated.tolist()],
            "de_raw": round(delta_e_color_raw, 1),
            "de_calibrated": round(delta_e_color_calibrated, 1),
            "threshold": round(color_change_threshold, 1),
            "changed": color_changed,
        },
        "step7_scale": {
            "factor": round(scale_factor, 4),
            "std": round(scale_factor_std, 4),
        },
        "step8_size": {
            "ref_area":                round(mole_info_reference['area_px'], 0),
            "test_area_calibrated":    round(mole_area_calibrated, 0),
            "area_change_pct":         round(area_change_percent, 1),
            "area_threshold_pct":      round(area_change_threshold, 1),
            "ref_diameter":            round(mole_info_reference['eq_diameter_px'], 1),
            "test_diameter_calibrated": round(mole_diameter_calibrated, 1),
            "diam_change_pct":         round(diameter_change_percent, 1),
            "diam_threshold_pct":      round(diameter_change_threshold, 1),
            "changed":                 size_changed,
        },
        "comparison_image": str(comparison_image_path),
    }, None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    """Health check - is the server running?"""
    return {"status": "ok", "message": "Mole Analysis API"}


@app.get("/app")
def web_app():
    """Serve the web application (app.html)."""
    html_path = Path(__file__).parent / "app.html"
    if not html_path.exists():
        raise HTTPException(404, "app.html not found")
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/moles")
def list_moles():
    """List every registered mole and its capture count."""
    moles_list = []
    for mole_directory in sorted(BASE_DIR.iterdir()):
        if mole_directory.is_dir():
            captures_in_mole = sorted([
                capture_dir.name for capture_dir in mole_directory.iterdir()
                if capture_dir.is_dir()
            ])
            moles_list.append({
                "name": mole_directory.name,
                "capture_count": len(captures_in_mole),
                "captures": captures_in_mole,
            })
    return {"moles": moles_list}


@app.get("/moles/{mole_name}/captures")
def list_captures(mole_name: str):
    """List every capture for a given mole."""
    mole_directory = BASE_DIR / mole_name
    if not mole_directory.exists():
        raise HTTPException(404, f"'{mole_name}' not found")

    captures_list = []
    for capture_dir in sorted(mole_directory.iterdir()):
        if capture_dir.is_dir():
            files_in_capture = [f.name for f in capture_dir.iterdir() if f.is_file()]
            has_analysis_data = "data.json" in files_in_capture
            captures_list.append({
                "timestamp": capture_dir.name,
                "files": files_in_capture,
                "analyzed": has_analysis_data,
            })
    return {"mole": mole_name, "captures": captures_list}


@app.delete("/moles/{mole_name}")
def delete_mole(mole_name: str):
    """Delete a mole and all of its captures (irreversible)."""
    mole_directory = BASE_DIR / mole_name
    if not mole_directory.exists():
        raise HTTPException(404, f"'{mole_name}' not found")
    shutil.rmtree(mole_directory)
    return {"status": "ok", "message": f"'{mole_name}' deleted"}


@app.delete("/moles/{mole_name}/{timestamp}")
def delete_capture(mole_name: str, timestamp: str):
    """Delete a single capture. If it was the last one, delete the mole too."""
    capture_directory = BASE_DIR / mole_name / timestamp
    if not capture_directory.exists():
        raise HTTPException(404, f"Capture not found: {timestamp}")
    shutil.rmtree(capture_directory)

    # Remove the mole folder if it's now empty
    mole_directory = BASE_DIR / mole_name
    remaining_captures = [c for c in mole_directory.iterdir() if c.is_dir()]
    if not remaining_captures:
        shutil.rmtree(mole_directory)

    return {"status": "ok", "message": f"Capture deleted: {timestamp}"}


@app.put("/moles/{mole_name}")
def rename_mole(mole_name: str, new_name: str = Form(...)):
    """Rename a mole."""
    mole_directory = BASE_DIR / mole_name
    if not mole_directory.exists():
        raise HTTPException(404, f"'{mole_name}' not found")

    new_mole_directory = BASE_DIR / new_name
    if new_mole_directory.exists():
        raise HTTPException(400, f"'{new_name}' already exists")

    mole_directory.rename(new_mole_directory)
    return {"status": "ok",
            "message": f"'{mole_name}' -> '{new_name}' renamed"}


@app.post("/analyze")
async def analyze(
    mole_name: str = Form(...),
    photo: UploadFile = File(...)
):
    """
    Upload a new photo and run Steps 1-3.

    Outputs are saved under MOLES/{mole_name}/{timestamp}/.
    Patch RGBs and pairwise distances are also written to data.json for
    quick reuse during later comparisons.
    """
    # Create the capture folder (each capture gets a unique timestamp)
    timestamp_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    capture_save_directory = BASE_DIR / mole_name / timestamp_string
    capture_save_directory.mkdir(parents=True, exist_ok=True)

    # Save the uploaded photo
    input_image_path = capture_save_directory / "input.jpg"
    uploaded_file_content = await photo.read()
    with open(input_image_path, "wb") as output_file:
        output_file.write(uploaded_file_content)

    # --- RUN STEPS 1-3 OF THE CODE ---
    pipeline_result, pipeline_error = run_pipeline_step1_3(
        input_image_path, capture_save_directory
    )
    if pipeline_error:
        print(f"  [ERROR] Steps 1-3 of the code: {pipeline_error}")
        shutil.rmtree(capture_save_directory, ignore_errors=True)
        raise HTTPException(400, f"Bad capture: {pipeline_error}")

    # --- PATCH DETECTION ---
    perspective_image = cv2.imread(str(capture_save_directory / "perspective.jpg"))
    if perspective_image is None:
        print(f"  [ERROR] Could not read the perspective image")
        shutil.rmtree(capture_save_directory, ignore_errors=True)
        raise HTTPException(400, "Bad capture: perspective image could not be produced")

    frame_bbox = find_label_frame(perspective_image)
    print(f"  Frame: {frame_bbox}")
    detected_patches = detect_patches_dynamic(perspective_image, frame_bbox)
    print(f"  Patches: {len(detected_patches)} found via dynamic detection")

    # If dynamic detection failed, fall back to fixed coordinates
    if len(detected_patches) < 14:
        frame_x, frame_y, frame_width, frame_height = frame_bbox
        detected_patches = []
        for normalized_cx, normalized_cy in PATCHES:
            x1 = int(frame_x + (normalized_cx - PATCH_SIZE / 2) * frame_width)
            y1 = int(frame_y + (normalized_cy - PATCH_SIZE / 2) * frame_height)
            x2 = int(frame_x + (normalized_cx + PATCH_SIZE / 2) * frame_width)
            y2 = int(frame_y + (normalized_cy + PATCH_SIZE / 2) * frame_height)
            detected_patches.append({'bbox': (x1, y1, x2 - x1, y2 - y1)})
        print(f"  Patches: {len(detected_patches)} fixed coordinates (fallback)")

    # --- MOLE DETECTION ---
    mole_information = extract_mole_info(perspective_image)
    print(f"  Mole: {'detected' if mole_information else 'not detected'}")

    # Validation is fully disabled - warn and still save
    if len(detected_patches) < 14:
        print(f"  [WARN] Low patch count: {len(detected_patches)}/16 - saving anyway")
    if mole_information is None:
        print(f"  [WARN] Mole not detected - saving anyway")

    # contour is a numpy array - not JSON-serializable, drop it
    mole_info_for_json = None
    if mole_information:
        mole_info_for_json = {
            key: value for key, value in mole_information.items()
            if key != 'contour'
        }

    # Extract patch RGBs and pairwise distances
    patch_rgb_values = extract_patch_rgb(perspective_image, detected_patches)
    patch_pairwise_distances = patch_distances(detected_patches)

    # data.json: quick-access data for later comparisons
    analysis_data = {
        "mole_name":        mole_name,
        "timestamp":        timestamp_string,
        "patch_count":      len(detected_patches),
        "patch_rgbs":       patch_rgb_values,
        "patch_distances":  patch_pairwise_distances,
        "mole":             mole_info_for_json,
        "files":            pipeline_result,
    }

    with open(capture_save_directory / "data.json", "w") as output_file:
        json.dump(analysis_data, output_file, indent=2)

    return {
        "status":         "ok",
        "mole":           mole_name,
        "timestamp":      timestamp_string,
        "patch_count":    len(detected_patches),
        "mole_detected":  True,
        "files":          list(pipeline_result.keys()),
    }


@app.post("/compare")
async def compare(
    mole_name: str = Form(...),
    ref_timestamp: str = Form(...),
    test_timestamp: str = Form(...)
):
    """
    Compare two captures of the same mole (Steps 4-8).

    Args:
        mole_name:      Mole name.
        ref_timestamp:  Reference capture timestamp (usually the older one).
        test_timestamp: Test capture timestamp (usually the newer one).
    """
    reference_directory = BASE_DIR / mole_name / ref_timestamp
    test_directory = BASE_DIR / mole_name / test_timestamp

    if not reference_directory.exists():
        raise HTTPException(404, f"Reference capture not found: {ref_timestamp}")
    if not test_directory.exists():
        raise HTTPException(404, f"Test capture not found: {test_timestamp}")

    # --- RUN STEPS 4-8 OF THE CODE ---
    comparison_result, comparison_error = run_comparison(reference_directory, test_directory)
    if comparison_error:
        raise HTTPException(500, comparison_error)

    # Persist results to a comparison JSON
    with open(test_directory / "comparison_data.json", "w") as output_file:
        json.dump(comparison_result, output_file, indent=2)

    return comparison_result


@app.get("/image/{mole_name}/{timestamp}/{filename}")
def get_image(mole_name: str, timestamp: str, filename: str):
    """Fetch a step image from a specific capture."""
    image_file_path = BASE_DIR / mole_name / timestamp / filename
    if not image_file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(image_file_path))


# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    print("\nStarting Mole Analysis API...")
    print(f"Data folder : {BASE_DIR.resolve()}")
    print(f"URL         : http://localhost:8000")
    print(f"Web app     : http://localhost:8000/app")
    print(f"API docs    : http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
