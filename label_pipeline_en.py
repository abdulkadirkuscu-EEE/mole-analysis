"""
============================================================================
SKIN LESION (MOLE) ANALYSIS - IMAGE PROCESSING MODULE
============================================================================
This module contains all image processing functions that operate on skin
lesion (mole) photos taken on top of a calibration card.

PROCESSING STEPS (8 steps total):
  Step 1: BW (black-and-white) image generation
  Step 2: Label (calibration card) bounding box detection
  Step 3: Perspective correction
  Step 4: Color drift detection between two captures
  Step 5: RGB channel calibration (least squares)
  Step 6: Mole color change detection (CIEDE2000)
  Step 7: Scale factor estimation (from inter-patch distances)
  Step 8: Mole size change detection

Steps 1-3 live in this module.
Steps 4-8 live in api_server.py, but reuse helpers from here.

HOW TO RUN:
  Running this file standalone starts a folder watcher: any new image
  dropped into input_images/ is processed automatically (steps 1-3).
============================================================================
"""

import cv2
import numpy as np
import time
import shutil
from pathlib import Path

# ============================================================================
# CONFIGURATION - constants that affect behavior
# ============================================================================

# Input/output folder names (used in standalone watcher mode)
INPUT_FOLDER  = "input_images"          # Original photos to process
BW_FOLDER     = "bw_images"             # Step 1 outputs: black-and-white images
OUTPUT_FOLDER = "output_images"         # Step 2 outputs: everything outside bbox whitened
CROP_FOLDER   = "crop_images"           # Tightly cropped images
PERSP_FOLDER  = "perspective_images"    # Step 3 outputs: perspective-corrected
COLOR_FOLDER  = "color_square_images"   # Detected color patches marked

# Supported image extensions (for the folder-watching mode)
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Polling interval for the folder watcher (seconds)
POLL_INTERVAL = 2

# ----------------------------------------------------------------------------
# Fixed reference points on our calibration card (normalized coords: 0..1)
# These values were measured from our printed template card. The card holds
# 16 color patches arranged in a U-shape:
#   - Top row:    4 horizontal patches (left -> right)
#   - Bottom row: 4 horizontal patches (left -> right)
#   - Left col:   4 vertical patches (top -> bottom)
#   - Right col:  4 vertical patches (top -> bottom)
# The empty middle area is where the lesion (mole) is photographed.
# ----------------------------------------------------------------------------
PATCHES = [
    # Top row - left to right, 4 patches
    (0.370, 0.269), (0.456, 0.269), (0.539, 0.268), (0.622, 0.269),
    # Bottom row - left to right, 4 patches
    (0.374, 0.756), (0.458, 0.760), (0.540, 0.756), (0.622, 0.755),
    # Left column - top to bottom, 4 patches
    (0.254, 0.386), (0.258, 0.471), (0.255, 0.553), (0.258, 0.634),
    # Right column - top to bottom, 4 patches
    (0.750, 0.385), (0.749, 0.468), (0.753, 0.549), (0.749, 0.633),
]

# Side length of each color patch (as a fraction of normalized card width)
PATCH_SIZE = 0.075

# Approximate center of the lesion (mole) area on the card (normalized coords)
LESION_CENTER = (0.497, 0.520)

# Approximate size of the lesion area (as a fraction of the card edge)
LESION_SIZE = 0.245

# Ratio between the inner frame and outer frame of the card.
# Measured from the template: inner_square_side / outer_square_side = 0.751
INNER_FRAME_RATIO = 0.751


# ============================================================================
# CIELAB COLOR SPACE + CIEDE2000 COLOR DIFFERENCE
# ----------------------------------------------------------------------------
# More perceptually accurate than RGB for measuring color differences as the
# human eye sees them. A delta-E2000 < 1 is imperceptible; 2-10 is visible at
# a glance. Since color changes carry clinical meaning for skin lesions, we
# keep our delta-E2000 thresholds on the conservative side.
# ============================================================================

def rgb_to_lab(rgb_color):
    """
    Convert sRGB (0-255) to CIELAB.

    Steps:
      1) Remove sRGB gamma (yields linear RGB)
      2) Linear RGB -> XYZ (sRGB->XYZ matrix, D65 white point)
      3) XYZ -> LAB (D65 white point reference, cube-root transform)

    Args:
        rgb_color: 3-element (R, G, B), each channel in 0..255.

    Returns:
        3-element numpy array: [L*, a*, b*]
    """
    rgb_normalized = np.asarray(rgb_color, dtype=np.float64) / 255.0

    # 1) Remove sRGB gamma: nonlinear sRGB -> linear RGB.
    # Threshold 0.04045 is per the sRGB standard.
    def linearize_srgb(channel):
        return np.where(channel > 0.04045,
                        ((channel + 0.055) / 1.055) ** 2.4,
                        channel / 12.92)

    r_lin = linearize_srgb(rgb_normalized[..., 0])
    g_lin = linearize_srgb(rgb_normalized[..., 1])
    b_lin = linearize_srgb(rgb_normalized[..., 2])

    # 2) Linear RGB -> XYZ: standard sRGB D65 matrix
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    # 3) XYZ -> LAB: normalize by the D65 white point
    white_x, white_y, white_z = 0.95047, 1.00000, 1.08883
    fx_arg, fy_arg, fz_arg = x / white_x, y / white_y, z / white_z

    # CIE LAB nonlinear transform (cube-root / linear blend)
    def lab_transform(t):
        # Threshold 0.008856 = (6/29)^3 per the CIE standard
        return np.where(t > 0.008856,
                        np.cbrt(t),
                        7.787 * t + 16.0 / 116.0)

    fx_val = lab_transform(fx_arg)
    fy_val = lab_transform(fy_arg)
    fz_val = lab_transform(fz_arg)

    # L*, a*, b* values
    lightness = 116.0 * fy_val - 16.0
    a_axis = 500.0 * (fx_val - fy_val)
    b_axis = 200.0 * (fy_val - fz_val)

    return np.stack([lightness, a_axis, b_axis], axis=-1)


def delta_e_ciede2000(lab1, lab2):
    """
    Compute the CIEDE2000 color difference (delta-E00) between two LAB colors.

    CIEDE2000 is the CIE's most refined color-difference formula and tracks
    human perception more closely than earlier metrics.

    Args:
        lab1: First color as [L*, a*, b*].
        lab2: Second color as [L*, a*, b*].

    Returns:
        delta-E2000 (float). Typical interpretation:
          < 1.0 : imperceptible to the human eye
          1-2   : noticeable on close inspection
          2-10  : noticeable at a glance
          > 10  : clearly different colors
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    # Mean lightness and chroma
    avg_lightness = (L1 + L2) / 2.0
    chroma1 = np.sqrt(a1 ** 2 + b1 ** 2)
    chroma2 = np.sqrt(a2 ** 2 + b2 ** 2)
    avg_chroma = (chroma1 + chroma2) / 2.0

    # G factor: low-chroma correction
    g_factor = 0.5 * (1 - np.sqrt(avg_chroma ** 7 / (avg_chroma ** 7 + 25 ** 7)))
    a1_prime = (1 + g_factor) * a1
    a2_prime = (1 + g_factor) * a2

    chroma1_prime = np.sqrt(a1_prime ** 2 + b1 ** 2)
    chroma2_prime = np.sqrt(a2_prime ** 2 + b2 ** 2)
    avg_chroma_prime = (chroma1_prime + chroma2_prime) / 2.0

    # Hue angles in degrees
    hue1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    hue2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360

    # Component differences
    delta_lightness_prime = L2 - L1
    delta_chroma_prime = chroma2_prime - chroma1_prime

    # Hue difference, wrapped to -180..+180
    delta_hue = hue2_prime - hue1_prime
    if abs(delta_hue) > 180:
        delta_hue -= 360 * np.sign(delta_hue)
    delta_hue_prime = 2 * np.sqrt(chroma1_prime * chroma2_prime) * np.sin(np.radians(delta_hue) / 2.0)

    # Mean hue
    avg_hue_prime = hue1_prime + hue2_prime
    if abs(hue1_prime - hue2_prime) > 180:
        avg_hue_prime += 360
    avg_hue_prime /= 2.0

    # T weighting function (hue correction)
    t_weight = (1 - 0.17 * np.cos(np.radians(avg_hue_prime - 30))
                  + 0.24 * np.cos(np.radians(2 * avg_hue_prime))
                  + 0.32 * np.cos(np.radians(3 * avg_hue_prime + 6))
                  - 0.20 * np.cos(np.radians(4 * avg_hue_prime - 63)))

    # Rotation factor (extra correction in the blue-purple region)
    rotation_angle = 30 * np.exp(-(((avg_hue_prime - 275) / 25) ** 2))
    chroma_correction = 2 * np.sqrt(avg_chroma_prime ** 7 / (avg_chroma_prime ** 7 + 25 ** 7))

    # Scaling coefficients (Sl, Sc, Sh)
    s_lightness = 1 + (0.015 * (avg_lightness - 50) ** 2) / np.sqrt(20 + (avg_lightness - 50) ** 2)
    s_chroma = 1 + 0.045 * avg_chroma_prime
    s_hue = 1 + 0.015 * avg_chroma_prime * t_weight
    rotation_term = -np.sin(np.radians(2 * rotation_angle)) * chroma_correction

    # Final delta-E2000
    delta_e = np.sqrt(
        (delta_lightness_prime / s_lightness) ** 2
        + (delta_chroma_prime / s_chroma) ** 2
        + (delta_hue_prime / s_hue) ** 2
        + rotation_term * (delta_chroma_prime / s_chroma) * (delta_hue_prime / s_hue)
    )
    return float(delta_e)


# ============================================================================
# FOLDER HELPERS
# ============================================================================

def clear_folders():
    """
    Wipe and recreate every output folder (input_images is left alone).
    Called once on startup in standalone watcher mode.
    """
    folders_to_clear = [BW_FOLDER, OUTPUT_FOLDER, CROP_FOLDER, PERSP_FOLDER, COLOR_FOLDER]
    for folder_name in folders_to_clear:
        folder_path = Path(folder_name)
        if folder_path.exists():
            shutil.rmtree(folder_path)
        folder_path.mkdir(exist_ok=True)
        print(f"  Cleared: {folder_path.resolve()}")


# ============================================================================
# STEP 1: BW (BLACK-AND-WHITE) IMAGE GENERATION
# ----------------------------------------------------------------------------
# Goal: produce a mask that isolates the label (calibration card) from the
#       background.
# Method: cascaded thresholding + rectangular contour search + histogram-based
#         fine thresholding inside the ROI.
# ============================================================================

def generate_bw(input_image_path, output_bw_path):
    """
    Produce a BW (black-and-white) image from the original RGB photo, with the
    calibration card isolated.

    Method:
      1) Try several global threshold values (80, 100, 70).
      2) At each threshold, look for 4-cornered rectangular contours.
      3) Pick the largest valid rectangle, then re-threshold within an ROI
         around it using a histogram-based approach.
      4) The card area ends up black; everything else is white.

    Args:
        input_image_path: Path to the original photo.
        output_bw_path:   Where to write the BW image.

    Returns:
        True on success; False if the image could not be read.
    """
    original_image = cv2.imread(str(input_image_path))
    if original_image is None:
        return False

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    image_height, image_width = gray_image.shape

    best_bbox = None
    best_bbox_area = 0

    # Try multiple thresholds; the card may look darker or lighter under
    # different lighting conditions.
    for threshold_value in [80, 100, 70]:
        # Inverse threshold: useful when the card is dark on a bright background
        _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour_area = cv2.contourArea(contour)
            # Skip very small (noise) or very large (whole frame) contours
            if contour_area < gray_image.size * 0.005 or contour_area > gray_image.size * 0.4:
                continue

            perimeter = cv2.arcLength(contour, True)

            # Try several epsilon values for the 4-corner approximation
            for epsilon_ratio in [0.03, 0.04, 0.05, 0.06]:
                approx_polygon = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)

                # Only accept quadrilaterals
                if len(approx_polygon) == 4:
                    bbox_x, bbox_y, bbox_width, bbox_height = cv2.boundingRect(approx_polygon)

                    # Aspect ratio > 0.8 means "close to square"
                    aspect_ratio = (min(bbox_width, bbox_height) / max(bbox_width, bbox_height)
                                    if max(bbox_width, bbox_height) > 0 else 0)
                    bbox_area = bbox_width * bbox_height

                    if aspect_ratio > 0.8 and bbox_area > best_bbox_area:
                        best_bbox_area = bbox_area
                        best_bbox = (bbox_x, bbox_y, bbox_width, bbox_height)
                    # Got a quadrilateral for this contour; skip remaining epsilons
                    break

    # If a best bbox was found, take a padded ROI around it
    if best_bbox is not None:
        bbox_x, bbox_y, bbox_width, bbox_height = best_bbox
        padding = int(max(bbox_width, bbox_height) * 0.15)
        roi_x1 = max(0, bbox_x - padding)
        roi_y1 = max(0, bbox_y - padding)
        roi_x2 = min(image_width, bbox_x + bbox_width + padding)
        roi_y2 = min(image_height, bbox_y + bbox_height + padding)
    else:
        # Otherwise treat the entire image as the ROI
        roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, image_width, image_height

    # Fine histogram-based threshold inside the ROI:
    # find the threshold that puts the darkest 15% of pixels below it.
    roi_gray = gray_image[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_histogram = cv2.calcHist([roi_gray], [0], None, [256], [0, 256]).flatten()
    cumulative_histogram = np.cumsum(roi_histogram)
    roi_threshold = int(np.searchsorted(cumulative_histogram, roi_gray.size * 0.15))

    # Inside the ROI: pixels below the threshold -> 0 (black), others -> 255
    roi_bw = np.where(roi_gray < roi_threshold, 0, 255).astype(np.uint8)

    # Full-size BW image: everything outside the ROI is white
    bw_full_image = np.ones_like(gray_image) * 255
    bw_full_image[roi_y1:roi_y2, roi_x1:roi_x2] = roi_bw

    cv2.imwrite(str(output_bw_path), bw_full_image)
    return True


# ============================================================================
# STEP 2: LABEL BOUNDING BOX DETECTION
# ============================================================================

def get_bbox_from_bw(bw_image):
    """
    Pick the best label-candidate black blob in the BW image and return its
    bounding box.

    Score formula (higher is better):
      score = 0.3 * area_ratio + 0.6 * aspect_ratio + 0.1 * proximity_to_center

    Args:
        bw_image: BW (single-channel) numpy array.

    Returns:
        (x, y, w, h) tuple, or None.
    """
    image_height, image_width = bw_image.shape

    # Turn black pixels into 255 (connectedComponents expects a positive mask)
    black_pixel_mask = (bw_image < 128).astype(np.uint8) * 255

    # 8-connected components with area, bbox, centroid
    num_components, label_image, stats, centroids = cv2.connectedComponentsWithStats(
        black_pixel_mask, connectivity=8
    )

    # Image center (the card usually sits near the photo center)
    image_center_x = image_width // 2
    image_center_y = image_height // 2

    best_bbox = None
    best_score = -1

    # Component 0 is the background — skip it
    for component_index in range(1, num_components):
        component_area = stats[component_index, cv2.CC_STAT_AREA]

        # Skip very small components (noise)
        if component_area < bw_image.size * 0.001:
            continue

        bbox_x = stats[component_index, 0]
        bbox_y = stats[component_index, 1]
        bbox_width = stats[component_index, 2]
        bbox_height = stats[component_index, 3]
        bbox_area = bbox_width * bbox_height

        # Aspect ratio (1.0 = perfect square = what we want)
        aspect_ratio = (min(bbox_width, bbox_height) / max(bbox_width, bbox_height)
                        if max(bbox_width, bbox_height) > 0 else 0)

        # Normalized distance from component center to image center
        component_center_x, component_center_y = centroids[component_index]
        distance_to_center = (np.sqrt((component_center_x - image_center_x) ** 2
                                       + (component_center_y - image_center_y) ** 2)
                               / np.sqrt(image_center_x ** 2 + image_center_y ** 2))

        # Combined score: prefer large + close-to-square + close-to-center
        score = ((bbox_area / bw_image.size) * 0.3
                 + aspect_ratio * 0.6
                 + (1 - distance_to_center) * 0.1)

        if score > best_score:
            best_score = score
            best_bbox = (int(bbox_x), int(bbox_y), int(bbox_width), int(bbox_height))

    return best_bbox


# ============================================================================
# STEP 3: PERSPECTIVE CORRECTION
# ============================================================================

def order_points(quadrilateral_points):
    """
    Sort the 4 corners of a quadrilateral clockwise, with the top-left corner
    appearing first.

    Method: sort by angle clockwise, then rotate so the corner with the
    smallest (x+y) — i.e. the top-left — comes first.

    Args:
        quadrilateral_points: numpy array of shape (4, 2).

    Returns:
        Reordered numpy array of shape (4, 2), dtype=float32.
    """
    geometric_center = quadrilateral_points.mean(axis=0)

    # Angle of each point relative to the centroid
    angles = np.arctan2(quadrilateral_points[:, 1] - geometric_center[1],
                        quadrilateral_points[:, 0] - geometric_center[0])
    sorted_order = np.argsort(angles)
    sorted_points = quadrilateral_points[sorted_order]

    # Smallest (x+y) marks the top-left corner
    sum_xy = sorted_points.sum(axis=1)
    top_left_index = np.argmin(sum_xy)

    # Roll the array so the top-left corner is index 0
    ordered_rectangle = np.roll(sorted_points, -top_left_index, axis=0).astype(np.float32)
    return ordered_rectangle


def perspective_correct(bw_image, full_color_image, label_bbox, output_path):
    """
    Find the card quadrilateral in the BW image, then warp the original color
    image into a square (perspective correction).

    Steps:
      1) Look for a 4-cornered contour in the BW mask (fallback: dilated mask).
      2) Order the 4 corners clockwise.
      3) Compute a perspective transform onto a square (side x side).
      4) Warp the original color image via warpPerspective.

    Args:
        bw_image:        Black-and-white image.
        full_color_image: Original BGR image (source for the warp).
        label_bbox:       Label bbox found in step 2 (x, y, w, h).
        output_path:      Where to save the warped image.

    Returns:
        True on success; False if the output size is too small (< 50 px).
    """
    bbox_x, bbox_y, bbox_width, bbox_height = label_bbox
    target_area = bbox_width * bbox_height
    bbox_center_x = bbox_x + bbox_width / 2
    bbox_center_y = bbox_y + bbox_height / 2

    def find_quadrilateral(binary_mask):
        """Find the best 4-cornered contour in the given binary mask."""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_quadrilateral = None
        best_area_diff = float('inf')

        for contour in contours:
            contour_area = cv2.contourArea(contour)
            # Skip if smaller than half the target area
            if contour_area < target_area * 0.5:
                continue

            perimeter = cv2.arcLength(contour, True)

            # Try a range of epsilons
            for epsilon_ratio in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
                approx_polygon = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)

                if len(approx_polygon) == 4:
                    bx, by, bw, bh = cv2.boundingRect(approx_polygon)
                    aspect_ratio = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
                    approx_bbox_area = bw * bh

                    # Center of the candidate quadrilateral
                    quad_center_x = bx + bw / 2
                    quad_center_y = by + bh / 2
                    center_distance = np.sqrt((quad_center_x - bbox_center_x) ** 2
                                              + (quad_center_y - bbox_center_y) ** 2)
                    area_difference = abs(approx_bbox_area - target_area) / target_area

                    # Best candidate: close-to-square + close-to-center + similar area
                    if (aspect_ratio > 0.8
                            and center_distance < max(bbox_width, bbox_height) * 0.3
                            and area_difference < best_area_diff):
                        best_area_diff = area_difference
                        best_quadrilateral = approx_polygon.reshape(4, 2).astype(np.float32)
                    break
        return best_quadrilateral

    # First try with the raw black mask
    black_mask = (bw_image < 128).astype(np.uint8) * 255
    best_corner_points = find_quadrilateral(black_mask)

    # If nothing, try a dilated mask (closes small breaks in the contour)
    if best_corner_points is None:
        dilation_kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(black_mask, dilation_kernel, iterations=2)
        best_corner_points = find_quadrilateral(dilated_mask)

    # Last resort: fall back to the raw bbox corners
    if best_corner_points is None:
        best_corner_points = np.array(
            [[bbox_x, bbox_y],
             [bbox_x + bbox_width, bbox_y],
             [bbox_x + bbox_width, bbox_y + bbox_height],
             [bbox_x, bbox_y + bbox_height]],
            dtype=np.float32
        )

    # Order corners clockwise with top-left first
    ordered_corners = order_points(best_corner_points)

    # Output side length: longest of the four edges
    side_length = int(max(
        np.linalg.norm(ordered_corners[1] - ordered_corners[0]),
        np.linalg.norm(ordered_corners[2] - ordered_corners[3]),
        np.linalg.norm(ordered_corners[3] - ordered_corners[0]),
        np.linalg.norm(ordered_corners[2] - ordered_corners[1])
    ))

    # Anything below 50 px isn't meaningful — treat as failure
    if side_length < 50:
        return False

    # Destination corners of a unit square (top-left first, clockwise)
    destination_points = np.array(
        [[0, 0],
         [side_length - 1, 0],
         [side_length - 1, side_length - 1],
         [0, side_length - 1]],
        dtype=np.float32
    )

    # Compute and apply the perspective transform
    perspective_matrix = cv2.getPerspectiveTransform(ordered_corners, destination_points)
    warped_image = cv2.warpPerspective(
        full_color_image, perspective_matrix, (side_length, side_length),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    cv2.imwrite(str(output_path), warped_image)
    return True


# ============================================================================
# POST-PERSPECTIVE: LABEL FRAME DETECTION
# ----------------------------------------------------------------------------
# After perspective correction, locate the actual "inner + outer frame" region
# of the card. This region is used as the reference for later patch detection.
# ============================================================================

def find_label_frame(perspective_image):
    """
    Find the outer-frame bbox of the label in a perspective-corrected image.

    Two methods are tried in order:
      1) High threshold + dilation to merge the outer frame into a single
         contour.
      2) If that fails, find the inner square and back-compute the outer
         frame from the known inner/outer size ratio.

    Args:
        perspective_image: Perspective-corrected BGR image.

    Returns:
        (x, y, w, h): outer-frame bbox.
    """
    image_height, image_width = perspective_image.shape[:2]
    gray_image = cv2.cvtColor(perspective_image, cv2.COLOR_BGR2GRAY)

    # --- METHOD 1: dilation + outer-frame contour ---
    _, binary_image = cv2.threshold(gray_image, 130, 255, cv2.THRESH_BINARY_INV)
    dilation_kernel = np.ones((10, 10), np.uint8)
    dilated_binary = cv2.dilate(binary_image, dilation_kernel, iterations=3)

    contours, _ = cv2.findContours(dilated_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    image_area = image_width * image_height
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        # Anything covering 10-95% of the image is a plausible frame candidate
        if contour_area > image_area * 0.1 and contour_area < image_area * 0.95:
            bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)
            aspect_ratio = min(bbox_w, bbox_h) / max(bbox_w, bbox_h)
            if aspect_ratio > 0.85:
                return (bbox_x, bbox_y, bbox_w, bbox_h)

    # --- METHOD 2: find the inner square, then back-compute the outer frame ---
    _, binary_low_thresh = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY_INV)
    inner_contours, _ = cv2.findContours(binary_low_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quadrilateral_candidates = []

    for contour in inner_contours:
        contour_area = cv2.contourArea(contour)
        # The inner square should cover 1-70% of the image
        if contour_area < image_area * 0.01 or contour_area > image_area * 0.7:
            continue

        perimeter = cv2.arcLength(contour, True)
        for epsilon_ratio in [0.02, 0.03, 0.04, 0.05]:
            approx_polygon = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
            if len(approx_polygon) == 4:
                bx, by, bw, bh = cv2.boundingRect(approx_polygon)
                aspect_ratio = min(bw, bh) / max(bw, bh)
                size_ratio_to_image = max(bw, bh) / image_width

                # Inner square should be roughly 60-85% of the card width
                if aspect_ratio > 0.85 and 0.60 < size_ratio_to_image < 0.85:
                    quadrilateral_candidates.append((max(bw, bh), bx, by, bw, bh))
                break

    if not quadrilateral_candidates:
        # No candidates — fall back to the entire image as the frame
        return (0, 0, image_width, image_height)

    # Pick the largest inner square
    quadrilateral_candidates.sort(reverse=True)
    _, inner_bbox_x, inner_bbox_y, inner_bbox_w, inner_bbox_h = quadrilateral_candidates[0]

    # Outer-frame size from the inner/outer ratio
    outer_frame_width = int(inner_bbox_w / INNER_FRAME_RATIO)
    outer_frame_height = int(inner_bbox_h / INNER_FRAME_RATIO)

    # The center of the inner square is also the center of the outer frame
    inner_center_x = inner_bbox_x + inner_bbox_w / 2
    inner_center_y = inner_bbox_y + inner_bbox_h / 2

    # 0.487 and 0.491 are the inner-center offsets measured from the template
    outer_frame_x = int(inner_center_x - 0.487 * outer_frame_width)
    outer_frame_y = int(inner_center_y - 0.491 * outer_frame_height)

    return (outer_frame_x, outer_frame_y, outer_frame_width, outer_frame_height)


# ============================================================================
# MOLE (LESION) DETECTION
# ============================================================================

def detect_mole(perspective_image):
    """
    Dynamically locate the mole (a dark blob in the center) in a
    perspective-corrected image.

    Algorithm:
      1) Define an ROI ~13% on each side of the image center.
      2) Try several thresholds (60, 80, 40, 100, 30).
      3) At each threshold, pick the blob with the highest
         (circularity * aspect * area) score.

    Args:
        perspective_image: Perspective-corrected BGR image.

    Returns:
        (x, y, w, h) of the mole bbox in absolute image coordinates, or None.
    """
    image_height, image_width = perspective_image.shape[:2]
    gray_image = cv2.cvtColor(perspective_image, cv2.COLOR_BGR2GRAY)

    # Search ROI defined around the image center
    center_x = image_width // 2
    center_y = image_height // 2
    roi_margin = int(image_width * 0.13)
    roi_x1 = max(0, center_x - roi_margin)
    roi_y1 = max(0, center_y - roi_margin)
    roi_x2 = min(image_width, center_x + roi_margin)
    roi_y2 = min(image_height, center_y + roi_margin)

    roi_gray = gray_image[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_height, roi_width = roi_gray.shape

    best_mole_bbox = None
    best_score = -1

    # Try multiple thresholds — the mole can be more or less dark
    for threshold_value in [60, 80, 40, 100, 30]:
        _, roi_binary = cv2.threshold(roi_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

        # Morphological cleanup: close small holes / open small spurs
        morph_kernel = np.ones((3, 3), np.uint8)
        roi_binary = cv2.morphologyEx(roi_binary, cv2.MORPH_CLOSE, morph_kernel)
        roi_binary = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, morph_kernel)

        contours, _ = cv2.findContours(roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour_area = cv2.contourArea(contour)

            # Skip if too small, or larger than half the ROI
            if contour_area < 80 or contour_area > roi_width * roi_height * 0.5:
                continue

            bx, by, bw, bh = cv2.boundingRect(contour)
            aspect_ratio = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0

            # Circularity: 4*pi*A/P^2 (perfect circle = 1, line = 0)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0

            # Score favors circular + close-to-square + large blobs
            score = circularity * aspect_ratio * contour_area

            if score > best_score:
                best_score = score
                # Convert ROI-local coords to absolute image coords
                best_mole_bbox = (roi_x1 + bx, roi_y1 + by, bw, bh)

    return best_mole_bbox


# ============================================================================
# COLOR PATCH DETECTION - BAND + PROJECTION METHOD
# ============================================================================

def detect_patches_dynamic(perspective_image, frame_bbox):
    """
    Dynamically detect all 16 color patches in a perspective-corrected image.

    Algorithm (Band + Projection method):
      1) Find 4 saturated-color bands (top / bottom / left / right) from the
         saturation channel.
      2) Inside each band, take a projection along the V (value) channel.
      3) Cut the band into segments using valleys (gaps between patches) and
         peaks (the patches themselves).
      4) Each segment becomes one color patch.

    This is more robust than fixed coordinates because perspective correction
    can slightly shift the exact patch positions.

    Args:
        perspective_image: Perspective-corrected BGR image.
        frame_bbox:        Outer-frame bbox of the label (x, y, w, h).

    Returns:
        List of dicts: each has key 'bbox' = (x, y, w, h).
        Expected count: 16 (4 bands x 4 patches). Sometimes fewer.
    """
    frame_x, frame_y, frame_width, frame_height = frame_bbox
    image_height, image_width = perspective_image.shape[:2]

    # Switch to HSV — we analyze the saturation and value channels
    hsv_image = cv2.cvtColor(perspective_image, cv2.COLOR_BGR2HSV)
    saturation_channel = hsv_image[:, :, 1]
    value_channel = hsv_image[:, :, 2]

    # ------------------------------------------------------------------------
    # PHASE 1: DETECT THE SATURATED BANDS
    # ------------------------------------------------------------------------
    # Color patches are highly saturated (unlike black/white/gray patches).
    # We look for 4 rectangular bands in the saturation mask.
    frame_area = frame_width * frame_height
    min_band_area = frame_area * 0.01       # 1% of the frame area
    max_band_area = frame_area * 0.08       # 8% of the frame area
    min_band_side = max(frame_width, frame_height) * 0.05  # 5% of the longest side

    detected_bands = []

    # Try several saturation thresholds — color intensity varies
    for saturation_threshold in [50, 40, 30, 60]:
        saturation_mask = (saturation_channel > saturation_threshold).astype(np.uint8) * 255
        mask_contours, _ = cv2.findContours(saturation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_bands = []
        for contour in mask_contours:
            contour_area = cv2.contourArea(contour)
            bx, by, bw, bh = cv2.boundingRect(contour)

            # To qualify as a band, the rectangle must be elongated
            aspect_long_to_short = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 999

            if (min_band_area < contour_area < max_band_area
                    and 1.8 < aspect_long_to_short < 7.0
                    and max(bw, bh) > min_band_side):
                # Horizontal (H) or vertical (V)?
                orientation = "H" if bw > bh else "V"
                detected_bands.append((bx, by, bw, bh, orientation))

        # If we found exactly 4 bands, no need to try other thresholds
        if len(detected_bands) == 4:
            break

    # If we didn't end up with 4 bands, treat as failure
    if len(detected_bands) != 4:
        return []

    # ------------------------------------------------------------------------
    # CLASSIFY BANDS: TOP / BOTTOM / LEFT / RIGHT
    # ------------------------------------------------------------------------
    frame_center_x = frame_x + frame_width / 2
    frame_center_y = frame_y + frame_height / 2

    classified_bands = {'top': None, 'bottom': None, 'left': None, 'right': None}

    for band in detected_bands:
        bx, by, bw, bh, orientation = band
        band_center_x = bx + bw / 2
        band_center_y = by + bh / 2

        if orientation == "H":
            # Horizontal band: top or bottom based on its y position
            if band_center_y < frame_center_y:
                classified_bands['top'] = band
            else:
                classified_bands['bottom'] = band
        else:
            # Vertical band: left or right based on its x position
            if band_center_x < frame_center_x:
                classified_bands['left'] = band
            else:
                classified_bands['right'] = band

    # Order: top (L->R), bottom (L->R), left (T->B), right (T->B)
    ordered_bands = [classified_bands['top'],    classified_bands['bottom'],
                     classified_bands['left'],   classified_bands['right']]

    # If any position is missing, the detection failed
    if None in ordered_bands:
        return []

    # ------------------------------------------------------------------------
    # PHASE 2: SPLIT EACH BAND INTO INDIVIDUAL PATCHES VIA PROJECTION
    # ------------------------------------------------------------------------
    all_detected_patches = []

    for bx, by, bw, bh, orientation in ordered_bands:
        # Expand the band ROI slightly so edge patches aren't clipped
        roi_margin = 3
        roi_y1 = max(0, by - roi_margin)
        roi_y2 = min(image_height, by + bh + roi_margin)
        roi_x1 = max(0, bx - roi_margin)
        roi_x2 = min(image_width, bx + bw + roi_margin)

        roi_value = value_channel[roi_y1:roi_y2, roi_x1:roi_x2]

        # 1D projection along the band's orientation.
        # Horizontal band: average each column. Vertical band: average each row.
        if orientation == "H":
            intensity_profile = np.mean(roi_value, axis=0)
        else:
            intensity_profile = np.mean(roi_value, axis=1)

        # 5-window moving average for noise smoothing
        smoothed_profile = np.convolve(intensity_profile, np.ones(5) / 5, mode='same')

        # Valley threshold (gaps between patches): 25th percentile of the profile
        valley_threshold = np.percentile(smoothed_profile, 25)

        # ---- FIND VALLEYS ----
        # Regions below the threshold that are at least 2 px wide count as valleys
        valleys = []
        currently_in_valley = False
        valley_start_index = 0

        for index, intensity_value in enumerate(smoothed_profile):
            if intensity_value < valley_threshold:
                if not currently_in_valley:
                    valley_start_index = index
                    currently_in_valley = True
            else:
                if currently_in_valley:
                    if index - valley_start_index >= 2:
                        valleys.append((valley_start_index, index))
                    currently_in_valley = False

        # Profile may still be inside a valley at the end
        if currently_in_valley and len(intensity_profile) - valley_start_index >= 2:
            valleys.append((valley_start_index, len(intensity_profile)))

        # ---- SEGMENTS BETWEEN VALLEYS = PATCHES ----
        segments = []
        previous_segment_end = 0

        for valley_start, valley_end in valleys:
            if valley_start > previous_segment_end + 5:
                segments.append((previous_segment_end, valley_start))
            previous_segment_end = valley_end

        # Trailing segment (if profile doesn't end on a valley)
        if previous_segment_end < len(intensity_profile) - 5:
            segments.append((previous_segment_end, len(intensity_profile)))

        # ---- SIZE FILTER WHEN WE HAVE TOO MANY SEGMENTS ----
        # Keep only segments between 0.4x and 2.5x the median size
        if len(segments) > 4:
            segment_sizes = [end - start for start, end in segments]
            median_segment_size = np.median(segment_sizes)
            segments = [(start, end) for start, end in segments
                        if (end - start) > median_segment_size * 0.4
                        and (end - start) < median_segment_size * 2.5]

        # ---- FINE-CROP EACH SEGMENT ----
        # Use saturated pixels (S > 40) to tighten the patch bounds
        for patch_segment_start, patch_segment_end in segments[:4]:
            if orientation == "H":
                # Horizontal band: x-axis is the segment direction, y is the band height
                patch_x = roi_x1 + patch_segment_start
                patch_y = roi_y1
                patch_width = patch_segment_end - patch_segment_start
                patch_height = roi_y2 - roi_y1

                # Find the active (saturated) range along y and crop
                segment_saturation = saturation_channel[roi_y1:roi_y2,
                                                        max(0, patch_x):min(image_width, patch_x + patch_width)]
                if segment_saturation.size > 0:
                    vertical_profile = np.mean(segment_saturation, axis=1)
                    active_indices = np.where(vertical_profile > 40)[0]
                    if len(active_indices) > 3:
                        patch_y = roi_y1 + active_indices[0]
                        patch_height = active_indices[-1] - active_indices[0] + 1
            else:
                # Vertical band: y-axis is the segment direction, x is the band width
                patch_x = roi_x1
                patch_y = roi_y1 + patch_segment_start
                patch_width = roi_x2 - roi_x1
                patch_height = patch_segment_end - patch_segment_start

                # Find the active range along x and crop
                segment_saturation = saturation_channel[max(0, patch_y):min(image_height, patch_y + patch_height),
                                                        roi_x1:roi_x2]
                if segment_saturation.size > 0:
                    horizontal_profile = np.mean(segment_saturation, axis=0)
                    active_indices = np.where(horizontal_profile > 40)[0]
                    if len(active_indices) > 3:
                        patch_x = roi_x1 + active_indices[0]
                        patch_width = active_indices[-1] - active_indices[0] + 1

            all_detected_patches.append({'bbox': (patch_x, patch_y, patch_width, patch_height)})

    return all_detected_patches


# ============================================================================
# COLOR-PATCH AND MOLE VISUALIZATION
# ============================================================================

def detect_color_squares(perspective_image_path, output_visualization_path):
    """
    Produce a visualization image that marks the detected color patches and
    the mole on a perspective-corrected image.

    Drawings:
      - Color patches: green rectangle + number + center circle
      - Mole: blue bbox + red Otsu segmentation contour

    If dynamic detection fails, fall back to the fixed PATCHES coordinates.

    Args:
        perspective_image_path:    Output path of step 3.
        output_visualization_path: Where to save the visualization image.

    Returns:
        True on success; False if the image could not be read.
    """
    perspective_image = cv2.imread(str(perspective_image_path))
    if perspective_image is None:
        return False

    # Locate the label frame
    frame_bbox = find_label_frame(perspective_image)
    frame_x, frame_y, frame_width, frame_height = frame_bbox

    # Output image starts as a copy of the original
    output_image = perspective_image.copy()

    # ------------------------------------------------------------------------
    # COLOR PATCHES - DYNAMIC OR FALLBACK
    # ------------------------------------------------------------------------
    detected_patches = detect_patches_dynamic(perspective_image, frame_bbox)

    if 14 <= len(detected_patches) <= 18:
        # Dynamic detection succeeded — draw the detected boxes
        for patch_index, patch in enumerate(detected_patches):
            patch_x, patch_y, patch_width, patch_height = patch['bbox']

            # Outer green rectangle
            cv2.rectangle(output_image,
                          (patch_x, patch_y),
                          (patch_x + patch_width, patch_y + patch_height),
                          (0, 255, 0), 2)
            # Index label
            cv2.putText(output_image, str(patch_index + 1),
                        (patch_x, patch_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

            # Center circle (RGB sampling region)
            patch_center_x = patch_x + patch_width // 2
            patch_center_y = patch_y + patch_height // 2
            sampling_radius = min(patch_width, patch_height) // 2
            cv2.circle(output_image,
                       (patch_center_x, patch_center_y),
                       sampling_radius, (0, 255, 0), 2)
            cv2.putText(output_image, f"{patch_index + 1}.1",
                        (patch_center_x - 10, patch_center_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 1)

        print(f"    [PATCH] Dynamic: {len(detected_patches)} patches detected")
    else:
        # Fallback: use the fixed PATCHES coordinates
        for patch_index, (normalized_cx, normalized_cy) in enumerate(PATCHES):
            x1 = int(frame_x + (normalized_cx - PATCH_SIZE / 2) * frame_width)
            y1 = int(frame_y + (normalized_cy - PATCH_SIZE / 2) * frame_height)
            x2 = int(frame_x + (normalized_cx + PATCH_SIZE / 2) * frame_width)
            y2 = int(frame_y + (normalized_cy + PATCH_SIZE / 2) * frame_height)
            patch_w = x2 - x1
            patch_h = y2 - y1

            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_image, str(patch_index + 1),
                        (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

            patch_center_x = (x1 + x2) // 2
            patch_center_y = (y1 + y2) // 2
            sampling_radius = min(patch_w, patch_h) // 2
            cv2.circle(output_image, (patch_center_x, patch_center_y),
                       sampling_radius, (0, 255, 0), 2)
            cv2.putText(output_image, f"{patch_index + 1}.1",
                        (patch_center_x - 10, patch_center_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 1)

        print(f"    [PATCH] Fallback: fixed coordinates ({len(detected_patches)} found dynamically)")

    # ------------------------------------------------------------------------
    # MOLE - BLUE BBOX + RED OTSU CONTOUR
    # ------------------------------------------------------------------------
    mole_bbox = detect_mole(perspective_image)
    if mole_bbox:
        mx, my, mw, mh = mole_bbox

        # Blue bbox with a small padding
        padding = 2
        cv2.rectangle(output_image,
                      (mx - padding, my - padding),
                      (mx + mw + padding, my + mh + padding),
                      (255, 0, 0), 2)

        # Otsu thresholding to outline the mole (red contour)
        gray_perspective = cv2.cvtColor(perspective_image, cv2.COLOR_BGR2GRAY)
        roi_gray = gray_perspective[my:my + mh, mx:mx + mw]

        # Otsu picks an optimal threshold automatically
        _, roi_binary = cv2.threshold(roi_gray, 0, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        morph_kernel = np.ones((3, 3), np.uint8)
        roi_binary = cv2.morphologyEx(roi_binary, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
        roi_binary = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, morph_kernel)

        mole_contours, _ = cv2.findContours(roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if mole_contours:
            # The biggest contour is the mole (Otsu sometimes returns multiple parts)
            largest_contour = max(mole_contours, key=cv2.contourArea)
            # Convert ROI-local coords to absolute coords
            largest_contour_absolute = largest_contour + np.array([mx, my])
            cv2.drawContours(output_image, [largest_contour_absolute], -1, (0, 0, 255), 2)

    cv2.imwrite(str(output_visualization_path), output_image)
    return True


# ============================================================================
# COMBINED PROCESS FUNCTION (FOR STANDALONE RUNS)
# ============================================================================

def process(original_image_path, bw_output_path, output_image_path,
            crop_output_path, perspective_output_path, color_marked_path):
    """
    Run all of steps 1-3 in order for a single original image and produce
    every output file.

    Args:
        original_image_path:    Input: original color photo.
        bw_output_path:         Output: BW (step 1).
        output_image_path:      Output: outside-bbox whitened (step 2).
        crop_output_path:       Output: tightly cropped.
        perspective_output_path: Output: perspective-corrected (step 3).
        color_marked_path:      Output: visualization with patches and mole.

    Returns:
        True if all steps succeeded; False if any step failed.
    """
    original_image = cv2.imread(str(original_image_path))
    if original_image is None:
        print(f"  [ERROR] Could not read: {original_image_path.name}")
        return False

    original_height, original_width = original_image.shape[:2]

    # --- STEP 1: BW GENERATION ---
    if not generate_bw(original_image_path, bw_output_path):
        print(f"  [WARN] BW could not be generated: {original_image_path.name}")
        return False

    # Re-read the BW image
    bw_image = cv2.imread(str(bw_output_path), cv2.IMREAD_GRAYSCALE)
    if bw_image is None:
        return False

    # Resize back to the original dimensions if needed
    bw_height, bw_width = bw_image.shape
    if bw_height != original_height or bw_width != original_width:
        bw_image = cv2.resize(bw_image, (original_width, original_height),
                              interpolation=cv2.INTER_NEAREST)

    # --- STEP 2: BBOX DETECTION ---
    label_bbox = get_bbox_from_bw(bw_image)
    if label_bbox is None:
        print(f"  [WARN] No bbox found: {original_image_path.name}")
        return False

    # Whiten everything outside the bbox (isolate the label)
    bbox_x, bbox_y, bbox_width, bbox_height = label_bbox
    isolated_image = original_image.copy()
    isolated_image[:bbox_y, :] = 255
    isolated_image[bbox_y + bbox_height:, :] = 255
    isolated_image[:, :bbox_x] = 255
    isolated_image[:, bbox_x + bbox_width:] = 255
    cv2.imwrite(str(output_image_path), isolated_image)

    # Tight crop around the non-white area
    gray_isolated = cv2.cvtColor(isolated_image, cv2.COLOR_BGR2GRAY)
    non_white_mask = gray_isolated < 250
    rows_with_content = np.any(non_white_mask, axis=1)
    cols_with_content = np.any(non_white_mask, axis=0)

    if not np.any(rows_with_content) or not np.any(cols_with_content):
        print(f"  [WARN] Crop failed")
        return False

    row_start, row_end = np.where(rows_with_content)[0][[0, -1]]
    col_start, col_end = np.where(cols_with_content)[0][[0, -1]]

    # 5 px safety padding
    crop_padding = 5
    row_start = max(0, row_start - crop_padding)
    row_end = min(original_height, row_end + crop_padding)
    col_start = max(0, col_start - crop_padding)
    col_end = min(original_width, col_end + crop_padding)

    cropped_image = isolated_image[row_start:row_end, col_start:col_end]
    cv2.imwrite(str(crop_output_path), cropped_image)

    # --- STEP 3: PERSPECTIVE CORRECTION ---
    perspective_ok = perspective_correct(bw_image, original_image, label_bbox,
                                          perspective_output_path)

    # --- PATCH AND MOLE MARKING ---
    color_marking_ok = False
    if perspective_ok:
        color_marking_ok = detect_color_squares(perspective_output_path, color_marked_path)

    print(f"  OK {original_image_path.name[-25:]} | "
          f"perspective={'OK' if perspective_ok else 'FAIL'} | "
          f"marking={'OK' if color_marking_ok else 'FAIL'}")
    return True


# ============================================================================
# FOLDER WATCHER MODE
# ----------------------------------------------------------------------------
# When this module is run standalone, it watches input_images/ and processes
# every new image automatically. For real usage, prefer the HTTP API in
# api_server.py.
# ============================================================================

def watch_folder():
    """
    Continuously watch input_images/, run steps 1-3 on every new image, and
    save outputs into the matching folders.

    Exit: loops until interrupted with Ctrl+C.
    """
    input_folder_path = Path(INPUT_FOLDER)
    bw_folder_path = Path(BW_FOLDER)
    output_folder_path = Path(OUTPUT_FOLDER)
    crop_folder_path = Path(CROP_FOLDER)
    perspective_folder_path = Path(PERSP_FOLDER)
    color_folder_path = Path(COLOR_FOLDER)

    input_folder_path.mkdir(exist_ok=True)

    print("Clearing folders...")
    clear_folders()
    print()

    print(f"  Originals     : {input_folder_path.resolve()}")
    print(f"  BW            : {bw_folder_path.resolve()}")
    print(f"  Output        : {output_folder_path.resolve()}")
    print(f"  Cropped       : {crop_folder_path.resolve()}")
    print(f"  Perspective   : {perspective_folder_path.resolve()}")
    print(f"  Color squares : {color_folder_path.resolve()}")
    print(f"  Poll interval : {POLL_INTERVAL}s\n")

    # Filenames we've already processed (to avoid reprocessing)
    already_processed = set()

    while True:
        for original_file in sorted(input_folder_path.iterdir()):
            # Skip unsupported extensions
            if original_file.suffix.lower() not in SUPPORTED_EXT:
                continue
            # Skip files we've already handled
            if original_file.name in already_processed:
                continue

            # Output file paths
            bw_file_path = bw_folder_path / (original_file.stem + "_bw.png")
            output_file_path = output_folder_path / (original_file.stem + "_output.jpg")
            crop_file_path = crop_folder_path / (original_file.stem + "_crop.jpg")
            perspective_file_path = perspective_folder_path / (original_file.stem + "_persp.jpg")
            color_file_path = color_folder_path / (original_file.stem + "_color.jpg")

            print(f"Processing: {original_file.name}")
            process(original_file, bw_file_path, output_file_path,
                    crop_file_path, perspective_file_path, color_file_path)
            already_processed.add(original_file.name)

        # Wait for new files
        time.sleep(POLL_INTERVAL)


# ============================================================================
# DIRECT-RUN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # When this file is run directly, start the folder watcher.
    # For the API server instead, run: python api_server.py
    watch_folder()
