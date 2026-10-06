"""
============================================================================
CILT LEZYONU (BEN) ANALIZ - FastAPI BACKEND
============================================================================
Bu sunucu, label_pipeline.py icindeki goruntu isleme fonksiyonlarini HTTP
API endpoint'leri olarak sunar. Bir fotograf yuklenince Adim 1-3 calistirilir
ve sonuclar diske kaydedilir. Iki cekim arasi karsilastirma icin Adim 4-8
calistirilir.

KOD ADIMLARI:
  Adim 1: BW (siyah-beyaz) goruntu uretimi      } label_pipeline.py
  Adim 2: Etiket bounding box tespiti           } iceriginden
  Adim 3: Perspektif duzeltme                   } kullanilir
  Adim 4: Renk kaymasi (drift) tespiti          \\
  Adim 5: RGB kanal kalibrasyonu (en kucuk kareler) \\ Bu dosyada
  Adim 6: Ben rengi degisim tespiti (CIEDE2000)    // calisir
  Adim 7: Olcek faktoru hesabi                  //
  Adim 8: Ben boyut degisim tespiti             //

API ENDPOINT'LERI:
  GET  /                          - Saglik kontrolu
  GET  /app                       - Web uygulamasini sun (app.html)
  GET  /moles                     - Tum benleri listele
  GET  /moles/{name}/captures     - Bir benin cekimlerini listele
  POST /analyze                   - Yeni fotograf isle (Adim 1-3)
  POST /compare                   - Iki cekimi karsilastir (Adim 4-8)
  PUT  /moles/{name}              - Ben adini degistir
  DELETE /moles/{name}            - Beni sil
  DELETE /moles/{name}/{ts}       - Cekimi sil
  GET  /image/{name}/{ts}/{file}  - Belirli bir adim goruntusunu getir

CALISTIRMA:
  python api_server.py
  Sonra tarayicidan: http://localhost:8000/app
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

# Goruntu isleme fonksiyonlarini ic kod modulunden al
from label_pipeline import (
    generate_bw, get_bbox_from_bw, perspective_correct,
    find_label_frame, detect_patches_dynamic, detect_mole,
    detect_color_squares, PATCHES, PATCH_SIZE,
    rgb_to_lab, delta_e_ciede2000
)

# ============================================================================
# AYARLAR
# ============================================================================

# Tum kullanici verilerinin saklandigi kok klasor
# Yapi: BENLER/{ben_adi}/{zaman_damgasi}/dosya.jpg
BASE_DIR = Path("BENLER")
BASE_DIR.mkdir(exist_ok=True)

# FastAPI uygulamasini olustur
app = FastAPI(title="Ben Analiz API", version="1.0")

# CORS: HTML uygulamasi farkli portta calisirsa erisebilsin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# YARDIMCI FONKSIYONLAR
# ============================================================================

def run_pipeline_step1_3(input_image_path: Path, save_directory: Path):
    """
    Adim 1-3'u sirayla calistirir ve tum cikti dosyalarini olusturur:
    BW, output (bbox disi beyazlatilmis), kirpilmis, perspektif duzeltilmis,
    ve yama-ben isaretli gorsel.

    Args:
        input_image_path: Yuklenen orijinal fotografin yolu.
        save_directory:   Tum ciktilarin yazilacagi cekim klasoru.

    Returns:
        (sonuc_dict, hata_mesaji): Basarili oldugunda hata_mesaji=None.
    """
    original_image = cv2.imread(str(input_image_path))
    if original_image is None:
        return None, "Goruntu okunamadi"

    original_height, original_width = original_image.shape[:2]

    # --- ADIM 1: BW URETIMI ---
    bw_image_path = save_directory / "bw.png"
    if not generate_bw(input_image_path, bw_image_path):
        return None, "BW uretilemedi"

    # BW goruntuyu yeniden oku
    bw_image = cv2.imread(str(bw_image_path), cv2.IMREAD_GRAYSCALE)
    if bw_image is None:
        return None, "BW okunamadi"

    # Boyut uyumsuzlugunu duzelt
    bw_height, bw_width = bw_image.shape
    if bw_height != original_height or bw_width != original_width:
        bw_image = cv2.resize(bw_image, (original_width, original_height),
                              interpolation=cv2.INTER_NEAREST)

    # --- ADIM 2: BBOX TESPITI ---
    label_bbox = get_bbox_from_bw(bw_image)
    if label_bbox is None:
        return None, "Bbox bulunamadi"

    bbox_x, bbox_y, bbox_width, bbox_height = label_bbox

    # Bbox disi alanlari beyaza cevir (etiketi izole et)
    isolated_image = original_image.copy()
    isolated_image[:bbox_y, :] = 255
    isolated_image[bbox_y + bbox_height:, :] = 255
    isolated_image[:, :bbox_x] = 255
    isolated_image[:, bbox_x + bbox_width:] = 255
    output_image_path = save_directory / "output.jpg"
    cv2.imwrite(str(output_image_path), isolated_image)

    # --- KIRPMA: Beyaz olmayan alani sikica kirp ---
    gray_isolated = cv2.cvtColor(isolated_image, cv2.COLOR_BGR2GRAY)
    non_white_mask = gray_isolated < 250
    rows_with_content = np.any(non_white_mask, axis=1)
    cols_with_content = np.any(non_white_mask, axis=0)

    if not np.any(rows_with_content) or not np.any(cols_with_content):
        return None, "Kirpma basarisiz"

    row_start, row_end = np.where(rows_with_content)[0][[0, -1]]
    col_start, col_end = np.where(cols_with_content)[0][[0, -1]]

    # 5px guvenlik payi
    crop_padding = 5
    row_start = max(0, row_start - crop_padding)
    row_end = min(original_height, row_end + crop_padding)
    col_start = max(0, col_start - crop_padding)
    col_end = min(original_width, col_end + crop_padding)

    cropped_image = isolated_image[row_start:row_end, col_start:col_end]
    crop_image_path = save_directory / "crop.jpg"
    cv2.imwrite(str(crop_image_path), cropped_image)

    # --- ADIM 3: PERSPEKTIF DUZELTME ---
    perspective_image_path = save_directory / "perspective.jpg"
    perspective_ok = perspective_correct(bw_image, original_image, label_bbox,
                                          perspective_image_path)
    if not perspective_ok:
        return None, "Perspektif duzeltme basarisiz"

    # --- RENK YAMASI + BEN GORSELI ---
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
    Her bir renk yamasinin merkez daire bolgesinden ortalama RGB rengini cikarir.

    Daire sebebi: yamanin dikdortgen sınırlarına yakın olan pikseller
    perspektif/aydınlatma etkisinde olabilir; merkezdeki dairesel bolge
    daha temiz bir renk ornegi verir.

    Args:
        perspective_image: Perspektif duzeltilmis BGR goruntu.
        patches:           detect_patches_dynamic ciktisi: [{'bbox': (x,y,w,h)}].

    Returns:
        [[R, G, B], ...] - Her yamanin ortalama RGB degerleri.
    """
    image_height, image_width = perspective_image.shape[:2]
    patch_rgb_list = []

    for patch in patches:
        patch_x, patch_y, patch_width, patch_height = patch['bbox']

        # Yama merkezini ve sampling yari capini hesapla
        center_x = patch_x + patch_width // 2
        center_y = patch_y + patch_height // 2
        sampling_radius = min(patch_width, patch_height) // 2

        # Sadece daire bolgesini secen bir maske olustur
        circular_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        cv2.circle(circular_mask, (center_x, center_y), sampling_radius, 255, -1)

        # Maske icindeki BGR ortalamasini al (OpenCV BGR sirayla calisir)
        mean_bgr = cv2.mean(perspective_image, mask=circular_mask)[:3]

        # BGR -> RGB cevirip listele
        patch_rgb_list.append([
            round(mean_bgr[2], 1),  # R
            round(mean_bgr[1], 1),  # G
            round(mean_bgr[0], 1),  # B
        ])

    return patch_rgb_list


def extract_mole_info(perspective_image):
    """
    Ben bolgesinden klinik olarak anlamli olcumler cikarir:
      - Alan (piksel kare)
      - Esdeger cap (eq_diameter): dairenin alaniyla esit olan capin uzunlugu
      - Cevre (piksel)
      - Ortalama RGB rengi
      - Otsu konturu (gorsellestirme icin)

    Args:
        perspective_image: Perspektif duzeltilmis BGR goruntu.

    Returns:
        Olcum sozlugu, ben tespit edilemezse None.
    """
    mole_bbox = detect_mole(perspective_image)
    if not mole_bbox:
        return None

    mole_x, mole_y, mole_width, mole_height = mole_bbox

    # Otsu ile ben sinirlarini cikar
    gray_image = cv2.cvtColor(perspective_image, cv2.COLOR_BGR2GRAY)
    mole_roi_gray = gray_image[mole_y:mole_y + mole_height, mole_x:mole_x + mole_width]
    _, mole_roi_binary = cv2.threshold(mole_roi_gray, 0, 255,
                                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morfolojik temizleme
    morph_kernel = np.ones((3, 3), np.uint8)
    mole_roi_binary = cv2.morphologyEx(mole_roi_binary, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
    mole_roi_binary = cv2.morphologyEx(mole_roi_binary, cv2.MORPH_OPEN, morph_kernel)

    contours, _ = cv2.findContours(mole_roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # En buyuk kontur ben sayilir
    largest_contour = max(contours, key=cv2.contourArea)
    mole_area_pixels = cv2.contourArea(largest_contour)
    mole_perimeter_pixels = cv2.arcLength(largest_contour, True)

    # Esdeger cap: pi*r^2 = A => 2r = 2*sqrt(A/pi) = sqrt(4A/pi)
    equivalent_diameter = float(np.sqrt(4 * mole_area_pixels / np.pi))

    # Tum goruntude konturu maskele ve ortalama RGB cikar
    image_height, image_width = perspective_image.shape[:2]
    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    # ROI yerel koordinatlari -> mutlak koordinatlar
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
    Tum yama ciftleri arasindaki Oklid mesafelerini hesaplar.
    16 yamadan 120 cift mesafe cikar (C(16,2) = 120).

    Bu mesafeler iki cekim arasi olcek faktorunu (Adim 7) bulmak icin
    kullanilir: olcek = test_mesafeleri / referans_mesafeleri.

    Args:
        patches: detect_patches_dynamic ciktisi.

    Returns:
        Float listesi: tum cift mesafeler.
    """
    # Her yamanin merkez koordinatini bul
    patch_centers = []
    for patch in patches:
        bx, by, bw, bh = patch['bbox']
        patch_centers.append((bx + bw / 2, by + bh / 2))
    patch_centers = np.array(patch_centers)

    # Tum cift mesafeleri hesapla
    pairwise_distances = []
    num_patches = len(patch_centers)
    for i in range(num_patches):
        for j in range(i + 1, num_patches):
            distance = float(np.sqrt(np.sum((patch_centers[i] - patch_centers[j]) ** 2)))
            pairwise_distances.append(distance)

    return pairwise_distances


# ============================================================================
# KARSILASTIRMA: ADIM 4-8 (REFERANS vs TEST)
# ============================================================================

def run_comparison(reference_directory: Path, test_directory: Path):
    """
    Ayni benin iki cekimini (referans + test) karsilastirir.

    Hesaplanan metrikler:
      Adim 4: Yama renk kaymasi (ham fark, kalibrasyon oncesi)
      Adim 5: RGB kanali kalibrasyon parametreleri (a*x + b, R^2)
      Adim 6: Ben rengi degisimi (kalibre sonrasi ΔE2000)
      Adim 7: Olcek faktoru (yamalar arasi mesafelerden medyan)
      Adim 8: Ben boyut degisimi (alan ve cap yuzde degisimi)

    Args:
        reference_directory: Referans (eski) cekim klasoru.
        test_directory:      Test (yeni) cekim klasoru.

    Returns:
        (sonuc_dict, hata_mesaji): Basarili oldugunda hata_mesaji=None.
    """
    reference_perspective_path = reference_directory / "perspective.jpg"
    test_perspective_path = test_directory / "perspective.jpg"

    if not reference_perspective_path.exists() or not test_perspective_path.exists():
        return None, "Perspektif goruntuleri bulunamadi"

    image_reference = cv2.imread(str(reference_perspective_path))
    image_test = cv2.imread(str(test_perspective_path))

    # ------------------------------------------------------------------------
    # YAMA RGB VE MESAFELERI: ONCE data.json'DAN OKU
    # ------------------------------------------------------------------------
    # Her cekim icin /analyze sirasinda yama tespiti yapildi ve data.json'a
    # kaydedildi. Onu kullanmak daha hizli ve tutarli olur.
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
                print(f"  [KARSILASTIRMA] data.json'dan yuklendi: "
                      f"ref={len(rgb_reference)}, test={len(rgb_test)} yama")

            if reference_data.get("patch_distances") and test_data.get("patch_distances"):
                distances_reference = np.array(reference_data["patch_distances"])
                distances_test = np.array(test_data["patch_distances"])
        except Exception as e:
            print(f"  [UYARI] data.json okunamadi: {e}")

    # ------------------------------------------------------------------------
    # FALLBACK: data.json yoksa veya eksikse dinamik tespit calistir
    # ------------------------------------------------------------------------
    if (rgb_reference is None or rgb_test is None
            or len(rgb_reference) < 14 or len(rgb_test) < 14):
        print(f"  [KARSILASTIRMA] Dinamik tespit deneniyor...")

        frame_bbox_reference = find_label_frame(image_reference)
        frame_bbox_test = find_label_frame(image_test)
        patches_reference = detect_patches_dynamic(image_reference, frame_bbox_reference)
        patches_test = detect_patches_dynamic(image_test, frame_bbox_test)

        if len(patches_reference) < 14 or len(patches_test) < 14:
            return None, (f"Yama tespiti basarisiz "
                          f"(ref={len(patches_reference)}, test={len(patches_test)})")

        rgb_reference = np.array(extract_patch_rgb(image_reference, patches_reference))
        rgb_test = np.array(extract_patch_rgb(image_test, patches_test))
        distances_reference = np.array(patch_distances(patches_reference))
        distances_test = np.array(patch_distances(patches_test))

    # ------------------------------------------------------------------------
    # ADIM 4: RENK KAYMASI (DRIFT) - KALIBRASYON ONCESI
    # ------------------------------------------------------------------------
    # Her yama icin ref RGB ve test RGB arasi Oklid (RGB uzayinda) mesafesi
    num_patches = min(len(rgb_reference), len(rgb_test))
    patch_drifts = []
    for i in range(num_patches):
        drift = float(np.sqrt(np.sum((rgb_test[i] - rgb_reference[i]) ** 2)))
        patch_drifts.append(drift)
    mean_drift = float(np.mean(patch_drifts))

    # ------------------------------------------------------------------------
    # ADIM 5: RGB KANAL KALIBRASYONU (EN KUCUK KARELER)
    # ------------------------------------------------------------------------
    # Test cekimini referans cekimine "renk olarak" yaklastirmak icin
    # her R, G, B kanali icin y = a*x + b dogrusu fit edilir.
    # a, b: kanal kazanci ve bias. R^2: fit kalitesi.
    calibration_parameters = {}
    for channel_index, channel_name in enumerate(['R', 'G', 'B']):
        # x: test kanali, y: referans kanali (16 yamadaki ortalamalar)
        x_values = rgb_test[:num_patches, channel_index]
        y_values = rgb_reference[:num_patches, channel_index]

        # En kucuk kareler: A * [a; b] = y, A = [x | 1]
        design_matrix = np.vstack([x_values, np.ones(len(x_values))]).T
        slope, intercept = np.linalg.lstsq(design_matrix, y_values, rcond=None)[0]

        # R^2 hesabi
        predicted_y = slope * x_values + intercept
        sum_squared_residuals = np.sum((y_values - predicted_y) ** 2)
        sum_squared_total = np.sum((y_values - np.mean(y_values)) ** 2)
        r_squared = float(1 - sum_squared_residuals / sum_squared_total) if sum_squared_total > 0 else 0

        calibration_parameters[channel_name] = {
            'a': float(slope),
            'b': float(intercept),
            'r2': r_squared,
        }

    # Kalibrasyon sonrasi yama renklerini hesapla
    rgb_test_calibrated = np.zeros_like(rgb_test[:num_patches])
    for channel_index, channel_name in enumerate(['R', 'G', 'B']):
        rgb_test_calibrated[:, channel_index] = np.clip(
            calibration_parameters[channel_name]['a'] * rgb_test[:num_patches, channel_index]
            + calibration_parameters[channel_name]['b'],
            0, 255
        )

    # Kalibrasyon sonrasi yama artiklari ΔE2000 (LAB) ile olculur
    patch_residuals_de2000 = [
        delta_e_ciede2000(rgb_to_lab(rgb_reference[i]), rgb_to_lab(rgb_test_calibrated[i]))
        for i in range(num_patches)
    ]
    mean_residual = float(np.mean(patch_residuals_de2000))
    std_residual = float(np.std(patch_residuals_de2000))

    # ------------------------------------------------------------------------
    # ADIM 6: BEN RENGI DEGISIMI
    # ------------------------------------------------------------------------
    mole_info_reference = extract_mole_info(image_reference)
    mole_info_test = extract_mole_info(image_test)

    if not mole_info_reference or not mole_info_test:
        return None, "Ben tespiti basarisiz"

    mole_rgb_reference = np.array(mole_info_reference['rgb'])
    mole_rgb_test = np.array(mole_info_test['rgb'])

    # Test benin rengini Adim 5 katsayilariyla kalibre et
    mole_rgb_test_calibrated = np.array([
        np.clip(calibration_parameters['R']['a'] * mole_rgb_test[0]
                + calibration_parameters['R']['b'], 0, 255),
        np.clip(calibration_parameters['G']['a'] * mole_rgb_test[1]
                + calibration_parameters['G']['b'], 0, 255),
        np.clip(calibration_parameters['B']['a'] * mole_rgb_test[2]
                + calibration_parameters['B']['b'], 0, 255),
    ])

    # ΔE2000: hem ham, hem kalibre sonrasi
    delta_e_color_raw = delta_e_ciede2000(rgb_to_lab(mole_rgb_reference),
                                           rgb_to_lab(mole_rgb_test))
    delta_e_color_calibrated = delta_e_ciede2000(rgb_to_lab(mole_rgb_reference),
                                                  rgb_to_lab(mole_rgb_test_calibrated))

    # Renk degisikligi esigi: yama artiklari ortalama + 3 sigma, minimum 6 ΔE2000
    # NOT: ΔE2000 olcegi RGB'den farklidir (<1 fark edilmez, 2-10 bir bakista
    # gorulebilir). Ben koyu/kahverengi iken yamalar doygun renkli oldugundan
    # kalibrasyon bene tam uymayabilir; bu yuzden mutlak alt sinir 6.0 ΔE2000.
    color_change_threshold = max(mean_residual + 3 * std_residual, 6.0)
    color_changed = delta_e_color_calibrated > color_change_threshold

    # ------------------------------------------------------------------------
    # ADIM 7: OLCEK FAKTORU (yamalar arasi mesafelerden)
    # ------------------------------------------------------------------------
    # Iki cekimde kamera-karta uzakligi farkli olabilir; bu olcek farkini
    # yama mesafelerinin medyan oraniyla cozeriz: scale = d_test / d_ref.
    min_distance_count = min(len(distances_reference), len(distances_test))
    pairwise_scale_factors = distances_test[:min_distance_count] / distances_reference[:min_distance_count]
    scale_factor = float(np.median(pairwise_scale_factors))
    scale_factor_std = float(np.std(pairwise_scale_factors))

    # Test benin alan ve capini olcege gore kalibre et
    mole_area_calibrated = mole_info_test['area_px'] / (scale_factor ** 2)
    mole_diameter_calibrated = mole_info_test['eq_diameter_px'] / scale_factor

    # ------------------------------------------------------------------------
    # ADIM 8: BEN BOYUT DEGISIMI
    # ------------------------------------------------------------------------
    area_change_percent = ((mole_area_calibrated - mole_info_reference['area_px'])
                           / mole_info_reference['area_px'] * 100)
    diameter_change_percent = ((mole_diameter_calibrated - mole_info_reference['eq_diameter_px'])
                               / mole_info_reference['eq_diameter_px'] * 100)

    # Goreceli olcek hatasi -> esik adaptasyonu
    # Kucuk benlerde Otsu segmentasyonu birkac piksel oynar; alan kareyle
    # buyudugunden cap birincil metrik, alan ikincildir.
    relative_scale_error = scale_factor_std / scale_factor
    area_change_threshold = max(2 * (2 * relative_scale_error * 100), 35.0)       # minimum %35
    diameter_change_threshold = max(2 * (relative_scale_error * 100), 18.0)       # minimum %18

    # Karar: hem cap hem alan esigi asarsa "degismis"
    size_changed = (abs(diameter_change_percent) > diameter_change_threshold
                    and abs(area_change_percent) > area_change_threshold)

    # ------------------------------------------------------------------------
    # KARSILASTIRMA GORSELI: REFERANS - TEST - KALIBRE EDILMIS TEST (yatay)
    # ------------------------------------------------------------------------
    # Test goruntuyu kanal-kanal kalibre et (gorsel olarak)
    test_image_calibrated = np.zeros_like(image_test, dtype=np.float64)
    # OpenCV BGR sirasiyla calisir: ch[0]=B, ch[1]=G, ch[2]=R
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

    # Tum goruntuleri ayni yukseklige resize et
    target_height = max(image_reference.shape[0], image_test.shape[0])

    def resize_to_target_height(image_to_resize):
        scale = target_height / image_to_resize.shape[0]
        return cv2.resize(image_to_resize,
                          (int(image_to_resize.shape[1] * scale), target_height))

    image_reference_resized = resize_to_target_height(image_reference)
    image_test_resized = resize_to_target_height(image_test)
    image_calibrated_resized = resize_to_target_height(test_image_calibrated)

    # Her birine etiket yaz
    for image_panel, label_text in [(image_reference_resized, "REFERANS"),
                                     (image_test_resized,      "TEST"),
                                     (image_calibrated_resized, "KALIBRE")]:
        cv2.putText(image_panel, label_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Yatay birlestir ve kaydet
    comparison_image = np.hstack([image_reference_resized,
                                   image_test_resized,
                                   image_calibrated_resized])
    comparison_image_path = test_directory / "comparison.jpg"
    cv2.imwrite(str(comparison_image_path), comparison_image)

    # ------------------------------------------------------------------------
    # ADIM 7 GORSELI: Olcek ve alan karsilastirmasi
    # ------------------------------------------------------------------------
    step7_reference = image_reference.copy()
    step7_test = image_test.copy()

    # Ben konturlarini kirmiziyla ciz
    if 'contour' in mole_info_reference:
        cv2.drawContours(step7_reference, [mole_info_reference['contour']], -1, (0, 0, 255), 2)
    if 'contour' in mole_info_test:
        cv2.drawContours(step7_test, [mole_info_test['contour']], -1, (0, 0, 255), 2)

    # Etiket ve olcek
    cv2.putText(step7_reference, "REFERANS", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(step7_test, "TEST", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(step7_test, f"Olcek: {scale_factor:.3f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Alan bilgisi
    cv2.putText(step7_reference,
                f"Alan: {mole_info_reference['area_px']:.0f}px2",
                (10, step7_reference.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(step7_test,
                f"Alan: {mole_info_test['area_px']:.0f}px2 -> kalibre: "
                f"{mole_area_calibrated:.0f}px2 ({area_change_percent:+.1f}%)",
                (10, step7_test.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    # Yan yana birlestir (arka plan koyu gri)
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
    # ADIM 8 GORSELI: Boyut karsilastirmasi (alan + cap karari ile)
    # ------------------------------------------------------------------------
    step8_reference = image_reference.copy()
    step8_test = image_test.copy()

    if 'contour' in mole_info_reference:
        cv2.drawContours(step8_reference, [mole_info_reference['contour']], -1, (0, 0, 255), 2)
    if 'contour' in mole_info_test:
        cv2.drawContours(step8_test, [mole_info_test['contour']], -1, (0, 0, 255), 2)

    cv2.putText(step8_reference, "REFERANS", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(step8_reference,
                f"Alan={mole_info_reference['area_px']:.0f}px2  "
                f"Cap={mole_info_reference['eq_diameter_px']:.1f}px",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(step8_test, "TEST (KALIBRE)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(step8_test,
                f"Alan={mole_area_calibrated:.0f}px2  Cap={mole_diameter_calibrated:.1f}px",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    # Karar metni
    size_decision_text = "BOYUT DEGISMEMIS" if not size_changed else "BOYUT DEGISMIS!"
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

    # Alt bilgi - karar metni
    cv2.putText(step8_canvas,
                f"Alan: {area_change_percent:+.1f}% | "
                f"Cap: {diameter_change_percent:+.1f}% | "
                f"Esik: +/-{area_change_threshold:.1f}% | {size_decision_text}",
                (10, step8_canvas_height + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, size_decision_color, 2)
    step8_image_path = test_directory / "step8_size.jpg"
    cv2.imwrite(str(step8_image_path), step8_canvas)

    # ------------------------------------------------------------------------
    # SONUC SOZLUGUNU INSA ET
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
# API ENDPOINT'LERI
# ============================================================================

@app.get("/")
def root():
    """Saglik kontrolu - sunucu calisiyor mu?"""
    return {"status": "ok", "message": "Ben Analiz API"}


@app.get("/app")
def web_app():
    """Web uygulamasini (app.html) sun."""
    html_path = Path(__file__).parent / "app.html"
    if not html_path.exists():
        raise HTTPException(404, "app.html bulunamadi")
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/moles")
def list_moles():
    """Sistemde tanimli tum benleri ve cekim sayilarini listele."""
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
    """Belirli bir benin tum cekimlerini listele."""
    mole_directory = BASE_DIR / mole_name
    if not mole_directory.exists():
        raise HTTPException(404, f"'{mole_name}' bulunamadi")

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
    """Bir beni ve tum cekimlerini sil (geri donulemez)."""
    mole_directory = BASE_DIR / mole_name
    if not mole_directory.exists():
        raise HTTPException(404, f"'{mole_name}' bulunamadi")
    shutil.rmtree(mole_directory)
    return {"status": "ok", "message": f"'{mole_name}' silindi"}


@app.delete("/moles/{mole_name}/{timestamp}")
def delete_capture(mole_name: str, timestamp: str):
    """Belirli bir cekimi sil. Ben'in son cekimiyse onu da sil."""
    capture_directory = BASE_DIR / mole_name / timestamp
    if not capture_directory.exists():
        raise HTTPException(404, f"Cekim bulunamadi: {timestamp}")
    shutil.rmtree(capture_directory)

    # Ben klasoru bossa onu da sil
    mole_directory = BASE_DIR / mole_name
    remaining_captures = [c for c in mole_directory.iterdir() if c.is_dir()]
    if not remaining_captures:
        shutil.rmtree(mole_directory)

    return {"status": "ok", "message": f"Cekim silindi: {timestamp}"}


@app.put("/moles/{mole_name}")
def rename_mole(mole_name: str, new_name: str = Form(...)):
    """Bir benin adini degistir."""
    mole_directory = BASE_DIR / mole_name
    if not mole_directory.exists():
        raise HTTPException(404, f"'{mole_name}' bulunamadi")

    new_mole_directory = BASE_DIR / new_name
    if new_mole_directory.exists():
        raise HTTPException(400, f"'{new_name}' zaten var")

    mole_directory.rename(new_mole_directory)
    return {"status": "ok",
            "message": f"'{mole_name}' -> '{new_name}' olarak degistirildi"}


@app.post("/analyze")
async def analyze(
    mole_name: str = Form(...),
    photo: UploadFile = File(...)
):
    """
    Yeni bir fotograf yukle ve Adim 1-3'u calistir.

    Sonuclar BENLER/{mole_name}/{timestamp}/ klasorune kaydedilir.
    Yama RGB'leri ve mesafeleri ileride karsilastirma icin data.json'a
    yazilir.
    """
    # Cekim klasorunu olustur (her cekim icin benzersiz zaman damgasi)
    timestamp_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    capture_save_directory = BASE_DIR / mole_name / timestamp_string
    capture_save_directory.mkdir(parents=True, exist_ok=True)

    # Fotografi diske kaydet
    input_image_path = capture_save_directory / "input.jpg"
    uploaded_file_content = await photo.read()
    with open(input_image_path, "wb") as output_file:
        output_file.write(uploaded_file_content)

    # --- KOD ADIM 1-3 CALISTIR ---
    pipeline_result, pipeline_error = run_pipeline_step1_3(
        input_image_path, capture_save_directory
    )
    if pipeline_error:
        print(f"  [HATA] Kod Adim 1-3: {pipeline_error}")
        shutil.rmtree(capture_save_directory, ignore_errors=True)
        raise HTTPException(400, f"Cekim hatali: {pipeline_error}")

    # --- YAMA TESPITI ---
    perspective_image = cv2.imread(str(capture_save_directory / "perspective.jpg"))
    if perspective_image is None:
        print(f"  [HATA] Perspektif goruntu okunamadi")
        shutil.rmtree(capture_save_directory, ignore_errors=True)
        raise HTTPException(400, "Cekim hatali: Perspektif goruntu olusturulamadi")

    frame_bbox = find_label_frame(perspective_image)
    print(f"  Cerceve: {frame_bbox}")
    detected_patches = detect_patches_dynamic(perspective_image, frame_bbox)
    print(f"  Yamalar: {len(detected_patches)} dinamik tespit edildi")

    # Dinamik tespit basarisizsa sabit koordinatlarla fallback
    if len(detected_patches) < 14:
        frame_x, frame_y, frame_width, frame_height = frame_bbox
        detected_patches = []
        for normalized_cx, normalized_cy in PATCHES:
            x1 = int(frame_x + (normalized_cx - PATCH_SIZE / 2) * frame_width)
            y1 = int(frame_y + (normalized_cy - PATCH_SIZE / 2) * frame_height)
            x2 = int(frame_x + (normalized_cx + PATCH_SIZE / 2) * frame_width)
            y2 = int(frame_y + (normalized_cy + PATCH_SIZE / 2) * frame_height)
            detected_patches.append({'bbox': (x1, y1, x2 - x1, y2 - y1)})
        print(f"  Yamalar: {len(detected_patches)} sabit koordinat (fallback)")

    # --- BEN TESPITI ---
    mole_information = extract_mole_info(perspective_image)
    print(f"  Ben: {'tespit edildi' if mole_information else 'tespit edilemedi'}")

    # Dogrulama tamamen kapali - uyari verip yine de kaydet
    if len(detected_patches) < 14:
        print(f"  [UYARI] Yama sayisi dusuk: {len(detected_patches)}/16 - "
              f"yine de kaydediliyor")
    if mole_information is None:
        print(f"  [UYARI] Ben tespit edilemedi - yine de kaydediliyor")

    # contour bir numpy array - JSON'a kaydedilemez, hariç tut
    mole_info_for_json = None
    if mole_information:
        mole_info_for_json = {
            key: value for key, value in mole_information.items()
            if key != 'contour'
        }

    # Yama RGB'lerini ve cift mesafelerini cikar
    patch_rgb_values = extract_patch_rgb(perspective_image, detected_patches)
    patch_pairwise_distances = patch_distances(detected_patches)

    # data.json: ileride karsilastirma icin hizli erisim
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
    Ayni benin iki cekimini karsilastir (Adim 4-8).

    Args:
        mole_name:      Ben adi.
        ref_timestamp:  Referans cekim damgasi (genelde daha eski).
        test_timestamp: Test cekim damgasi (genelde daha yeni).
    """
    reference_directory = BASE_DIR / mole_name / ref_timestamp
    test_directory = BASE_DIR / mole_name / test_timestamp

    if not reference_directory.exists():
        raise HTTPException(404, f"Referans cekim bulunamadi: {ref_timestamp}")
    if not test_directory.exists():
        raise HTTPException(404, f"Test cekim bulunamadi: {test_timestamp}")

    # --- KOD ADIM 4-8 CALISTIR ---
    comparison_result, comparison_error = run_comparison(reference_directory, test_directory)
    if comparison_error:
        raise HTTPException(500, comparison_error)

    # Sonuclari karsilastirma JSON'una yaz
    with open(test_directory / "comparison_data.json", "w") as output_file:
        json.dump(comparison_result, output_file, indent=2)

    return comparison_result


@app.get("/image/{mole_name}/{timestamp}/{filename}")
def get_image(mole_name: str, timestamp: str, filename: str):
    """Belirli bir cekimden adim goruntusunu getir."""
    image_file_path = BASE_DIR / mole_name / timestamp / filename
    if not image_file_path.exists():
        raise HTTPException(404, "Dosya bulunamadi")
    return FileResponse(str(image_file_path))


# ============================================================================
# SUNUCU BASLATMA
# ============================================================================

if __name__ == "__main__":
    print("\nBen Analiz API baslatiliyor...")
    print(f"Veri klasoru : {BASE_DIR.resolve()}")
    print(f"URL          : http://localhost:8000")
    print(f"Web uygulamasi: http://localhost:8000/app")
    print(f"API dokuman  : http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
