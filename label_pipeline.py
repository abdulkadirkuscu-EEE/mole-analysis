"""
============================================================================
CILT LEZYONU (BEN) ANALIZ KODU - GORUNTU ISLEME MODULU
============================================================================
Bu modul, kalibrasyon karti uzerine konulmus cilt lezyonu (ben) fotograflarini
isleyen tum goruntu isleme fonksiyonlarini icerir.

KOD ADIMLARI (8 adim):
  Adim 1: BW (siyah-beyaz) goruntu uretimi
  Adim 2: Etiket (kalibrasyon karti) sinir kutusu tespiti
  Adim 3: Perspektif duzeltme
  Adim 4: Iki cekim arasi renk kaymasi (drift) tespiti
  Adim 5: RGB kanal kalibrasyonu (en kucuk kareler)
  Adim 6: Ben rengi degisim tespiti (CIEDE2000)
  Adim 7: Olcek faktoru hesabi (yamalar arasi mesafelerden)
  Adim 8: Ben boyut degisim tespiti

Adim 1-3 bu modulde yer alir.
Adim 4-8 api_server.py icinde, bu modulun yardimci fonksiyonlariyla calisir.

CALISTIRMA:
  Tek basina calistirilirsa: input_images/ klasorunu izler ve gelen
  her yeni resmi otomatik olarak isler (Adim 1-3).
============================================================================
"""

import cv2
import numpy as np
import time
import shutil
from pathlib import Path

# ============================================================================
# AYARLAR - Calismayi etkileyen sabitler
# ============================================================================

# Girdi/cikti klasor adlari (tek basina calistirildiginda kullanilir)
INPUT_FOLDER  = "input_images"          # Orijinal fotograflarin bulundugu klasor
BW_FOLDER     = "bw_images"             # Adim 1 ciktilari: siyah-beyaz goruntuler
OUTPUT_FOLDER = "output_images"         # Adim 2 ciktilari: bbox disi beyazlatilmis
CROP_FOLDER   = "crop_images"           # Kirpilmis goruntuler
PERSP_FOLDER  = "perspective_images"    # Adim 3 ciktilari: perspektif duzeltilmis
COLOR_FOLDER  = "color_square_images"   # Tespit edilen renk yamalari isaretli

# Desteklenen goruntu uzantilari (klasor izleme modu icin)
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Klasor izleme modunda kontrol araligi (saniye)
POLL_INTERVAL = 2

# ----------------------------------------------------------------------------
# Kalibrasyon karti uzerindeki sabit noktalar (normalize koordinat: 0..1)
# Bu degerler bizim hazirladigimiz sablon kalibrasyon kartindan olculmustur.
# Karttaki 16 renk yamasi U-sekli bir yerlesim olusturur:
#   - Ust sirada 4 yatay yama (sol -> sag)
#   - Alt sirada 4 yatay yama (sol -> sag)
#   - Sol sutunda 4 dikey yama (yukari -> asagi)
#   - Sag sutunda 4 dikey yama (yukari -> asagi)
# Ortadaki bos alan ise lezyonun (benin) durdugu bolgedir.
# ----------------------------------------------------------------------------
PATCHES = [
    # Ust sira - sol'dan saga 4 yama
    (0.370, 0.269), (0.456, 0.269), (0.539, 0.268), (0.622, 0.269),
    # Alt sira - sol'dan saga 4 yama
    (0.374, 0.756), (0.458, 0.760), (0.540, 0.756), (0.622, 0.755),
    # Sol sutun - yukaridan asagi 4 yama
    (0.254, 0.386), (0.258, 0.471), (0.255, 0.553), (0.258, 0.634),
    # Sag sutun - yukaridan asagi 4 yama
    (0.750, 0.385), (0.749, 0.468), (0.753, 0.549), (0.749, 0.633),
]

# Her bir yama bolgesinin kenar uzunlugu (normalize edilmis kart genisliginin %)
PATCH_SIZE = 0.075

# Ben (lezyon) bolgesinin tahmini merkezi (kart uzerinde normalize koordinat)
LESION_CENTER = (0.497, 0.520)

# Ben (lezyon) bolgesinin tahmini boyutu (kart kenarina oran)
LESION_SIZE = 0.245

# Ic cerceve ile dis cerceve arasi olcek orani
# Kart sablonu olculurken bulundu: ic_kare_kenar / dis_kare_kenar = 0.751
INNER_FRAME_RATIO = 0.751


# ============================================================================
# CIELAB RENK UZAYI + CIEDE2000 RENK FARKI
# ----------------------------------------------------------------------------
# Insan gozunun algiladigi renk farkini olcmek icin RGB'den daha dogru bir
# yontemdir. ΔE2000 < 1 ise fark insan gozune fark edilmez; 2-10 arasinda ise
# bir bakista gorulebilen bir farktir. Cilt lezyonunda renk degisikliginin
# klinik anlami buyuk oldugundan ΔE2000 esiklerimizi muhafazakar tutuyoruz.
# ============================================================================

def rgb_to_lab(rgb_color):
    """
    sRGB (0-255) renk uzayindan CIELAB uzayina donusum yapar.

    Adimlar:
      1) sRGB gamma kaldirma (lineer RGB elde edilir)
      2) Lineer RGB -> XYZ (sRGB->XYZ matrisi, D65 beyaz nokta)
      3) XYZ -> LAB (D65 beyaz nokta referansi, kup-kok donusumu)

    Args:
        rgb_color: 3 elemanli (R, G, B) dizi, her kanal 0-255 araliginda.

    Returns:
        3 elemanli numpy dizisi: [L*, a*, b*]
    """
    rgb_normalized = np.asarray(rgb_color, dtype=np.float64) / 255.0

    # 1) sRGB gamma kaldirma: nonlinear sRGB -> lineer RGB
    # Esik 0.04045 sRGB standardindandir
    def linearize_srgb(channel):
        return np.where(channel > 0.04045,
                        ((channel + 0.055) / 1.055) ** 2.4,
                        channel / 12.92)

    r_lin = linearize_srgb(rgb_normalized[..., 0])
    g_lin = linearize_srgb(rgb_normalized[..., 1])
    b_lin = linearize_srgb(rgb_normalized[..., 2])

    # 2) Lineer RGB -> XYZ: sRGB D65 standart matrisi
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    # 3) XYZ -> LAB: D65 beyaz nokta normalizasyonu
    white_x, white_y, white_z = 0.95047, 1.00000, 1.08883
    fx_arg, fy_arg, fz_arg = x / white_x, y / white_y, z / white_z

    # CIE LAB nonlinear donusumu (kup-kok / lineer kombinasyonu)
    def lab_transform(t):
        # Esik 0.008856 = (6/29)^3, CIE standardi
        return np.where(t > 0.008856,
                        np.cbrt(t),
                        7.787 * t + 16.0 / 116.0)

    fx_val = lab_transform(fx_arg)
    fy_val = lab_transform(fy_arg)
    fz_val = lab_transform(fz_arg)

    # L*, a*, b* hesabi
    lightness = 116.0 * fy_val - 16.0
    a_axis = 500.0 * (fx_val - fy_val)
    b_axis = 200.0 * (fy_val - fz_val)

    return np.stack([lightness, a_axis, b_axis], axis=-1)


def delta_e_ciede2000(lab1, lab2):
    """
    Iki LAB rengi arasindaki CIEDE2000 renk farkini (ΔE00) hesaplar.

    CIEDE2000, CIE'nin tavsiye ettigi en gelismis renk fark formuludur.
    Insan gozunun algi egrisine en yakin sonucu verir.

    Args:
        lab1: Birinci rengin [L*, a*, b*] dizisi.
        lab2: Ikinci rengin [L*, a*, b*] dizisi.

    Returns:
        ΔE2000 degeri (float). Tipik yorumlama:
          < 1.0 : Fark insan gozune fark edilmez
          1-2   : Yakindan bakildiginda fark edilir
          2-10  : Bir bakista fark edilir
          > 10  : Belirgin fark
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    # L ve C (kroma) ortalamalari
    avg_lightness = (L1 + L2) / 2.0
    chroma1 = np.sqrt(a1 ** 2 + b1 ** 2)
    chroma2 = np.sqrt(a2 ** 2 + b2 ** 2)
    avg_chroma = (chroma1 + chroma2) / 2.0

    # G faktoru: dusuk kroma duzeltmesi
    g_factor = 0.5 * (1 - np.sqrt(avg_chroma ** 7 / (avg_chroma ** 7 + 25 ** 7)))
    a1_prime = (1 + g_factor) * a1
    a2_prime = (1 + g_factor) * a2

    chroma1_prime = np.sqrt(a1_prime ** 2 + b1 ** 2)
    chroma2_prime = np.sqrt(a2_prime ** 2 + b2 ** 2)
    avg_chroma_prime = (chroma1_prime + chroma2_prime) / 2.0

    # Hue acilari (derece)
    hue1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    hue2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360

    # Fark bilesenleri
    delta_lightness_prime = L2 - L1
    delta_chroma_prime = chroma2_prime - chroma1_prime

    # Hue farki (-180..+180 araligina indirgenir)
    delta_hue = hue2_prime - hue1_prime
    if abs(delta_hue) > 180:
        delta_hue -= 360 * np.sign(delta_hue)
    delta_hue_prime = 2 * np.sqrt(chroma1_prime * chroma2_prime) * np.sin(np.radians(delta_hue) / 2.0)

    # Ortalama hue
    avg_hue_prime = hue1_prime + hue2_prime
    if abs(hue1_prime - hue2_prime) > 180:
        avg_hue_prime += 360
    avg_hue_prime /= 2.0

    # T agirlik fonksiyonu (hue duzeltmesi)
    t_weight = (1 - 0.17 * np.cos(np.radians(avg_hue_prime - 30))
                  + 0.24 * np.cos(np.radians(2 * avg_hue_prime))
                  + 0.32 * np.cos(np.radians(3 * avg_hue_prime + 6))
                  - 0.20 * np.cos(np.radians(4 * avg_hue_prime - 63)))

    # Donus faktoru (mavi-mor bolgede ekstra duzeltme)
    rotation_angle = 30 * np.exp(-(((avg_hue_prime - 275) / 25) ** 2))
    chroma_correction = 2 * np.sqrt(avg_chroma_prime ** 7 / (avg_chroma_prime ** 7 + 25 ** 7))

    # Olcekleme katsayilari (Sl, Sc, Sh)
    s_lightness = 1 + (0.015 * (avg_lightness - 50) ** 2) / np.sqrt(20 + (avg_lightness - 50) ** 2)
    s_chroma = 1 + 0.045 * avg_chroma_prime
    s_hue = 1 + 0.015 * avg_chroma_prime * t_weight
    rotation_term = -np.sin(np.radians(2 * rotation_angle)) * chroma_correction

    # Final ΔE2000 hesabi
    delta_e = np.sqrt(
        (delta_lightness_prime / s_lightness) ** 2
        + (delta_chroma_prime / s_chroma) ** 2
        + (delta_hue_prime / s_hue) ** 2
        + rotation_term * (delta_chroma_prime / s_chroma) * (delta_hue_prime / s_hue)
    )
    return float(delta_e)


# ============================================================================
# YARDIMCI KLASOR ISLEMLERI
# ============================================================================

def clear_folders():
    """
    input_images haric tum cikti klasorlerini siler ve yeniden olusturur.
    Klasor izleme moduyla calistirildiginda her acilista cagrilir.
    """
    folders_to_clear = [BW_FOLDER, OUTPUT_FOLDER, CROP_FOLDER, PERSP_FOLDER, COLOR_FOLDER]
    for folder_name in folders_to_clear:
        folder_path = Path(folder_name)
        if folder_path.exists():
            shutil.rmtree(folder_path)
        folder_path.mkdir(exist_ok=True)
        print(f"  Temizlendi: {folder_path.resolve()}")


# ============================================================================
# ADIM 1: BW (SIYAH-BEYAZ) GORUNTU URETIMI
# ----------------------------------------------------------------------------
# Amac: Etiketi (kalibrasyon kartini) on plana cikaran bir maske olusturmak.
# Yontem: Kademeli esikleme + dikdortgen kontur bulma + ROI icinde histogram
#         tabanli ince esikleme.
# ============================================================================

def generate_bw(input_image_path, output_bw_path):
    """
    Verilen RGB fotograftan etiketi (kalibrasyon kartini) izole eden BW
    (siyah-beyaz) goruntu uretir.

    Yontem:
      1) Birden fazla esik degeri denenir (80, 100, 70).
      2) Her birinde 4-noktali dikdortgen konturlar aranir.
      3) En buyuk gecerli dikdortgen sectikten sonra etrafini kirpip
         o bolge icin tekrar histogram-bazli ince esikleme yapilir.
      4) Bu sayede etiket bolgesi kara, geri kalan tum alan beyaz olur.

    Args:
        input_image_path: Orijinal fotografin yolu.
        output_bw_path:   BW goruntunun kaydedilecegi yol.

    Returns:
        True: Basarili. False: Goruntu okunamadi.
    """
    original_image = cv2.imread(str(input_image_path))
    if original_image is None:
        return False

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    image_height, image_width = gray_image.shape

    best_bbox = None
    best_bbox_area = 0

    # Birden fazla esik dene; kart farkli aydinlatmalarda farkli koyulukta gorunebilir
    for threshold_value in [80, 100, 70]:
        # Ters esikleme: kart koyu, arka plan parlak ise bu ise yarar
        _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour_area = cv2.contourArea(contour)
            # Cok kucuk (gurultu) veya cok buyuk (tum kadr) konturlari atla
            if contour_area < gray_image.size * 0.005 or contour_area > gray_image.size * 0.4:
                continue

            perimeter = cv2.arcLength(contour, True)

            # Farkli epsilon degerleriyle 4-koseli yaklasim ara
            for epsilon_ratio in [0.03, 0.04, 0.05, 0.06]:
                approx_polygon = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)

                # Sadece dortgen sekilleri kabul et
                if len(approx_polygon) == 4:
                    bbox_x, bbox_y, bbox_width, bbox_height = cv2.boundingRect(approx_polygon)

                    # En-boy orani 0.8'den buyukse "kareye yakin" demektir
                    aspect_ratio = (min(bbox_width, bbox_height) / max(bbox_width, bbox_height)
                                    if max(bbox_width, bbox_height) > 0 else 0)
                    bbox_area = bbox_width * bbox_height

                    if aspect_ratio > 0.8 and bbox_area > best_bbox_area:
                        best_bbox_area = bbox_area
                        best_bbox = (bbox_x, bbox_y, bbox_width, bbox_height)
                    # Bu konturda bir dortgen bulunduysa diger epsilonlari atla
                    break

    # En iyi bbox bulunduysa onun etrafini biraz pay birakarak ROI olarak al
    if best_bbox is not None:
        bbox_x, bbox_y, bbox_width, bbox_height = best_bbox
        padding = int(max(bbox_width, bbox_height) * 0.15)
        roi_x1 = max(0, bbox_x - padding)
        roi_y1 = max(0, bbox_y - padding)
        roi_x2 = min(image_width, bbox_x + bbox_width + padding)
        roi_y2 = min(image_height, bbox_y + bbox_height + padding)
    else:
        # Hicbir uygun bbox bulunamadiysa tum goruntuyu ROI kabul et
        roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, image_width, image_height

    # ROI icinde histogram tabanli ince esikleme:
    # Toplam piksellerin %15'ini "kara" yapacak esigi bul.
    roi_gray = gray_image[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_histogram = cv2.calcHist([roi_gray], [0], None, [256], [0, 256]).flatten()
    cumulative_histogram = np.cumsum(roi_histogram)
    roi_threshold = int(np.searchsorted(cumulative_histogram, roi_gray.size * 0.15))

    # ROI icinde esik altindaki pikselleri 0, ustundekileri 255 yap
    roi_bw = np.where(roi_gray < roi_threshold, 0, 255).astype(np.uint8)

    # Tam boyutlu BW goruntu: ROI disi beyaz
    bw_full_image = np.ones_like(gray_image) * 255
    bw_full_image[roi_y1:roi_y2, roi_x1:roi_x2] = roi_bw

    cv2.imwrite(str(output_bw_path), bw_full_image)
    return True


# ============================================================================
# ADIM 2: ETIKET BOUNDING BOX TESPITI
# ============================================================================

def get_bbox_from_bw(bw_image):
    """
    BW goruntudeki en iyi etiket adayi siyah blogunu bulup bounding box'ini
    dondurur.

    Skor formulu (yuksek = daha iyi):
      score = 0.3 * (alan_orani) + 0.6 * (en-boy_orani) + 0.1 * (merkeze_yakinlik)

    Args:
        bw_image: BW (siyah-beyaz) goruntu (gri-tonlamali numpy dizisi).

    Returns:
        (x, y, w, h) tuple veya None.
    """
    image_height, image_width = bw_image.shape

    # Siyah pikselleri 255'e cevir (connectedComponents pozitif maske bekler)
    black_pixel_mask = (bw_image < 128).astype(np.uint8) * 255

    # 8-bagli bileskenleri bul (alan, bbox, merkez)
    num_components, label_image, stats, centroids = cv2.connectedComponentsWithStats(
        black_pixel_mask, connectivity=8
    )

    # Goruntu merkezi (kart genelde fotografin ortasinda olur)
    image_center_x = image_width // 2
    image_center_y = image_height // 2

    best_bbox = None
    best_score = -1

    # 0. bilesken arka plandir, atla
    for component_index in range(1, num_components):
        component_area = stats[component_index, cv2.CC_STAT_AREA]

        # Cok kucuk bilesenleri (gurultu) atla
        if component_area < bw_image.size * 0.001:
            continue

        bbox_x = stats[component_index, 0]
        bbox_y = stats[component_index, 1]
        bbox_width = stats[component_index, 2]
        bbox_height = stats[component_index, 3]
        bbox_area = bbox_width * bbox_height

        # En-boy orani (1'e yakin = kareye yakin = istenen)
        aspect_ratio = (min(bbox_width, bbox_height) / max(bbox_width, bbox_height)
                        if max(bbox_width, bbox_height) > 0 else 0)

        # Bilesen merkezinin goruntu merkezine olan normalize mesafesi
        component_center_x, component_center_y = centroids[component_index]
        distance_to_center = (np.sqrt((component_center_x - image_center_x) ** 2
                                       + (component_center_y - image_center_y) ** 2)
                               / np.sqrt(image_center_x ** 2 + image_center_y ** 2))

        # Birlesik skor: buyuk, kareye yakin ve merkeze yakin tercih edilir
        score = ((bbox_area / bw_image.size) * 0.3
                 + aspect_ratio * 0.6
                 + (1 - distance_to_center) * 0.1)

        if score > best_score:
            best_score = score
            best_bbox = (int(bbox_x), int(bbox_y), int(bbox_width), int(bbox_height))

    return best_bbox


# ============================================================================
# ADIM 3: PERSPEKTIF DUZELTME
# ============================================================================

def order_points(quadrilateral_points):
    """
    4 noktalik bir dortgenin koselerini saat yonunde, sol-ust kose ilk
    olacak sekilde siralar.

    Yontem: Once aci ile saat yonunde sirala, sonra (x+y) en kucuk olan
    nokta sol-ust kose oldugu icin onunla baslat.

    Args:
        quadrilateral_points: 4 nokta iceren numpy dizisi, sekil (4, 2).

    Returns:
        Siralanmis 4 nokta, dtype=float32, sekil (4, 2).
    """
    geometric_center = quadrilateral_points.mean(axis=0)

    # Her noktanin merkeze gore acisini hesapla
    angles = np.arctan2(quadrilateral_points[:, 1] - geometric_center[1],
                        quadrilateral_points[:, 0] - geometric_center[0])
    sorted_order = np.argsort(angles)
    sorted_points = quadrilateral_points[sorted_order]

    # (x+y) en kucuk olan nokta sol-ust koseyi gosterir
    sum_xy = sorted_points.sum(axis=1)
    top_left_index = np.argmin(sum_xy)

    # Sol-ust koseyi ilk yapacak sekilde diziyi kaydir
    ordered_rectangle = np.roll(sorted_points, -top_left_index, axis=0).astype(np.float32)
    return ordered_rectangle


def perspective_correct(bw_image, full_color_image, label_bbox, output_path):
    """
    BW goruntude tespit edilen kart dortgenini bularak orijinal renkli
    goruntuyu kareye duzlestirir (perspektif duzeltme).

    Adimlar:
      1) BW'den 4-koseli kontur ara (ve fallback olarak dilate edilmis maske).
      2) Bulunan 4 koseyi saat yonunde sirala.
      3) Kareye (side x side) cevirmek icin perspektif matrisi hesapla.
      4) warpPerspective ile orijinal renkli goruntuyu duzlestir.

    Args:
        bw_image:        Siyah-beyaz goruntu.
        full_color_image: Orijinal BGR goruntu (kaynagi olusturulan goruntu).
        label_bbox:       Adim 2'de bulunan etiket bbox'i (x, y, w, h).
        output_path:      Duzeltilmis goruntunun kaydedilecegi yol.

    Returns:
        True: Basarili. False: Cikti boyutu cok kucuk (50px alti).
    """
    bbox_x, bbox_y, bbox_width, bbox_height = label_bbox
    target_area = bbox_width * bbox_height
    bbox_center_x = bbox_x + bbox_width / 2
    bbox_center_y = bbox_y + bbox_height / 2

    def find_quadrilateral(binary_mask):
        """Verilen binary maskede en iyi 4-koseli konturu bulur."""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_quadrilateral = None
        best_area_diff = float('inf')

        for contour in contours:
            contour_area = cv2.contourArea(contour)
            # Hedef alanin yarisindan kucukse atla
            if contour_area < target_area * 0.5:
                continue

            perimeter = cv2.arcLength(contour, True)

            # Farkli epsilon degerleri dene
            for epsilon_ratio in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
                approx_polygon = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)

                if len(approx_polygon) == 4:
                    bx, by, bw, bh = cv2.boundingRect(approx_polygon)
                    aspect_ratio = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
                    approx_bbox_area = bw * bh

                    # Quadrilateralin merkezi
                    quad_center_x = bx + bw / 2
                    quad_center_y = by + bh / 2
                    center_distance = np.sqrt((quad_center_x - bbox_center_x) ** 2
                                              + (quad_center_y - bbox_center_y) ** 2)
                    area_difference = abs(approx_bbox_area - target_area) / target_area

                    # En iyi aday: kareye yakin + merkeze yakin + benzer alanli
                    if (aspect_ratio > 0.8
                            and center_distance < max(bbox_width, bbox_height) * 0.3
                            and area_difference < best_area_diff):
                        best_area_diff = area_difference
                        best_quadrilateral = approx_polygon.reshape(4, 2).astype(np.float32)
                    break
        return best_quadrilateral

    # Once dogrudan siyah maske ile dene
    black_mask = (bw_image < 128).astype(np.uint8) * 255
    best_corner_points = find_quadrilateral(black_mask)

    # Bulunamadiysa dilate edilmis maske dene (kucuk kopukluklari kapatir)
    if best_corner_points is None:
        dilation_kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(black_mask, dilation_kernel, iterations=2)
        best_corner_points = find_quadrilateral(dilated_mask)

    # Hala bulunamadiysa bbox'in koselerini kullan (son care)
    if best_corner_points is None:
        best_corner_points = np.array(
            [[bbox_x, bbox_y],
             [bbox_x + bbox_width, bbox_y],
             [bbox_x + bbox_width, bbox_y + bbox_height],
             [bbox_x, bbox_y + bbox_height]],
            dtype=np.float32
        )

    # Koseleri saat yonunde, sol-ust ilk olacak sekilde sirala
    ordered_corners = order_points(best_corner_points)

    # Kare cikti boyutunu belirle: 4 kenarin maksimumu
    side_length = int(max(
        np.linalg.norm(ordered_corners[1] - ordered_corners[0]),
        np.linalg.norm(ordered_corners[2] - ordered_corners[3]),
        np.linalg.norm(ordered_corners[3] - ordered_corners[0]),
        np.linalg.norm(ordered_corners[2] - ordered_corners[1])
    ))

    # 50px altinda anlamli olmaz - basarisiz say
    if side_length < 50:
        return False

    # Hedef kare koordinatlari (sol-ust ilk, saat yonunde)
    destination_points = np.array(
        [[0, 0],
         [side_length - 1, 0],
         [side_length - 1, side_length - 1],
         [0, side_length - 1]],
        dtype=np.float32
    )

    # Perspektif matrisi hesapla ve uygula
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
# PERSPEKTIF SONRASI: ETIKET CERCEVE TESPITI
# ----------------------------------------------------------------------------
# Perspektif duzeltme sonrasi goruntu icindeki kartin gercek "ic+dis cerceve"
# bolgesini bulur. Bu bolge sonradan yama tespiti icin referans alanir.
# ============================================================================

def find_label_frame(perspective_image):
    """
    Perspektif duzeltilmis goruntude etiketin dis cerceve bbox'ini bulur.

    Iki yontem denenir (siralamayla):
      1) Yuksek esik + dilate ile birlesik dis cerceveyi tek kontur olarak ara.
      2) Bulamazsa, ic kareyi bul ve bilinen IC/DIS oranindan dis cerceveyi
         geriye dogru hesapla.

    Args:
        perspective_image: Perspektif duzeltilmis BGR goruntu.

    Returns:
        (x, y, w, h): Dis cerceve bbox'i.
    """
    image_height, image_width = perspective_image.shape[:2]
    gray_image = cv2.cvtColor(perspective_image, cv2.COLOR_BGR2GRAY)

    # --- YONTEM 1: Dilate ile birlesik dis cerceve ---
    _, binary_image = cv2.threshold(gray_image, 130, 255, cv2.THRESH_BINARY_INV)
    dilation_kernel = np.ones((10, 10), np.uint8)
    dilated_binary = cv2.dilate(binary_image, dilation_kernel, iterations=3)

    contours, _ = cv2.findContours(dilated_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    image_area = image_width * image_height
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        # Goruntunun %10-95'i arasi alanlar makul cerceve adaylaridir
        if contour_area > image_area * 0.1 and contour_area < image_area * 0.95:
            bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)
            aspect_ratio = min(bbox_w, bbox_h) / max(bbox_w, bbox_h)
            if aspect_ratio > 0.85:
                return (bbox_x, bbox_y, bbox_w, bbox_h)

    # --- YONTEM 2: Ic kareyi bul, oranla dis cerceveyi hesapla ---
    _, binary_low_thresh = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY_INV)
    inner_contours, _ = cv2.findContours(binary_low_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quadrilateral_candidates = []

    for contour in inner_contours:
        contour_area = cv2.contourArea(contour)
        # Ic kare goruntunun %1-70'i araliginda olmali
        if contour_area < image_area * 0.01 or contour_area > image_area * 0.7:
            continue

        perimeter = cv2.arcLength(contour, True)
        for epsilon_ratio in [0.02, 0.03, 0.04, 0.05]:
            approx_polygon = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
            if len(approx_polygon) == 4:
                bx, by, bw, bh = cv2.boundingRect(approx_polygon)
                aspect_ratio = min(bw, bh) / max(bw, bh)
                size_ratio_to_image = max(bw, bh) / image_width

                # Ic kare kart genisliginin yaklasik %60-85'i kadar olmali
                if aspect_ratio > 0.85 and 0.60 < size_ratio_to_image < 0.85:
                    quadrilateral_candidates.append((max(bw, bh), bx, by, bw, bh))
                break

    if not quadrilateral_candidates:
        # Hicbir aday yok - tum goruntuyu cerceve kabul et
        return (0, 0, image_width, image_height)

    # En buyuk ic kareyi sec
    quadrilateral_candidates.sort(reverse=True)
    _, inner_bbox_x, inner_bbox_y, inner_bbox_w, inner_bbox_h = quadrilateral_candidates[0]

    # IC/DIS oranindan dis cerceve boyutlarini hesapla
    outer_frame_width = int(inner_bbox_w / INNER_FRAME_RATIO)
    outer_frame_height = int(inner_bbox_h / INNER_FRAME_RATIO)

    # Ic karenin merkezi dis cervenin de merkezi
    inner_center_x = inner_bbox_x + inner_bbox_w / 2
    inner_center_y = inner_bbox_y + inner_bbox_h / 2

    # 0.487 ve 0.491: sablonu olcerken bulunan ic merkez konumlari
    outer_frame_x = int(inner_center_x - 0.487 * outer_frame_width)
    outer_frame_y = int(inner_center_y - 0.491 * outer_frame_height)

    return (outer_frame_x, outer_frame_y, outer_frame_width, outer_frame_height)


# ============================================================================
# BEN (LEZYON) TESPITI
# ============================================================================

def detect_mole(perspective_image):
    """
    Perspektif goruntude orta bolgede yer alan beni (koyu blob) dinamik
    olarak bulur.

    Algoritma:
      1) Goruntu merkezinde +/-%13'luk bir ROI tanimla.
      2) Birden fazla esik (60, 80, 40, 100, 30) dene.
      3) Her birinde en yuksek (dairesellik * aspect * alan) skorlu blogu sec.

    Args:
        perspective_image: Perspektif duzeltilmis BGR goruntu.

    Returns:
        (x, y, w, h): Ben bbox'i (mutlak koordinatlar) veya None.
    """
    image_height, image_width = perspective_image.shape[:2]
    gray_image = cv2.cvtColor(perspective_image, cv2.COLOR_BGR2GRAY)

    # Goruntu merkezinde tanimlanan arama bolgesi (ROI)
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

    # Birden fazla esik dene - ben farkli koyuluklarda olabilir
    for threshold_value in [60, 80, 40, 100, 30]:
        _, roi_binary = cv2.threshold(roi_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

        # Morfolojik temizleme: kucuk delik/cikintilari kapat
        morph_kernel = np.ones((3, 3), np.uint8)
        roi_binary = cv2.morphologyEx(roi_binary, cv2.MORPH_CLOSE, morph_kernel)
        roi_binary = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, morph_kernel)

        contours, _ = cv2.findContours(roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour_area = cv2.contourArea(contour)

            # Cok kucuk veya ROI'nin yarisindan buyukse atla
            if contour_area < 80 or contour_area > roi_width * roi_height * 0.5:
                continue

            bx, by, bw, bh = cv2.boundingRect(contour)
            aspect_ratio = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0

            # Dairesellik: 4*pi*A/P^2 (mukemmel daire = 1, cizgi = 0)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0

            # Skor: dairesel + kareye yakin + buyuk hepsi tercih edilir
            score = circularity * aspect_ratio * contour_area

            if score > best_score:
                best_score = score
                # ROI yerel koordinatlarindan goruntu mutlak koordinatlarina cevir
                best_mole_bbox = (roi_x1 + bx, roi_y1 + by, bw, bh)

    return best_mole_bbox


# ============================================================================
# RENK YAMASI TESPITI - BANT + PROJEKSIYON YONTEMI
# ============================================================================

def detect_patches_dynamic(perspective_image, frame_bbox):
    """
    Perspektif goruntude 16 renk yamasini dinamik olarak tespit eder.

    Algoritma (Bant + Projeksiyon yontemi):
      1) Doygunluk (S) kanalindan 4 renk yama bandini bul (ust/alt/sol/sag).
      2) Her bant icinde V (parlaklik) kanal projeksiyonu al.
      3) Vadiler (yama arasi bosluklar) ve tepeler arasini segmentlere ayir.
      4) Her segment bir renk yamasi olarak isaretlenir.

    Bu yontem sabit koordinatlardan daha saglamdir cunku perspektif duzeltme
    sonrasi yamalarin tam konumlari hafifce kayabilir.

    Args:
        perspective_image: Perspektif duzeltilmis BGR goruntu.
        frame_bbox:        Etiket dis cerceve bbox'i (x, y, w, h).

    Returns:
        Liste: her elemani {'bbox': (x, y, w, h)} olan sozluklerin listesi.
        Beklenen sayi: 16 (4 bant x 4 yama). Az olabilir.
    """
    frame_x, frame_y, frame_width, frame_height = frame_bbox
    image_height, image_width = perspective_image.shape[:2]

    # HSV uzayina cevir - doygunluk ve parlaklik kanallari analiz edilir
    hsv_image = cv2.cvtColor(perspective_image, cv2.COLOR_BGR2HSV)
    saturation_channel = hsv_image[:, :, 1]
    value_channel = hsv_image[:, :, 2]

    # ------------------------------------------------------------------------
    # 1. ASAMA: DOYGUN BANT TESPITI
    # ------------------------------------------------------------------------
    # Renk yamalari yuksek doygunluga sahip (siyah/beyaz/gri yamalardan farkli
    # olarak renkli). Doygunluk maskesinden 4 dikdortgen bant ararız.
    frame_area = frame_width * frame_height
    min_band_area = frame_area * 0.01      # Frame alaninin %1'i
    max_band_area = frame_area * 0.08      # Frame alaninin %8'i
    min_band_side = max(frame_width, frame_height) * 0.05  # En kisa kenar %5

    detected_bands = []

    # Farkli doygunluk esikleriyle dene (renk yogunlugu degiskendir)
    for saturation_threshold in [50, 40, 30, 60]:
        saturation_mask = (saturation_channel > saturation_threshold).astype(np.uint8) * 255
        mask_contours, _ = cv2.findContours(saturation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_bands = []
        for contour in mask_contours:
            contour_area = cv2.contourArea(contour)
            bx, by, bw, bh = cv2.boundingRect(contour)

            # Bant olabilmesi icin dikdortgenin uzun olmasi lazim
            aspect_long_to_short = max(bw, bh) / min(bw, bh) if min(bw, bh) > 0 else 999

            if (min_band_area < contour_area < max_band_area
                    and 1.8 < aspect_long_to_short < 7.0
                    and max(bw, bh) > min_band_side):
                # Yatay mi (H), dikey mi (V)?
                orientation = "H" if bw > bh else "V"
                detected_bands.append((bx, by, bw, bh, orientation))

        # Tam 4 bant bulduysak diger esikleri denemeye gerek yok
        if len(detected_bands) == 4:
            break

    # 4 bant bulamadiysak basarisiz say
    if len(detected_bands) != 4:
        return []

    # ------------------------------------------------------------------------
    # BANTLARI SINIFLANDIR: UST / ALT / SOL / SAG
    # ------------------------------------------------------------------------
    frame_center_x = frame_x + frame_width / 2
    frame_center_y = frame_y + frame_height / 2

    classified_bands = {'top': None, 'bottom': None, 'left': None, 'right': None}

    for band in detected_bands:
        bx, by, bw, bh, orientation = band
        band_center_x = bx + bw / 2
        band_center_y = by + bh / 2

        if orientation == "H":
            # Yatay bant: y'sine gore ust mu alt mi
            if band_center_y < frame_center_y:
                classified_bands['top'] = band
            else:
                classified_bands['bottom'] = band
        else:
            # Dikey bant: x'ine gore sol mu sag mi
            if band_center_x < frame_center_x:
                classified_bands['left'] = band
            else:
                classified_bands['right'] = band

    # Sira: ust (L->R), alt (L->R), sol (T->B), sag (T->B)
    ordered_bands = [classified_bands['top'],    classified_bands['bottom'],
                     classified_bands['left'],   classified_bands['right']]

    # Bir konum eksikse basarisiz
    if None in ordered_bands:
        return []

    # ------------------------------------------------------------------------
    # 2. ASAMA: HER BANT ICINDE PROJEKSIYON ILE YAMA AYIRMA
    # ------------------------------------------------------------------------
    all_detected_patches = []

    for bx, by, bw, bh, orientation in ordered_bands:
        # Bant ROI'sini hafif genislet (kenardaki yama da girsin)
        roi_margin = 3
        roi_y1 = max(0, by - roi_margin)
        roi_y2 = min(image_height, by + bh + roi_margin)
        roi_x1 = max(0, bx - roi_margin)
        roi_x2 = min(image_width, bx + bw + roi_margin)

        roi_value = value_channel[roi_y1:roi_y2, roi_x1:roi_x2]

        # Bant yonune gore 1D profil al
        # Yatay bantta sutun ortalamasi, dikey bantta satir ortalamasi
        if orientation == "H":
            intensity_profile = np.mean(roi_value, axis=0)
        else:
            intensity_profile = np.mean(roi_value, axis=1)

        # 5-pencereli hareketli ortalama ile yumusatma (gurultu azaltma)
        smoothed_profile = np.convolve(intensity_profile, np.ones(5) / 5, mode='same')

        # Vadi (yama arasi bosluk) esigi: profilin 25. persentili
        valley_threshold = np.percentile(smoothed_profile, 25)

        # ---- VADILERI BUL ----
        # Esik altinda kalan ve en az 2px genis olan bolgeler vadi sayilir
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

        # Profil sonunda hala bir vadi aciksa onu da ekle
        if currently_in_valley and len(intensity_profile) - valley_start_index >= 2:
            valleys.append((valley_start_index, len(intensity_profile)))

        # ---- VADILER ARASINDAKI SEGMENTLER = YAMALAR ----
        segments = []
        previous_segment_end = 0

        for valley_start, valley_end in valleys:
            if valley_start > previous_segment_end + 5:
                segments.append((previous_segment_end, valley_start))
            previous_segment_end = valley_end

        # Son segment (eger profil sonu bir vadiyle bitmiyorsa)
        if previous_segment_end < len(intensity_profile) - 5:
            segments.append((previous_segment_end, len(intensity_profile)))

        # ---- COK FAZLA SEGMENT VARSA BOYUT FILTRESI ----
        # Median'in 0.4-2.5 kati araliginda olanlari koru
        if len(segments) > 4:
            segment_sizes = [end - start for start, end in segments]
            median_segment_size = np.median(segment_sizes)
            segments = [(start, end) for start, end in segments
                        if (end - start) > median_segment_size * 0.4
                        and (end - start) < median_segment_size * 2.5]

        # ---- HER SEGMENT ICIN INCE KIRPMA ----
        # Doygun (S > 40) aktif piksellerle yama kenarlarini netlestir
        for patch_segment_start, patch_segment_end in segments[:4]:
            if orientation == "H":
                # Yatay bantta: x ekseninde segment, y ekseninde tam bant
                patch_x = roi_x1 + patch_segment_start
                patch_y = roi_y1
                patch_width = patch_segment_end - patch_segment_start
                patch_height = roi_y2 - roi_y1

                # Y ekseninde aktif (doygun) bolgeyi bul ve kirp
                segment_saturation = saturation_channel[roi_y1:roi_y2,
                                                        max(0, patch_x):min(image_width, patch_x + patch_width)]
                if segment_saturation.size > 0:
                    vertical_profile = np.mean(segment_saturation, axis=1)
                    active_indices = np.where(vertical_profile > 40)[0]
                    if len(active_indices) > 3:
                        patch_y = roi_y1 + active_indices[0]
                        patch_height = active_indices[-1] - active_indices[0] + 1
            else:
                # Dikey bantta: y ekseninde segment, x ekseninde tam bant
                patch_x = roi_x1
                patch_y = roi_y1 + patch_segment_start
                patch_width = roi_x2 - roi_x1
                patch_height = patch_segment_end - patch_segment_start

                # X ekseninde aktif bolgeyi bul ve kirp
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
# RENK YAMASI VE BEN GORSELLESTIRMESI
# ============================================================================

def detect_color_squares(perspective_image_path, output_visualization_path):
    """
    Perspektif goruntude tespit edilen renk yamalarini ve beni renkli
    isaretlerle bir gorsel uretir.

    Cizimler:
      - Renk yamalari: yesil dikdortgen + numara + merkez daire
      - Ben: mavi dikdortgen + kirmizi Otsu segmentasyon konturu

    Dinamik tespit basarisizsa sabit PATCHES koordinatlarina geri duser.

    Args:
        perspective_image_path:    Adim 3'un cikti yolu.
        output_visualization_path: Olusturulacak gorsellestirmenin yolu.

    Returns:
        True: Basarili. False: Goruntu okunamadi.
    """
    perspective_image = cv2.imread(str(perspective_image_path))
    if perspective_image is None:
        return False

    # Etiket cercevesini bul
    frame_bbox = find_label_frame(perspective_image)
    frame_x, frame_y, frame_width, frame_height = frame_bbox

    # Cikti gorseli orijinalin kopyasi olarak baslar
    output_image = perspective_image.copy()

    # ------------------------------------------------------------------------
    # RENK YAMALARI - DINAMIK TESPIT VEYA FALLBACK
    # ------------------------------------------------------------------------
    detected_patches = detect_patches_dynamic(perspective_image, frame_bbox)

    if 14 <= len(detected_patches) <= 18:
        # Dinamik tespit basarili - tespit edilen kutularda ciz
        for patch_index, patch in enumerate(detected_patches):
            patch_x, patch_y, patch_width, patch_height = patch['bbox']

            # Yesil dis dikdortgen
            cv2.rectangle(output_image,
                          (patch_x, patch_y),
                          (patch_x + patch_width, patch_y + patch_height),
                          (0, 255, 0), 2)
            # Numara etiketi
            cv2.putText(output_image, str(patch_index + 1),
                        (patch_x, patch_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

            # Merkeze yesil daire (RGB ornekleme bolgesi)
            patch_center_x = patch_x + patch_width // 2
            patch_center_y = patch_y + patch_height // 2
            sampling_radius = min(patch_width, patch_height) // 2
            cv2.circle(output_image,
                       (patch_center_x, patch_center_y),
                       sampling_radius, (0, 255, 0), 2)
            cv2.putText(output_image, f"{patch_index + 1}.1",
                        (patch_center_x - 10, patch_center_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 1)

        print(f"    [YAMA] Dinamik: {len(detected_patches)} yama bulundu")
    else:
        # Fallback: sabit PATCHES koordinatlarini kullan
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

        print(f"    [YAMA] Fallback: sabit koordinatlar ({len(detected_patches)} dinamik bulunmustu)")

    # ------------------------------------------------------------------------
    # BEN (MOLE) - MAVI BBOX + KIRMIZI OTSU KONTURU
    # ------------------------------------------------------------------------
    mole_bbox = detect_mole(perspective_image)
    if mole_bbox:
        mx, my, mw, mh = mole_bbox

        # Mavi sinir kutusu (hafif paddingli)
        padding = 2
        cv2.rectangle(output_image,
                      (mx - padding, my - padding),
                      (mx + mw + padding, my + mh + padding),
                      (255, 0, 0), 2)

        # Otsu esikleme ile ben sınırlarini cikar (kirmizi kontur)
        gray_perspective = cv2.cvtColor(perspective_image, cv2.COLOR_BGR2GRAY)
        roi_gray = gray_perspective[my:my + mh, mx:mx + mw]

        # Otsu: optimal esigi otomatik bulur
        _, roi_binary = cv2.threshold(roi_gray, 0, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        morph_kernel = np.ones((3, 3), np.uint8)
        roi_binary = cv2.morphologyEx(roi_binary, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
        roi_binary = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, morph_kernel)

        mole_contours, _ = cv2.findContours(roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if mole_contours:
            # En buyuk konturu ben sayar (Otsu sometimes returns multiple parts)
            largest_contour = max(mole_contours, key=cv2.contourArea)
            # ROI yerel koordinatlari -> mutlak koordinatlar
            largest_contour_absolute = largest_contour + np.array([mx, my])
            cv2.drawContours(output_image, [largest_contour_absolute], -1, (0, 0, 255), 2)

    cv2.imwrite(str(output_visualization_path), output_image)
    return True


# ============================================================================
# TUM ADIMLARI BIRLESTIREN PROCESS FONKSIYONU (TEK BASINA CALISTIRMA ICIN)
# ============================================================================

def process(original_image_path, bw_output_path, output_image_path,
            crop_output_path, perspective_output_path, color_marked_path):
    """
    Tek bir orijinal goruntu icin tum Adim 1-3'u sirayla calistirir ve
    tum cikti dosyalarini olusturur.

    Args:
        original_image_path:    Girdi: orijinal renkli fotograf.
        bw_output_path:         Cikti: BW (Adim 1).
        output_image_path:      Cikti: bbox disi beyazlatilmis (Adim 2).
        crop_output_path:       Cikti: kirpilmis.
        perspective_output_path: Cikti: perspektif duzeltilmis (Adim 3).
        color_marked_path:      Cikti: yama ve ben isaretli gorsellestirme.

    Returns:
        True: Tum adimlar basarili. False: En az bir adim hata verdi.
    """
    original_image = cv2.imread(str(original_image_path))
    if original_image is None:
        print(f"  [HATA] Okunamadi: {original_image_path.name}")
        return False

    original_height, original_width = original_image.shape[:2]

    # --- ADIM 1: BW URETIMI ---
    if not generate_bw(original_image_path, bw_output_path):
        print(f"  [UYARI] BW uretilemedi: {original_image_path.name}")
        return False

    # BW'yi yeniden oku
    bw_image = cv2.imread(str(bw_output_path), cv2.IMREAD_GRAYSCALE)
    if bw_image is None:
        return False

    # Boyut uyumsuzlugu varsa orijinal boyuta resize et
    bw_height, bw_width = bw_image.shape
    if bw_height != original_height or bw_width != original_width:
        bw_image = cv2.resize(bw_image, (original_width, original_height),
                              interpolation=cv2.INTER_NEAREST)

    # --- ADIM 2: BBOX TESPITI ---
    label_bbox = get_bbox_from_bw(bw_image)
    if label_bbox is None:
        print(f"  [UYARI] Bbox bulunamadi: {original_image_path.name}")
        return False

    # Bbox disindaki bolgeleri beyaza cevir (etiketi izole et)
    bbox_x, bbox_y, bbox_width, bbox_height = label_bbox
    isolated_image = original_image.copy()
    isolated_image[:bbox_y, :] = 255
    isolated_image[bbox_y + bbox_height:, :] = 255
    isolated_image[:, :bbox_x] = 255
    isolated_image[:, bbox_x + bbox_width:] = 255
    cv2.imwrite(str(output_image_path), isolated_image)

    # Beyaz olmayan alani sikica kirp
    gray_isolated = cv2.cvtColor(isolated_image, cv2.COLOR_BGR2GRAY)
    non_white_mask = gray_isolated < 250
    rows_with_content = np.any(non_white_mask, axis=1)
    cols_with_content = np.any(non_white_mask, axis=0)

    if not np.any(rows_with_content) or not np.any(cols_with_content):
        print(f"  [UYARI] Kirpma basarisiz")
        return False

    row_start, row_end = np.where(rows_with_content)[0][[0, -1]]
    col_start, col_end = np.where(cols_with_content)[0][[0, -1]]

    # 5px guvenlik payi ekle
    crop_padding = 5
    row_start = max(0, row_start - crop_padding)
    row_end = min(original_height, row_end + crop_padding)
    col_start = max(0, col_start - crop_padding)
    col_end = min(original_width, col_end + crop_padding)

    cropped_image = isolated_image[row_start:row_end, col_start:col_end]
    cv2.imwrite(str(crop_output_path), cropped_image)

    # --- ADIM 3: PERSPEKTIF DUZELTME ---
    perspective_ok = perspective_correct(bw_image, original_image, label_bbox,
                                          perspective_output_path)

    # --- RENK YAMASI VE BEN ISARETLEME ---
    color_marking_ok = False
    if perspective_ok:
        color_marking_ok = detect_color_squares(perspective_output_path, color_marked_path)

    print(f"  OK {original_image_path.name[-25:]} | "
          f"perspektif={'OK' if perspective_ok else 'HATA'} | "
          f"isaretleme={'OK' if color_marking_ok else 'HATA'}")
    return True


# ============================================================================
# KLASOR IZLEME MODU
# ----------------------------------------------------------------------------
# Bu modul tek basina calistirildiginda input_images/ klasorunu izler ve
# her yeni eklenen resmi otomatik isler. Gercek kullanim icin api_server.py
# uzerinden HTTP API tercih edilmelidir.
# ============================================================================

def watch_folder():
    """
    input_images/ klasorunu surekli izle, yeni resim girdiginde tum Adim
    1-3'u calistir ve ciktilari ilgili klasorlere kaydet.

    Cikis: Ctrl+C ile sonlandirilana kadar dongude calisir.
    """
    input_folder_path = Path(INPUT_FOLDER)
    bw_folder_path = Path(BW_FOLDER)
    output_folder_path = Path(OUTPUT_FOLDER)
    crop_folder_path = Path(CROP_FOLDER)
    perspective_folder_path = Path(PERSP_FOLDER)
    color_folder_path = Path(COLOR_FOLDER)

    input_folder_path.mkdir(exist_ok=True)

    print("Klasorler temizleniyor...")
    clear_folders()
    print()

    print(f"  Orijinal     : {input_folder_path.resolve()}")
    print(f"  BW           : {bw_folder_path.resolve()}")
    print(f"  Output       : {output_folder_path.resolve()}")
    print(f"  Kirpilmis    : {crop_folder_path.resolve()}")
    print(f"  Perspektif   : {perspective_folder_path.resolve()}")
    print(f"  Renk Kareler : {color_folder_path.resolve()}")
    print(f"  Kontrol araligi : {POLL_INTERVAL}s\n")

    # Daha once islenmis dosya isimleri (tekrar islememek icin)
    already_processed = set()

    while True:
        for original_file in sorted(input_folder_path.iterdir()):
            # Desteklenmeyen uzantilari atla
            if original_file.suffix.lower() not in SUPPORTED_EXT:
                continue
            # Daha onceden islenmisse atla
            if original_file.name in already_processed:
                continue

            # Her bir cikti dosya yolu
            bw_file_path = bw_folder_path / (original_file.stem + "_bw.png")
            output_file_path = output_folder_path / (original_file.stem + "_output.jpg")
            crop_file_path = crop_folder_path / (original_file.stem + "_crop.jpg")
            perspective_file_path = perspective_folder_path / (original_file.stem + "_persp.jpg")
            color_file_path = color_folder_path / (original_file.stem + "_color.jpg")

            print(f"Isleniyor: {original_file.name}")
            process(original_file, bw_file_path, output_file_path,
                    crop_file_path, perspective_file_path, color_file_path)
            already_processed.add(original_file.name)

        # Yeni dosyalari bekle
        time.sleep(POLL_INTERVAL)


# ============================================================================
# DOGRUDAN CALISTIRMA NOKTASI
# ============================================================================

if __name__ == "__main__":
    # Bu dosya dogrudan calistirildiginda klasor izleme modunu baslat.
    # API sunucusu kullanmak icin: python api_server.py
    watch_folder()
