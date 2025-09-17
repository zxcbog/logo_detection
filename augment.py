import math
import os
import random

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from PIL import Image
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def adjust_texture_aspect_center_crop(texture, max_aspect_ratio=2.0, min_aspect_ratio=None, debug=False):
    """
    Если texture имеет aspect (w/h) вне диапазона [min_aspect_ratio, max_aspect_ratio],
    делаем центрированный crop по более длинной стороне, чтобы привести aspect в допустимый диапазон.
    Возвращаем новую текстуру и флаг changed.
    """
    h, w = texture.shape[:2]
    if h == 0 or w == 0:
        return texture, False

    aspect = float(w) / float(h)
    if min_aspect_ratio is None:
        min_aspect_ratio = 1.0 / max_aspect_ratio

    # уже в пределах
    if min_aspect_ratio <= aspect <= max_aspect_ratio:
        return texture, False

    # слишком широкий -> crop width
    if aspect > max_aspect_ratio:
        desired_w = int(round(h * max_aspect_ratio))
        if desired_w < 1: desired_w = 1
        start_x = max(0, (w - desired_w) // 2)
        cropped = texture[:, start_x:start_x + desired_w].copy()
        if debug:
            print(f"adjust_texture_aspect: crop wide {w}x{h} -> {cropped.shape[1]}x{cropped.shape[0]} (aspect {aspect:.2f} -> {max_aspect_ratio:.2f})")
        return cropped, True

    # слишком высокий (узкий) -> crop height
    if aspect < min_aspect_ratio:
        desired_h = int(round(w / min_aspect_ratio))
        if desired_h < 1: desired_h = 1
        start_y = max(0, (h - desired_h) // 2)
        cropped = texture[start_y:start_y + desired_h, :].copy()
        if debug:
            print(f"adjust_texture_aspect: crop tall {w}x{h} -> {cropped.shape[1]}x{cropped.shape[0]} (aspect {aspect:.2f} -> {min_aspect_ratio:.2f})")
        return cropped, True

    return texture, False

def palette_shift_texture(texture, strength=0.08, debug=False):
    """
    Применить небольшую линейную трансформацию цветовых каналов:
    M = I + N(0, strength) (3x3), затем многократно применить к пикселям.
    strength ~ 0.05..0.15 даёт заметный, но аккуратный сдвиг палитры.
    """
    img = texture.astype(np.float32)
    # случайная матрица около единицы
    M = np.eye(3, dtype=np.float32) + np.random.normal(scale=strength, size=(3,3)).astype(np.float32)
    # опционально нормализуем так, чтобы средняя яркость сохранялась близко к оригиналу:
    # посчитаем среднюю яркость (lum) в линейной комбинации каналов (простая аппроксимация)
    # но для простоты оставим как есть — сильные изменения стоят уменьшить strength.
    h, w = img.shape[:2]
    flat = img.reshape(-1, 3)  # N x 3
    transformed = (flat @ M.T).reshape(h, w, 3)
    transformed = np.clip(transformed, 0, 255).astype(np.uint8)
    if debug:
        print(f"palette_shift_texture: applied M with strength={strength:.3f}")
    return transformed


def estimate_plane_from_depth(depth_map, mask=None, threshold=0.1, sample_step=10):
    """
    Оценка параметров плоскости с использованием RANSAC с выборкой точек
    """
    # Получаем координаты точек с шагом выборки
    if mask is None:
        points = np.argwhere(depth_map > 0)
    else:
        points = np.argwhere(mask)

    # Выборка точек для уменьшения объема вычислений
    if len(points) > 10000:
        indices = np.random.choice(len(points), 10000, replace=False)
        points = points[indices]
    else:
        indices = np.arange(len(points))
        if len(points) > 1000:
            indices = indices[::sample_step]
            points = points[indices]

    z = depth_map[points[:, 0], points[:, 1]]
    points_3d = np.column_stack((points[:, 1], points[:, 0], z))

    # Масштабирование данных для улучшения численной стабильности
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points_3d[:, :2])

    # Используем RANSAC для подбора плоскости
    try:
        ransac = RANSACRegressor(residual_threshold=threshold, max_trials=100)
        ransac.fit(points_scaled, points_3d[:, 2])

        # Получаем параметры плоскости (ax + by + c = z)
        a, b = ransac.estimator_.coef_
        c = ransac.estimator_.intercept_

        # Обратное масштабирование коэффициентов
        mean_x, mean_y = scaler.mean_
        std_x, std_y = np.sqrt(scaler.var_)
        a /= std_x
        b /= std_y
        c = c - a*mean_x - b*mean_y

        normal = np.array([a, b, -1])
        normal /= np.linalg.norm(normal)

        return normal, (a, b, c)
    except:
        # В случае ошибки возвращаем плоскость по умолчанию
        z_mean = np.mean(z) if len(z) > 0 else 1.0
        return np.array([0, 0, 1]), (0, 0, z_mean)

def compute_homography(depth_map, plane_mask, K):
    """
    Вычисление гомографии для проекции на плоскость
    """
    # Оцениваем плоскость
    normal, params = estimate_plane_from_depth(depth_map, plane_mask)

    # Вычисляем поворот для выравнивания с плоскостью
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, normal)
    rotation_axis /= np.linalg.norm(rotation_axis)

    rotation_angle = np.arccos(np.dot(z_axis, normal))
    R, _ = cv2.Rodrigues(rotation_axis * rotation_angle)

    # Вычисляем матрицу гомографии
    H = K @ R @ np.linalg.inv(K)

    return H

def find_dominant_planes(depth_map, num_planes=15):
    """
    Поиск доминирующих плоскостей в карте глубины
    """
    planes = []
    masks = []

    current_mask = depth_map > 0  # Только точки с положительной глубиной

    for _ in range(num_planes):
        if not np.any(current_mask):
            break

        # Оцениваем плоскость на оставшихся точках
        normal, params = estimate_plane_from_depth(depth_map, current_mask)

        # Вычисляем расстояния до плоскости для всех точек
        points = np.argwhere(current_mask)
        z_pred = params[0] * points[:, 1] + params[1] * points[:, 0] + params[2]
        distances = np.abs(depth_map[points[:, 0], points[:, 1]] - z_pred)

        # Создаем маску для inliers
        inlier_mask = distances < 0.05
        plane_mask = np.zeros_like(depth_map, dtype=bool)
        plane_mask[points[inlier_mask, 0], points[inlier_mask, 1]] = True

        # Фильтруем слишком маленькие плоскости
        if np.sum(plane_mask) < 100:  # Минимум 100 точек
            break

        planes.append(params)
        masks.append(plane_mask)

        # Убираем inliers из дальнейшего рассмотрения
        current_mask[plane_mask] = False

    return planes, masks

def draw_plane_contours(image, masks, colors=None, thickness=2, min_area_ratio=0.1):
    """
    Рисует контуры найденных плоскостей на изображении

    Параметры:
    image: исходное изображение (BGR или RGB)
    masks: список масок плоскостей
    colors: список цветов для каждой плоскости (по умолчанию случайные)
    thickness: толщина линий контуров
    """

    total_pixels = image.shape[0] * image.shape[1]
    min_area = int(min_area_ratio * total_pixels)
    if colors is None:
        # Генерируем случайные цвета для каждой плоскости
        colors = []
        for i in range(len(masks)):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors.append(color)

    # Создаем копию изображения для рисования
    result_image = image.copy()

    for i, mask in enumerate(masks):
        # Находим контуры маски
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                filtered_contours.append(contour)

        # Рисуем контуры на изображении
        cv2.drawContours(result_image, filtered_contours, -1, colors[i], thickness)

        # Добавляем номер плоскости
        if filtered_contours:
            # Находим наибольший контур
            largest_contour = max(filtered_contours, key=cv2.contourArea)

            # Находим центр масс контура
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Вычисляем площадь
                area = cv2.contourArea(largest_contour)
                area_percent = (area / total_pixels) * 100

                # Добавляем текст
                text = f"P{i+1}: {area_percent:.1f}%"
                cv2.putText(result_image, text, (cx, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)


    return result_image

def plane_basis_from_mask(depth_map, plane_mask, K, min_points=30):
    """
    Возвращает centroid(3,), axis1(3,), axis2(3,), coords_2d (Nx2) — координаты точек в базисе (axis1,axis2)
    или None если мало точек.
    """
    pts = np.argwhere(plane_mask)
    if pts.shape[0] < min_points:
        return None

    us = pts[:, 1].astype(np.float64)
    vs = pts[:, 0].astype(np.float64)
    zs = depth_map[pts[:, 0], pts[:, 1]].astype(np.float64)

    K_inv = np.linalg.inv(K)
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=1)  # N x 3
    cam_pts = (K_inv @ pix.T).T * zs[:, None]  # N x 3

    centroid = cam_pts.mean(axis=0)
    X = cam_pts - centroid
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    axis1 = Vt[0, :]
    axis2 = Vt[1, :]
    coords_2d = np.dot(X, np.vstack([axis1, axis2]).T)  # N x 2

    return {
        "centroid": centroid,
        "axis1": axis1,
        "axis2": axis2,
        "coords_2d": coords_2d,
        "cam_pts": cam_pts,
        "pts_idx": pts
    }

def polygon_mask_from_pts(shape, pts):
    """
    pts: Nx2 (float or int) - polygon vertices in pixel coordinates
    Возвращает uint8 mask (0/1)
    """
    mask = np.zeros(shape, dtype=np.uint8)
    pts_int = np.round(pts).astype(np.int32)
    cv2.fillConvexPoly(mask, pts_int, 1)
    return mask.astype(bool)

def polygon_area(pts):
    return abs(cv2.contourArea(pts.reshape(-1,1,2).astype(np.float32)))

def corners_inside_plane_mask(plane_mask, dst_pts):
    """
    Проверить, что все пиксели полигона dst_pts лежат внутри plane_mask.
    Возвращает True если полигон полностью внутри plane_mask.
    """
    h, w = plane_mask.shape
    poly_mask = polygon_mask_from_pts((h, w), dst_pts)
    # если есть пиксели полигона, которые не принадлежат плоскости -> False
    outside = np.logical_and(poly_mask, ~plane_mask)
    return not np.any(outside)

def match_texture_exposure_to_region(texture, target_region):
    """
    Подогнать средние и стандартные отклонения (по каналам) текстуры к целевой области.
    texture, target_region: BGR uint8 arrays (можно разного размера, но 3-канальные).
    Возвращает отмасштабированную текстуру uint8.
    Формула: t' = (t - mean_t) * (std_r / std_t) + mean_r
    guard: если std_t < eps -> scale = 1
    """
    # Привести к float
    tex = texture.astype(np.float32)
    tgt = target_region.astype(np.float32)

    # Если целевая область пустая (возможно) — вернуть оригинал
    if tgt.size == 0:
        return texture

    # По каналам BGR
    mean_tex = tex.reshape(-1, 3).mean(axis=0)
    std_tex = tex.reshape(-1, 3).std(axis=0)
    mean_tgt = tgt.reshape(-1, 3).mean(axis=0)
    std_tgt = tgt.reshape(-1, 3).std(axis=0)

    eps = 1e-2
    std_tex_safe = np.maximum(std_tex, eps)

    scaled = (tex - mean_tex[None, None, :]) * (std_tgt[None, None, :] / std_tex_safe[None, None, :]) + mean_tgt[None, None, :]
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    return scaled

def color_jitter_texture(texture, hue_shift_deg=0, sat_scale=1.0, bri_mul=1.0,
                         contrast_mul=1.0, to_gray_prob=0.0, gray_alpha=1.0, debug=False):
    """
    Цветовая аугментация + опциональное преобразование в ч/б.
    Параметры:
      - hue_shift_deg: смещение оттенка в градусах (±)
      - sat_scale: множитель насыщенности
      - bri_mul: множитель яркости (V в HSV)
      - contrast_mul: множитель контраста (в RGB)
      - to_gray_prob: вероятность превратить результат в ч/б (0..1)
      - gray_alpha: при 0..1 — смешивание (0 — оставляем цвет, 1 — полностью ч/б)
    Возвращает uint8 BGR изображение.
    """
    img = texture.copy()
    # HSV adjustments
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    except Exception:
        # Если одно-канальное, просто скопировать
        hsv = None

    if hsv is not None:
        # H: [0,179], S,V: [0,255]
        if hue_shift_deg != 0:
            dh = (hue_shift_deg / 360.0) * 180.0
            hsv[:, :, 0] = (hsv[:, :, 0] + dh) % 180.0
        if sat_scale != 1.0:
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
        if bri_mul != 1.0:
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * bri_mul, 0, 255)

        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # contrast in RGB
    if contrast_mul != 1.0:
        img = img.astype(np.float32)
        img = (img - 128.0) * contrast_mul + 128.0
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Возможное преобразование в ч/б (по вероятности)
    if to_gray_prob and random.random() < float(to_gray_prob):
        # конвертация в grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_3 = np.stack([gray, gray, gray], axis=2)  # HxWx3 float
        if gray_alpha < 1.0:
            # смешиваем: out = (1-alpha)*color + alpha*gray
            out = (1.0 - gray_alpha) * img.astype(np.float32) + gray_alpha * gray_3
            out = np.clip(out, 0, 255).astype(np.uint8)
            if debug:
                print(f"Applied partial grayscale: prob triggered, gray_alpha={gray_alpha}")
            return out
        else:
            if debug:
                print("Applied full grayscale: prob triggered")
            return np.clip(gray_3, 0, 255).astype(np.uint8)

    return img

def polygon_area(pts):
    return abs(cv2.contourArea(pts.reshape(-1,1,2).astype(np.float32)))

def min_edge_length_px(pts):
    # pts: 4x2 or Nx2 ordered polygon (in pixels)
    dists = []
    n = pts.shape[0]
    for i in range(n):
        a = pts[i]
        b = pts[(i+1) % n]
        d = np.linalg.norm(a - b)
        dists.append(d)
    return min(dists) if dists else 0.0

def place_small_texture_on_plane_randomly(
    image,
    depth_map,
    plane_mask,
    K,
    texture,
    scale_range=(0.05, 0.15),
    rotation_range=(-30, 30),
    min_points_for_pca=50,
    require_within_plane=True,
    max_attempts=10,
    exposure_match=True,
    color_jitter_params=None,
    max_aspect_ratio=4.0,
    min_aspect_ratio=None,
    allow_aspect_auto_adjust=True,
    # NEW PARAMETERS (to avoid needles)
    min_projected_size_px=8,       # min width & height in px for projected box
    min_projected_area_px=30,      # min polygon area in px
    max_projected_aspect=6.0,      # restrict projected aspect (w/h or h/w)
    min_edge_px=4,                 # minimal length of any polygon edge (px)
    min_dot_with_camera=0.18,      # minimal |normal·z| (z is camera forward); low -> plane edge-on
    debug=False
):
    """
    Возвращает (composed_image, H, bbox) — bbox = (x_min,y_min,x_max,y_max) (ints),
    либо (image, None, None) если не получилось разместить.
    """
    if min_aspect_ratio is None:
        min_aspect_ratio = 1.0 / max_aspect_ratio

    # 1) auto adjust aspect if requested
    if allow_aspect_auto_adjust:
        texture_adj, changed = adjust_texture_aspect_center_crop(texture,
                                                                max_aspect_ratio=max_aspect_ratio,
                                                                min_aspect_ratio=min_aspect_ratio,
                                                                debug=debug)
        if changed:
            texture = texture_adj
            if debug:
                print("Texture auto-aspect adjusted.")
    else:
        # filter by input aspect
        h_tex, w_tex = texture.shape[:2]
        if h_tex == 0 or w_tex == 0:
            if debug: print("Texture has zero size.")
            return image, None, None
        aspect = float(w_tex) / float(h_tex)
        if aspect > max_aspect_ratio or aspect < min_aspect_ratio:
            if debug: print(f"Texture filtered by aspect: {aspect:.3f}")
            return image, None, None

    # 2) compute plane basis
    info = plane_basis_from_mask(depth_map, plane_mask, K, min_points=min_points_for_pca)
    if info is None:
        if debug: print("Not enough points for plane PCA.")
        return image, None, None

    centroid = info["centroid"]
    axis1 = info["axis1"]
    axis2 = info["axis2"]
    coords_2d = info["coords_2d"]

    # plane normal
    normal = np.cross(axis1, axis2)
    n_norm = np.linalg.norm(normal)
    if n_norm == 0:
        if debug: print("Zero normal.")
        return image, None, None
    normal = normal / n_norm
    # dot with camera z-axis (camera forward is [0,0,1])
    normal_z_abs = abs(normal[2])
    if normal_z_abs < min_dot_with_camera:
        if debug:
            ang = np.degrees(np.arccos(np.clip(normal_z_abs, -1, 1)))
            print(f"Plane too edge-on to camera: |nz|={normal_z_abs:.3f} (angle from front ≈ {90-ang:.1f}°). Skip.")
        return image, None, None

    min_xy = coords_2d.min(axis=0)
    max_xy = coords_2d.max(axis=0)
    extent = max_xy - min_xy
    if np.any(extent <= 1e-6):
        if debug: print("Tiny plane extent.")
        return image, None, None

    h_tex, w_tex = texture.shape[:2]
    h_img, w_img = image.shape[:2]

    attempt = 0
    while attempt < max_attempts:
        attempt += 1

        # random center & size & rotation
        cx = random.uniform(min_xy[0], max_xy[0])
        cy = random.uniform(min_xy[1], max_xy[1])
        center_plane = centroid + cx * axis1 + cy * axis2

        scale = random.uniform(scale_range[0], scale_range[1])
        width_plane = scale * extent[0]
        height_plane = scale * extent[1]

        theta_deg = random.uniform(rotation_range[0], rotation_range[1])
        theta = np.deg2rad(theta_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        new_axis1 = cos_t * axis1 + sin_t * axis2
        new_axis2 = -sin_t * axis1 + cos_t * axis2

        half_w = width_plane / 2.0
        half_h = height_plane / 2.0

        corners_3d = np.array([
            center_plane + (-half_w) * new_axis1 + (-half_h) * new_axis2,  # TL
            center_plane + ( half_w) * new_axis1 + (-half_h) * new_axis2,  # TR
            center_plane + ( half_w) * new_axis1 + ( half_h) * new_axis2,  # BR
            center_plane + (-half_w) * new_axis1 + ( half_h) * new_axis2,  # BL
        ])  # 4x3

        proj = (K @ corners_3d.T).T
        proj_xy = proj[:, :2] / proj[:, 2:3]

        # validity checks
        if np.any(np.isnan(proj_xy)) or np.any(np.isinf(proj_xy)):
            if debug: print(f"Attempt {attempt}: invalid projection.")
            continue

        # clip projected polygon bbox to image
        x_min = float(np.min(proj_xy[:,0])); x_max = float(np.max(proj_xy[:,0]))
        y_min = float(np.min(proj_xy[:,1])); y_max = float(np.max(proj_xy[:,1]))
        w_px = x_max - x_min
        h_px = y_max - y_min

        # 1) min size in pixels
        if w_px < min_projected_size_px or h_px < min_projected_size_px:
            if debug: print(f"Attempt {attempt}: projected size too small w={w_px:.1f}, h={h_px:.1f}.")
            continue

        # 2) min area px
        area = polygon_area(proj_xy)
        if area < min_projected_area_px:
            if debug: print(f"Attempt {attempt}: projected area too small {area:.1f}.")
            continue

        # 3) projected aspect
        if h_px <= 0:
            if debug: print(f"Attempt {attempt}: zero projected height.")
            continue
        proj_aspect = max(w_px / h_px, h_px / w_px)
        if proj_aspect > max_projected_aspect:
            if debug: print(f"Attempt {attempt}: projected aspect too large {proj_aspect:.2f} > {max_projected_aspect}.")
            continue

        # 4) min edge length
        min_edge = min_edge_length_px(proj_xy)
        if min_edge < min_edge_px:
            if debug: print(f"Attempt {attempt}: min edge {min_edge:.2f} < {min_edge_px}.")
            continue

        # 5) optionally require polygon fully inside plane_mask (in-image)
        if require_within_plane:
            # check polygon inside plane_mask in image coords
            poly_mask = polygon_mask_from_pts((h_img, w_img), proj_xy)
            if not np.any(poly_mask):
                if debug: print(f"Attempt {attempt}: polygon does not overlap image or plane_mask.")
                continue
            # ensure all pixels of poly belong to plane_mask region
            outside = np.logical_and(poly_mask, ~plane_mask)
            if np.any(outside):
                if debug: print(f"Attempt {attempt}: polygon not fully inside plane_mask.")
                continue

        # OK — build H and composite
        src_pts = np.array([[0, 0], [w_tex - 1, 0], [w_tex - 1, h_tex - 1], [0, h_tex - 1]], dtype=np.float32)
        dst_pts = proj_xy.astype(np.float32)
        try:
            H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        except Exception as e:
            if debug: print(f"Attempt {attempt}: getPerspectiveTransform failed: {e}")
            continue

        # prepare texture (color jitter, palette, exposure)
        tex_aug = texture.copy()
        if color_jitter_params is not None:
            prob = color_jitter_params.get("prob", 0.5)
            if random.random() < prob:
                hue = color_jitter_params.get("hue_shift_deg", 0)
                sat = color_jitter_params.get("sat_scale", 1.0)
                bri = color_jitter_params.get("bri_mul", 1.0)
                cont = color_jitter_params.get("contrast_mul", 1.0)
                to_gray_prob = color_jitter_params.get("to_gray_prob", 0.0)
                gray_alpha = color_jitter_params.get("gray_alpha", 1.0)

                hue_rand = hue * random.uniform(-1, 1)
                sat_rand = sat ** random.uniform(-0.5, 0.5) if sat != 1.0 else 1.0
                bri_rand = bri ** random.uniform(-0.2, 0.2) if bri != 1.0 else 1.0
                cont_rand = cont ** random.uniform(-0.2, 0.2) if cont != 1.0 else 1.0
                tex_aug = color_jitter_texture(tex_aug,
                                               hue_shift_deg=hue_rand,
                                               sat_scale=sat_rand,
                                               bri_mul=bri_rand,
                                               contrast_mul=cont_rand,
                                               to_gray_prob=to_gray_prob,
                                               gray_alpha=gray_alpha)
            pal = color_jitter_params.get("palette_shift", None)
            if pal is not None:
                pal_prob = pal.get("prob", 0.3)
                pal_strength = pal.get("strength", 0.08)
                if random.random() < pal_prob:
                    tex_aug = palette_shift_texture(tex_aug, strength=pal_strength, debug=debug)

        if exposure_match:
            # compute target pixels inside polygon on original image
            poly_mask = polygon_mask_from_pts((h_img, w_img), proj_xy)
            if np.any(poly_mask):
                ys, xs = np.where(poly_mask)
                y0, y1 = ys.min(), ys.max()
                x0, x1 = xs.min(), xs.max()
                target_region = image[y0:y1+1, x0:x1+1]
                submask = poly_mask[y0:y1+1, x0:x1+1]
                if target_region.size != 0 and np.any(submask):
                    tgt_pixels = target_region[submask]
                    tex_aug = match_texture_exposure_to_region(tex_aug, tgt_pixels.reshape(-1,3))
                else:
                    if debug: print(f"Attempt {attempt}: target region empty for exposure match.")
            else:
                if debug: print(f"Attempt {attempt}: polygon mask empty for exposure match.")

        # warp and composite (soft alpha)
        warped = cv2.warpPerspective(tex_aug, H, (w_img, h_img))
        if warped.ndim == 3:
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            gray = warped
        _, mask_w = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        # soft mask
        kernel = int(max(1, min(21, math.ceil(np.sqrt(area)/6))))
        if kernel % 2 == 0:
            kernel += 1
        soft_mask = cv2.GaussianBlur(mask_w.astype(np.float32)/255.0, (kernel, kernel), 0)
        soft_mask_3 = np.repeat(soft_mask[:, :, None], 3, axis=2)
        composed = (image.astype(np.float32) * (1 - soft_mask_3) + warped.astype(np.float32) * soft_mask_3).astype(np.uint8)

        # compute bbox in integer and clip
        x0 = max(0, int(np.floor(x_min)))
        y0 = max(0, int(np.floor(y_min)))
        x1 = min(w_img - 1, int(np.ceil(x_max)))
        y1 = min(h_img - 1, int(np.ceil(y_max)))
        bbox = (x0, y0, x1, y1)

        if debug:
            print(f"Placed after {attempt} attempts: bbox={bbox}, proj_area={area:.1f}, proj_aspect={proj_aspect:.2f}, min_edge={min_edge:.2f}, |nz|={normal_z_abs:.3f}")

        return composed, H, bbox

    if debug:
        print("Failed to place texture after max_attempts.")
    return image, None, None

# ---------------------------
# Wrapper: apply texture(s) to all planes and collect boxes
def apply_textures_to_planes(image, depth_map, planes_masks, K, texture,
                             **place_kwargs):
    """
    Проходит по списку plane_masks (например из find_dominant_planes),
    вызывает place_small_texture_on_plane_randomly для каждой плоскости,
    собирает метаданные для успешно размещённых текстур.

    Возвращает:
      composed_image, placed_items
    где placed_items = [ { 'plane_idx': idx, 'H': H, 'bbox': (x0,y0,x1,y1) }, ... ]
    """
    composed = image.copy()
    placed_items = []
    for i, mask in enumerate(planes_masks):
        composed_new, H, bbox = place_small_texture_on_plane_randomly(
            composed, depth_map, mask, K, texture, **place_kwargs
        )
        if H is not None and bbox is not None:
            placed_items.append({'plane_idx': i, 'H': H, 'bbox': bbox})
            composed = composed_new  # update image with newly added texture
    return composed, placed_items

def draw_bboxes_matplotlib(image_bgr, placed_items, figsize=(12,8), linewidth=2, alpha=0.6, save_path=None, show=True):
    """
    Рисует bbox'ы из placed_items поверх image (BGR) с помощью matplotlib.
    placed_items: list of dicts with keys {'plane_idx':int, 'H':..., 'bbox':(x0,y0,x1,y1)}
    Возвращает: matplotlib Figure и Axes (fig, ax)
    """
    # Конвертация BGR -> RGB
    if image_bgr is None:
        raise ValueError("image_bgr is None")
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img_rgb)
    ax.axis('off')

    # Генерируем цвет для каждого bbox (детерминированно по plane_idx для повторяемости)
    for item in placed_items:
        bbox = item.get('bbox', None)
        plane_idx = item.get('plane_idx', None)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        w = x1 - x0
        h = y1 - y0

        # цвет — по индексу, иначе случайный
        if plane_idx is not None:
            rnd = (plane_idx * 123457) % 0xFFFFFF
            # преобразуем в RGB tuple [0-1]
            r = ((rnd >> 16) & 255) / 255.0
            g = ((rnd >> 8) & 255) / 255.0
            b = (rnd & 255) / 255.0
            color = (r, g, b)
        else:
            color = (random.random(), random.random(), random.random())

        # прямоугольник
        rect = patches.Rectangle((x0, y0), w, h, linewidth=linewidth,
                                 edgecolor=color, facecolor=color+(0.0,), alpha=alpha)
        ax.add_patch(rect)

        # подпись: plane index и размер
        label = f"P{plane_idx}" if plane_idx is not None else "P?"
        ax.text(x0 + 2, y0 + 12, label, color='white', fontsize=10,
                bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor='none'))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.01)
    if show:
        plt.show()
    return fig, ax

from pathlib import Path

image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

def create_depth_map(image):


    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    result = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    return result

def augment_scene(logo: Path, scene_path: Path, scene_size: tuple[int, int] = (500, 500), draw_results=False):
    image = Image.open(scene_path)
    image = image.resize(scene_size)
    depth_map = create_depth_map(
        image=image
    )
    if draw_results:
      plt.imshow(depth_map[0, 0])
      plt.show()

    # Загружаем карту глубины (предполагаем, что она уже загружена)
    depth_map = depth_map.cpu().detach().numpy()[0, 0]

    # Параметры камеры (замените на реальные значения)
    K = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    # Находим доминирующие плоскости
    planes, masks = find_dominant_planes(depth_map)

    # Загружаем исходное изображение
    #image = cv2.imread(image_path)
    #cv2.resize()

    texture = cv2.imread(logo)
    texture = cv2.resize(texture, (50, 50), interpolation=cv2.INTER_CUBIC)

    color_jitter_params = {
        "prob": 0.7,
        "sat_scale": 1.05,
        "bri_mul": 1.05,
        "contrast_mul": 1.15,
        "palette_shift": {
            "prob": 0.5,
            "strength": 0.1
        },
        "gray_alpha": 1,
        "to_gray_prob": 0.5
    }
    image_cv = np.array(image)[:, :, ::-1].copy()

    composed, items = apply_textures_to_planes(
        image_cv,
        depth_map,
        masks,
        K,
        texture,
        scale_range=(0.03, 0.12),
        rotation_range=(-45, 45),
        min_points_for_pca=40,
        require_within_plane=True,
        max_attempts=20,
        exposure_match=False,
        color_jitter_params=color_jitter_params,
        max_aspect_ratio=4.0,
        allow_aspect_auto_adjust=True,
        # needle-avoidance params
        min_projected_size_px=10,
        min_projected_area_px=100,
        max_projected_aspect=6.0,
        min_edge_px=4,
        min_dot_with_camera=0.18,
        debug=False
    )

    return composed, items

def _is_bgr_image(img: np.ndarray) -> bool:
    """
    Простая эвристика: если у изображения 3 канала и значения выглядят как BGR из OpenCV,
    мы не можем 100% определить, поэтому считаем 3-канальную матрицу как BGR by default.
    We'll convert BGR->RGB before saving so colors look correct.
    """
    return img.ndim == 3 and img.shape[2] == 3

def _clip_box_to_image(box, w, h):
    x0, y0, x1, y1 = box
    x0c = max(0, min(w-1, int(round(x0))))
    x1c = max(0, min(w-1, int(round(x1))))
    y0c = max(0, min(h-1, int(round(y0))))
    y1c = max(0, min(h-1, int(round(y1))))
    # ensure x0<=x1 and y0<=y1
    if x1c < x0c:
        x0c, x1c = x1c, x0c
    if y1c < y0c:
        y0c, y1c = y1c, y0c
    return x0c, y0c, x1c, y1c

def save_yolo_dataset(
    images_with_boxes,
    output_dir: str,
    prefix: str = "aug",
    start_index: int = 0,
    class_id_default: int = 0,
    image_ext: str = ".jpg",
    save_rgb: bool = True,
    quality: int = 95,
):
    """
    Saves augmented images + annotations in YOLOv11 format.

    Parameters
    ----------
    images_with_boxes : list
        Each element can be either:
          - (image: np.ndarray, boxes: list of (x_min,y_min,x_max,y_max,class_id))
          - or dict {'image': image, 'placed_items': placed_items_list}
            where each placed_item is a dict containing 'bbox':(x0,y0,x1,y1)
            (class_id_default will be used unless you provide class_id inside placed_item)
    output_dir : str
        Directory where "images/" and "labels/" will be created.
    prefix : str
        Filename prefix for saved images (prefix_00001.jpg, etc.).
    start_index : int
        Starting index for filenames.
    class_id_default : int
        Default class id to write for boxes (if not provided).
    image_ext : str
        Image extension (".jpg" or ".png").
    save_rgb : bool
        If True and image appears BGR (OpenCV), convert to RGB prior to saving.
    quality : int
        JPEG quality (used by PIL save).
    Returns
    -------
    saved : list of dict
        [{'image_path':..., 'label_path':..., 'boxes':[...]}, ...]
    """
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    saved_records = []
    idx = start_index

    for item in images_with_boxes:
        # Normalize input to (img, boxes) form
        if isinstance(item, tuple) or isinstance(item, list):
            img, boxes = item
        elif isinstance(item, dict):
            img = item.get('image', None)
            if img is None:
                raise ValueError("Dict item must contain key 'image'")
            placed_items = item.get('placed_items', None)
            boxes = []
            if placed_items:
                for p in placed_items:
                    # p can be dict with 'bbox' and optional 'class_id'
                    bbox = p.get('bbox') if isinstance(p, dict) else None
                    if bbox is None:
                        continue
                    cid = p.get('class_id', class_id_default) if isinstance(p, dict) else class_id_default
                    boxes.append((bbox[0], bbox[1], bbox[2], bbox[3], cid))
        else:
            raise ValueError("Unsupported item type in images_with_boxes")

        if img is None:
            continue

        # Convert BGR->RGB for saving if needed (PIL expects RGB)
        img_to_save = img
        if _is_bgr_image(img) and save_rgb:
            # assume OpenCV BGR -> convert to RGB
            img_to_save = img[..., ::-1]

        # Ensure dtype is uint8
        if img_to_save.dtype != np.uint8:
            img_to_save = np.clip(img_to_save, 0, 255).astype(np.uint8)

        # Save image
        img_name = f"{prefix}_{idx:06d}{image_ext}"
        image_path = os.path.join(output_dir, "images", img_name)
        pil_img = Image.fromarray(img_to_save)
        if image_ext.lower().endswith(".jpg") or image_ext.lower().endswith(".jpeg"):
            pil_img.save(image_path, quality=quality, subsampling=0)
        else:
            pil_img.save(image_path)

        # Write label file
        h, w = img.shape[:2]
        label_name = f"{prefix}_{idx:06d}.txt"
        label_path = os.path.join(output_dir, "labels", label_name)

        lines = []
        for b in boxes:
            if len(b) == 5:
                x_min, y_min, x_max, y_max, cid = b
            elif len(b) == 4:
                x_min, y_min, x_max, y_max = b
                cid = class_id_default
            else:
                raise ValueError("Box must be (x_min,y_min,x_max,y_max[,class_id])")

            # clip box
            x0c, y0c, x1c, y1c = _clip_box_to_image((x_min, y_min, x_max, y_max), w, h)
            # ignore degenerate boxes
            if x1c <= x0c or y1c <= y0c:
                continue

            # YOLO format: class x_center y_center width height (normalized 0..1)
            x_center = ((x0c + x1c) / 2.0) / w
            y_center = ((y0c + y1c) / 2.0) / h
            bw = (x1c - x0c) / float(w)
            bh = (y1c - y0c) / float(h)

            lines.append(f"{int(cid)} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        saved_records.append({
            "image_path": image_path,
            "label_path": label_path,
            "boxes": boxes
        })

        idx += 1

    return saved_records


# Пример использования
if __name__ == "__main__":
    images_with_boxes = []
    scenes = os.listdir("scenes")

    for scene in tqdm.tqdm(scenes):
      scene_path = f"./scenes/{scene}"
      print(scene_path)
      try:
        composed, placed_items = augment_scene(
            logo="./tbank logo.png",
            scene_path=scene_path,
            scene_size=(500, 500),
            draw_results=False
        )
      except:
        continue

      images_with_boxes.append(
          {'image': composed, 'placed_items': placed_items}
      )

    saved = save_yolo_dataset(
        images_with_boxes,
        output_dir="yolo_aug_dataset",
        prefix="aug",
        start_index=0,
        class_id_default=0,   # класс логотипа
        image_ext=".jpg",
    )
    print("Saved:", saved)
