"""Ekstraksi fitur klasik (tanpa deep learning) dengan OpenCV.

Fitur gabungan:
1. Warna   : histogram HSV 3D + mean/std kanal Lab
2. Tekstur : histogram LBP sederhana, kepadatan tepi Canny, variansi Laplacian
3. Bentuk  : Hu moments, solidity, aspect ratio dari kontur terbesar
"""
import cv2
import numpy as np

IMG_SIZE = 224


def preprocess(img_bgr):
    """Resize + Gaussian blur untuk mengurangi noise."""
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(img, (5, 5), 0)


def color_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    mean, std = cv2.meanStdDev(lab)
    return np.concatenate([hist, mean.flatten() / 255.0, std.flatten() / 255.0])


def _lbp_hist(gray):
    """LBP 8 tetangga, histogram 32 bin (kode >> 3)."""
    g = gray.astype(np.int16)
    c = g[1:-1, 1:-1]
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    code = np.zeros_like(c, dtype=np.uint8)
    for i, (dy, dx) in enumerate(shifts):
        nb = g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
        code |= ((nb >= c).astype(np.uint8) << i)
    hist, _ = np.histogram(code >> 3, bins=32, range=(0, 32))
    return hist / max(hist.sum(), 1)


def texture_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = _lbp_hist(gray)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = edges.mean() / 255.0
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return np.concatenate([lbp, [edge_density, np.log1p(lap_var), gray.std() / 255.0]])


def shape_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Pastikan objek = putih (asumsi objek lebih kecil dari latar)
    if th.mean() > 127:
        th = cv2.bitwise_not(th)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros(7 + 3)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    hull_area = cv2.contourArea(cv2.convexHull(c)) or 1.0
    hu = cv2.HuMoments(cv2.moments(c)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    extra = [area / (IMG_SIZE * IMG_SIZE), area / hull_area, w / max(h, 1)]
    return np.concatenate([hu, extra])


def extract_features(img_bgr):
    img = preprocess(img_bgr)
    return np.concatenate([color_features(img), texture_features(img), shape_features(img)])
