#!/usr/bin/env python3
import os
import sys
import json
import math
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ultralytics import YOLO
import reachability_ranking_circular as rrc
import reachability_ranking as rrb

model = YOLO(rrc.MODEL_PATH)
images = sorted([f for f in os.listdir(os.path.join(BASE_DIR, "immaginiPerAlgoritmo")) if f.lower().endswith((".jpg", ".png"))])

differences = []

for img_name in images:
    img_path = os.path.join(BASE_DIR, "immaginiPerAlgoritmo", img_name)
    results = model.predict(img_path, conf=rrc.CONF, imgsz=rrc.IMGSZ, verbose=False)[0]
    img = results.orig_img.copy()
    h, w = img.shape[:2]

    if results.masks is None:
        continue

    for i, mask_tensor in enumerate(results.masks.data):
        if int(results.boxes.cls[i]) != rrc.TARGET_CLASS:
            continue
        m = mask_tensor.cpu().numpy()
        m = cv2.resize(m, (w, h))
        binary = (m > 0.5).astype(np.uint8) * 255

        area = int(np.sum(binary > 0))
        if area < rrc.AREA_MIN:
            continue

        # Colleague (raw contour)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        p_raw = cv2.arcLength(contours[0], True)
        circ_raw = min(1.0, (4 * math.pi * area) / (p_raw ** 2)) if p_raw > 0 else 0

        # Ours (convex hull)
        pts = cv2.findNonZero(binary)
        if pts is None or len(pts) < 5:
            continue
        hull = cv2.convexHull(pts)
        p_hull = cv2.arcLength(hull, True)
        circ_hull = min(1.0, (4 * math.pi * area) / (p_hull ** 2)) if p_hull > 0 else 0

        diff_circ = circ_hull - circ_raw
        differences.append({
            "image": img_name,
            "area": area,
            "p_raw": p_raw,
            "p_hull": p_hull,
            "circ_raw": circ_raw,
            "circ_hull": circ_hull,
            "diff_circ": diff_circ,
            "diff_geo": diff_circ * rrc.W_CIRCULARITY
        })

differences.sort(key=lambda x: x["diff_circ"], reverse=True)

print("=" * 80)
print("TOP 10 DIFFERENZE DI CIRCOLARITÀ (CONVEX HULL vs CONTORNO GREZZO)")
print("=" * 80)
for idx, d in enumerate(differences[:10], 1):
    print(f"\n[{idx:02d}] Immagine: {d['image']}")
    print(f"     Area: {d['area']} px")
    print(f"     Perimetro Grezzo (Collega):  {d['p_raw']:.1f} px  -> Circolarità: {d['circ_raw']:.3f}")
    print(f"     Perimetro Convex Hull (Tuo): {d['p_hull']:.1f} px  -> Circolarità: {d['circ_hull']:.3f}")
    print(f"     Delta Circolarità: +{d['diff_circ']:.3f} | Incremento GeoScore: +{d['diff_geo']:.3f}")
