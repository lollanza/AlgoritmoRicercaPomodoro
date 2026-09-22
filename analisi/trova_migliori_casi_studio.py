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

diff_files = [
    "col_2023-08-23-12-38-59_4_png.rf.be72469f32246237418e41dbce50b0cd.jpg",
    "col_2023-08-23-12-40-20_27_png.rf.ae04f6fd459f597ba825d488b2400c64.jpg",
    "col_2023-08-23-12-40-44_34_png.rf.a56a54e747d8445e0138e676f3e5d10c.jpg",
    "col_2023-08-23-12-45-20_9_png.rf.99872f990c2cbaabe69b290abff13006.jpg",
    "col_2023-08-23-12-48-32_5_png.rf.d5cbbb495d445c5bb6c9b2b997bddf77.jpg",
    "col_2023-08-23-12-50-26_16_png.rf.5c464a8b79804e8c6906b009832f05c2.jpg",
    "col_2023-08-23-12-51-18_31_png.rf.73f3758b3ea5ff83c389dd23d130c35f.jpg",
    "col_2023-08-23-13-02-29_16_png.rf.3980c6f7dc99773bf7b6b00b502bd5d2.jpg",
    "col_2023-09-01-11-39-31_17_png.rf.408b180a9f91823a15826267fb7f9eac.jpg",
    "col_2023-09-01-12-06-39_13_png.rf.08e80c9033d997eaa73aa4edf053e0cc.jpg",
    "col_2024-01-11-12-37-13_3_png.rf.721ac8d4115fb1eea24c1d76194afc49.jpg",
    "col_2024-01-11-16-36-05_35_png.rf.5bc99fb0a6c6aee3a8e749fa2e44de8e.jpg"
]

model = YOLO(rrc.MODEL_PATH)

print("=" * 80)
print("ANALISI DETTAGLIATA DELLE SCENE DIVERGENTI IN immaginiPerAlgoritmo")
print("=" * 80)

for fname in diff_files:
    img_path = os.path.join(BASE_DIR, "immaginiPerAlgoritmo", fname)
    results = model.predict(img_path, conf=rrc.CONF, imgsz=rrc.IMGSZ, verbose=False)[0]
    img = results.orig_img.copy()
    h, w = img.shape[:2]

    candidates = []
    if results.masks is not None:
        for i, mask_tensor in enumerate(results.masks.data):
            if int(results.boxes.cls[i]) != rrc.TARGET_CLASS: 
                continue
            conf_val = float(results.boxes.conf[i])
            m = mask_tensor.cpu().numpy()
            m = cv2.resize(m, (w, h))
            binary = (m > 0.5).astype(np.uint8) * 255
            res = rrc.compute_geometric_score(binary, h, w)
            if res is not None:
                res["conf"] = conf_val
                res["bbox"] = results.boxes.xyxy[i].cpu().numpy()
                candidates.append(res)

    top_c, free_c = rrc.select_targets([dict(c) for c in candidates], img)
    top_b, free_b = rrb.select_targets([dict(c) for c in candidates], img)

    seq_c = [(t["centroid"][0], t["centroid"][1], f"R{r}" + (f" [Unl#{t['unlocked_by']}]" if "unlocked_by" in t else "")) for r, t in enumerate(top_c, 1)]
    seq_b = [(t["centroid"][0], t["centroid"][1], f"R{r}" + (f" [Unl#{t['unlocked_by']}]" if "unlocked_by" in t else "")) for r, t in enumerate(top_b, 1)]

    print(f"\nIMMAGINE: {fname}")
    print("  ELLISSI (NOSTRO): ", seq_c)
    print("  BBOX (BASELINE):  ", seq_b)
