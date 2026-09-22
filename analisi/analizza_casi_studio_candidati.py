#!/usr/bin/env python3
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import cv2
import json
import math
import numpy as np
from ultralytics import YOLO
import reachability_ranking_circular as rrc
import reachability_ranking as rrb

candidate_files = [
    "col_2023-08-23-12-38-59_4_png.rf.be72469f32246237418e41dbce50b0cd.jpg",
    "col_2023-08-23-12-39-06_6_png.rf.b20e615a9ea0a175ebc7119ddec29bbd.jpg",
    "col_2023-08-23-12-45-02_4_png.rf.c3ae4db6eea134c5e4fa1074315efb52.jpg",
    "col_2023-08-23-12-51-18_31_png.rf.73f3758b3ea5ff83c389dd23d130c35f.jpg",
    "col_2023-08-23-13-01-39_3_png.rf.4a1d00f7efa3ec4f003cbd4d3ddd2ebc.jpg"
]

model = YOLO(rrc.MODEL_PATH)

for fname in candidate_files:
    img_path = os.path.join("immaginiPerAlgoritmo", fname)
    results = model.predict(img_path, conf=rrc.CONF, imgsz=rrc.IMGSZ, verbose=False)[0]
    img = results.orig_img.copy()
    h, w = img.shape[:2]

    # Process candidates
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

    print("=" * 70)
    print("FILE:", fname)
    
    # Run circular selection
    top_c, free_c = rrc.select_targets([dict(c) for c in candidates], img)
    print("--- CIRCULAR (ELLISSI) ---")
    for r, t in enumerate(top_c, 1):
        unl = f" (Liberato da #{t['unlocked_by']})" if "unlocked_by" in t else ""
        cx, cy = t["centroid"]
        geo = t["geo_score"]
        fin = t.get("final_score", 0)
        print(f"  Rank {r}: pos=({cx}, {cy}), geo={geo:.3f}, final={fin:.3f}{unl}")
    
    # Run bbox selection
    top_b, free_b = rrb.select_targets([dict(c) for c in candidates], img)
    print("--- BBOX (BASELINE) ---")
    for r, t in enumerate(top_b, 1):
        unl = f" (Liberato da #{t['unlocked_by']})" if "unlocked_by" in t else ""
        cx, cy = t["centroid"]
        geo = t["geo_score"]
        fin = t.get("final_score", 0)
        print(f"  Rank {r}: pos=({cx}, {cy}), geo={geo:.3f}, final={fin:.3f}{unl}")
