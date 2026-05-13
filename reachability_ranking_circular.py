"""
Algoritmo di Task Planning per la Raccolta Robotizzata di Pomodori.
Versione con Approssimazione ELLITTICA per il calcolo delle occlusioni.
(Evoluzione del file reachability_ranking_circular.py)

Pipeline Decisionale:
  1. Screening Geometrico: Valutazione raggiungibilità e fitting Ellisse.
  2. Filtro di Maturità HSV: Criterio decisionale primario (Maturo vs Acerbo).
  3. Analisi Occlusioni Ellittiche: Stima della sovrapposizione tramite ellissi ruotate.
  4. Selezione Dinamica: Loop "Pick & Update" per definire la sequenza di raccolta ottimale.
"""

import cv2
import json
import numpy as np
import math
import os
from ultralytics import YOLO

# ── Configurazione ──────────────────────────────────────────────────
MODEL_PATH   = "runs/yolo11NewData6/weights/best.pt"
IMAGE_DIR    = "immaginiPerAlgoritmo"
OUTPUT_DIR   = "output_reachability_circular" 
TARGET_CLASS = 3          
CONF         = 0.3        
IMGSZ        = 800        
AREA_MIN     = 1500       
MAX_TARGETS  = 3          

# pesi del ranking geometrico
W_AREA        = 0.45
W_CIRCULARITY = 0.35
W_CENTRALITY  = 0.20

# ── Parametri Task Planning ─────────────────────────────────────────
MATURITY_THRESHOLD = 0.5
IOU_THRESHOLD = 0.15
DEDUPLICATION_THRESHOLD = 0.70
PENALTY_OCCLUDED = 0.7
BONUS_UNLOCKED   = 1.2
GEOMETRIC_POOL = 8

RANK_COLORS = {
    1: (0, 255, 0),    
    2: (0, 200, 255),  
    3: (0, 100, 255),  
}

# fase 1: criterio geometrico

def compute_geometric_score(mask_binary, img_h, img_w):
    """
    Calcola lo score di raggiungibilità geometrica e fitta un'ellisse.
    """
    area = int(np.sum(mask_binary > 0))
    if area < AREA_MIN:
        return None

    # 1. Recuperiamo tutti i punti della maschera (più robusto di un singolo contorno se frammentata)
    all_points = cv2.findNonZero(mask_binary)
    if all_points is None or len(all_points) < 5:
        return None

    # 2. Calcoliamo il Convex Hull (Involucro Convesso)
    # Questo "riempie" virtualmente i tagli causati da rametti o occlusioni
    hull = cv2.convexHull(all_points)
    
    # 3. Fit Ellisse sull'involucro convesso
    try:
        ellipse = cv2.fitEllipse(hull)
    except:
        return None

    # Calcolo metriche per lo score
    perimeter = cv2.arcLength(hull, True)
    circularity = min(1.0, (4 * math.pi * area) / (perimeter ** 2)) if perimeter > 0 else 0

    coords = np.where(mask_binary > 0)
    cy, cx = int(np.mean(coords[0])), int(np.mean(coords[1]))
    max_dist = math.sqrt((img_w / 2) ** 2 + (img_h / 2) ** 2)
    dist = math.sqrt((cx - img_w / 2) ** 2 + (cy - img_h / 2) ** 2)
    centrality = 1.0 - (dist / max_dist)

    area_norm = min(1.0, area / (img_w * img_h * 0.15))

    score = (W_AREA * area_norm
           + W_CIRCULARITY * circularity
           + W_CENTRALITY * centrality)

    return {
        "geo_score": score,
        "area": area,
        "area_norm": area_norm,
        "circularity": circularity,
        "centrality": centrality,
        "centroid": (cx, cy),
        "ellipse": ellipse, # ((x,y), (w,h), angle)
        "mask": mask_binary,
    }

def check_occlusion_ellipse(target_maybe_occluded, target_maybe_blocker, img_shape):
    """
    Verifica se target_maybe_occluded è coperto da target_maybe_blocker
    utilizzando le ellissi ruotate.
    """
    h, w = img_shape[:2]
    
    # Creiamo due maschere per le ellissi per calcolare l'intersezione reale
    mask_a = np.zeros((h, w), dtype=np.uint8)
    mask_b = np.zeros((h, w), dtype=np.uint8)
    
    cv2.ellipse(mask_a, target_maybe_occluded["ellipse"], 255, -1)
    cv2.ellipse(mask_b, target_maybe_blocker["ellipse"], 255, -1)
    
    intersection = cv2.bitwise_and(mask_a, mask_b)
    inter_area = np.sum(intersection > 0)
    area_a = np.sum(mask_a > 0)
    
    if area_a == 0: return False
    
    overlap_ratio = inter_area / float(area_a)
    
    # Heuristic: l'occluso ha area visibile minore
    if overlap_ratio > IOU_THRESHOLD and target_maybe_occluded["area"] < target_maybe_blocker["area"]:
        return True
    
    return False


# fase 2: filtro di maturità cromatica 

def compute_maturity(image_bgr, binary_mask):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask_red1  = cv2.inRange(hsv, (0, 50, 50),   (15, 255, 255))
    mask_red2  = cv2.inRange(hsv, (165, 50, 50),  (180, 255, 255))
    mask_red   = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, (35, 40, 40),   (85, 255, 255))

    red_pixels   = int(np.sum((mask_red > 0) & (binary_mask > 0)))
    green_pixels = int(np.sum((mask_green > 0) & (binary_mask > 0)))
    total = red_pixels + green_pixels

    if total == 0:
        return 0.5
    return red_pixels / total

# visualizzazione 

def overlay_mask(image, mask, color, alpha=0.45):
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def draw_rank_marker(image, centroid, rank, color):
    cx, cy = int(centroid[0]), int(centroid[1])
    cv2.circle(image, (cx, cy), 12, color, -1)
    cv2.circle(image, (cx, cy), 12, (255, 255, 255), 2)
    cv2.putText(image, str(rank), (cx - 5, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def draw_legend(image, targets, original_h):
    x0 = 50
    y_ptr = original_h + 50
    for i, det in enumerate(targets):
        rank = i + 1
        color = RANK_COLORS[rank]
        maturity = det.get("maturity", 0)
        cv2.rectangle(image, (x0 - 30, y_ptr - 20), (x0 - 10, y_ptr + 5), color, -1)
        cv2.putText(image, f"RANK {rank} (ELLISSE)", (x0, y_ptr),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y_ptr += 35
        # Mostriamo l'angolo di rotazione dell'ellisse per verifica
        metrics_text = f" > Score Geo: {det['geo_score']:.2f} | Mat.: {maturity:.0%} | Rot: {det['ellipse'][2]:.1f}deg"
        cv2.putText(image, metrics_text, (x0, y_ptr),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        y_ptr += 30
        details_text = f" > Area: {det['area']}px"
        if "unlocked_by" in det:
            details_text += f" | Liberato da #{det['unlocked_by']}"
        cv2.putText(image, details_text, (x0, y_ptr),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        y_ptr += 60

# pipeline principale

def select_targets(candidates, image_bgr):
    candidates.sort(key=lambda x: x.get("conf", 0), reverse=True)#ordinando per confidenza decrescente prima di deduplicare
    unique_candidates = []
    for cand in candidates:
        is_duplicate = False
        for unique in unique_candidates:
            inter = np.logical_and(cand["mask"], unique["mask"])
            union = np.logical_or(cand["mask"], unique["mask"])
            iou = np.sum(inter) / np.sum(union) if np.sum(union) > 0 else 0
            if iou > DEDUPLICATION_THRESHOLD:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_candidates.append(cand)
    
    candidates = unique_candidates
    candidates.sort(key=lambda x: x["geo_score"], reverse=True)
    pool = candidates[:GEOMETRIC_POOL]

    if not pool:
        return [], []

    for det in pool:
        det["maturity"] = compute_maturity(image_bgr, det["mask"])
        det["is_selected"] = False
        det["bonus"] = 1.0
        det["occluded_by_me"] = []

    # Analisi occlusioni ELLITTICHE
    h, w = image_bgr.shape[:2]
    for i, a in enumerate(pool):
        for j, b in enumerate(pool):
            if i == j: continue
            if check_occlusion_ellipse(b, a, (h, w)):
                a["occluded_by_me"].append(b)

    selected = []
    free_candidates = [d for d in pool if d["maturity"] >= MATURITY_THRESHOLD]
    free_candidates.sort(key=lambda x: x["geo_score"], reverse=True)
    free_targets = free_candidates[:5]

    while len(selected) < MAX_TARGETS:
        current_candidates = [d for d in pool if not d["is_selected"] and d["maturity"] >= MATURITY_THRESHOLD]
        if not current_candidates: break

        for d in current_candidates:
            is_currently_occluded = False
            for other in pool: 
                if not other["is_selected"] and d in other["occluded_by_me"]:
                    is_currently_occluded = True
                    break
            penalty = PENALTY_OCCLUDED if is_currently_occluded else 1.0
            d["final_score"] = d["geo_score"] * penalty * d["bonus"]

        current_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        best = current_candidates[0]
        best["is_selected"] = True
        selected.append(best)
        for under in best["occluded_by_me"]:
            under["bonus"] = BONUS_UNLOCKED
            under["unlocked_by"] = len(selected)

    return selected, free_targets

def process_folder(model_path=MODEL_PATH, image_dir=IMAGE_DIR, output_dir=OUTPUT_DIR):
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(extensions)])

    if not images:
        return

    algo_results = {}
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        results = model.predict(img_path, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        img = results.orig_img.copy()
        h, w = img.shape[:2]

        candidates = []
        if results.masks is not None:
            for i, mask_tensor in enumerate(results.masks.data):
                if int(results.boxes.cls[i]) != TARGET_CLASS: continue
                conf_val = float(results.boxes.conf[i])
                m = mask_tensor.cpu().numpy()
                m = cv2.resize(m, (w, h))
                binary = (m > 0.5).astype(np.uint8) * 255
                result = compute_geometric_score(binary, h, w)
                if result is not None:
                    result["conf"] = conf_val
                    result["bbox"] = results.boxes.xyxy[i].cpu().numpy()
                    candidates.append(result)

        if not candidates:
            cv2.imwrite(os.path.join(output_dir, f"ranked_{img_name}"), img)
            continue

        top, free = select_targets(candidates, img)
        algo_results[img_name] = {
            "targets": [{"x": d["centroid"][0], "y": d["centroid"][1], "rank": r, "ellipse": d["ellipse"], "bbox": d["bbox"].tolist()} for r, d in enumerate(top, 1)],
            "free_targets": [{"x": d["centroid"][0], "y": d["centroid"][1], "geo_score": d["geo_score"], "maturity": d["maturity"], "area": d["area"], "ellipse": d["ellipse"], "bbox": d["bbox"].tolist()} for d in free]
        }

        margin_bottom = 500
        output = cv2.copyMakeBorder(img, 0, margin_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        for rank, det in enumerate(top, 1):
            color = RANK_COLORS[rank]
            # Disegno l'ellisse fitata per debug visivo
            cv2.ellipse(output[0:h, 0:w], det["ellipse"], color, 2, cv2.LINE_AA)
            draw_rank_marker(output, det["centroid"], rank, color)
        
        draw_legend(output, top, h)
        cv2.imwrite(os.path.join(output_dir, f"ranked_{img_name}"), output)

    with open(os.path.join(output_dir, "algorithm_results_circular.json"), "w") as f:
        json.dump(algo_results, f, indent=2)

if __name__ == "__main__":
    process_folder()
