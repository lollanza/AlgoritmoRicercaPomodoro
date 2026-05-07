"""
Algoritmo di Task Planning per la Raccolta Robotizzata di Pomodori.

Pipeline Decisionale:
  1. Screening Geometrico: Valutazione raggiungibilità (Area, Circolarità, Centralità).
  2. Filtro di Maturità HSV: Criterio decisionale primario (Maturo vs Acerbo).
  3. Analisi Occlusioni IoU: Identificazione delle dipendenze spaziali (Chi copre chi).
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
OUTPUT_DIR   = "output_reachability"
TARGET_CLASS = 3          # la classe tomato
CONF         = 0.3        # soglia di confidenza scelta
IMGSZ        = 800        
AREA_MIN     = 1500       # area minima in pixel, scartando quelli troppo piccoli 
MAX_TARGETS  = 3          # numero di target che verranno selezionati (top 3)

# pesi del ranking geometrico
W_AREA        = 0.45
W_CIRCULARITY = 0.35
W_CENTRALITY  = 0.20

# ── Parametri Task Planning ─────────────────────────────────────────
# soglia di maturità per il filtro cromatico: 0.0 = acerbo, 1.0 = maturo
MATURITY_THRESHOLD = 0.5

# soglia IoU per rilevamento occlusioni
IOU_THRESHOLD = 0.15

# soglia IoU per de-duplicazione (se due maschere sono quasi identiche)
DEDUPLICATION_THRESHOLD = 0.70

# penalità per frutti occlusi e bonus per frutti liberati
PENALTY_OCCLUDED = 0.7
BONUS_UNLOCKED   = 1.2

# sottoinsieme di candidati geometrici da cui filtrare (> MAX_TARGETS)
GEOMETRIC_POOL = 8

# colori di visualizzazione per i rank (BGR)
RANK_COLORS = {
    1: (0, 255, 0),    # Verde  — Rank 1 (miglior target)
    2: (0, 200, 255),  # Arancio — Rank 2
    3: (0, 100, 255),  # Rosso-arancio — Rank 3
}


# fase 1: criterio geometrico

def compute_geometric_score(mask_binary, img_h, img_w):
    """
    Calcola lo score di raggiungibilità geometrica per una singola istanza.
    Restituisce il dizionario con metriche e score, oppure None se filtrata.
    """
    area = int(np.sum(mask_binary > 0))
    if area < AREA_MIN:
        return None

    # circolarità: 
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#trova i contorni della maschera binaria,
    #RETR_EXTERNAL: recupera solo i contorni esterni, ignorando quelli interni (es. buchi)
    #CHAIN_APPROX_SIMPLE: comprime i contorni, memorizzando solo i punti essenziali (es. angoli), riducendo la memoria e velocizzando i calcoli successivi.
    if not contours:
        return None
    perimeter = cv2.arcLength(contours[0], True) #calcola la lunghezza del contorno, True indica che il contorno è chiuso
    circularity = min(1.0, (4 * math.pi * area) / (perimeter ** 2)) if perimeter > 0 else 0 # evito divisione per zero con il controllo

    # centralità: distanza normalizzata del centroide dal centro ottico
    coords = np.where(mask_binary > 0)
    cy, cx = int(np.mean(coords[0])), int(np.mean(coords[1]))
    max_dist = math.sqrt((img_w / 2) ** 2 + (img_h / 2) ** 2)
    dist = math.sqrt((cx - img_w / 2) ** 2 + (cy - img_h / 2) ** 2)
    centrality = 1.0 - (dist / max_dist)

    # area normalizzata, indice di vicinanza
    area_norm = min(1.0, area / (img_w * img_h * 0.15))

    # score geometrico 
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
        "mask": mask_binary,
    }


def check_occlusion(target_maybe_occluded, target_maybe_blocker):
    """
    Verifica se target_maybe_occluded è coperto da target_maybe_blocker.
    Utilizza le Bounding Box per una maggiore sensibilità nei grappoli.
    """
    # [x1, y1, x2, y2]
    box_a = target_maybe_occluded["bbox"]
    box_b = target_maybe_blocker["bbox"]

    # Calcolo area intersezione tra i rettangoli
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    if interArea == 0:
        return False

    # Calcoliamo il rapporto rispetto all'area del pomodoro occluso (il più piccolo)
    area_occluded = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    overlap_ratio = interArea / float(area_occluded)

    # Se l'intersezione copre più del 15% del rettangolo dell'occluso
    if overlap_ratio > IOU_THRESHOLD and target_maybe_occluded["area"] < target_maybe_blocker["area"]:
        return True
    
    return False


# fase 2: filtro di maturità cromatica 

def compute_maturity(image_bgr, binary_mask):
    """
    Stima la maturità del pomodoro analizzando la dominanza cromatica
    nella regione della maschera, nello spazio HSV.

    Pomodori maturi (rossi):  H ∈ [0, 15] ∪ [165, 180], S > 50, V > 50
    Pomodori acerbi (verdi):  H ∈ [35, 85],  S > 40, V > 40

    Restituisce un valore in [0, 1]: 1.0 = maturo, 0.0 = acerbo.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV) # converte immagine da BGR a HSV, per analizzare meglio il rosso e il verde.

    # maschere cromatiche
    mask_red1  = cv2.inRange(hsv, (0, 50, 50),   (15, 255, 255))
    mask_red2  = cv2.inRange(hsv, (165, 50, 50),  (180, 255, 255))
    mask_red   = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, (35, 40, 40),   (85, 255, 255))

    # intersezione con la maschera del pomodoro
    red_pixels   = int(np.sum((mask_red > 0) & (binary_mask > 0)))
    green_pixels = int(np.sum((mask_green > 0) & (binary_mask > 0)))
    total = red_pixels + green_pixels

    if total == 0:
        return 0.5  # indeterminato (es. pomodoro arancione in transizione)

    return red_pixels / total

# visualizzazione 

def overlay_mask(image, mask, color, alpha=0.45):
    """sovrappone una maschera colorata semi-trasparente sull'immagine."""
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0) #sovrappone immagine con trasparenza.


def draw_rank_marker(image, centroid, rank, color):
    """disegna solo il numero di rank sul centroide."""
    cx, cy = centroid
    cv2.circle(image, (cx, cy), 12, color, -1)
    cv2.circle(image, (cx, cy), 12, (255, 255, 255), 2)
    cv2.putText(image, str(rank), (cx - 5, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def draw_free_legend(image, free_targets, original_h):
    """disegna la legenda per l'approccio Naive (sinistra)."""
    x0 = 50
    y_ptr = original_h + 50
    
    cv2.putText(image, (x0, y_ptr),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    y_ptr += 45

    for i, det in enumerate(free_targets):
        num = i + 1
        # Riga 1: Identificativo
        cv2.putText(image, f"TARGET LIBERO #{num}", (x0, y_ptr),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y_ptr += 25
        # Riga 2: Dati
        metrics = f" > G: {det['geo_score']:.2f} | M: {det['maturity']:.0%} | A: {det['area']}px"
        cv2.putText(image, metrics, (x0, y_ptr),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y_ptr += 40

def draw_legend(image, targets, original_h):
    """disegna una legenda a blocchi verticali per garantire la leggibilità completa."""
    x0 = 50
    y_ptr = original_h + 50  # Puntatore verticale dinamico

    for i, det in enumerate(targets):
        rank = i + 1
        color = RANK_COLORS[rank]
        maturity = det.get("maturity", 0)
        
        # 1. RIGA RANK (con indicatore colorato)
        cv2.rectangle(image, (x0 - 30, y_ptr - 20), (x0 - 10, y_ptr + 5), color, -1)
        cv2.putText(image, f"POMODORO RANK {rank}", (x0, y_ptr),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y_ptr += 35

        # 2. RIGA METRICHE
        metrics_text = f" > Score Geo: {det['geo_score']:.2f} | Maturita': {maturity:.0%}"
        cv2.putText(image, metrics_text, (x0, y_ptr),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        y_ptr += 30

        # 3. RIGA AREA E PREMIO (solo se sbloccato)
        details_text = f" > Area: {det['area']}px"
        if "unlocked_by" in det:
            details_text += f" | Liberato da Target #{det['unlocked_by']}"
        
        cv2.putText(image, details_text, (x0, y_ptr),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        
        # Spazio tra un pomodoro e l'altro
        y_ptr += 60


# pipeline principale

def select_targets(candidates, image_bgr):
    """
    Pianificatore di task iterativo:
      1. Elimina doppie rilevazioni (De-duplicazione).
      2. Screma i migliori GEOMETRIC_POOL candidati.
      3. Valuta la Maturità (Criterio Decisionale Primario).
      4. Analizza le Occlusioni e esegue il Task Planning.
    """
    # fase 0: De-duplicazione (NMS su maschere)
    # Ordiniamo per confidenza YOLO per tenere la rilevazione più affidabile
    candidates.sort(key=lambda x: x.get("conf", 0), reverse=True)
    
    unique_candidates = []
    for cand in candidates:
        is_duplicate = False
        for unique in unique_candidates:
            # Calcolo IoU rapido per de-duplicazione
            inter = np.logical_and(cand["mask"], unique["mask"])
            union = np.logical_or(cand["mask"], unique["mask"])
            iou = np.sum(inter) / np.sum(union) if np.sum(union) > 0 else 0
            
            if iou > DEDUPLICATION_THRESHOLD:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_candidates.append(cand)
    
    candidates = unique_candidates

    # fase 1: ranking geometrico iniziale per scremare i candidati più "raggiungibili"
    candidates.sort(key=lambda x: x["geo_score"], reverse=True)
    pool = candidates[:GEOMETRIC_POOL]

    if not pool:
        return []

    # fase 2: calcolo maturità (Bussola della decisione)
    for det in pool:
        det["maturity"] = compute_maturity(image_bgr, det["mask"])
        det["is_selected"] = False
        det["bonus"] = 1.0
        det["occluded_by_me"] = [] # lista di chi questo pomodoro sta coprendo

    # fase 3: analisi occlusioni (Identificazione delle dipendenze nel grappolo)
    for i, a in enumerate(pool):
        for j, b in enumerate(pool):
            if i == j: continue
            # se A occlude B (A è più grande e si sovrappongono per IoU > 0.15)
            if check_occlusion(b, a):
                a["occluded_by_me"].append(b)

    selected = []
    
    # fase 4: loop di selezione iterativa (Task Planning Dinamico)
    # Prima però calcoliamo le "Scelte Libere" per il confronto richiesto dall'utente
    free_candidates = [d for d in pool if d["maturity"] >= MATURITY_THRESHOLD]
    free_candidates.sort(key=lambda x: x["geo_score"], reverse=True)
    # Ne prendiamo 5 per una visione più ampia nel confronto (come richiesto dall'utente)
    free_targets = free_candidates[:5]

    while len(selected) < MAX_TARGETS:
        # FILTRO RIGOROSO: Solo candidati non selezionati E che superano la soglia di maturità
        current_candidates = [d for d in pool if not d["is_selected"] and d["maturity"] >= MATURITY_THRESHOLD]
        
        if not current_candidates: 
            break # Se non ci sono pomodori maturi liberi, interrompi la selezione

        for d in current_candidates:
            # controlla se d è occluso da qualcuno NON ANCORA raccolto (anche se acerbo!)
            is_currently_occluded = False
            for other in pool: 
                if not other["is_selected"] and d in other["occluded_by_me"]:
                    is_currently_occluded = True
                    break
            
            penalty = PENALTY_OCCLUDED if is_currently_occluded else 1.0
            
            # score finale = geometrico * penalità occlusione * bonus sblocco
            d["final_score"] = d["geo_score"] * penalty * d["bonus"]

        # ORDINE DI RACCOLTA tra i soli maturi disponibili
        current_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        best = current_candidates[0]
        best["is_selected"] = True
        selected.append(best)

        # "PREMIO DI SBLOCCO": Recupero dei frutti che erano coperti dal target appena scelto
        for under in best["occluded_by_me"]:
            under["bonus"] = BONUS_UNLOCKED
            under["unlocked_by"] = len(selected) # ricorda chi l'ha sbloccato per la legenda

    return selected, free_targets


def process_folder(model_path=MODEL_PATH, image_dir=IMAGE_DIR, output_dir=OUTPUT_DIR):
    """Processa tutte le immagini nella cartella e salva i risultati."""

    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)

    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(extensions)])

    if not images:
        print(f"Nessuna immagine trovata in {image_dir}")
        return

    print(f"Elaborazione di {len(images)} immagini...")

    algo_results = {}

    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        results = model.predict(img_path, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        img = results.orig_img.copy()
        h, w = img.shape[:2]

        # fase 1: estrazione metriche geometriche
        candidates = []
        if results.masks is not None:
            for i, mask_tensor in enumerate(results.masks.data):
                if int(results.boxes.cls[i]) != TARGET_CLASS:
                    continue

                conf_val = float(results.boxes.conf[i])
                bbox_val = results.boxes.xyxy[i].cpu().numpy() # coordinate [x1, y1, x2, y2]
                
                m = mask_tensor.cpu().numpy()
                m = cv2.resize(m, (w, h))
                binary = (m > 0.5).astype(np.uint8) * 255

                result = compute_geometric_score(binary, h, w)
                if result is not None:
                    result["conf"] = conf_val
                    result["bbox"] = bbox_val
                    candidates.append(result)

        if not candidates:
            cv2.imwrite(os.path.join(output_dir, f"ranked_{img_name}"), img) #Salva immagine
            continue

        # fase 2: selezione con filtro maturità
        top, free = select_targets(candidates, img)

        algo_results[img_name] = {
            "targets": [
                {
                    "x": det["centroid"][0], 
                    "y": det["centroid"][1], 
                    "rank": rank,
                    "geo_score": det["geo_score"],
                    "maturity": det["maturity"],
                    "bbox": det["bbox"].tolist() if hasattr(det["bbox"], "tolist") else det["bbox"]
                }
                for rank, det in enumerate(top, 1)
            ],
            "free_targets": [
                {
                    "x": det["centroid"][0], 
                    "y": det["centroid"][1], 
                    "geo_score": det["geo_score"],
                    "maturity": det["maturity"],
                    "area": det["area"],
                    "bbox": det["bbox"].tolist() if hasattr(det["bbox"], "tolist") else det["bbox"]
                }
                for det in free
            ]
        }

        # --- VISUALIZZAZIONE E CREAZIONE LEGENDA ---
        h, w = img.shape[:2]
        
        # Aggiungiamo solo il margine nero sotto per la legenda
        margin_bottom = 500
        output = cv2.copyMakeBorder(img, 0, margin_bottom, 0, 0, 
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Disegniamo i marker e le maschere
        for rank, det in enumerate(top, 1):
            color = RANK_COLORS[rank]
            # Overlay maschera sull'area dell'immagine
            output[0:h, 0:w] = overlay_mask(output[0:h, 0:w], det["mask"], color)
            draw_rank_marker(output, det["centroid"], rank, color)
        
        # Disegniamo la legenda
        draw_legend(output, top, h)

        save_path = os.path.join(output_dir, f"ranked_{img_name}")
        cv2.imwrite(save_path, output)

    print(f"Completato. Risultati salvati in: {output_dir}/")
    with open(os.path.join(output_dir, "algorithm_results.json"), "w") as f:
        json.dump(algo_results, f, indent=2)
    print(f"JSON salvato in: {output_dir}/algorithm_results.json")


if __name__ == "__main__":
    process_folder()