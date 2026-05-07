import cv2
import os
import numpy as np
import json

# Configurazione cartelle
ORIGINAL_DIR = "immaginiPerAlgoritmo"
PROCESSED_BBOX_DIR = "output_reachability"
PROCESSED_CIRC_DIR = "output_reachability_circular"

JSON_BBOX_PATH = os.path.join(PROCESSED_BBOX_DIR, "algorithm_results.json")
JSON_CIRC_PATH = os.path.join(PROCESSED_CIRC_DIR, "algorithm_results_circular.json")

# Colori per i rank (BGR)
RANK_COLORS = {
    1: (0, 255, 0),    # Verde
    2: (0, 200, 255),  # Giallo/Arancio
    3: (0, 100, 255),  # Arancio/Rosso
}

def compare_images():
    # ... (caricamento JSON invariato)
    algo_bbox_data = {}
    algo_circ_data = {}
    
    if os.path.exists(JSON_BBOX_PATH):
        try:
            with open(JSON_BBOX_PATH, "r") as f:
                algo_bbox_data = json.load(f)
        except Exception as e:
            print(f"Errore caricamento JSON BBox: {e}")

    if os.path.exists(JSON_CIRC_PATH):
        try:
            with open(JSON_CIRC_PATH, "r") as f:
                algo_circ_data = json.load(f)
        except Exception as e:
            print(f"Errore caricamento JSON Circolare: {e}")

    files = sorted([f for f in os.listdir(ORIGINAL_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not files:
        print("Nessuna immagine trovata.")
        return

    idx = 0
    total = len(files)

    while True:
        filename = files[idx]
        path_orig = os.path.join(ORIGINAL_DIR, filename)
        path_circ = os.path.join(PROCESSED_CIRC_DIR, f"ranked_{filename}")

        img_orig = cv2.imread(path_orig)
        img_circ = cv2.imread(path_circ)

        if img_orig is None or img_circ is None:
            print(f"Mancano immagini per {filename}")
            idx = (idx + 1) % total
            continue

        h_orig, w_orig = img_orig.shape[:2]
        
        # Creiamo i tre pannelli partendo dall'originale per i primi due
        margin_bottom = 500
        def create_panel(base_img):
            return cv2.copyMakeBorder(base_img, 0, margin_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))

        img_naive_p = create_panel(img_orig.copy())
        img_bbox_task_p = create_panel(img_orig.copy())
        img_circ_task_p = cv2.copyMakeBorder(img_circ, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0)) # Già ha la sua legenda

        # --- PANNELLO 1: NAIVE (BBOX CIANO + ELLISSI BIANCHE) ---
        if filename in algo_bbox_data:
            free_targets = algo_bbox_data[filename].get("free_targets", [])
            for i, target in enumerate(free_targets, 1):
                tx, ty = target.get("x"), target.get("y")
                
                # Bounding Box (Ciano chiaro, spessore 2)
                bbox = target.get("bbox")
                
                if bbox:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(img_naive_p, (x1, y1), (x2, y2), (255, 255, 0), 2, cv2.LINE_AA)
                
                # Ellisse (Bianca, spessore 2)
                # FALLBACK: Cerchiamo l'ellisse nel JSON circolare per disegnarla (solo estetica)
                ellipse = None
                if filename in algo_circ_data:
                    for other in algo_circ_data[filename].get("free_targets", []):
                        ox, oy = other.get("x"), other.get("y")
                        if abs(tx - ox) < 8 and abs(ty - oy) < 8:
                            ellipse = other.get("ellipse")
                            break
                
                if ellipse:
                    center = (int(ellipse[0][0]), int(ellipse[0][1]))
                    axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
                    angle = ellipse[2]
                    cv2.ellipse(img_naive_p, center, axes, angle, 0, 360, (255, 255, 255), 2, cv2.LINE_AA)
                
                if tx is not None and ty is not None:
                    cx, cy = int(tx), int(ty)
                    cv2.circle(img_naive_p, (cx, cy), 10, (255, 255, 255), -1)
                    cv2.putText(img_naive_p, str(i), (cx - 4, cy + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

            # Legenda Naive
            x_leg, y_ptr = 50, h_orig + 50
            for i, det in enumerate(free_targets):
                info_text = f"#{i+1}  G: {det['geo_score']:.2f} | M: {det['maturity']:.0%} | A: {det['area']}px"
                cv2.rectangle(img_naive_p, (x_leg - 30, y_ptr - 15), (x_leg - 10, y_ptr + 2), (255, 255, 255), 1)
                cv2.putText(img_naive_p, info_text, (x_leg, y_ptr), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
                y_ptr += 35

        # --- PANNELLO 2: BBOX TASK (SOLO BBOX COLORATE) ---
        if filename in algo_bbox_data:
            targets = algo_bbox_data[filename].get("targets", [])
            for target in targets:
                rank = target.get("rank")
                color = RANK_COLORS.get(rank, (255, 255, 255))
                bbox = target.get("bbox")
                if bbox:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(img_bbox_task_p, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                
                tx, ty = target.get("x"), target.get("y")
                if tx is not None and ty is not None:
                    cx, cy = int(tx), int(ty)
                    cv2.circle(img_bbox_task_p, (cx, cy), 12, color, -1)
                    cv2.circle(img_bbox_task_p, (cx, cy), 12, (255, 255, 255), 2)
                    cv2.putText(img_bbox_task_p, str(rank), (cx - 5, cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Legenda BBox Task
            x_leg, y_ptr = 50, h_orig + 50
            for i, target in enumerate(targets):
                rank = i + 1
                color = RANK_COLORS.get(rank, (255, 255, 255))
                geo = target.get("geo_score", 0)
                mat = target.get("maturity", 0)
                
                cv2.rectangle(img_bbox_task_p, (x_leg - 30, y_ptr - 20), (x_leg - 10, y_ptr + 5), color, -1)
                cv2.putText(img_bbox_task_p, f"RANK {rank} (BBOX)", (x_leg, y_ptr), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_ptr += 35
                cv2.putText(img_bbox_task_p, f" > G: {geo:.2f} | M: {mat:.0%}", (x_leg, y_ptr), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_ptr += 60

        # --- PANNELLO 3: TASK CIRCOLARE (ORIGINALE DA DISCO: MASCHERA + ELLISSE) ---
        # img_circ_task_p contiene già img_circ con maschere ed ellissi colorate. Non aggiungiamo altro.



        # Testi identificativi
        cv2.putText(img_naive_p, "1. NAIVE (MATURI)", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img_bbox_task_p, "2. TASK (BBOX ONLY)", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img_circ_task_p, "3. TASK (CIRCLE + MASK)", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --- SICUREZZA: Allineamento altezze prima di hconcat ---
        h1, h2, h3 = img_naive_p.shape[0], img_bbox_task_p.shape[0], img_circ_task_p.shape[0]
        max_h = max(h1, h2, h3)
        
        def final_pad(img, target_h):
            h, w = img.shape[:2]
            if h < target_h:
                return cv2.copyMakeBorder(img, 0, target_h - h, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            return img

        img_naive_p = final_pad(img_naive_p, max_h)
        img_bbox_task_p = final_pad(img_bbox_task_p, max_h)
        img_circ_task_p = final_pad(img_circ_task_p, max_h)

        # Affiancamento dei tre pannelli
        combined = cv2.hconcat([img_naive_p, img_bbox_task_p, img_circ_task_p])

        
        # Header Status Bar
        header_h = 100
        header = np.zeros((header_h, combined.shape[1], 3), dtype=np.uint8)
        title_text = f"COMPARISON [{idx+1}/{total}]: {filename}"
        cmd_text = "N: Next | P: Prev | Q: Quit"
        cv2.putText(header, title_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(header, cmd_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        
        final_view = cv2.vconcat([header, combined])

        # Resize per lo schermo (molto largo ora che sono 3 immagini)
        screen_w = 1800
        if final_view.shape[1] > screen_w:
            scale = screen_w / final_view.shape[1]
            final_view = cv2.resize(final_view, (0, 0), fx=scale, fy=scale)

        cv2.imshow("Reviewer: Naive vs BBox vs Circular", final_view)
        
        key = cv2.waitKeyEx(0)
        if key == 27 or key == ord('q') or key == ord('Q'):
            break
        elif key == ord('n') or key == ord('N') or key in [2555904, 83, 3, 124, 63235, 65363]:
            idx = (idx + 1) % total
        elif key == ord('p') or key == ord('P') or key in [2424832, 81, 2, 123, 63234, 65361]:
            idx = (idx - 1) % total

    cv2.destroyAllWindows()

if __name__ == "__main__":
    compare_images()
