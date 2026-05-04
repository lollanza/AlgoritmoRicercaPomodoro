import cv2
import os
import numpy as np
import json

# Configurazione cartelle
ORIGINAL_DIR = "immaginiPerAlgoritmo"
PROCESSED_DIR = "output_reachability"
JSON_PATH = os.path.join(PROCESSED_DIR, "algorithm_results.json")

# Colori per i Rank (coerenti con reachability_ranking.py)
RANK_COLORS = {
    1: (0, 255, 0),    # Verde
    2: (0, 200, 255),  # Arancio
    3: (0, 100, 255),  # Rosso-arancio
}

def compare_images():
    # Carica i risultati dell'algoritmo dal JSON
    algo_data = {}
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r") as f:
                algo_data = json.load(f)
        except Exception as e:
            print(f"Errore nel caricamento del JSON: {e}")

    # Prendi la lista dei file originali
    files = sorted([f for f in os.listdir(ORIGINAL_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not files:
        print("Nessuna immagine trovata.")
        return

    idx = 0
    total = len(files)

    while True:
        filename = files[idx]
        path_orig = os.path.join(ORIGINAL_DIR, filename)
        path_proc = os.path.join(PROCESSED_DIR, f"ranked_{filename}")

        img_orig = cv2.imread(path_orig)
        img_proc = cv2.imread(path_proc)

        if img_proc is None:
            idx = (idx + 1) % total
            continue

        # Allineamento altezza (senza offset orizzontale)
        h_proc, w_proc = img_proc.shape[:2]
        h_orig, w_orig = img_orig.shape[:2]
        
        diff_h = max(0, h_proc - h_orig)
        img_orig_padded = cv2.copyMakeBorder(img_orig, 0, diff_h, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))

        # --- DISEGNO MARKER E LEGENDE ---
        if filename in algo_data:
            # 1. IMMAGINE SINISTRA (NAIVE)
            free_targets = algo_data[filename].get("free_targets", [])
            for i, target in enumerate(free_targets, 1):
                tx, ty = target.get("x"), target.get("y")
                bbox = target.get("bbox")
                
                # Disegno Bounding Box
                if bbox:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(img_orig_padded, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    # Label piccola per la box
                    cv2.putText(img_orig_padded, f"#{i}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                if tx is not None and ty is not None:
                    cx, cy = int(tx), int(ty)
                    cv2.circle(img_orig_padded, (cx, cy), 10, (255, 255, 255), -1)
                    cv2.putText(img_orig_padded, str(i), (cx - 4, cy + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            
            # DISEGNO LEGENDA NAIVE (Sotto l'immagine di sinistra)
            x_leg, y_ptr = 40, h_orig + 40
            for i, det in enumerate(free_targets):
                # Rimosso il disegno ridondante dei marker qui, ora è gestito nel loop sopra
                
                # Scrivo i dati in legenda per tutti e 5
                cv2.putText(img_orig_padded, f"TARGET #{i+1} -> G: {det['geo_score']:.2f} | M: {det['maturity']:.0%}", (x_leg, y_ptr),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
                y_ptr += 25
                cv2.putText(img_orig_padded, f"  Area: {det['area']}px", (x_leg, y_ptr),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
                y_ptr += 35

            # 2. IMMAGINE DESTRA (TASK PLANNING)
            # (Le legende e le maschere sono già impresse in img_proc da reachability_ranking.py)
            # Aggiungiamo solo i marker se vogliamo extra visibilità o lasciamo quelli originali
            pass

        # Testo identificativo sulle immagini
        cv2.putText(img_orig_padded, "ORIGINALE", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img_proc, "RANKING", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Affiancamento
        combined = cv2.hconcat([img_orig_padded, img_proc])
        
        # --- CREAZIONE BARRA DEI COMANDI (Status Bar) ---
        header_h = 110
        header = np.zeros((header_h, combined.shape[1], 3), dtype=np.uint8)
        
        # Info immagine e Comandi
        title_text = f"IMMAGINE {idx+1}/{total}: {filename}"
        legend_text = "[SINISTRA] Box Bianche = Scelte Libere (Naive) | [DESTRA] Colorate = Task Planning"
        cmd_text = "Tasti: [FRECCIA DX / N] Avanti | [FRECCIA SX / P] Indietro | [Q / ESC] Esci"
        
        cv2.putText(header, title_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(header, legend_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(header, cmd_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        final_view = cv2.vconcat([header, combined])

        # Ridimensionamento per lo schermo
        screen_w = 1400
        if final_view.shape[1] > screen_w:
            scale = screen_w / final_view.shape[1]
            final_view = cv2.resize(final_view, (0, 0), fx=scale, fy=scale)

        cv2.imshow("Reviewer Algoritmo di Ranking", final_view)
        
        # waitKey(0) restituisce un intero a 32 bit. Su molti sistemi le frecce 
        # occupano i bit superiori o hanno codici specifici.
        key = cv2.waitKeyEx(0)
        
        # ESCI
        if key == 27 or key == ord('q') or key == ord('Q'):
            break
            
        # AVANTI: Freccia Destra (codici comuni: 2555904, 83, 3, 124, 63235) o tasto 'N'
        elif key == ord('n') or key == ord('N') or key in [2555904, 83, 3, 124, 63235, 65363]:
            idx = (idx + 1) % total
            
        # INDIETRO: Freccia Sinistra (codici comuni: 2424832, 81, 2, 123, 63234) o tasto 'P'
        elif key == ord('p') or key == ord('P') or key in [2424832, 81, 2, 123, 63234, 65361]:
            idx = (idx - 1) % total

    cv2.destroyAllWindows()

if __name__ == "__main__":
    compare_images()
