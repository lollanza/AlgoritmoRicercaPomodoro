#!/usr/bin/env python3
"""
diagnostica_filtro_hsv.py
=========================
Riesegue la pipeline YOLO + filtro maturità HSV sulle 49 immagini e identifica
i candidati SCARTATI dal filtro (maturity < 0.50) — ovvero pomodori rilevati
da YOLO ma classificati come "non maturi" a causa di condizioni di illuminazione.

Output: stampa le immagini con candidati scartati, con dettagli sul punteggio
di maturità e analisi dei pixel HSV (S_mean, V_mean).
"""

import cv2
import numpy as np
import os
from ultralytics import YOLO

# Configurazione (stessi parametri dell'algoritmo)
MODEL_PATH = "runs/yolo11NewData6/weights/best.pt"
IMAGE_DIR = "immaginiPerAlgoritmo"
TARGET_CLASS = 3
CONF = 0.3
IMGSZ = 800
AREA_MIN = 1500
MATURITY_THRESHOLD = 0.5

def compute_maturity_detailed(image_bgr, binary_mask):
    """Calcola maturità + diagnostica HSV dettagliata."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask_red1  = cv2.inRange(hsv, (0, 50, 50),   (15, 255, 255))
    mask_red2  = cv2.inRange(hsv, (165, 50, 50),  (180, 255, 255))
    mask_red   = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, (35, 40, 40),   (85, 255, 255))

    roi_pixels = binary_mask > 0
    red_pixels   = int(np.sum((mask_red > 0) & roi_pixels))
    green_pixels = int(np.sum((mask_green > 0) & roi_pixels))
    total_mask   = int(np.sum(roi_pixels))
    total_classified = red_pixels + green_pixels
    unclassified = total_mask - total_classified

    if total_classified == 0:
        maturity = 0.0
    else:
        maturity = red_pixels / total_classified

    # Diagnostica HSV media sulla ROI
    h_vals = hsv[:,:,0][roi_pixels]
    s_vals = hsv[:,:,1][roi_pixels]
    v_vals = hsv[:,:,2][roi_pixels]

    return {
        "maturity": maturity,
        "red_px": red_pixels,
        "green_px": green_pixels,
        "total_mask_px": total_mask,
        "unclassified_px": unclassified,
        "unclassified_pct": unclassified / total_mask * 100 if total_mask > 0 else 0,
        "H_mean": float(np.mean(h_vals)) if len(h_vals) > 0 else 0,
        "S_mean": float(np.mean(s_vals)) if len(s_vals) > 0 else 0,
        "V_mean": float(np.mean(v_vals)) if len(v_vals) > 0 else 0,
        "S_below50_pct": float(np.sum(s_vals < 50) / len(s_vals) * 100) if len(s_vals) > 0 else 0,
        "V_below50_pct": float(np.sum(v_vals < 50) / len(v_vals) * 100) if len(v_vals) > 0 else 0,
    }


def main():
    model = YOLO(MODEL_PATH)
    images = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png"))])

    print(f"Analisi di {len(images)} immagini...\n")
    print("=" * 100)

    scartati_totali = []

    for img_name in images:
        img_path = os.path.join(IMAGE_DIR, img_name)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue
        
        h, w = image_bgr.shape[:2]
        results = model.predict(img_path, conf=CONF, imgsz=IMGSZ, verbose=False)

        if not results or results[0].masks is None:
            continue

        masks = results[0].masks.data.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        scartati = []
        validi = []

        for i, cls in enumerate(classes):
            if int(cls) != TARGET_CLASS:
                continue

            mask = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_LINEAR)
            binary = (mask > 0.5).astype(np.uint8) * 255
            area = int(np.sum(binary > 0))

            if area < AREA_MIN:
                continue

            info = compute_maturity_detailed(image_bgr, binary)
            info["area"] = area
            info["mask_idx"] = i

            if info["maturity"] < MATURITY_THRESHOLD:
                scartati.append(info)
            else:
                validi.append(info)

        if scartati:
            short_name = img_name[:50]
            print(f"\n🔴 {short_name}")
            print(f"   Validi: {len(validi)} | SCARTATI: {len(scartati)}")
            for s in scartati:
                print(f"   ❌ Candidato scartato:")
                print(f"      Maturità = {s['maturity']:.1%} (soglia: {MATURITY_THRESHOLD:.0%})")
                print(f"      Area = {s['area']}px")
                print(f"      Pixel rossi = {s['red_px']}, verdi = {s['green_px']}")
                print(f"      Non classificati = {s['unclassified_px']} ({s['unclassified_pct']:.1f}%)")
                print(f"      HSV medio: H={s['H_mean']:.1f}, S={s['S_mean']:.1f}, V={s['V_mean']:.1f}")
                print(f"      Pixel con S<50: {s['S_below50_pct']:.1f}%")
                print(f"      Pixel con V<50: {s['V_below50_pct']:.1f}%")
                scartati_totali.append((img_name, s))

    print("\n" + "=" * 100)
    print(f"\nRIEPILOGO: {len(scartati_totali)} candidati scartati su {len(images)} immagini")
    
    if scartati_totali:
        print("\nImmagini con scartati (ordinate per % pixel non classificati):")
        scartati_totali.sort(key=lambda x: x[1]["unclassified_pct"], reverse=True)
        for img_name, s in scartati_totali:
            print(f"  {img_name[:55]:55s} mat={s['maturity']:.1%}  uncl={s['unclassified_pct']:.0f}%  S_mean={s['S_mean']:.0f}  V_mean={s['V_mean']:.0f}")
    else:
        print("\n⚠️  Nessun candidato scartato trovato nel dataset.")
        print("   Il limite del filtro HSV descritto in §4.3.3 NON è empiricamente")
        print("   dimostrabile sulle 49 immagini del test set.")


if __name__ == "__main__":
    main()
