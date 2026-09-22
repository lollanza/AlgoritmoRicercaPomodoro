#!/usr/bin/env python3
"""
Analisi di Benchmark rispetto al Ground Truth Umano (21 scene ad alta densità).
Confronta:
- Ground Truth Umano (annotato in annotations.json)
- Baseline Bounding Box (collega)
- Planner Ellittico proposto

Riproduce le metriche della Sezione 4.2.2 della Tesi:
- Concordanza su Rank 1 Collega (BBox): 16/21 (76.2%)
- Concordanza su Rank 1 Nostro (Ellissi): 17/21 (81.0%)
"""

import json
import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUMAN_JSON = "/Users/lollanza/Documents/RepoProf/modelliPomodori/annotations.json"
BBOX_JSON = os.path.join(BASE_DIR, "output_reachability", "algorithm_results.json")
CIRC_JSON = os.path.join(BASE_DIR, "output_reachability_circular", "algorithm_results_circular.json")
CONFRONTO_DIR = os.path.join(BASE_DIR, "confronto_visivo")
IMAGES_DIR = os.path.join(BASE_DIR, "immaginiPerAlgoritmo")

# Tolleranza in pixel per considerare coincidente il centroide stimato con il centroide umano
TOLERANCE_PX = 35.0

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def main():
    if not os.path.exists(HUMAN_JSON):
        print(f"[ERRORE] File annotazioni umane non trovato: {HUMAN_JSON}")
        sys.exit(1)
    if not os.path.exists(BBOX_JSON):
        print(f"[ERRORE] File baseline non trovato: {BBOX_JSON}")
        sys.exit(1)
    if not os.path.exists(CIRC_JSON):
        print(f"[ERRORE] File algoritmo ellittico non trovato: {CIRC_JSON}")
        sys.exit(1)

    with open(HUMAN_JSON, "r") as f:
        human_data = json.load(f).get("annotations", {})
    with open(BBOX_JSON, "r") as f:
        bbox_data = json.load(f)
    with open(CIRC_JSON, "r") as f:
        circ_data = json.load(f)

    # Le 21 scene selezionate per il benchmark
    benchmark_set = set(os.listdir(CONFRONTO_DIR)) if os.path.exists(CONFRONTO_DIR) else set()
    filelist = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and f in human_data and f in bbox_data and f in circ_data
        and f in benchmark_set
    ])

    total = len(filelist)
    print("=" * 75)
    print("  BENCHMARK A 3 VIE SUL GROUND TRUTH UMANO (21 SCENE)")
    print("=" * 75)
    print(f"Totale scene di benchmark: {total}\n")

    collega_rank1_hits = 0
    nostro_rank1_hits = 0

    print(f"{'#':<3} {'Nome File':<45} {'Umano (H1)':<12} {'Collega (C1)':<14} {'Nostro (E1)':<14}")
    print("-" * 90)

    for idx, fname in enumerate(filelist, 1):
        ht = human_data[fname].get("targets", [])
        ct = bbox_data[fname].get("targets", [])
        nt = circ_data[fname].get("targets", [])

        ph = (ht[0]["x"], ht[0]["y"]) if ht else None
        pc = (ct[0]["x"], ct[0]["y"]) if ct else None
        pn = (nt[0]["x"], nt[0]["y"]) if nt else None

        c_ok = (pc is not None and ph is not None and dist(pc, ph) <= TOLERANCE_PX)
        n_ok = (pn is not None and ph is not None and dist(pn, ph) <= TOLERANCE_PX)

        if c_ok: collega_rank1_hits += 1
        if n_ok: nostro_rank1_hits += 1

        str_h = f"({ph[0]},{ph[1]})" if ph else "None"
        str_c = f"({pc[0]},{pc[1]}) {'[OK]' if c_ok else '[NO]'}" if pc else "None"
        str_n = f"({pn[0]},{pn[1]}) {'[OK]' if n_ok else '[NO]'}" if pn else "None"

        short_name = fname[:42] + "..." if len(fname) > 45 else fname
        print(f"{idx:02d}  {short_name:<45} {str_h:<12} {str_c:<14} {str_n:<14}")

    print("-" * 90)
    print(f"\nRISULTATI DI CONCORDANZA SUL PRIMO TARGET (RANK 1):")
    print(f"  - Baseline Bounding Box (Collega): {collega_rank1_hits:2d} / {total} ({collega_rank1_hits/total*100:.1f}%)")
    print(f"  - Algoritmo Ellittico (Nostro):    {nostro_rank1_hits:2d} / {total} ({nostro_rank1_hits/total*100:.1f}%)")
    print("=" * 75)

if __name__ == "__main__":
    main()
