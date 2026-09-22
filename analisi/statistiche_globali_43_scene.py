#!/usr/bin/env python3
"""
Analisi Quantitativa Globale sulle 43 scene di test.
Confronto tra la baseline Bounding Box (collega) e l'approccio Ellittico proposto.
Riproduce i dati della Sezione 4.2.2 della Tesi:
- 27 scene a sequenza identica (62.8%)
- 12 scene divergenti (27.9%)
- 4 scene vuote/safe-fail (9.3%)
"""

import json
import os
import sys

# Percorsi ai risultati JSON
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BBOX_JSON = os.path.join(BASE_DIR, "output_reachability", "algorithm_results.json")
CIRC_JSON = os.path.join(BASE_DIR, "output_reachability_circular", "algorithm_results_circular.json")

def main():
    if not os.path.exists(BBOX_JSON):
        print(f"[ERRORE] File non trovato: {BBOX_JSON}")
        sys.exit(1)
    if not os.path.exists(CIRC_JSON):
        print(f"[ERRORE] File non trovato: {CIRC_JSON}")
        sys.exit(1)

    with open(BBOX_JSON, "r") as f:
        bbox_data = json.load(f)
    with open(CIRC_JSON, "r") as f:
        circ_data = json.load(f)

    common_keys = sorted(set(bbox_data.keys()).intersection(circ_data.keys()))
    total = len(common_keys)

    identical = []
    divergent = []
    empty = []

    for k in common_keys:
        tb = [(t["x"], t["y"], t.get("rank", idx + 1)) for idx, t in enumerate(bbox_data[k].get("targets", []))]
        tc = [(t["x"], t["y"], t.get("rank", idx + 1)) for idx, t in enumerate(circ_data[k].get("targets", []))]

        if not tb and not tc:
            empty.append(k)
        elif tb == tc:
            identical.append(k)
        else:
            divergent.append((k, tb, tc))

    print("=" * 70)
    print("  STATISTICHE GLOBALI: BASELINE BBOX vs PLANNER ELLITTICO PROPOSTO")
    print("=" * 70)
    print(f"Totale scene analizzate: {total}")
    print(f"  - Sequenze IDENTICHE:  {len(identical):2d} / {total} ({len(identical)/total*100:.1f}%)")
    print(f"  - Sequenze DIVERGENTI: {len(divergent):2d} / {total} ({len(divergent)/total*100:.1f}%)")
    print(f"  - Scene VUOTE (Safe):  {len(empty):2d} / {total} ({len(empty)/total*100:.1f}%)")
    print("=" * 70)

    print("\n--- DETTAGLIO DELLE 12 SCENE DIVERGENTI ---")
    for idx, (k, tb, tc) in enumerate(divergent, 1):
        print(f"\n[{idx:02d}] {k}")
        print(f"     BBox:    {tb}")
        print(f"     Ellissi: {tc}")

if __name__ == "__main__":
    main()
