# Script di Analisi Quantitativa per la Tesi

Questa cartella contiene gli script Python per estrarre e verificare in modo riproducibile tutti i dati quantitativi citati nel **Capitolo 4** della tesi (*Validazione Sperimentale e Discussione*).

---

## 1. Statistiche Globali sulle 43 Scene (`statistiche_globali_43_scene.py`)
Confronta le sequenze di raccolta generate per tutte le 43 scene di test tra:
- Baseline a **Bounding Box** (`output_reachability/algorithm_results.json`)
- Planner a **Ellissi Orientate** (`output_reachability_circular/algorithm_results_circular.json`)

### Esecuzione:
```bash
./venv/bin/python3 analisi/statistiche_globali_43_scene.py
```

### Valori verificati (Tesi Sez. 4.2.2):
- **Scene identiche**: 27 / 43 (62.8%)
- **Scene divergenti**: 12 / 43 (27.9%)
- **Scene vuote / safe-fail**: 4 / 43 (9.3%)

---

## 2. Benchmark con Ground Truth Umano sulle 21 Scene (`benchmark_ground_truth_21_scene.py`)
Valuta l'accuratezza nella scelta del primo pomodoro da raccogliere (Rank 1) rispetto alle annotazioni umane memorizzate in `/Users/lollanza/Documents/RepoProf/modelliPomodori/annotations.json` per il sottoinsieme di 21 scene critiche ad alta densità (`confronto_visivo/`).

### Esecuzione:
```bash
./venv/bin/python3 analisi/benchmark_ground_truth_21_scene.py
```

### Valori verificati (Tesi Sez. 4.2.2):
- **Concordanza Baseline Collega (BBox)**: 17 / 21 (81.0%)
- **Concordanza Planner Ellittico (Nostro)**: 17 / 21 (81.0%)

---

## 3. Analisi della Circolarità: Convex Hull vs Contorno Grezzo (`trova_differenze_circolarita.py`)
Confronta la circolarità $C = \frac{4\pi A}{P^2}$ calcolata sul contorno grezzo della maschera (`findContours`, baseline) rispetto all'involucro convesso (`convexHull`, nostro approccio) su tutte le maschere del dataset.

### Esecuzione:
```bash
./venv/bin/python3 analisi/trova_differenze_circolarita.py
```

### Valori verificati (Tesi Sez. 4.3.2, Caso Studio 2 `col_2023-08-23-12-48-32_5`):
- **Perimetro grezzo (Collega)**: $P = 290.1\text{ px} \implies C = 0.417$
- **Perimetro Convex Hull (Nostro)**: $P = 216.9\text{ px} \implies C = 0.746$
- **Delta Circolarità**: $+0.329$ ($+78.9\%$)
- **Incremento GeoScore $G$** ($w_{\text{circ}} = 0.35$): $+0.115$
