# Algoritmo di Task Planning per la Raccolta Robotizzata di Pomodori

Questo modulo implementa un sistema di selezione intelligente (Ranking) per identificare i migliori target di raccolta. L'algoritmo trasforma le maschere di segmentazione di YOLO in una **sequenza logica di azioni**, ottimizzando la raccolta in base alla maturità, alla facilità di presa e alle dipendenze fisiche tra i frutti.

Il progetto si è evoluto da una stima basata su rettangoli (BBox) a una più precisa **Approssimazione Ellittica**.

---

## 🧩 Logica Core e Stadi Comuni

Entrambi gli algoritmi (`reachability_ranking.py` e `reachability_ranking_circular.py`) condividono la struttura decisionale di base.

### 1. Screening Geometrico e GEOMETRIC_POOL
L'algoritmo non lavora su tutte le rilevazioni contemporaneamente per ottimizzare le prestazioni e la precisione.
- **De-duplicazione**: Per prima cosa, elimina maschere sovrapposte (IoU > 0.7) causate da doppie rilevazioni di YOLO.
- **Selezione del Pool (Top 8)**: Viene estratto un **GEOMETRIC_POOL di 8 candidati** con lo score geometrico più alto. 
- **Il ruolo dei frutti acerbi**: In questo pool vengono inclusi **sia i pomodori maturi che quelli acerbi**. Questa è una scelta critica: un pomodoro verde non può essere raccolto, ma se si trova davanti a uno maturo, rappresenta un **ostacolo fisico insormontabile**. Ignorarlo porterebbe il robot a collidere con il frutto verde nel tentativo di raggiungere quello rosso.

---

### 2. Fitting dell'Ellisse tramite Convex Hull
Per ottenere una geometria solida anche da maschere imperfette, l'algoritmo utilizza il concetto matematico di **Convex Hull (Involucro Convesso)**.

#### Cos'è la Convex Hull?
Immagina di piantare dei chiodi in corrispondenza di ogni pixel della maschera del pomodoro e di tendere un elastico che li circondi tutti: la forma assunta dall'elastico è la Convex Hull. È il più piccolo poligono convesso che racchiude tutti i punti.

#### Perché si usa nel progetto?
1.  **Robustezza al rumore**: La segmentazione YOLO può presentare bordi frastagliati o "buchi" interni. La Convex Hull uniforma questi errori creando una forma continua.
2.  **Gestione delle Occlusioni Sottili**: Se un piccolo rametto o un picciolo taglia visivamente in due un pomodoro, la maschera risulterà frammentata. Calcolare l'ellisse su due frammenti separati fallirebbe; calcolarla sulla Convex Hull dei due frammenti permette di **"ricostruire" virtualmente la forma intera** del frutto sottostante.
3.  **Fitting Ellittico Stabile**: La funzione `cv2.fitEllipse` richiede un set di punti coerente. Operare sulla Convex Hull garantisce che l'ellisse fittata rappresenti l'ingombro reale del frutto e non solo una porzione parziale.

---

### 5. Il Ruolo dei Frutti Acerbi come Ostacoli Permanenti

Una delle caratteristiche più avanzate del sistema è la gestione dei pomodori acerbi (verdi). Anche se non sono target di raccolta, essi influenzano drasticamente la pianificazione.

#### La Logica nel Codice
Sia in `reachability_ranking.py` che in `reachability_ranking_circular.py`, la logica segue questi passaggi chiave:

1.  **Inclusione nel Pool (Senza Filtro)**:
    Il pool dei migliori candidati viene creato basandosi solo sulla geometria, prima di conoscere il colore.
    ```python
    # reachability_ranking_circular.py (Riga 203)
    pool = candidates[:GEOMETRIC_POOL] 
    ```

2.  **Analisi delle Dipendenze Totali**:
    L'algoritmo calcola chi copre chi tra *tutti* i membri del pool.
    ```python
    # reachability_ranking_circular.py (Righe 215-220)
    for i, a in enumerate(pool):
        for j, b in enumerate(pool):
            if check_occlusion_ellipse(b, a, (h, w)):
                a["occluded_by_me"].append(b)
    ```

3.  **Penalità Inevitabile**:
    Durante la selezione, un frutto maturo (`d`) viene penalizzato se è occluso da un qualsiasi `other` presente nel pool che non sia stato rimosso (`is_selected`). Poiché i frutti acerbi non superano mai il filtro di maturità, **non vengono mai selezionati** e quindi la loro flag `is_selected` rimane sempre `False`.
    ```python
    # reachability_ranking_circular.py (Righe 231-236)
    for d in current_candidates: # Solo i maturi
        for other in pool:       # Tutti (Inclusi gli Acerbi)
            if not other["is_selected"] and d in other["occluded_by_me"]:
                is_currently_occluded = True # Scatta la penalità 0.7x
    ```

**Conseguenza**: Un pomodoro maturo dietro a uno acerbo verrà sempre visto come "occluso" e riceverà un punteggio penalizzato (0.7x). Il robot preferirà quindi raccogliere prima altri pomodori maturi liberi, garantendo la sicurezza del grappolo.

---

## 🚀 Sviluppi Futuri

L'attuale architettura è predisposta per l'integrazione di moduli avanzati per aumentare l'efficienza robotica:

1.  **Allineamento Dinamico del Gripper (Angolo di Approccio)**:
    Utilizzando l'orientamento reale fornito dal fitting ellittico (`det["ellipse"][2]`), è possibile calcolare l'angolo di rotazione ottimale per l'effettore finale del robot. Questo permette di approcciare il frutto seguendo il suo asse naturale, aumentando la stabilità della presa.

2.  **Ottimizzazione dei Percorsi (Bonus di Prossimità)**:
    Introduzione di un moltiplicatore di score basato sulla **Distanza Euclidea** tra il target appena raccolto e i candidati rimanenti. Minimizzare lo spostamento tra Rank 1 e Rank 2 riduce i tempi di ciclo e il consumo energetico del sistema.

3.  **Gestione della Maturità Graduale (Orange/Breaker Stage)**:
    Evoluzione del filtro HSV per distinguere non solo tra rosso e verde, ma anche per identificare lo stadio di invaiatura (pomodori arancioni). Questo consentirebbe una pianificazione logistica su più giorni (es. raccolta prioritaria per i frutti da mercato fresco e secondaria per quelli in fase di maturazione post-raccolta).

---

## 📊 Visualizzazione e Comparazione (`compare_results.py`)

Lo script di comparazione permette di analizzare contemporaneamente tre diverse strategie:

1.  **PANNELLO 1: NAIVE (MATURI)**
    *   **Cosa mostra**: I 5 pomodori maturi con lo score geometrico più alto.
    *   **Marker**: Bounding Box e Ellissi bianche.
    *   **Obiettivo**: Mostrare cosa sceglierebbe un robot senza logica di planning (spesso target occluser o irraggiungibili).

2.  **PANNELLO 2: TASK (BBOX ONLY)**
    *   **Cosa mostra**: La sequenza di raccolta pianificata usando i rettangoli.
    *   **Dettaglio**: Solo box colorate per Rank (1=Verde, 2=Giallo, 3=Arancio).
    *   **Obiettivo**: Evidenziare come la Bbox tenda a sovrastimare le occlusioni a causa dei suoi angoli retti.

3.  **PANNELLO 3: TASK (ELLIPSE + MASK)**
    *   **Cosa mostra**: La strategia di raccolta basata su ellissi e maschere di segmentazione.
    *   **Dettaglio**: Visualizza la maschera semitrasparente e l'ellisse di fitting.
    *   **Obiettivo**: Dimostrare la precisione superiore nel gestire le dipendenze nei grappoli complessi.

---

## ⚙️ Parametri Tecnici (Default)
| Parametro | Valore | Descrizione |
| :--- | :--- | :--- |
| `AREA_MIN` | 1500 px | Dimensione minima della maschera per la validità. |
| `IOU_THRESHOLD` | 0.15 | Soglia di sovrapposizione per stabilire un'occlusione. |
| `MATURITY_THRESHOLD`| 0.5 | Soglia minima di pixel rossi (HSV) per la raccolta. |
| `PENALTY_OCCLUDED` | 0.7x | Riduzione dello score per frutti coperti. |
| `BONUS_UNLOCKED` | 1.2x | Premio per frutti liberati dalla raccolta precedente. |

---
*Sviluppato per il progetto di Tirocinio: Deep Learning per Robotic Object Recognition.*
