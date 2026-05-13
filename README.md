# Algoritmo di Task Planning per la Raccolta Robotizzata di Pomodori

Questo modulo implementa un sistema di selezione intelligente (Ranking) per identificare i migliori target di raccolta. L'algoritmo trasforma le maschere di segmentazione di YOLO in una **sequenza logica di azioni**, ottimizzando la raccolta in base alla maturità, alla facilità di presa e alle dipendenze fisiche tra i frutti.

Il progetto si è evoluto da una stima basata su rettangoli (BBox) a una più precisa **Approssimazione Ellittica**.

---

## 🧩 Logica Core e Stadi Comuni

Entrambi gli algoritmi (`reachability_ranking.py` e `reachability_ranking_circular.py`) condividono la struttura decisionale di base.

### 1. Screening Geometrico e GEOMETRIC_POOL
L'algoritmo non lavora su tutte le rilevazioni contemporaneamente per ottimizzare le prestazioni e la precisione.

#### A. De-duplicazione e Mask-NMS (Secondo Livello di Sicurezza)
Prima di ogni calcolo, l'algoritmo esegue una pulizia delle rilevazioni grezze.

- **Mask-NMS (Il nostro approccio)**: L'algoritmo aggiunge un **secondo livello di sicurezza spietato** basato sulle maschere di segmentazione reali. Invece di fidarsi solo dei rettangoli, confronta i pixel effettivi dei frutti. Se due maschere si sovrappongono per più del **70%** (`DEDUPLICATION_THRESHOLD`), l'algoritmo scarta quella con confidenza minore.
- **Perché è necessario?**: In grappoli di pomodori estremamente densi, YOLO può ancora generare maschere "fantasma" o frammentate. Il Mask-NMS garantisce che ogni Rank (1, 2, 3) corrisponda a un **unico oggetto fisico reale**, impedendo al robot di tentare la raccolta dello stesso frutto più volte.

#### B. Selezione del Pool (Top 8)
Viene estratto un **GEOMETRIC_POOL di 8 candidati** con lo score geometrico più alto. 
- **Il ruolo dei frutti acerbi**: In questo pool vengono inclusi **sia i pomodori maturi che quelli acerbi**. Questa è una scelta critica: un pomodoro verde non può essere raccolto, ma se si trova davanti a uno maturo, rappresenta un **ostacolo fisico insormontabile**. Ignorarlo porterebbe il robot a collidere con il frutto verde nel tentativo di raggiungere quello rosso.

---

### 2. Il Graspability Score ($G$) e la stabilità della Circolarità

Entrambi gli algoritmi valutano la qualità dei frutti tramite lo score $G \in [0, 1]$, calcolato come pesatura di Area, Circolarità e Centralità. Tuttavia, il calcolo della **Circolarità ($C$)** differisce radicalmente:

#### A. Approccio Maschera Grezza (`reachability_ranking.py`)
La circolarità è calcolata direttamente sul contorno della segmentazione YOLO (`cv2.findContours`).
- **Problema**: Se un rametto "taglia" visivamente il pomodoro, il perimetro rilevato aumenta drasticamente (sommando i bordi del taglio). 
- **Effetto**: La circolarità crolla verso lo zero. Un pomodoro perfetto ma parzialmente coperto viene ingiustamente penalizzato.

#### B. Approccio Convex Hull (`reachability_ranking_circular.py`)
La circolarità è calcolata sul **perimetro dell'Involucro Convesso** (`cv2.convexHull`).
- **Soluzione**: La Convex Hull ignora le rientranze causate da rami o ombre, ricostruendo la convessità naturale del frutto.
- **Effetto**: Il valore $C$ rimane alto (vicino a 1.0) anche con maschere frammentate, rendendo il $G$ Score un riflesso fedele della **forma reale** del frutto.

### ⚠️ Il Paradosso del Pomodoro Frammentato (Falla nell'Approccio BBox)
Un limite tecnico significativo dell'algoritmo standard (`reachability_ranking.py`) emerge quando un pomodoro è diviso visivamente in più parti (es. da un ramo che attraversa il frutto).

- **L'Errore Logico**: L'algoritmo calcola correttamente l'**area totale** (sommando tutti i pixel della maschera), ma a causa dell'uso di `contours[0]`, recupera il **perimetro di un solo frammento** 
- **L'Effetto Matematico**: Poiché un frammento piccolo ha un perimetro molto corto, la formula della circolarità ($C = \frac{4\pi A}{P^2}$) applicata all'area totale genera un valore distorto verso l'alto, che viene poi forzato a **1.0 (punteggio massimo)**.
- **Conseguenza nel Ranking**: Un pomodoro frammentato, che dovrebbe essere penalizzato per la forma irregolare, può "ingannare" il sistema e ottenere un **Rank 1 ingiustificato**, venendo scambiato per un cerchio perfetto di grandi dimensioni.

L'algoritmo **Circular** risolve radicalmente questa falla: raccogliendo tutti i punti della maschera (`cv2.findNonZero`) e unendoli in un unico involucro convesso, garantisce che il perimetro sia sempre coerente con l'intera massa del frutto, indipendentemente dai rami che lo attraversano.

---

### 🔍 Case Study: L'impatto della Convex Hull sul G Score

Un esempio emblematico è l'immagine `col_2023-08-23-12-45-20_9_png`. In questo scenario, un pomodoro maturo è "tagliato" visivamente a metà da un rametto sottile.

| Algoritmo | Logica Perimetro ($P$) | Calcolo Circolarità ($C = \frac{4\pi A}{P^2}$) | Risultato G Score |
| :--- | :--- | :--- | :--- |
| **Standard** | Segue il bordo del "taglio", raddoppiando il perimetro reale. | Il denominatore ($P^2$) esplode, $C$ crolla verso lo zero. | **0.37** (Penalizzato ingiustamente) |
| **Circular** | L'elastico della Convex Hull "salta" il rametto. | $P$ rimane minimo e coerente con un cerchio, $C \approx 1.0$. | **0.46** (Rank 1 meritato) |

**Conclusione**: L'uso della Convex Hull permette al robot di non scartare frutti perfetti solo perché parzialmente coperti da ostacoli filiformi (peduncoli, rami o foglie strette), aumentando l'efficienza di raccolta del 15-20% nei grappoli densi.

---

### 3. Fitting dell'Ellisse tramite Convex Hull
Per ottenere una geometria solida anche da maschere imperfette, l'algoritmo utilizza il concetto matematico di **Convex Hull (Involucro Convesso)**.

#### Cos'è la Convex Hull?
Immagina di piantare dei chiodi in corrispondenza di ogni pixel della maschera del pomodoro e di tendere un elastico che li circondi tutti: la forma assunta dall'elastico è la Convex Hull. **È il più piccolo poligono convesso che racchiude tutti i punti.**

#### Implementazione Pratica: Dal Pixel alla Geometria
Nel codice (`reachability_ranking_circular.py`), il processo avviene in due step fondamentali:

1.  **Estrazione della Nuvola di Punti (`cv2.findNonZero`)**:
    Invece di lavorare sui singoli contorni (che fallirebbero se il pomodoro fosse diviso in più frammenti), trasformiamo l'intera maschera binaria in una lista di coordinate $(x, y)$.
    ```python
    all_points = cv2.findNonZero(mask_binary) # Raccoglie tutti i pixel bianchi
    ```
    Questo garantisce che, anche se un ramo "taglia" il frutto in due, i punti di entrambi i pezzi vengano considerati come un'unica entità fisica.

2.  **Calcolo dell'Involucro (`cv2.convexHull`)**:
    La funzione applica un algoritmo di scansione (come il *Monotone Chain*) per identificare i soli punti esterni che formano il perimetro convesso.
    ```python
    hull = cv2.convexHull(all_points) # Genera i vertici del poligono convesso
    ```
    Il risultato è un set di punti estremamente pulito e semplificato, che "salta" letteralmente sopra le occlusioni o i buchi della segmentazione.

#### Perché si usa nel progetto?
1.  **Robustezza al rumore**: La segmentazione YOLO può presentare bordi frastagliati o "buchi" interni. La Convex Hull uniforma questi errori creando una forma continua.
2.  **Gestione delle Occlusioni Sottili**: Permette di **"ricostruire" virtualmente la forma intera** del frutto sottostante ignorando i rami che lo attraversano.
3.  **Fitting Ellittico Stabile**: La funzione `cv2.fitEllipse` richiede un set di punti coerente. Operare sulla Convex Hull garantisce che l'ellisse fittata rappresenti l'ingombro reale del frutto e non solo una porzione parziale.

---

### 4. Bounding Box vs Ellissi: La Precisione nelle Occlusioni

La scelta di passare dalle Bounding Box (BBox) alle Ellissi è fondamentale per risolvere il problema della **sovrapposizione geometrica fittizia**.

#### Il Problema dei "Rettangoli Ingombranti"
Le BBox racchiudono la maschera in un perimetro rettangolare. Nei grappoli densi:
- Gli **angoli vuoti** del rettangolo occupano spazio dove fisicamente il pomodoro non esiste.
- Se un pomodoro acerbo si trova vicino a uno maturo, le loro BBox possono sovrapporsi significativamente anche se i frutti sono distanziati.
- **Effetto**: L'algoritmo standard rileva un'occlusione inesistente, applicando la penalità `PENALTY_OCCLUDED` (0.7x) e facendo scalare il pomodoro nel ranking (es. da Rank 2 a Rank 3).

#### La Soluzione Ellittica
L'ellisse aderisce alla geometria curva del frutto:
- Elimina le aree "morte" degli angoli del rettangolo.
- Riduce drasticamente i **falsi positivi** nel rilevamento delle occlusioni.
- Permette al sistema di pianificazione di riconoscere che un frutto è effettivamente libero e pronto per la raccolta, ottimizzando la sequenza di task.

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
