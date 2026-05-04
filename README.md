# Algoritmo di Task Planning per la Raccolta Robotizzata di Pomodori

Questo modulo implementa un sistema di selezione intelligente (Ranking) per identificare i migliori target di raccolta. L'algoritmo trasforma le maschere di segmentazione di YOLO in una **sequenza logica di azioni**, ottimizzando la raccolta in base alla maturità, alla facilità di presa e alle dipendenze fisiche tra i frutti.

## 🧩 Logica dell'Algoritmo (`reachability_ranking.py`)

Il sistema opera attraverso una pipeline multi-stadio progettata per la massima efficienza robotica.

### 0. De-duplicazione (Filtro NMS)
Prima di ogni analisi, l'algoritmo elimina le **doppie rilevazioni** dello stesso frutto:
- **Analisi IoU**: Se due maschere si sovrappongono per più del **70%** (`DEDUPLICATION_THRESHOLD`), vengono considerate come lo stesso oggetto.
- **Selezione per Confidenza**: Tra i duplicati, viene mantenuta solo la rilevazione con il valore di confidenza di YOLO più elevato, scartando i "fantasmi" meno certi.

### 1. Screening Geometrico: Graspability Score ($G$)
Filtro iniziale basato sulla qualità geometrica. Viene applicata una soglia di **Area Minima ($\ge 1500$ pixel)**. Per i candidati validi, si calcola lo score $G \in [0, 1]$:
- **Area (45%)**: Proxy della vicinanza (normalizzata rispetto al 15% dell'area totale).
- **Circolarità (35%)**: $C = \frac{4\pi \cdot Area}{Perimetro^2}$. Identifica frutti isolati ($C \approx 1.0$) rispetto a quelli coperti.
- **Centralità (20%)**: Favorisce i target vicini al centro dell'immagine per facilitare il movimento del braccio.

### 2. Stadio Cromatico: Il Filtro di Maturità (HSV)
Questo è il **criterio decisionale bloccante**. L'algoritmo valuta la qualità del frutto per stabilire se sia idoneo alla raccolta:
- **Analisi Regionale**: L'analisi avviene nello spazio HSV all'interno della maschera, isolando i pixel rossi ($H \in [0, 15] \cup [165, 180]$) e verdi ($H \in [35, 85]$).
- **Indice di Maturità ($M$)**: Rapporto tra pixel rossi e la somma dei pixel cromaticamente identificati (Rossi / [Rossi + Verdi]).
- **Soglia di Accettazione ($\ge 0.5$)**: Un frutto viene considerato per la raccolta **solo se** l'indice $M$ è almeno del 50%. Se non rilevati pixel cromatici, viene assegnato un valore neutro di 0.5.
- **Filtro Rigoroso**: Se nessun frutto nel pool dei candidati supera questa soglia, l'algoritmo non seleziona alcun target.

### 3. Analisi delle Occlusioni (Bbox Overlap)
Per determinare l'ordine logico di raccolta, l'algoritmo analizza le relazioni spaziali tra i migliori 8 candidati utilizzando le **Bounding Box (Bbox)** fornite da YOLO:
- **Perché le Bbox?** Mentre le maschere rappresentano solo i pixel visibili, le Bbox catturano l'ingombro fisico totale del frutto. Nei grappoli densi, i rettangoli si sovrappongono molto più facilmente delle maschere, permettendo di identificare occlusioni anche minime che altrimenti passerebbero inosservate.
- **Rapporto di Sovrapposizione**: L'algoritmo calcola l'area di intersezione tra due rettangoli e la rapporta all'area del frutto potenzialmente occluso.
- **Rilevamento ($> 0.15$)**: Se un pomodoro "davanti" (più grande) copre più del 15% dello spazio del rettangolo di un pomodoro "dietro" (più piccolo), viene stabilita una dipendenza gerarchica.
- **Criterio di Profondità**: Il frutto con l'area della maschera maggiore è considerato l'**Occludente** (davanti), il più piccolo è l'**Occluso** (dietro).

### 4. Task Planning: Gestione Occlusioni e Sblocco
Una volta identificati i frutti idonei, il sistema pianifica la **sequenza logica** per gestire i grappoli:
1. **Selezione Esclusiva**: Il loop di selezione opera **esclusivamente sui frutti maturi**. Se non ci sono maturi liberi, la selezione si interrompe.
2. **Consapevolezza degli Ostacoli**: Anche se i frutti acerbi non vengono selezionati, l'algoritmo ne tiene conto come ostacoli fisici. Un frutto maturo riceve la **Penalità Occlusione (0.7x)** se è coperto da un qualsiasi altro frutto (anche acerbo) ancora presente sulla pianta.
3. **Bonus di Sblocco (1.2x)**: I frutti maturi sbloccati dalla rimozione di un target precedente ricevono un bonus del 20% per favorire la raccolta in profondità.

## 📊 Output e Integrazione

1.  **Visualizzazione con Margine Tecnico**: Per garantire la massima visibilità, l'algoritmo non sovrappone la legenda ai frutti. L'immagine di output viene espansa con un **margine nero di 160px** sul fondo:
    - **Overlay**: Le maschere colorate e i marker sono disegnati esclusivamente sull'area della foto originale.
    - **Legenda Esterna**: Tutti i dati tecnici ($G$, $M$, Area) sono ospitati nel margine inferiore, evitando di coprire i pixel dei frutti.
2.  **Gerarchia Visiva**: I target sono evidenziati per priorità:
    - 🟢 **Rank 1**: Verde (Miglior target)
    - 🟠 **Rank 2**: Arancio
    - 🔴 **Rank 3**: Rosso-arancio
3.  **Legenda Informativa**: Segnala esplicitamente se un frutto è stato selezionato grazie allo sblocco (es. `[Sbloccato da #1]`).
4.  **Export JSON**: Coordinate $(x, y)$ dei centroidi per l'interfacciamento con il software del braccio robotico.

## 🔍 Validazione e Confronto (`compare_results.py`)

Per valutare l'efficacia del sistema di Task Planning, il progetto include uno strumento di revisione che permette di confrontare visivamente due diverse filosofie di scelta.

### Analisi Comparativa (Side-by-Side)
Eseguendo `python compare_results.py`, l'interfaccia mostra:

1.  **A SINISTRA: Scelte Libere (Approccio Naive)**
    *   **Logica**: Rappresenta un robot che segue un istinto "estetico". Filtra i pomodori maturi e sceglie semplicemente i 3 più grandi, circolari e centrali.
    *   **Limite**: Ignora totalmente le occlusioni. Nelle immagini con grappoli densi, questo approccio evidenzierebbe spesso frutti fisicamente irraggiungibili perché coperti da altri (anche acerbi).
    *   **Marker**: Box sottili bianche (`FREE #1`, `FREE #2`, etc.).

2.  **A DESTRA: Task Planning (Algoritmo Proposto)**
    *   **Logica**: Implementa la sequenza logica di raccolta. Valuta se un frutto è "libero" o "bloccato" e pianifica l'ordine di rimozione per minimizzare le interferenze.
    *   **Vantaggio**: Dimostra come il ranking cambi dinamicamente: un frutto geometricamente perfetto può finire in Rank 3 se è occluso, lasciando il posto a un frutto più piccolo ma immediatamente accessibile.
    *   **Marker**: Box spesse colorate (`R1`, `R2`, `R3`) coerenti con la legenda tecnica.

### Comandi del Viewer
*   **Frecce Direzionali / N-P**: Navigazione rapida tra le immagini del dataset.
*   **Q / ESC**: Esce dal programma.

## ⚙️ Parametri Tecnici (Default)
| Parametro | Valore | Descrizione |
| :--- | :--- | :--- |
| `AREA_MIN` | 1500 px | Dimensione minima del frutto |
| `DEDUPLICATION_THRESHOLD`| 0.70 | Soglia per eliminare doppie rilevazioni |
| `IOU_THRESHOLD`| 0.15 | Soglia per rilevamento occlusioni |
| `MATURITY_THRESHOLD` | 0.5 | Rapporto minimo pixel rossi |
| `GEOMETRIC_POOL`| 8 | Numero di candidati per analisi dinamica |
| `MAX_TARGETS` | 3 | Target selezionati per immagine |

---
*Sviluppato per il progetto di Tirocinio: Deep Learning per Robotic Object Recognition.*
