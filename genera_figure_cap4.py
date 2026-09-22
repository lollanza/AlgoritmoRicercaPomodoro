#!/usr/bin/env python3
"""
genera_figure_cap4.py
=====================
Genera figure composite per il Capitolo 4 usando le immagini GIÀ ANNOTATE
dall'algoritmo (ellissi, ranking, score, confronti triptych).

Figure prodotte:
  fig4_4_confronto_bbox_ellisse.png  → 2×2: BBox vs Ellipse su 2 scene (§4.2)
  fig4_5_limiti_maturita.png         → 1×3: output algoritmico su casi limite (§4.3)
  fig4_6_safefail_acerbi.png         → 1×2: output safe-fail vs caso nominale (§4.2/§4.3)

Utilizzo:
    python genera_figure_cap4.py            # genera tutte
    python genera_figure_cap4.py --show     # anteprima senza salvare
    python genera_figure_cap4.py --dpi 600  # alta risoluzione

Output in: tesi/figures/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ============================================================================
# PERCORSI
# ============================================================================

ROOT = Path(__file__).resolve().parent
OUT_ELLIPTIC = ROOT / "output_reachability_circular"
OUT_BBOX = ROOT / "output_reachability"
OUT_TRIPTYCH = ROOT / "confronto_trittico"
OUT_DIR = ROOT / "tesi" / "figures"

# ============================================================================
# STILE
# ============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.facecolor": "white",
})

LABEL_BBOX_STYLE = dict(
    boxstyle="round,pad=0.2",
    facecolor="white",
    edgecolor="black",
    alpha=0.85,
)


def _load(path: Path):
    """Carica immagine, con errore chiaro se non esiste."""
    if not path.exists():
        raise FileNotFoundError(f"Non trovata: {path}")
    return mpimg.imread(str(path))


def _add_label(ax, label: str):
    """Etichetta (a), (b) ecc. in alto a sinistra."""
    ax.text(
        0.03, 0.97, label,
        transform=ax.transAxes, fontsize=12, fontweight="bold",
        verticalalignment="top", bbox=LABEL_BBOX_STYLE,
    )


# ============================================================================
# FIGURA 4.4 — Confronto BBox vs Ellisse su 2 scene chiave (§4.2)
# ============================================================================
#
# Layout 2×2:
#   Riga 1: Scena con frutti allungati (B1)
#     (a) BBox baseline     | (b) Elliptic planner
#   Riga 2: Scena con maturo/acerbo contigui (C3)
#     (c) BBox baseline     | (d) Elliptic planner
#
# Queste 2 scene mostrano chiaramente dove i metodi divergono.

SCENE_B1 = "col_2023-08-23-12-39-06_6_png.rf.b20e615a9ea0a175ebc7119ddec29bbd.jpg"
SCENE_C3 = "col_2023-08-23-12-45-02_4_png.rf.c3ae4db6eea134c5e4fa1074315efb52.jpg"


def genera_fig4_4(dpi: int = 300, show: bool = False) -> Path:
    """Pannello 2×2: BBox vs Ellipse su 2 scene."""

    images = [
        (OUT_BBOX / f"ranked_{SCENE_B1}", "(a)", "BBox — frutti allungati"),
        (OUT_ELLIPTIC / f"ranked_{SCENE_B1}", "(b)", "Ellisse — frutti allungati"),
        (OUT_BBOX / f"ranked_{SCENE_C3}", "(c)", "BBox — maturo/acerbo contigui"),
        (OUT_ELLIPTIC / f"ranked_{SCENE_C3}", "(d)", "Ellisse — maturo/acerbo contigui"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7, 10), dpi=dpi)
    fig.subplots_adjust(wspace=0.06, hspace=0.12)

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (path, label, title), (r, c) in zip(images, positions):
        ax = axes[r][c]
        ax.imshow(_load(path))
        ax.set_axis_off()
        _add_label(ax, label)
        ax.set_title(title, fontsize=9, pad=4)

    out_path = OUT_DIR / "fig4_4_confronto_bbox_ellisse.png"
    if show:
        plt.show()
    else:
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        print(f"[OK] {out_path.name}")
    plt.close(fig)
    return out_path


# ============================================================================
# FIGURA 4.5 — Limiti del filtro maturità con output algoritmico (§4.3.3)
# ============================================================================
#
# 3 triptych affiancati verticalmente:
#   (a) Ombra profonda — tutti e 3 i valutatori concordano su 1 solo frutto
#   (b) Luce HPS — ordini divergenti tra umano e algoritmo
#   (c) Occlusione fogliare — 1 solo frutto rilevato
#
# Usiamo i triptych (Human|Colleague|Elliptic) perché mostrano sia
# l'output dell'algoritmo sia il benchmark umano.

TRIPTYCH_SHADOW = "col_2024-01-11-12-55-54_9_png.rf.6dac84c10e57bcd8dca00b03b3a90b53.jpg"
TRIPTYCH_HPS = "col_2024-01-11-16-36-05_35_png.rf.5bc99fb0a6c6aee3a8e749fa2e44de8e.jpg"
TRIPTYCH_OCCL = "1690474092232180118_color_jpeg.rf.90e84ca3b38f8a51d466333c9c5a6526.jpg"


def genera_fig4_5(dpi: int = 300, show: bool = False) -> Path:
    """3 triptych impilati verticalmente: condizioni limite."""

    triplets = [
        (OUT_TRIPTYCH / TRIPTYCH_SHADOW, "(a)", "Ombra profonda — singolo frutto visibile"),
        (OUT_TRIPTYCH / TRIPTYCH_HPS, "(b)", "Illuminazione HPS — ordini divergenti"),
        (OUT_TRIPTYCH / TRIPTYCH_OCCL, "(c)", "Occlusione fogliare — 1 candidato rilevato"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), dpi=dpi)
    fig.subplots_adjust(hspace=0.18)

    for ax, (path, label, title) in zip(axes, triplets):
        ax.imshow(_load(path))
        ax.set_axis_off()
        _add_label(ax, label)
        ax.set_title(title, fontsize=9, pad=4)

    out_path = OUT_DIR / "fig4_5_limiti_maturita.png"
    if show:
        plt.show()
    else:
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        print(f"[OK] {out_path.name}")
    plt.close(fig)
    return out_path


# ============================================================================
# FIGURA 4.6 — Safe-fail: output vuoto su solo acerbi vs caso nominale (§4.2.2)
# ============================================================================
#
# Layout 1×2:
#   (a) Output elliptic su scena "solo acerbi" → nessun rank, area nera vuota
#   (b) Output elliptic su scena nominale → ranking completo
#
# Il contrasto visivo dimostra che il pipeline NON produce falsi positivi.

SAFE_FAIL = "ranked_frame-4888_color_jpg.rf.519d7047e7c14d7bb79e8620d12ff6e7.jpg"
NOMINAL = "ranked_col_2023-08-23-13-55-02_29_png.rf.c77b98324b41a51a66776a1951fdd7ee.jpg"


def genera_fig4_6(dpi: int = 300, show: bool = False) -> Path:
    """1×2: safe-fail (solo acerbi) vs caso nominale."""

    panels = [
        (OUT_ELLIPTIC / SAFE_FAIL, "(a)", "Safe-fail: solo acerbi — output vuoto"),
        (OUT_ELLIPTIC / NOMINAL, "(b)", "Caso nominale — ranking completo"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7, 5.5), dpi=dpi)
    fig.subplots_adjust(wspace=0.08)

    for ax, (path, label, title) in zip(axes, panels):
        ax.imshow(_load(path))
        ax.set_axis_off()
        _add_label(ax, label)
        ax.set_title(title, fontsize=9, pad=4)

    out_path = OUT_DIR / "fig4_6_safefail_acerbi.png"
    if show:
        plt.show()
    else:
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        print(f"[OK] {out_path.name}")
    plt.close(fig)
    return out_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Genera figure composite Cap4 con output algoritmici."
    )
    parser.add_argument("--show", action="store_true",
                        help="Anteprima senza salvare.")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Risoluzione (default: 300).")
    parser.add_argument("--only", choices=["4_4", "4_5", "4_6"],
                        help="Genera solo una figura.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generators = {
        "4_4": genera_fig4_4,
        "4_5": genera_fig4_5,
        "4_6": genera_fig4_6,
    }

    if args.only:
        generators[args.only](dpi=args.dpi, show=args.show)
    else:
        for gen in generators.values():
            gen(dpi=args.dpi, show=args.show)

    if not args.show:
        print(f"\nOutput in: {OUT_DIR}")
        print("Comandi LaTeX:")
        for name in ["fig4_4_confronto_bbox_ellisse",
                      "fig4_5_limiti_maturita",
                      "fig4_6_safefail_acerbi"]:
            print(f"  \\includegraphics[width=\\textwidth]{{figures/{name}.png}}")


if __name__ == "__main__":
    main()
