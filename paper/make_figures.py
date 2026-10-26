"""Every figure in the paper, from paper/data/paper_data.json.

Run:  .venv/Scripts/python.exe paper/make_figures.py

Design rules followed (dataviz skill):
  * the categorical palette passed scripts/validate_palette.js in light mode: lightness band,
    chroma floor, CVD separation, normal-vision floor and contrast. The green-orange adjacent pair
    sits in the 6-8 CVD floor band, which is legal only WITH secondary encoding, so every series
    also carries its own marker and dash pattern and is direct-labelled. That also makes the
    figures readable in grayscale print, which a journal will do to them anyway.
  * no dual-axis chart anywhere: where two measures of different scale belong together they are
    two panels.
  * sequential magnitude = one hue, light to dark. Never a rainbow.
  * grid and spines are recessive; numbers and labels wear ink colours, never a series colour.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent
D = json.loads((ROOT / "data" / "paper_data.json").read_text(encoding="utf-8"))
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# validated categorical palette, in fixed order -- never cycled, never reassigned by rank
C = {"Pool": "#1f6fb4", "MHFA": "#c4571f", "LFCC": "#2a8a4a", "Graph": "#7a4fa3",
     "AdvWav": "#8a7320"}
MARK = {"Pool": "o", "MHFA": "s", "LFCC": "^", "Graph": "D", "AdvWav": "v"}
DASH = {"Pool": (0, ()), "MHFA": (0, (5, 1.5)), "LFCC": (0, (1.5, 1.5)),
        "Graph": (0, (5, 1.5, 1.5, 1.5)), "AdvWav": (0, (3, 1, 1, 1, 1, 1))}
INK, INK2, MUTED = "#1a1a1a", "#4a4a4a", "#8a8a8a"
SEQ = ["#eaf2fa", "#c3dcf0", "#8fbfe2", "#559dd0", "#2b7ab8", "#14517f"]  # one hue, light->dark

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "grid.color": "#dcdcdc", "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "lines.linewidth": 1.4, "lines.markersize": 4,
})
W = 6.5  # single-column text width, letter paper with 1 in margins
PCT = FuncFormatter(lambda v, _: f"{v * 100:g}")


def finish(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, zorder=0)
        ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=220)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- F1: system schematic
def f1_architecture():
    fig, ax = plt.subplots(figsize=(W, 2.5))
    ax.set_xlim(0, 100), ax.set_ylim(0, 44), ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f6f8", ec=MUTED, fs=7.5, lw=0.7, ls="-"):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                             fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3,
                color=INK, linespacing=1.35)

    def arrow(x1, y1, x2, y2, ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, linestyle=ls,
                                    shrinkA=1, shrinkB=1))

    box(1, 18, 13, 8, "waveform\n16 kHz")
    box(17, 18, 17, 8, "augmentation\ncodec + EnCodec", fc="#fdf6ec")
    box(37, 18, 17, 8, "WavLM-Large\n+ LoRA r=16", fc="#eaf2fa")
    ax.text(45.5, 12.5, "25 layers x 1024", ha="center", fontsize=6.8, color=MUTED)
    ax.text(45.5, 8.5, "frames [L, T, D]", ha="center", fontsize=6.8, color=MUTED)

    y = [34, 24.5, 15, 5.5]
    names = ["Pool:  time-mean of layers", "MHFA:  attentive pooling",
             "LFCC:  + cepstra from wave", "Graph:  time/layer node graph"]
    for k, (yy, nm) in enumerate(zip(y, names)):
        short = nm.split(":")[0]
        box(60, yy - 3.2, 24, 6.4, nm, fc="#ffffff", ec=C[short], lw=1.1, fs=6.8)
        arrow(54.5, 22, 59.5, yy)
        arrow(84.5, yy, 89.5, 22)
    box(90, 18, 9, 8, "mean\nfusion", fc="#f0ece f".replace(" ", ""))
    ax.text(94.5, 14.5, "score", ha="center", fontsize=6.8, color=MUTED)
    arrow(14, 22, 17, 22), arrow(34, 22, 37, 22)
    ax.text(72, 42.5, "four back-ends on ONE frozen-feature stack", ha="center", fontsize=7.2,
            color=INK2, style="italic")
    save(fig, "fig1_architecture")


# ---------------------------------------------------------------- F2: training curves
def f2_curves():
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"wspace": 0.3})
    runs = [("lora_enc", "Pool"), ("lora_mhfa", "MHFA"), ("lora_lfcc", "LFCC"),
            ("lora_graph", "Graph"), ("advwav", "AdvWav")]
    for ax, key, lab in ((axes[0], "select", "dev select EER (\\%)"),
                         (axes[1], "train_bce", "training BCE")):
        for run, name in runs:
            if run not in D["curves"]:
                continue
            c = D["curves"][run]
            ax.plot(c["epoch"], c[key], color=C[name], marker=MARK[name], linestyle=DASH[name],
                    label=name, markeredgecolor="white", markeredgewidth=0.4)
        ax.set_xlabel("epoch")
        finish(ax)
    axes[0].yaxis.set_major_formatter(PCT)
    axes[0].set_ylabel("dev select EER (%)")
    axes[1].set_ylabel("training BCE")
    axes[1].set_yscale("log")
    # Five series, so the legend carries identity on its own: the dataviz rule to ALSO direct-label
    # applies at four or fewer, and here the end-of-line labels collided with the right panel.
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(1.12, 1.30), handlelength=2.2,
                   columnspacing=1.1)
    save(fig, "fig2_training_curves")


# ---------------------------------------------------------------- F3: DET curves
def f3_det():
    fig, ax = plt.subplots(figsize=(3.2, 2.9))
    keymap = {"Pool": "Pool", "MHFA": "MHFA", "LFCC": "LFCC",
              "Pool+MHFA (selected)": "Pool", "Pool+LFCC (oracle)": "LFCC"}
    for name, cur in D["det_eval"].items():
        base = keymap[name]
        fused = "+" in name
        ax.plot(np.array(cur["far"]) * 100, np.array(cur["frr"]) * 100,
                color=C[base], linestyle="-" if fused else DASH[base],
                lw=1.8 if fused else 1.1, label=name, zorder=3 if fused else 2)
    ax.plot([0.01, 60], [0.01, 60], color=MUTED, lw=0.6, ls=":", zorder=1)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(0.05, 60), ax.set_ylim(0.05, 60)
    ax.set_xlabel("false alarm rate (%)"), ax.set_ylabel("miss rate (%)")
    ax.legend(loc="lower left", fontsize=6.5, handlelength=2.2)
    finish(ax, grid="both")
    save(fig, "fig3_det")


# ---------------------------------------------------------------- F4: per codec
def f4_codec():
    order = ["clean"] + [f"C{i:02d}" for i in range(1, 12)]
    series = ["Pool", "MHFA", "LFCC"]
    fig, ax = plt.subplots(figsize=(W, 2.3))
    x = np.arange(len(order))
    w = 0.26
    for k, s in enumerate(series):
        vals = [D["per_codec"][s].get(c, np.nan) for c in order]
        ax.bar(x + (k - 1) * w, vals, w * 0.88, color=C[s], label=s, zorder=2,
               edgecolor="white", linewidth=0.7)
    for c in ("C04", "C07"):
        ax.axvspan(order.index(c) - 0.45, order.index(c) + 0.45, color="#f3e9db", zorder=0)
    ax.annotate("neural codecs", (order.index("C04") + 0.5, 0.135), ha="center", fontsize=6.8,
                color=INK2)
    ax.set_xticks(x), ax.set_xticklabels(order)
    ax.set_ylabel("EER (%)"), ax.yaxis.set_major_formatter(PCT)
    ax.legend(ncol=3, loc="upper left", handlelength=1.4)
    finish(ax)
    save(fig, "fig4_per_codec")


# ---------------------------------------------------------------- F5: per attack
def f5_attack():
    items = sorted(D["per_attack"].items(), key=lambda kv: kv[1]["eer"])
    fig, ax = plt.subplots(figsize=(3.2, 3.1))
    y = np.arange(len(items))
    for i, (a, v) in enumerate(items):
        ax.barh(i, v["eer"], 0.72, color=C["Pool"] if not v["adversarial"] else "white",
                edgecolor=C["Pool"], hatch="" if not v["adversarial"] else "////", lw=0.8, zorder=2)
    ax.set_yticks(y), ax.set_yticklabels([a for a, _ in items])
    ax.set_xlabel("EER (%)"), ax.xaxis.set_major_formatter(PCT)
    ax.set_ylim(-0.7, len(items) - 0.3)
    for i, (a, v) in enumerate(items):
        ax.annotate(f"{v['eer'] * 100:.2f}", (v["eer"], i), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=6.3, color=INK2)
    ax.set_xlim(0, max(v["eer"] for _, v in items) * 1.22)
    handles = [mpatches.Patch(facecolor=C["Pool"], edgecolor=C["Pool"], label="TTS / VC"),
               mpatches.Patch(facecolor="white", edgecolor=C["Pool"], hatch="////",
                              label="adversarial")]
    ax.legend(handles=handles, loc="lower right", handlelength=1.3)
    finish(ax, grid="x")
    save(fig, "fig5_per_attack")


# ---------------------------------------------------------------- F6: correlation + spectrum
def f6_spectrum():
    names = ["Pool", "MHFA", "LFCC", "Graph"]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.1),
                             gridspec_kw={"width_ratios": [1.15, 1, 1], "wspace": 0.62})
    Cm = np.array(D["spectrum_eval"]["corr"])
    cmap = mpl.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    im = axes[0].imshow(Cm, cmap=cmap, vmin=0.85, vmax=1.0)
    axes[0].set_xticks(range(4)), axes[0].set_xticklabels(names, rotation=35, ha="right")
    axes[0].set_yticks(range(4)), axes[0].set_yticklabels(names)
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f"{Cm[i, j]:.3f}", ha="center", va="center", fontsize=6.2,
                         color="white" if Cm[i, j] > 0.955 else INK)
    axes[0].set_title("(a) score correlation, eval", fontsize=8, pad=9)
    for s in axes[0].spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.045, pad=0.06)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.8, width=0.5, pad=1.5)
    cb.set_ticks([0.85, 0.90, 0.95, 1.0])

    share = np.array(D["spectrum_eval"]["variance_share"]) * 100
    axes[1].bar(np.arange(1, 5), share, 0.6, color=SEQ[4], edgecolor="white", lw=0.7, zorder=2)
    for i, v in enumerate(share):
        axes[1].annotate(f"{v:.1f}", (i + 1, v), xytext=(0, 2), textcoords="offset points",
                         ha="center", fontsize=6.5, color=INK2)
    axes[1].set_xticks(range(1, 5)), axes[1].set_xlabel("principal component")
    axes[1].set_ylabel("variance explained (%)"), axes[1].set_ylim(0, 108)
    axes[1].set_title("(b) eigen-spectrum", fontsize=8, pad=9)
    finish(axes[1])

    pc1 = [D["spectrum_eval"]["pc1_loadings"][f"lora_{n.lower()}" if n != "Pool" else "lora_enc"]
           for n in names]
    axes[2].bar(np.arange(4), pc1, 0.6, color=[C[n] for n in names], edgecolor="white", lw=0.7,
                zorder=2)
    axes[2].axhline(0.5, color=MUTED, lw=0.7, ls="--", zorder=3)
    axes[2].annotate("0.50 = identical\ncontribution", (3.35, 0.5), xytext=(0, 6),
                     textcoords="offset points", ha="right", fontsize=6.2, color=INK2)
    axes[2].set_xticks(range(4)), axes[2].set_xticklabels(names, rotation=35, ha="right")
    axes[2].set_ylabel("|PC1 loading|"), axes[2].set_ylim(0, 0.62)
    axes[2].set_title("(c) PC1 loadings", fontsize=8, pad=9)
    finish(axes[2])
    save(fig, "fig6_spectrum")


# ---------------------------------------------------------------- F7: the selection failure
def f7_selection():
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True)
    for ax, tag, title in ((axes[0], "selection_2view", "(a) selection on clean + codec"),
                           (axes[1], "selection_3view", "(b) + a neural-codec view")):
        rk = D[tag]["ranking"]
        dev = np.array([r["dev_select"] for r in rk]) * 100
        ev = np.array([r["eval_eer"] for r in rk]) * 100
        ax.scatter(dev, ev, s=26, facecolor="white", edgecolor=MUTED, lw=0.9, zorder=2)
        chosen, bestev = rk[0], min(rk, key=lambda r: r["eval_eer"])
        for r, col, mk, lab in ((chosen, C["MHFA"], "o", "dev-selected"),
                                (bestev, C["LFCC"], "s", "best on eval (oracle)")):
            ax.scatter(r["dev_select"] * 100, r["eval_eer"] * 100, s=72, color=col, marker=mk,
                       zorder=4, edgecolor="white", lw=0.8, label=lab)
            ax.annotate(r["label"], (r["dev_select"] * 100, r["eval_eer"] * 100),
                        xytext=(6, -8), textcoords="offset points", fontsize=6.8, color=INK2)
        b = np.polyfit(dev, ev, 1)
        xs = np.linspace(dev.min(), dev.max(), 10)
        ax.plot(xs, np.polyval(b, xs), color=MUTED, lw=0.8, ls=":", zorder=1)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("dev select EER (%)")
        ax.annotate(f"Spearman $\\rho$ = {D[tag]['spearman']:+.3f}\n($p$ = {D[tag]['p']:.2f})",
                    (0.97, 0.05), xycoords="axes fraction", ha="right", fontsize=7, color=INK)
        finish(ax, grid="both")
    axes[0].set_ylabel("eval EER (%)")
    axes[0].legend(loc="upper left", handlelength=1.2)
    save(fig, "fig7_selection")


# ---------------------------------------------------------------- F8: in-domain vs OOD
def f8_transfer():
    fig, ax = plt.subplots(figsize=(3.2, 2.7))
    rows = sorted(D["members"], key=lambda r: r["eval_eer"])
    for r in rows:
        n = r["short"]
        ax.plot([0, 1], [r["eval_eer"] * 100, r["itw_eer"] * 100], color=C[n], marker=MARK[n],
                linestyle=DASH[n], markeredgecolor="white", markeredgewidth=0.5, label=n,
                markersize=5)
        ax.annotate(f"{n}", (1, r["itw_eer"] * 100), xytext=(6, 0), textcoords="offset points",
                    fontsize=7, color=INK2, va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ASVspoof 5 eval\n(in domain)", "In-the-Wild\n(out of domain)"])
    ax.set_xlim(-0.12, 1.32)
    ax.set_ylabel("EER (%)")
    ax.legend(loc="upper left", ncol=2, handlelength=2.0)
    finish(ax)
    save(fig, "fig8_transfer")


# ---------------------------------------------------------------- F9: progression
def f9_progress():
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    labels = ["frozen\nsingle", "4 bands\n+ full", "Pool", "Pool\n+MHFA", "Pool\n+LFCC*", "T45"]
    eer = [4.06, 3.80, 3.31, 3.21, 2.93, 2.59]
    err = [0.33, 0.17, None, None, None, None]
    cols = [MUTED, MUTED, C["Pool"], C["Pool"], C["LFCC"], "#444444"]
    xs = np.arange(len(labels))
    ax.bar(xs, eer, 0.62, color=cols, edgecolor="white", lw=0.8, zorder=2,
           hatch=["", "", "", "", "////", "...."])
    for x, (v, e) in enumerate(zip(eer, err)):
        if e is not None:
            ax.errorbar(x, v, yerr=e, color=INK2, lw=0.9, capsize=2.5, zorder=3)
        ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 4 if e is None else 9),
                    textcoords="offset points", ha="center", fontsize=6.8, color=INK2)
    ax.set_xticks(xs), ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylabel("eval EER (%)"), ax.set_ylim(0, 4.9)
    ax.annotate("* oracle-selected,\nnot claimed", (4, 2.93), xytext=(-2, 26),
                textcoords="offset points", ha="center", fontsize=6.2, color=INK2)
    finish(ax)
    save(fig, "fig9_progress")


# ---------------------------------------------------------------- F10: dataset composition
def f10_dataset():
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.9),
                             gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.3})
    parts = ["train", "dev", "eval", "itw"]
    names = ["train", "dev", "eval", "In-the-Wild"]
    bona = [D["dataset"][p]["bona"] for p in parts]
    spoof = [D["dataset"][p]["clips"] - b for p, b in zip(parts, bona)]
    y = np.arange(len(parts))[::-1]
    axes[0].barh(y, bona, 0.6, color=SEQ[4], label="bona fide", zorder=2, edgecolor="white", lw=0.8)
    axes[0].barh(y, spoof, 0.6, left=bona, color=SEQ[1], label="spoof", zorder=2,
                 edgecolor="white", lw=0.8)
    for yy, p in zip(y, parts):
        tot = D["dataset"][p]["clips"]
        axes[0].annotate(f"{tot:,}", (tot, yy), xytext=(4, 0), textcoords="offset points",
                         va="center", fontsize=6.5, color=INK2)
    axes[0].set_yticks(y), axes[0].set_yticklabels(names)
    axes[0].set_xlabel("clips"), axes[0].set_xlim(0, 820000)
    axes[0].legend(loc="lower right", handlelength=1.2)
    axes[0].set_title("(a) partition composition", fontsize=8)
    finish(axes[0], grid="x")

    cod = D["dataset"]["eval"]["codecs"]
    order = ["clean"] + [f"C{i:02d}" for i in range(1, 12)]
    vals = [cod.get(c, 0) for c in order]
    axes[1].bar(np.arange(len(order)), vals, 0.62,
                color=[SEQ[5] if c == "clean" else SEQ[3] for c in order],
                edgecolor="white", lw=0.8, zorder=2)
    axes[1].set_xticks(np.arange(len(order))), axes[1].set_xticklabels(order, fontsize=6.8)
    axes[1].set_ylabel("clips"), axes[1].set_title("(b) eval channel conditions", fontsize=8)
    axes[1].annotate("train and dev are 100% uncoded", (0.98, 0.9), xycoords="axes fraction",
                     ha="right", fontsize=6.8, color=INK)
    finish(axes[1])
    save(fig, "fig10_dataset")


for fn in (f1_architecture, f2_curves, f3_det, f4_codec, f5_attack, f6_spectrum, f7_selection,
           f8_transfer, f9_progress, f10_dataset):
    fn()
print("all figures in", FIG)
