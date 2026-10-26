"""Collect every number the paper reports into one JSON, so text and figures cannot disagree."""
from __future__ import annotations

import itertools
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "D:/AUDIO_DEEPFAKE/src")
from caesd.metrics import compute_det_curve, compute_eer, min_cllr, min_dcf  # noqa: E402

R = Path("D:/AUDIO_DEEPFAKE/experiments/runs_partial")
OUT = Path("D:/AUDIO_DEEPFAKE/paper/data")
OUT.mkdir(parents=True, exist_ok=True)
MEMBERS = ["lora_enc", "lora_mhfa", "lora_lfcc", "lora_graph"]
SHORT = {"lora_enc": "Pool", "lora_mhfa": "MHFA", "lora_lfcc": "LFCC", "lora_graph": "Graph"}
EER = lambda s, y: compute_eer(s, y)[0]  # noqa: E731

d: dict = {}

# ---- calib statistics, used for every z-score ------------------------------------------------
stats = {}
for m in MEMBERS:
    c = pd.concat([pd.read_parquet(R / m / f"scores_dev_{v}.parquet") for v in ("clean", "codec")])
    c = c[c["split"] == "calib"]["score"]
    stats[m] = (float(c.mean()), float(c.std()))
d["calib_stats"] = stats


def zframe(setname: str, members=MEMBERS) -> pd.DataFrame:
    f = None
    for m in members:
        p = pd.read_parquet(R / m / f"scores_{setname}.parquet")[["utt", "label", "score"]]
        p[m] = (p["score"] - stats[m][0]) / stats[m][1]
        p = p.drop(columns="score")
        f = p if f is None else f.merge(p.drop(columns="label"), on="utt")
    return f


# ---- per-member headline numbers --------------------------------------------------------------
ev, itw = zframe("eval_clean"), zframe("itw_clean")
best = {m: json.loads((R / m / "best.json").read_text()) for m in MEMBERS}
rows = []
for m in MEMBERS:
    ye, yi = ev["label"].to_numpy(), itw["label"].to_numpy()
    rows.append({
        "run": m, "short": SHORT[m],
        "dev_select": best[m]["select"], "best_epoch": best[m]["epoch"],
        "eval_eer": EER(ev[m].to_numpy(), ye), "eval_min_dcf": min_dcf(ev[m].to_numpy(), ye),
        "eval_min_cllr": min_cllr(ev[m].to_numpy(), ye),
        "itw_eer": EER(itw[m].to_numpy(), yi), "itw_min_dcf": min_dcf(itw[m].to_numpy(), yi),
    })
d["members"] = rows

# ---- correlation, eigen-spectrum, PC1 ---------------------------------------------------------
for tag, frame in (("eval", ev), ("itw", itw)):
    X = frame[MEMBERS].to_numpy()
    X = (X - X.mean(0)) / X.std(0)
    C = np.corrcoef(X, rowvar=False)
    w, v = np.linalg.eigh(C)
    w, v = w[::-1], v[:, ::-1]
    share = w / w.sum()
    d[f"spectrum_{tag}"] = {
        "corr": C.tolist(), "eigenvalues": w.tolist(), "variance_share": share.tolist(),
        "participation_ratio": float(w.sum() ** 2 / (w ** 2).sum()),
        "entropy_rank": float(np.exp(-(share * np.log(share)).sum())),
        "pc1_loadings": dict(zip(MEMBERS, np.abs(v[:, 0]).tolist())),
    }

# ---- every subset, ranked by dev select on 2 and on 3 views ----------------------------------
devsets = {v: zframe(f"dev_{v}") for v in ("clean", "codec", "encodec")}
for v, f in devsets.items():
    sel = pd.read_parquet(R / MEMBERS[0] / f"scores_dev_{v}.parquet")[["utt", "split"]]
    devsets[v] = f.merge(sel, on="utt")
    devsets[v] = devsets[v][devsets[v]["split"] == "select"]

for tag, views in (("2view", ["clean", "codec"]), ("3view", ["clean", "codec", "encodec"])):
    table = []
    for k in range(1, len(MEMBERS) + 1):
        for combo in itertools.combinations(MEMBERS, k):
            per = [EER(devsets[v][list(combo)].mean(axis=1).to_numpy(),
                       devsets[v]["label"].to_numpy()) for v in views]
            s = ev[list(combo)].mean(axis=1).to_numpy()
            si = itw[list(combo)].mean(axis=1).to_numpy()
            table.append({"members": list(combo),
                          "label": "+".join(SHORT[m] for m in combo),
                          "dev_select": float(np.mean(per)),
                          "eval_eer": EER(s, ev["label"].to_numpy()),
                          "eval_min_dcf": min_dcf(s, ev["label"].to_numpy()),
                          "itw_eer": EER(si, itw["label"].to_numpy())})
    table.sort(key=lambda r: r["dev_select"])
    from scipy.stats import spearmanr
    rho, pval = spearmanr([r["dev_select"] for r in table], [r["eval_eer"] for r in table])
    d[f"selection_{tag}"] = {"ranking": table, "spearman": float(rho), "p": float(pval),
                             "views": views}

# ---- DET curves for the figure ----------------------------------------------------------------
head = ["lora_enc", "lora_mhfa"]
curves = {}
for name, cols in [("Pool", ["lora_enc"]), ("MHFA", ["lora_mhfa"]), ("LFCC", ["lora_lfcc"]),
                   ("Pool+MHFA (selected)", head), ("Pool+LFCC (oracle)", ["lora_enc", "lora_lfcc"])]:
    s = ev[cols].mean(axis=1).to_numpy()
    frr, far, _ = compute_det_curve(s, ev["label"].to_numpy())
    step = max(1, len(frr) // 1200)
    curves[name] = {"frr": frr[::step].tolist(), "far": far[::step].tolist()}
d["det_eval"] = curves

# ---- per-codec and per-attack, from the official protocol ------------------------------------
with tarfile.open("F:/datasets/audio/asvspoof5/ASVspoof5_protocols.tar") as t:
    proto = pd.read_csv(t.extractfile("ASVspoof5.eval.track_1.tsv"), sep=" ", header=None,
                        dtype=str, names=["speaker", "utt", "gender", "codec", "codec_q",
                                          "codec_seed", "attack_tag", "generator", "key", "tmp"])
m = ev.merge(proto[["utt", "codec", "generator", "attack_tag", "codec_seed"]], on="utt")
m["codec"] = m["codec"].replace("-", "clean")
m["sel"] = m[head].mean(axis=1)
bona = m["label"] == 1

d["per_codec"] = {}
for col, label in [("lora_enc", "Pool"), ("lora_mhfa", "MHFA"), ("lora_lfcc", "LFCC"),
                   ("sel", "Pool+MHFA")]:
    d["per_codec"][label] = {c: EER(sub[col].to_numpy(), sub["label"].to_numpy())
                             for c, sub in m.groupby("codec")}

d["per_attack"] = {}
for a in sorted(set(m["generator"]) - {"bonafide"}):
    keep = bona | (m["generator"] == a)
    d["per_attack"][a] = {"eer": EER(m.loc[keep, "sel"].to_numpy(), m.loc[keep, "label"].to_numpy()),
                          "n": int((m["generator"] == a).sum())}
ADV = {"A18", "A20", "A23", "A27", "A30", "A31", "A32"}
for a in d["per_attack"]:
    d["per_attack"][a]["adversarial"] = a in ADV

d["per_attack_tag"] = {}
for tg in sorted(set(m["attack_tag"]) - {"-"}):
    keep = bona | (m["attack_tag"] == tg)
    d["per_attack_tag"][tg] = {"eer": EER(m.loc[keep, "sel"].to_numpy(),
                                          m.loc[keep, "label"].to_numpy()),
                               "n": int((m["attack_tag"] == tg).sum())}

# ---- the clustered-bootstrap problem, quantified ---------------------------------------------
src = m["codec_seed"].where(m["codec_seed"] != "-", m["utt"])
d["eval_structure"] = {"clips": int(len(m)), "sources": int(src.nunique()),
                       "clips_per_source": float(len(m) / src.nunique()),
                       "coded": int((m["codec_seed"] != "-").sum())}

# ---- training curves -------------------------------------------------------------------------
d["curves"] = {}
for run in MEMBERS + ["advwav", "lora_probe"]:
    p = R / run / "metrics.jsonl"
    if not p.exists():
        continue
    ms = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    d["curves"][run] = {"epoch": [x["epoch"] for x in ms],
                        "select": [x["select"] for x in ms],
                        "train_bce": [x["train/bce"] for x in ms],
                        "clean": [x["dev_clean/eer"] for x in ms],
                        "codec": [x["dev_codec/eer"] for x in ms]}

# ---- dataset statistics ----------------------------------------------------------------------
d["dataset"] = {}
for part in ("train", "dev", "eval", "itw"):
    f = pd.read_parquet(f"D:/AUDIO_DEEPFAKE/experiments/data_full/{part}.parquet")
    d["dataset"][part] = {"clips": int(len(f)), "bona": int((f["label"] == 1).sum()),
                          "speakers": int(f["speaker"].nunique()),
                          "attacks": sorted(set(f["generator"]) - {"-", "bonafide"}),
                          "codecs": f["codec"].value_counts().to_dict()}

(OUT / "paper_data.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
print("wrote", OUT / "paper_data.json")
print("members:", [(r["short"], round(r["eval_eer"] * 100, 2), round(r["itw_eer"] * 100, 2)) for r in d["members"]])
print("eval effective systems: %.2f" % d["spectrum_eval"]["participation_ratio"])
print("2view spearman %.3f | 3view spearman %.3f" % (d["selection_2view"]["spearman"], d["selection_3view"]["spearman"]))
print("eval sources:", d["eval_structure"])
