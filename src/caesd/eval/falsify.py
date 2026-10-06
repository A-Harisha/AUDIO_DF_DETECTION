"""Phase 0 falsification logic. Separated from the script so it is unit-testable.

The whole project rests on one assumption: the calibration gap under shift is large, and a
simple per-condition lookup table does not close it on held-out codecs and languages. This
measures both and returns a go/no-go verdict.

Expected columns in ``df``: score, label, condition, codec, language, split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..calibration import LogisticCalibrator, LookupTableCalibrator
from ..metrics import act_dcf, min_dcf, threshold_transfer


def _mean_gap(df: pd.DataFrame, llr_col: str) -> float:
    """Mean of (actDCF - minDCF) across conditions present in df."""
    gaps = []
    for _, sub in df.groupby("condition"):
        if np.unique(sub["label"]).size < 2:
            continue
        gaps.append(act_dcf(sub[llr_col].to_numpy(), sub["label"].to_numpy())
                    - min_dcf(sub["score"].to_numpy(), sub["label"].to_numpy()))
    return float(np.mean(gaps)) if gaps else float("nan")


def run_falsification(
    df: pd.DataFrame,
    far_target: float = 0.01,
    holdout_codecs: list[str] | None = None,
    holdout_languages: list[str] | None = None,
    gap_is_large: float = 0.10,
    lookup_closes_gap: float = 0.50,
    min_per_cell: int = 50,
) -> dict:
    holdout_codecs = holdout_codecs or []
    holdout_languages = holdout_languages or []
    dev = df[df["split"] == "dev"]
    test = df[df["split"] == "test"].copy()

    clean_dev = dev[dev["codec"] == "clean"]
    if np.unique(clean_dev["label"]).size < 2:
        raise ValueError("need clean dev data with both classes to fit the global calibrator")

    # --- global calibration, fit once on clean dev (the universal prior art) ---
    global_cal = LogisticCalibrator().fit(clean_dev["score"].to_numpy(), clean_dev["label"].to_numpy())
    test["llr_global"] = global_cal.transform(test["score"].to_numpy())

    # --- per-condition lookup table, fit on NON-held-out dev cells ---
    seen_dev = dev[~dev["codec"].isin(holdout_codecs) & ~dev["language"].isin(holdout_languages)]
    table = LookupTableCalibrator(min_per_cell=min_per_cell).fit(
        seen_dev["score"].to_numpy(), seen_dev["label"].to_numpy(), seen_dev["condition"].to_numpy()
    )
    llr_lookup, covered = table.transform(test["score"].to_numpy(), test["condition"].to_numpy())
    test["llr_lookup"] = np.where(covered, llr_lookup, test["llr_global"])  # explicit, not hidden
    test["covered"] = covered

    held_mask = test["codec"].isin(holdout_codecs) | test["language"].isin(holdout_languages)
    seen, held = test[~held_mask], test[held_mask]

    gap_global_seen = _mean_gap(seen, "llr_global")
    gap_lookup_seen = _mean_gap(seen, "llr_lookup")
    gap_global_held = _mean_gap(held, "llr_global")
    frac_closed_seen = (
        (gap_global_seen - gap_lookup_seen) / gap_global_seen if gap_global_seen > 0 else 0.0
    )

    # threshold transfer: fix on clean dev, apply unchanged per held-out condition
    tt: dict[str, dict] = {}
    for cond, sub in held.groupby("condition"):
        if np.unique(sub["label"]).size == 2:
            tt[str(cond)] = threshold_transfer(
                clean_dev["score"].to_numpy(), clean_dev["label"].to_numpy(),
                sub["score"].to_numpy(), sub["label"].to_numpy(), far_target,
            )

    lookup_is_competitive = frac_closed_seen >= lookup_closes_gap  # the reviewer's objection, on seen cells
    premise_holds = (gap_global_held >= gap_is_large) and (float(held["covered"].mean()) < 1.0)
    decision = "GO" if premise_holds else "NO-GO"
    reason = (
        f"held-out gap {gap_global_held:.3f} (>= {gap_is_large} needed); "
        f"lookup table covers {held['covered'].mean():.0%} of held-out cells; "
        f"on seen cells the table closes {frac_closed_seen:.0%} of the gap "
        f"({'competitive' if lookup_is_competitive else 'weak'} baseline there)"
    )
    return {
        "decision": decision,
        "reason": reason,
        "gap_global_seen": gap_global_seen,
        "gap_lookup_seen": gap_lookup_seen,
        "gap_global_heldout": gap_global_held,
        "fraction_closed_seen": float(frac_closed_seen),
        "lookup_is_competitive_on_seen": bool(lookup_is_competitive),
        "heldout_covered_fraction": float(held["covered"].mean()) if len(held) else 0.0,
        "threshold_transfer_heldout": tt,
    }
