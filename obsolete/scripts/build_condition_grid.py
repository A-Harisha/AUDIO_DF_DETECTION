"""Apply the frozen condition grid to a manifest of clean audio, writing processed clips and a
new manifest with free condition labels.

    uv run python scripts/build_condition_grid.py input_manifest=... output_dir=...

This is run once. After Phase 0 the grid config must not change.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from caesd.data.condition_grid import Condition, ConditionGrid, apply_codec
from caesd.utils.io import read_audio, write_audio
from caesd.utils.logging import get_logger

log = get_logger("caesd.grid")


def _codec_cells(grid: ConditionGrid) -> list[Condition]:
    """The codec axis only -- the part we can build correctly with ffmpeg alone.
    Noise, reverb and benign edits are added once their corpora (MUSAN, RIRs, a denoiser)
    are wired in; they are NOT emitted here, to avoid writing mislabelled clips."""
    cells = [Condition(codec="clean")]
    for codec in grid.codecs:
        if codec == "clean":
            continue
        for bitrate in grid.bitrates_kbps:
            cells.append(Condition(codec=codec, bitrate_kbps=bitrate))
    return cells


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    input_manifest = Path(cfg.input_manifest)
    output_dir = Path(cfg.output_dir)
    grid = ConditionGrid(**cfg.condition_grid)
    cells = _codec_cells(grid)
    log.info("building %d codec cells (noise/reverb/benign pending their corpora)", len(cells))

    src = pd.read_parquet(input_manifest) if input_manifest.suffix == ".parquet" else pd.read_csv(input_manifest)
    records = []
    for _, row in src.iterrows():
        wave = read_audio(row["path"])
        for cond in cells:
            processed = wave if cond.codec == "clean" else apply_codec(wave, cond.codec, cond.bitrate_kbps or 16)
            out_path = output_dir / cond.cond_id / Path(row["path"]).name
            write_audio(out_path, processed)
            records.append({**row.to_dict(), "path": str(out_path), "condition": cond.cond_id,
                            "codec": cond.codec})
    out_manifest = output_dir / "manifest.parquet"
    pd.DataFrame(records).to_parquet(out_manifest)
    log.info("wrote %d clips, manifest -> %s", len(records), out_manifest)


if __name__ == "__main__":
    main()
