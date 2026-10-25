"""Live terminal view of the running jobs. Read-only: it never touches a job.

    uv run python scripts/monitor.py            # refresh each 5 s; Ctrl+C stops the view only

Claim served: none (observability). For the newest logs in experiments/logs_partial it shows a
progress bar, the last progress lines, the per-epoch metrics of the newest run, and the GPU load.
Feature extraction reports clips done of total; training reports epochs; both reach 100% when the
job writes its result. Standard library only, so it starts fast and cannot hang on the pandas
import.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent / "experiments"
KEEP = ("clips", "epoch", "scored", "wrote", "Error", "shape |")
CLIPS = re.compile(r"clips (?:already )?done:? (\d+) / (\d+)")
STEP = re.compile(r"epoch (\d+) step (\d+)/(\d+)")
DONE = ("best epoch", "| wrote ")
STAMP = "%Y-%m-%d %H:%M:%S"
WINDOW = 6  # ticks averaged for the rate: enough to ride out one stalled log period
SPARK = "▁▂▃▄▅▆▇█"
DIM, BOLD, RESET = "\x1b[2m", "\x1b[1m", "\x1b[0m"
RED, GREEN, CYAN = "\x1b[31m", "\x1b[32m", "\x1b[36m"


def prepare_stdout() -> None:
    """Switch the Windows console to virtual-terminal mode and to UTF-8, so the frames draw."""
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)


def width_of(text: str) -> int:
    """Printed width: the ANSI escapes take no columns."""
    out, i = 0, 0
    while i < len(text):
        if text[i] == "\x1b":
            i = text.index("m", i) + 1
        else:
            out, i = out + 1, i + 1
    return out


def panel(title: str, rows: list[str], width: int, note: str = "") -> list[str]:
    """One framed block: a title on the top rule, an optional note at its right."""
    inner = width - 2
    head = f"─ {BOLD}{title}{RESET}{DIM} "
    foot = f" {note[:inner - width_of(head) - 4]} ─" if note else "─"
    rule = "─" * max(0, inner - width_of(head) - width_of(foot))
    out = [f"{DIM}┌{head}{rule}{foot}┐{RESET}"]
    for row in rows:
        pad = " " * max(0, inner - 1 - width_of(row))
        out.append(f"{DIM}│{RESET} {row}{pad}{DIM}│{RESET}")
    out.append(f"{DIM}└{'─' * inner}┘{RESET}")
    return out


def bar(fraction: float, cells: int = 24) -> str:
    full = round(min(max(fraction, 0.0), 1.0) * cells)
    colour = CYAN if full == cells else GREEN
    return f"{colour}{'█' * full}{DIM}{'·' * (cells - full)}{RESET}"


def sparkline(values: list[float]) -> str:
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    return "".join(SPARK[int((v - lo) / span * (len(SPARK) - 1))] for v in values)


def clock(seconds: float) -> str:
    return f"{int(seconds) // 3600}h {int(seconds) % 3600 // 60:02d}m"


def stamp(line: str) -> float | None:
    """The line's timestamp, or None if it carries none.

    This view races the writers it reads: a log just truncated by a shell redirect can be read
    while the file is extended but not yet flushed, so its head is NUL bytes. A read-only monitor
    must not die of that."""
    try:
        return time.mktime(time.strptime(line[:19], STAMP))
    except ValueError:
        return None


def marks(lines: list[str]) -> list[tuple[float, int, int]]:
    """Every periodic ``clips done X / Y`` line as (when, done, total). Feature extraction only.

    The ``clips already done`` line is a resume marker written at startup, not a periodic tick, so
    it is excluded: counting it puts the log period at seconds and makes a live job look quiet."""
    return [(at, int(m[1]), int(m[2])) for ln in lines
            if (m := CLIPS.search(ln)) and "already done" not in ln and (at := stamp(ln)) is not None]


def total_epochs(name: str) -> int | None:
    """The configured epochs of the run this log belongs to, by matching name."""
    path = ROOT / "runs_partial" / name / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))["experiment"]["epochs"] if path.exists() else None


def progress(lines: list[str], name: str) -> tuple[float, str] | None:
    """Fraction done and its label, from the newest progress line. None if the log shows neither."""
    finished = any(key in ln for ln in lines[-40:] for key in DONE)
    steps = [m for ln in lines if (m := STEP.search(ln))]
    if steps:
        epochs = total_epochs(name)
        if epochs is None:
            return None
        epoch, step, of = int(steps[-1][1]), int(steps[-1][2]), int(steps[-1][3])
        return (1.0 if finished else (epoch + step / of) / epochs), f"epoch {epoch + 1} of {epochs}"
    if ticks := [m for ln in lines if (m := CLIPS.search(ln))]:
        done, total = int(ticks[-1][1]), int(ticks[-1][2])
        return (1.0 if finished else done / total), f"{done:,} of {total:,} clips"
    return None


def pace(lines: list[str], mtime: float) -> str:
    """The span the log covers, and, while it is still live, the end clock its recent rate points to.

    The rate comes from the last ``WINDOW`` ticks, not from the whole log. Averaging from the first
    tick lets one stalled period bias every later estimate: in the eval extraction of 2026-10-10 a
    single 32-minute stall held the estimate about three times slow for hours after the job had
    recovered. Under three ticks there is no estimate at all, because one interval cannot be told
    apart from a stall."""
    started = next((at for ln in lines if (at := stamp(ln)) is not None), None)
    if started is None:
        return "no timestamped lines yet"
    span = f"spans {clock(mtime - started)}"
    found = marks(lines)
    if not found:  # a training log: it ticks per epoch, not in clips
        return span
    if len(found) < 3:
        return f"{span}   {len(found)} tick(s): too few for an end estimate"
    recent = found[-WINDOW:]
    (first_at, first_done, _), (last_at, last_done, total) = recent[0], recent[-1]
    if last_done <= first_done:
        return span
    gap = (last_at - first_at) / (len(recent) - 1)
    if time.time() - mtime > 2 * gap:
        return f"{span}   quiet for more than two log periods, so no end estimate"
    rate = (last_done - first_done) / (last_at - first_at)
    left = (total - last_done) / rate
    end = time.strftime("%H:%M", time.localtime(time.time() + left))
    return (f"{span}   {60 * rate:,.0f} clips/min over {len(recent)} ticks   "
            f"{clock(left)} left   ends about {end}")


def gpu_rows() -> list[str]:
    query = "name,utilization.gpu,memory.used,memory.total,temperature.gpu"
    text = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout.strip()
    rows = []
    for line in text.splitlines():
        name, util, used, total, temp = (field.strip() for field in line.split(","))
        rows.append(f"{name.removeprefix('NVIDIA ')[:20]:<20}  {bar(int(util) / 100, 20)} {int(util):>3}%   "
                    f"{int(used) / 1024:>5.1f}/{int(total) / 1024:.0f} GiB   {temp}°C")
    return rows


def log_rows(path: Path, n: int, width: int) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    kept = [ln[:width] for ln in lines if any(key in ln for key in KEEP)][-n:]
    rows = [f"{RED}{ln}{RESET}" if "Error" in ln else ln for ln in kept]
    if (now := progress(lines, path.stem)) is not None:
        rows = [f"{bar(now[0])} {100 * now[0]:>5.1f}%   {now[1]}", ""] + rows
    return rows + ["", f"{DIM}{pace(lines, path.stat().st_mtime)}{RESET}"]


def epoch_rows(path: Path) -> list[str]:
    records = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not records:
        return [f"{DIM}no epochs written yet{RESET}"]
    eers = [100 * m["select"] for m in records]
    best = min(eers)
    rows = [f"{DIM}epoch   train BCE   select EER{RESET}"]
    for record, eer in zip(records[-8:], eers[-8:]):
        mark = f"  {CYAN}best{RESET}" if eer == best else ""
        rows.append(f"{record['epoch']:>5}   {record['train/bce']:>9.4f}   {eer:>8.2f}%{mark}")
    return rows + ["", (f"{DIM}EER{RESET} {sparkline(eers[-40:])}  "
                        f"{eers[0]:.2f}% → {eers[-1]:.2f}%   best {best:.2f}%  (lower is better)")]


def newest(pattern: str, n: int) -> list[Path]:
    return sorted(ROOT.glob(pattern), key=os.path.getmtime, reverse=True)[:n]


def main(period: float = 5.0) -> None:
    prepare_stdout()
    while True:
        width = min(shutil.get_terminal_size((100, 30)).columns, 110)
        out = [(f"{BOLD}CAESD monitor{RESET}  {time.strftime('%H:%M:%S')}"
                f"{DIM}   Ctrl+C stops the view, not the jobs{RESET}"), ""]
        out += panel("GPU", gpu_rows(), width) + [""]
        for log in newest("logs_partial/*.log", 3):
            age = (time.time() - log.stat().st_mtime) / 60
            out += panel(log.name, log_rows(log, 4, width - 4), width,
                         f"last write {age:.0f} min ago") + [""]
        for run in newest("runs_partial/*/metrics.jsonl", 1):
            out += panel(run.parent.name, epoch_rows(run), width)
        sys.stdout.write("\x1b[H\x1b[2J" + "\n".join(out) + "\n")
        sys.stdout.flush()
        time.sleep(period)


if __name__ == "__main__":
    main()
