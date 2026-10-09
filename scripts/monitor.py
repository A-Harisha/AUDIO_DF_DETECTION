"""Live terminal view of the running jobs. Read-only: it never touches a job.

    uv run python scripts/monitor.py            # refresh each 5 s; Ctrl+C stops the view only

Claim served: none (observability). It shows, for the newest logs in experiments/logs_partial:
the last progress lines, the per-epoch metrics of the newest run, and the GPU load.
Standard library only, so it starts fast and cannot hang on the pandas import.
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
KEEP = ("clips done", "epoch", "scored", "wrote", "Error", "shape |")
BATCH = re.compile(r"batch (\d+) / (\d+)")
STAMP = "%Y-%m-%d %H:%M:%S"
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


def sparkline(values: list[float]) -> str:
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    return "".join(SPARK[int((v - lo) / span * (len(SPARK) - 1))] for v in values)


def clock(seconds: float) -> str:
    return f"{int(seconds) // 3600}h {int(seconds) % 3600 // 60:02d}m"


def pace(lines: list[str], mtime: float) -> str:
    """The span the log covers, and, while it is still live, the end clock its batch rate points to."""
    if not lines:
        return "no lines yet"
    span = f"spans {clock(mtime - time.mktime(time.strptime(lines[0][:19], STAMP)))}"
    marks = [(time.mktime(time.strptime(ln[:19], STAMP)), int(m[1]), int(m[2]))
             for ln in lines if (m := BATCH.search(ln))]
    if len(marks) < 2 or marks[-1][1] <= marks[0][1]:
        return span
    (first_at, first_batch, _), (last_at, last_batch, total) = marks[0], marks[-1]
    done = f"{span}   {last_batch}/{total} batches"
    gap = (last_at - first_at) / (len(marks) - 1)
    if time.time() - mtime > 2 * gap:
        return f"{done}   quiet for more than two log periods, so no end estimate"
    left = (total - last_batch) * (last_at - first_at) / (last_batch - first_batch)
    end = time.strftime("%H:%M", time.localtime(time.time() + left))
    return f"{done}   {clock(left)} left   ends about {end}"


def gpu_rows() -> list[str]:
    query = "name,utilization.gpu,memory.used,memory.total,temperature.gpu"
    text = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout.strip()
    rows = []
    for line in text.splitlines():
        name, util, used, total, temp = (field.strip() for field in line.split(","))
        full = round(int(util) / 5)
        bar = f"{GREEN}{'█' * full}{DIM}{'·' * (20 - full)}{RESET}"
        rows.append(f"{name.removeprefix('NVIDIA ')[:20]:<20}  {bar} {int(util):>3}%   "
                    f"{int(used) / 1024:>5.1f}/{int(total) / 1024:.0f} GiB   {temp}°C")
    return rows


def log_rows(path: Path, n: int, width: int) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    kept = [ln[:width] for ln in lines if any(key in ln for key in KEEP)][-n:]
    rows = [f"{RED}{ln}{RESET}" if "Error" in ln else ln for ln in kept]
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
    return rows + ["", f"{DIM}EER{RESET} {sparkline(eers[-40:])}  "
                       f"{eers[0]:.2f}% → {eers[-1]:.2f}%   best {best:.2f}%  (lower is better)"]


def newest(pattern: str, n: int) -> list[Path]:
    return sorted(ROOT.glob(pattern), key=os.path.getmtime, reverse=True)[:n]


def main(period: float = 5.0) -> None:
    prepare_stdout()
    while True:
        width = min(shutil.get_terminal_size((100, 30)).columns, 110)
        out = [f"{BOLD}CAESD monitor{RESET}  {time.strftime('%H:%M:%S')}"
               f"{DIM}   Ctrl+C stops the view, not the jobs{RESET}", ""]
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
