"""
Matches the device name strings reported by system_specs.py (via nvidia-smi /
py-cpuinfo) against the static hardware_specs_lookup.csv table, since
psutil/py-cpuinfo/nvidia-smi report device NAME but never peak FLOPS or
memory bandwidth -- those are looked up here, not measured live.

Matching is fuzzy: reported names rarely match the spec-sheet name exactly
("NVIDIA GeForce RTX 4090" vs "RTX 4090"), so this normalizes both sides and
scores by token overlap rather than requiring an exact string match.
"""

import csv
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "hardware_specs_lookup.csv"

_NOISE_TOKENS = {"nvidia", "geforce", "amd", "radeon", "intel", "apple", "gpu",
                  "cpu", "series", "graphics", "founders", "edition"}


def _load_table():
    with open(DATA_PATH, newline="") as f:
        return list(csv.DictReader(f))


_TABLE = _load_table()


import re as _re

def _normalize(name: str) -> set:
    if not name:
        return set()
    cleaned = _re.sub(r"[^a-z0-9]+", " ", name.lower())
    tokens = cleaned.split()
    return {t for t in tokens if t not in _NOISE_TOKENS}


def _score(query_tokens: set, candidate_tokens: set) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = len(query_tokens & candidate_tokens)
    return overlap / max(len(query_tokens), len(candidate_tokens))


def match_gpu(gpu_name: str):
    """Best-effort fuzzy match against GPU/workstation/datacenter rows.
    Returns the matched row dict, or None if nothing scores above threshold."""
    query_tokens = _normalize(gpu_name)
    if not query_tokens:
        return None

    best_row, best_score = None, 0.0
    for row in _TABLE:
        if row["device_type"] not in ("gpu",):
            continue
        cand_tokens = _normalize(row["device_name"])
        score = _score(query_tokens, cand_tokens)
        if score > best_score:
            best_row, best_score = row, score

    if best_score < 0.4:  # too weak a match to trust
        return None
    return best_row


# CPU archetype fallback: real reported CPU brand strings ("AMD Ryzen 9 7900X
# 12-Core Processor") don't reliably match our fixed CPU rows exactly, and
# unlike GPUs there's no small fixed catalog to fuzzy-match against -- CPU
# models number in the thousands. Instead we bucket by core count + ISA
# support (which py-cpuinfo DOES report reliably) into one of our researched
# CPU archetype rows. This is intentionally coarse -- flagged in the response.
def match_cpu(physical_cores, supports_avx512, supports_avx2):
    cores = physical_cores or 4
    if supports_avx512 and cores >= 12:
        target_name = "AMD Ryzen 9 9950X (DDR5)"
    elif supports_avx2 and cores >= 12:
        target_name = "Intel Core i9-13900K (DDR5)"
    elif supports_avx2 and cores >= 6:
        target_name = "AMD Ryzen 5 7600 (DDR5)"
    elif supports_avx2:
        target_name = "Intel Core i5-1135G7 (laptop, DDR4)"
    else:
        target_name = "Generic 4-core laptop (DDR4, no AVX2)"

    for row in _TABLE:
        if row["device_name"] == target_name:
            return row
    return None


def resolve_hardware(system_specs: dict):
    """Given the dict from system_specs.collect_system_specs(), return the
    best-matched hardware_specs_lookup.csv row plus a human-readable note on
    match confidence, so the frontend/response can be honest about it."""
    gpu = system_specs.get("gpu")
    if gpu and gpu.get("name"):
        row = match_gpu(gpu["name"])
        if row:
            return row, f"matched '{gpu['name']}' -> '{row['device_name']}' (spec lookup)"
        # GPU present but not in our table -- fall through to CPU archetype
        # rather than silently guessing a random GPU row
    cpu = system_specs.get("cpu", {})
    row = match_cpu(
        cpu.get("physical_cores"),
        cpu.get("supports_avx512"),
        cpu.get("supports_avx2"),
    )
    note = (
        "no matching GPU spec found, falling back to CPU archetype"
        if gpu else
        "no GPU detected, using CPU archetype based on core count + ISA support"
    )
    return row, note
