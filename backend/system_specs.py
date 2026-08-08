"""Collects the current machine's hardware specs, for feeding real numbers into
the section-4 (latency & resource profile) part of the model-mock prompt, and
later as the feature vector for an actual hardware-aware latency predictor.

Every field is best-effort: if a library/tool isn't available or a read fails,
that field is just None rather than raising. Nothing here requires elevated
permissions, and the GPU/Ollama checks no-op cleanly if absent.
"""

import platform
import shutil
import subprocess

try:
    import psutil
except ImportError:
    psutil = None

try:
    import cpuinfo  # py-cpuinfo
except ImportError:
    cpuinfo = None


def _bytes_to_gb(n):
    if n is None:
        return None
    return round(n / (1024 ** 3), 1)


def get_cpu_specs():
    physical_cores = logical_cores = max_freq_mhz = None
    if psutil:
        try:
            physical_cores = psutil.cpu_count(logical=False)
            logical_cores = psutil.cpu_count(logical=True)
        except Exception:
            pass
        try:
            freq = psutil.cpu_freq()
            max_freq_mhz = round(freq.max) if freq and freq.max else None
        except Exception:
            pass

    brand = None
    flags = []
    if cpuinfo:
        try:
            info = cpuinfo.get_cpu_info()
            brand = info.get("brand_raw")
            flags = info.get("flags", []) or []
            if not max_freq_mhz:
                hz = info.get("hz_advertised_friendly")
                max_freq_mhz = hz  # already human-formatted string in this fallback case
        except Exception:
            pass

    if not brand:
        # crude stdlib fallback — often unreliable on Windows, last resort only
        brand = platform.processor() or "unknown"

    return {
        "brand": brand,
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "max_freq_mhz": max_freq_mhz,
        "supports_avx2": "avx2" in flags,
        "supports_avx512": any(f.startswith("avx512") for f in flags),
        "supports_fma": "fma" in flags,
    }


def get_memory_specs():
    total_gb = available_gb = None
    if psutil:
        try:
            vm = psutil.virtual_memory()
            total_gb = _bytes_to_gb(vm.total)
            available_gb = _bytes_to_gb(vm.available)
        except Exception:
            pass
    return {"total_ram_gb": total_gb, "available_ram_gb": available_gb}


def get_disk_specs():
    free_gb = None
    if psutil:
        try:
            path = "C:\\" if platform.system() == "Windows" else "/"
            free_gb = _bytes_to_gb(psutil.disk_usage(path).free)
        except Exception:
            pass
    return {"free_disk_gb": free_gb}


def get_os_specs():
    return {
        "os": platform.system() or "unknown",
        "os_version": platform.release() or "unknown",
        "architecture": platform.machine() or "unknown",
    }


def get_gpu_specs():
    """Best-effort NVIDIA GPU check via nvidia-smi. Returns None if no NVIDIA
    GPU/driver is present — most of this tool's target users are CPU-only,
    but if a GPU IS present it changes the latency story enough to flag."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        name, mem = [p.strip() for p in out.stdout.strip().split(",")[:2]]
        return {"name": name, "vram": mem}
    except Exception:
        return None


def get_ollama_runtime_info():
    """Ground-truth check via `ollama ps`: confirms whether a currently loaded
    model is actually running on CPU or GPU, and its live memory footprint.
    Returns None if ollama isn't on PATH or nothing is currently loaded."""
    if not shutil.which("ollama"):
        return None
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        lines = [l for l in out.stdout.strip().splitlines() if l.strip()]
        if len(lines) <= 1:
            return None  # header only, nothing loaded right now
        return lines[1:]  # raw rows; caller can parse further if needed
    except Exception:
        return None


def collect_system_specs():
    """Single entry point: gathers everything above into one dict. Cheap
    enough (<50ms typically) to call per-request, but callers should cache
    this at startup/on-demand rather than re-collecting on every mock call."""
    return {
        "cpu": get_cpu_specs(),
        "memory": get_memory_specs(),
        "disk": get_disk_specs(),
        "os": get_os_specs(),
        "gpu": get_gpu_specs(),
        "ollama_runtime": get_ollama_runtime_info(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(collect_system_specs(), indent=2))
