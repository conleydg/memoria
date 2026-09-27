"""Peak-memory sampling for processes we don't control (the Ollama
server/runner), plus helpers for in-process frameworks.

On Apple Silicon, GPU memory is unified RAM, but Metal buffers don't
always show up in a process's RSS. For Ollama we therefore record both
the peak RSS of its processes and the peak size Ollama itself reports
via /api/ps, and use the larger.
"""

import threading
import time

import psutil
import requests

OLLAMA = "http://127.0.0.1:11434"


def ollama_ps_bytes(model: str | None = None) -> int:
    """Memory Ollama reports for `model` (or all loaded models). Filtering
    matters: anything else loaded at the same time (e.g. the Ask page's
    model) would otherwise be counted against the model being measured."""
    try:
        models = requests.get(f"{OLLAMA}/api/ps", timeout=2).json().get("models", [])
        return sum(m.get("size", 0) for m in models if model is None or m.get("name") == model)
    except requests.RequestException:
        return 0


def ollama_rss_bytes() -> int:
    total = 0
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if "ollama" in (p.info["name"] or "").lower():
                total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


class OllamaPeak:
    def __init__(self, interval=0.5, model: str | None = None):
        self.interval, self.model = interval, model
        self.peak_rss = self.peak_ps = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, ollama_rss_bytes())
            self.peak_ps = max(self.peak_ps, ollama_ps_bytes(self.model))
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    @property
    def peak(self):
        return max(self.peak_rss, self.peak_ps)


class ProcessPeak:
    """Peak RSS of this process (for torch/MLX models loaded in-process)."""

    def __init__(self, interval=0.25):
        self.interval = interval
        self.peak = 0
        self._p = psutil.Process()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._p.memory_info().rss)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def now():
    return time.time()
