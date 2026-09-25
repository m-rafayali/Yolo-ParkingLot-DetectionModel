"""Plate readers and the asynchronous plate service.

Design rules (from the field constraints):
  * the LLM is called once per *vehicle*, never per frame, and never on the video thread;
  * 429 / 503 from the provider must never block the pipeline: the model variant is
    put on cooldown, the next variant or the free local reader is tried instead;
  * every read is validated (e.g. Singapore checksum) so a misread costs no extra call.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import queue
import re
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np

from .formats import PlateFormat

log = logging.getLogger("parkwatch.plates")

PROMPT = (
    "You are reading a vehicle licence plate for a car-park access log. The image is a CCTV crop of ONE vehicle. "
    "Find its licence (number) plate and read it exactly.\n"
    "Reply with ONLY a JSON object, no markdown: {\"plate\": \"<characters without spaces>\" or null, "
    "\"confidence\": <number 0..1>}.\n"
    "Use null if no plate is visible or it is unreadable. Never guess characters you cannot see."
)


@dataclass
class PlateResult:
    plate: str | None
    conf: float = 0.0
    valid: bool = False
    source: str = ""
    latency_ms: float = 0.0
    attempt: int = 1
    error: str | None = None
    raw: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d


def parse_llm_reply(text: str) -> tuple[str | None, float]:
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            plate = obj.get("plate")
            conf = float(obj.get("confidence", 0.5) or 0.0)
            return (str(plate) if plate else None), max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            pass
    # free-text reply: join up to 3 neighbouring words ("SBA 1234 K") and keep the longest plate-like one
    words = re.findall(r"[A-Z0-9]+", text.upper())
    cands = ["".join(words[i:i + n]) for n in (1, 2, 3) for i in range(len(words) - n + 1)]
    cands = [c for c in cands if 4 <= len(c) <= 10 and re.search(r"\d", c) and re.search(r"[A-Z]", c)]
    return (max(cands, key=len) if cands else None), 0.3


class LLMPlateReader:
    """Vision-LLM reader through the Ultralytics `LLM` interface (OpenAI-compatible).

    Works with Gemini (OpenAI-compatible endpoint), OpenAI, OpenRouter, or a local
    server such as Ollama / vLLM serving Qwen2.5-VL for on-premise, no-data-egress use.
    """

    name = "llm"

    def __init__(self, cfg: dict, hint: str = ""):
        self.models = [cfg.get("model", "gemini-2.5-flash-lite"), *cfg.get("fallback_models", [])]
        self.base_url = cfg.get("base_url")
        self.api = cfg.get("api", "chat.completions")
        self.api_key = os.environ.get(cfg.get("api_key_env", "GEMINI_API_KEY") or "", "") or cfg.get("api_key")
        if not self.api_key and self.base_url and ("localhost" in self.base_url or "127.0.0.1" in self.base_url):
            self.api_key = "local"  # Ollama / vLLM ignore the key
        self.timeout = float(cfg.get("timeout_s", 20))
        self.cost_per_call = float(cfg.get("cost_per_call_usd", 0.001))
        self.request_params = dict(cfg.get("request_params") or {})
        self.prompt = PROMPT + (f"\n{hint}" if hint else "")
        self.available = bool(self.api_key)
        self.why_unavailable = None if self.available else f"set {cfg.get('api_key_env', 'GEMINI_API_KEY')} to enable"
        self._clients: dict = {}
        self._cooldown: dict[str, float] = {}
        self._lock = threading.Lock()

    def _client(self, model: str):
        if model not in self._clients:
            try:
                from ultralytics import LLM
            except ImportError as e:  # ultralytics < 8.4 has no LLM class
                raise RuntimeError("ultralytics>=8.4 with the LLM class is required: pip install -U 'ultralytics[llm]'") from e
            llm = LLM(model, api=self.api, base_url=self.base_url, api_key=self.api_key)
            try:  # fail fast: we rotate models ourselves instead of letting the SDK sleep and retry
                from openai import OpenAI

                kw = {k: v for k, v in {"api_key": self.api_key, "base_url": self.base_url}.items() if v}
                llm.client = OpenAI(max_retries=0, timeout=self.timeout, **kw)
            except ImportError:
                pass
            self._clients[model] = llm
        return self._clients[model]

    def read(self, img: np.ndarray) -> PlateResult:
        errors = []
        for model in self.models:
            with self._lock:
                if time.time() < self._cooldown.get(model, 0):
                    continue
            t0 = time.time()
            try:
                resp = self._client(model)(self.prompt, image=img, temperature=0, **self.request_params)
                text = resp.choices[0].message.content if self.api == "chat.completions" else resp.output_text
                plate, conf = parse_llm_reply(text)
                return PlateResult(plate, conf, source=f"llm:{model}", latency_ms=(time.time() - t0) * 1000, raw=text)
            except Exception as e:  # noqa: BLE001 - provider errors are data here, never fatal
                status = getattr(e, "status_code", None)
                errors.append(f"{model}:{status or type(e).__name__}")
                with self._lock:
                    if status in (401, 403) or (status == 400 and "key" in str(e).lower()):
                        self.available, self.why_unavailable = False, f"auth failed ({status}) - check the API key"
                        log.warning("LLM plate reader disabled: %s", self.why_unavailable)
                        break
                    if status in (400, 404):  # bad model name / unsupported parameter: stop using this variant
                        self._cooldown[model] = float("inf")
                        log.warning("LLM variant %s rejected the request (%s): %s", model, status, str(e)[:200])
                        continue
                    # 429 rate limit / 5xx overload / timeouts: park this variant, try the next one
                    self._cooldown[model] = time.time() + (60 if status == 429 else 20)
        return PlateResult(None, source="llm", error=", ".join(errors) or "all LLM variants cooling down")


class LocalPlateReader:
    """Free, offline reader: fast-alpr (ONNX plate detector + OCR, CPU friendly)."""

    name = "local"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._alpr = None
        self._lock = threading.Lock()
        self.cost_per_call = 0.0
        if importlib.util.find_spec("fast_alpr") is not None:
            self.available, self.why_unavailable = True, None
        else:
            self.available, self.why_unavailable = False, "pip install fast-alpr to enable"

    def _load(self):
        if self._alpr is None:
            from fast_alpr import ALPR

            kw = {"detector_conf_thresh": float(self.cfg.get("detector_conf", 0.3)), "ocr_device": "cpu"}
            if self.cfg.get("detector_model"):
                kw["detector_model"] = self.cfg["detector_model"]
            if self.cfg.get("ocr_model"):
                kw["ocr_model"] = self.cfg["ocr_model"]
            self._alpr = ALPR(**kw)
        return self._alpr

    def read(self, img: np.ndarray) -> PlateResult:
        t0 = time.time()
        try:
            with self._lock:
                results = self._load().predict(img)
        except Exception as e:  # noqa: BLE001
            return PlateResult(None, source="local", error=f"{type(e).__name__}: {e}")
        best, best_score = None, -1.0
        for r in results:
            if r.ocr is None or not r.ocr.text:
                continue
            c = r.ocr.confidence
            oc = float(np.mean(c)) if isinstance(c, (list, tuple, np.ndarray)) else float(c or 0)
            score = oc * float(r.detection.confidence)
            if score > best_score:
                best, best_score = (r.ocr.text, oc), score
        ms = (time.time() - t0) * 1000
        if best is None:
            return PlateResult(None, source="local:fast-alpr", latency_ms=ms)
        return PlateResult(best[0], best[1], source="local:fast-alpr", latency_ms=ms)


class PlateService:
    """Background workers that turn vehicle crops into validated plate strings."""

    def __init__(self, cfg: dict, fmt: PlateFormat):
        self.fmt = fmt
        self.max_crops = int(cfg.get("max_attempts_per_vehicle", 2))
        self.min_conf = float(cfg.get("min_confidence", 0.4))
        hint = cfg.get("hint", "")
        builders = {"llm": lambda: LLMPlateReader(cfg.get("llm", {}), hint), "local": lambda: LocalPlateReader(cfg.get("local", {}))}
        self.readers = []
        for name in cfg.get("providers", ["llm", "local"]):
            if name not in builders:
                raise ValueError(f"unknown plate provider {name!r} (use llm / local)")
            r = builders[name]()
            self.readers.append(r)
            state = "enabled" if r.available else f"DISABLED ({r.why_unavailable})"
            log.info("plate reader %-5s %s", name, state)
        self.stats = {r.name: {"calls": 0, "ok": 0, "errors": 0, "ms": 0.0, "cost_usd": 0.0, "last_error": None}
                      for r in self.readers}
        self.stats["vehicles"] = {"requested": 0, "read": 0, "valid": 0, "unreadable": 0}
        self._q: queue.Queue = queue.Queue()
        self._out: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._workers = [threading.Thread(target=self._work, daemon=True, name=f"plate-{i}")
                         for i in range(int(cfg.get("workers", 2)))]
        for w in self._workers:
            w.start()

    # -------------------------------------------------------------- public API (called from the video thread)
    def submit(self, ref: str, cam: str, crops: list[np.ndarray]) -> None:
        self.stats["vehicles"]["requested"] += 1
        self._q.put((ref, cam, [c for c in crops if c is not None and c.size]))

    def poll(self) -> list[tuple[str, str, PlateResult]]:
        out = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    def pending(self) -> int:
        return self._q.qsize() + sum(1 for w in self._workers if getattr(w, "busy", False))

    def wait_idle(self, timeout: float = 60.0) -> None:
        t0 = time.time()
        while self.pending() and time.time() - t0 < timeout:
            time.sleep(0.05)

    def close(self) -> None:
        self._stop.set()

    def summary(self) -> dict:
        s = {k: dict(v) for k, v in self.stats.items()}
        s["readers"] = [{"name": r.name, "available": r.available, "note": r.why_unavailable} for r in self.readers]
        s["llm_cost_usd"] = round(sum(v.get("cost_usd", 0.0) for k, v in self.stats.items() if k != "vehicles"), 4)
        return s

    # -------------------------------------------------------------- internals
    def _work(self) -> None:
        me = threading.current_thread()
        while not self._stop.is_set():
            try:
                ref, cam, crops = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            me.busy = True
            try:
                res = self._read(crops)
            except Exception as e:  # noqa: BLE001
                res = PlateResult(None, error=f"{type(e).__name__}: {e}")
            v = self.stats["vehicles"]
            v["read"] += res.plate is not None
            v["valid"] += bool(res.valid)
            v["unreadable"] += res.plate is None
            self._out.put((ref, cam, res))
            me.busy = False

    def _read(self, crops: list[np.ndarray]) -> PlateResult:
        best: PlateResult | None = None
        errors = []
        for i, crop in enumerate(crops[: self.max_crops]):
            for r in self.readers:
                if not r.available:
                    continue
                res = r.read(crop)
                st = self.stats[r.name]
                st["calls"] += 1
                st["ms"] += res.latency_ms
                if res.error:
                    st["errors"] += 1
                    st["last_error"] = res.error
                    errors.append(res.error)
                    continue
                st["ok"] += 1
                st["cost_usd"] += r.cost_per_call
                res.attempt = i + 1
                res.plate, res.valid = self.fmt.validate(res.plate)
                if res.valid and res.conf >= self.min_conf:
                    return res
                if res.plate and (best is None or (res.valid, res.conf) > (best.valid, best.conf)):
                    best = res
        if best is not None:
            return best
        return PlateResult(None, error="; ".join(errors) if errors else "no plate found")
