"""Alert fan-out: console, alerts.jsonl, and an optional webhook (Slack / Teams / Discord / your API).

Webhook calls run on a background thread so a slow endpoint never stalls the video loop.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path

log = logging.getLogger("parkwatch.alerts")

ICON = {"overstay": "OVERSTAY", "overstay_exit": "OVERSTAY-EXIT", "lot_full": "LOT FULL"}


class AlertManager:
    def __init__(self, cfg: dict, out_dir: Path):
        self.console = bool(cfg.get("console", True))
        self.webhook = cfg.get("webhook_url")
        self.file = open(out_dir / "alerts.jsonl", "a", buffering=1)
        self._q: queue.Queue = queue.Queue()
        if self.webhook:
            threading.Thread(target=self._post_loop, daemon=True, name="webhook").start()

    def send(self, alert: dict) -> None:
        self.file.write(json.dumps(alert, default=float) + "\n")
        if self.console:
            print(f"\n\033[1;31m[ALERT {ICON.get(alert['kind'], alert['kind'])}] {alert['time']}  {alert['message']}\033[0m",
                  flush=True)
        if self.webhook:
            self._q.put(alert)

    def _post_loop(self) -> None:
        import requests

        while True:
            a = self._q.get()
            try:  # `text` is understood by Slack, Teams (legacy connectors) and Discord-compatible hooks
                requests.post(self.webhook, json={"text": f"[{a['kind']}] {a['message']}", "alert": a}, timeout=10)
            except Exception as e:  # noqa: BLE001
                log.warning("webhook failed: %s", e)

    def close(self) -> None:
        self.file.close()
