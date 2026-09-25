# Approach: what to build, why, and what to do next

## 1. The short answer

Your edge-and-brain, event-driven design is the right one. Keep it. The changes worth making are about **robustness and cost**, not a different architecture:

| Area | Recommendation | Where it is in the code |
|------|----------------|-------------------------|
| Plate reads | Async worker pool; **never** call the LLM on the video thread; one read per vehicle with the best 2 crops | `parkwatch/plates/readers.py` (`PlateService`) |
| LLM outages (C5) | Rotate model variants on 429/503 with per-variant cooldown; fall back to a free local reader; a 400/401 bad key disables the LLM instead of retrying | `LLMPlateReader.read` |
| Misreads | Validate with the **Singapore checksum** (your two measured plates `SHD1476M` and `SLN9145R` both pass); repair only O↔0, I↔1, S↔5, B↔8 in grammar positions; fuzzy exit match at edit distance 1 | `parkwatch/plates/formats.py` |
| Cost | Put `local` first in `plates.providers` so the LLM only sees plates the free reader could not validate | `configs/demo.yaml` |
| Tracker churn (C4) | Position re-ID for parked cars; short-horizon motion re-ID for moving cars; duplicate-crossing suppression at gates; duplicate plate within 60 s merges the visit | `edge.py`, `central.py` |
| Lot ↔ gate linking | FIFO arrival matching in a time window (entry → lot entry zone); lot departures back up exits with unreadable plates | `CentralService._link_arrivals`, `_resolve_exit` |
| Unknown arrivals | Cars already parked at start-up get a `≥` timer and **no** overstay alert by default (you cannot prove an overstay without an arrival time) | `site.alert_unknown_arrival` |
| Time | Media clock for recorded footage (results do not depend on CPU speed); `time_scale` for demos; wall clock for RTSP | `parkwatch/clock.py` |
| Evidence | Plate crop saved for every read; lot snapshot attached to every overstay alert | `evidence/` |
| Proof | A synthetic twin with ground truth, so every run prints measured accuracy | `simulator/`, `parkwatch/evaluate.py` |

## 2. Camera roles (unchanged, because your constraints are right)

* **C1 (one camera cannot do both jobs):** gate cameras are low and close for plates; the lot camera is high and wide for occupancy. Never try to read plates from the lot view.
* **C2 (a gate cannot measure parking):** duration is `exit time − entry time` from the gates. The lot camera adds *which bay* and *when it stopped*, and it is the only source for cars that were already inside at start-up.
* **C3 (overhead breaks COCO):** use an OBB model for the lot. The twin reproduces this exactly: COCO calls the rendered top-down cars "cell phone". Stock DOTA-OBB works on real aerial photos but needed a short fine-tune for the rendered site. That same fine-tune step (with real labelled frames) is what you do per client site when accuracy matters.

**Gate camera spec to give the client**, because this decides plate accuracy more than any model:

* Mount 1–1.5 m high (or 3–4 m for a pole), no more than ~30° horizontal and vertical angle to the plate. The vehicle should stop or slow at the barrier inside the field of view.
* Plate at least ~130 px wide (Latin characters ≥ 20 px tall) in the frame you send for reading.
* Shutter faster than 1/500 s, IR illuminator or an ANPR camera with IR for night, and WDR against headlight glare.
* Aim the camera at the stop line, not the barrier arm: a car waiting at the barrier gives ~2 s of perfect frames.

## 3. Models

| Job | Default | When to change |
|-----|---------|----------------|
| Gate detection | YOLO11s COCO (`car/truck/bus/motorcycle`) @ 512–640 | YOLO26s is the newer Ultralytics family and a drop-in change of `weights:`. Use `n` on CPU-only edge boxes. |
| Lot detection | YOLO11s-OBB (DOTA) @ 1024 for real aerial footage | Fine-tune on 200–500 labelled frames from the client's own lot camera (`tools/train_twin.py --data your.yaml`). |
| Tracking | BoT-SORT, `gmc_method: none` (fixed cameras), `track_buffer: 60` | ByteTrack is lighter; identity is held by plate and position anyway. |
| Plate reading | Gemini Flash-Lite via `ultralytics.LLM` → fast-alpr fallback | For volume: a fine-tuned plate detector + OCR (below) with the LLM only for low-confidence cases. |

### Cutting LLM calls to near zero (roadmap step 2)

1. **Plate detector.** fast-alpr already ships a YOLOv9-t plate detector (ONNX). Fine-tune one on your gate footage if plates are small or angled.
2. **OCR.** Fine-tune [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) on Singapore plates. Its models are ~3 MB and read in ~2 ms on CPU. The simulator's `plate_texture()` can generate unlimited synthetic SG plates for pre-training, and real crops from the `evidence/` folder (labelled with the LLM's answers after spot checks) make the fine-tune set.
3. **Cascade.** `providers: [local, llm]`. The LLM is called only when the local read fails the checksum or has low confidence, typically a few percent of vehicles.

## 4. Real-time budget

| Hardware | Gate cams (YOLO11s @ 512) | Lot cam (OBB @ 640) | Verdict |
|----------|---------------------------|---------------------|---------|
| 4-core laptop CPU (measured) | ~17 inferences/s each | ~22 inferences/s | 3 cams at stride 2/2/3 ≈ 0.6× real time → use `--realtime` (drops frames) or YOLO11n |
| NVIDIA RTX-class GPU | >100/s | >100/s | real time with margin |
| Jetson Orin Nano / NX (TensorRT FP16) | 30–60/s | 20–40/s | one Jetson per gate is a good edge unit |

Export for speed: `yolo export model=yolo11s.pt format=engine half=True` (TensorRT), `format=openvino` (Intel CPUs), or `format=onnx`.

Only line crossings and "parked" transitions matter, so gates need ~5–8 inferences/s and the lot 1–5/s. Use `stride` rather than a bigger GPU.

## 5. Cross-camera re-identification (roadmap step 4, do last)

The FIFO linking in this repo is correct when cars enter the lot view in the order they crossed the gate: one entry lane, with the lot camera covering the entrance. For multi-lane sites or partial coverage:

1. Add an appearance embedding per track (e.g. an OSNet or FastReID model trained on VeRi-776 / CityFlow-ReID; see DATASETS.md).
2. Match lot arrivals to recent entries by `time window × colour/appearance similarity` with Hungarian assignment instead of FIFO.
3. Keep the plate as the only *authoritative* identity. Re-ID only links views, and the exit plate closes the visit.

## 6. Operating it for real

* **Initial occupancy:** gate counting needs the count at start-up (`site.initial_occupancy`). Take it from the lot camera's parked count at the start of the day, or from a nightly manual check. Drift from missed crossings is corrected at that daily sync.
* **Alerts:** `alerts.webhook_url` posts JSON (with a `text` field, so Slack, Teams and Discord hooks work). Repeat alerts with `repeat_every_min`.
* **Privacy (Singapore PDPA):** show signage, keep plates and crops only as long as needed (e.g. 30 days, or until disputes close), restrict dashboard access (bind to `127.0.0.1` or put it behind the client's VPN or SSO), and log who exports data. With a local LLM or fast-alpr, no images leave the site.
* **Monitoring:** the dashboard's plate-reader panel shows calls, errors and latency per provider. Unreadable plates stay in the table with their evidence image for manual review.
