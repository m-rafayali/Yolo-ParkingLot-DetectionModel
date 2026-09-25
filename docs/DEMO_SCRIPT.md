# Client demo: run sheet and talk track (about 15 minutes)

## Before the meeting (the day before)

```bash
python tools/make_demo_videos.py        # once; renders assets/demo/*.mp4
python run_demo.py --max-seconds 20     # warm-up: downloads yolo11s.pt and the fast-alpr models, checks everything runs
python run_demo.py                      # full run -> runs/<ts>/annotated_mosaic.mp4 = your BACKUP recording
```

* Set `GEMINI_API_KEY` if you want to show the live LLM cost counter. Without a key the free local reader is used and the demo still works.
* The laptop must be on power. On a CPU-only laptop, use `--realtime` (it drops frames to stay live) or play the backup recording.
* Have one of the client's own clips ready (see "Part 3").

## Part 1: the problem (2 min, no screen)

"You have no automated view of who is in the lot, for how long, or how much space is left. We answer four questions with the CCTV you already have, with no sensors, barriers or tickets: **free spaces, who arrived when, how long they've stayed, and when they left or overstayed.**"

## Part 2: live run (8 min)

```bash
python run_demo.py --realtime --hold          # then open http://localhost:8000 full-screen
```

| When (site clock) | Show | Say |
|-------------------|------|-----|
| start | the three camera tiles | "One virtual site, three cameras with different jobs: a gate camera must be low and close to read plates, the lot camera high and wide to see bays. One camera can't do both." (C1) |
| 08:00 | lot bays turn red, timers with a `+` | "Ten cars were already here when we switched on. We don't know when they arrived, so we show a minimum and don't accuse anyone." |
| 08:09 | entry: box, green line, plate appears | "The plate is read **once per car**, not per frame, from the best crop. Here it's read locally for free. With Gemini it's about a tenth of a cent per car." Point at the LLM counter. |
| 08:10–08:20 | the car drives into T01 and turns green, then red | "The lot camera follows it to its bay. Tracker IDs flicker, so identity is held by the plate and by position: a parked car can't move." (C4) |
| 08:58 | exit: plate read, row shows EXITED 48 min | "Visit closed: in, out, duration, bay, with the plate images as evidence." |
| 09:12 | a car parked before start-up leaves | "No entry record, so it's matched through the lot camera and flagged 'arrived before system start'. Nothing is silently dropped." |
| **≈09:18** | **red alert, OVERSTAY in B02, evidence thumbnails** | "One hour. This alert can go to Slack, Teams or your enforcement app, with the plate crop and a snapshot of the bay." |
| ≈10:14 | the overstayer exits: EXITED_OVERSTAY 1h 55m and an alert | "And we know exactly when it left and by how much it overstayed." |
| end | terminal: ground-truth report | "Because this site is simulated, we know what really happened, and the system is scored against it every run." |

Say clearly: **"This footage is simulated, and the clock runs 60× faster so a 1-hour rule fits in 3 minutes. The logic and the 60-minute limit are the production ones."**

## Part 3: their footage (3 min)

```bash
python tools/draw_zones.py --source client_gate.mp4 --camera entry --out configs/client_geometry.json
python run_demo.py --config configs/client.yaml --realtime
```

"Same pipeline on your camera. The biggest accuracy factor is where the gate camera is mounted, so the next step is a site survey."

## Part 4: cost, privacy, next steps (2 min)

* **Cost:** LLM calls scale with *vehicles*, not frames: 500 cars/day ≈ 1,000 reads (in and out) ≈ US$1/day with Gemini Flash-Lite. With the local reader first (`providers: [local, llm]`) it drops to cents. A fully offline option exists (local Qwen2.5-VL or fast-alpr), so no images leave the site.
* **Reliability:** if the LLM provider is rate-limited or down (you saw 429/503 in testing), the system rotates model variants and falls back to the local reader. It never stops counting.
* **Privacy:** plates and timestamps are personal data (PDPA). We agree retention, signage and access before launch.
* **Next:** site survey (gate camera at plate height) → one-week pilot on real cameras with measured accuracy → bay polygons for all lot cameras → alerts integration → edge hardware (Jetson / TensorRT).

## If something goes wrong

| Symptom | Fix |
|---------|-----|
| Dashboard empty | the footage may have finished: rerun with `--hold`. Or port 8000 is busy: `--port 8080`. |
| Very slow on the laptop | `--realtime` (drops frames), or set `ground.weights: yolo11n.pt`, or play `runs/<ts>/annotated_mosaic.mp4` |
| "missing video" | `python tools/make_demo_videos.py` |
| LLM counter stays 0 | no `GEMINI_API_KEY`: the local reader is doing the work (see the "Plate reading" panel) |
