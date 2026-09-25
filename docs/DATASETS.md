# Datasets and test footage: where to look and what to use each one for

The stock models already cover generic vehicle detection (COCO for ground views, DOTA for aerial views). You only need extra data for three jobs:

1. **Plate detection and OCR**, to cut LLM calls (roadmap step 2).
2. **Site adaptation** of the lot camera, if the client's overhead angle differs from drone imagery.
3. **Cross-camera re-ID**, only if FIFO linking is not enough (roadmap step 4).

Check each dataset's licence before commercial use. Many are research-only. Links were checked when this was written.

## 1. Licence plates (detection + OCR)

| Dataset | Size / content | Use it for | Link |
|---------|----------------|------------|------|
| **Roboflow Universe: License Plate Recognition** | ~10k images, many countries, YOLO export built in | plate *detector* fine-tune, fastest start | https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e |
| Roboflow Universe search "singapore license plate" | small community sets of SG plates | SG-specific OCR and detector validation | https://universe.roboflow.com/search?q=singapore%20license%20plate |
| **Open Images V7**, class *Vehicle registration plate* | thousands of boxes on varied real photos | detector robustness (angles, clutter) | https://storage.googleapis.com/openimages/web/index.html |
| **CCPD** (Chinese City Parking Dataset) | 300k parking-lot images, plate corners + text | detector at scale, tilt and blur, parking context | https://github.com/detectRecog/CCPD |
| **UFPR-ALPR** | 4,500 frames from 150 vehicle *videos* | tracking + ALPR on video, realistic motion | https://web.inf.ufpr.br/vri/databases/ufpr-alpr/ |
| **RodoSol-ALPR** | 20k toll-booth images (cars and motorcycles) | gate/toll-like viewpoint, day and night | https://github.com/raysonlaroca/rodosol-alpr-dataset |
| OpenALPR benchmarks | small US/EU sets with plate text | quick OCR sanity tests | https://github.com/openalpr/benchmarks |
| Kaggle "Car License Plate Detection" | 433 images (Pascal VOC) | tiny smoke-test set | https://www.kaggle.com/datasets/andrewmvd/car-plate-detection |
| **Synthetic SG plates (this repo)** | unlimited: `simulator.vehicles.plate_texture()` renders valid-checksum plates | pre-training the OCR on the exact SG format | `simulator/vehicles.py` |

Ready-made models to start from, so you don't train from zero:
* **fast-alpr / fast-plate-ocr / open-image-models**, already the local reader here. ONNX, runs on CPU, fine-tunable. https://github.com/ankandrew/fast-alpr and https://github.com/ankandrew/fast-plate-ocr

**Best source of all: the client's own gate footage.** Run this pipeline with the LLM on for a week. Every read is saved in `evidence/` with the LLM's answer, and checksum-valid answers are near-certainly correct. After a spot check that is a free labelled OCR set from the exact cameras, angles and lighting you will deploy on.

## 2. Vehicles from above (lot camera)

| Dataset | Content | Use it for | Link |
|---------|---------|------------|------|
| **DOTA v1.5 / v2** | aerial images, oriented boxes; *small vehicle* / *large vehicle* | what `yolo11s-obb.pt` is trained on; extend with your frames | https://captain-whu.github.io/DOTA/ |
| **CARPK** and **PUCPR+** | ~90k cars in drone / high-camera **parking-lot** images | closest match to a lot camera; counting | https://lafi.github.io/LPN/ |
| **PKLot** | 12k images of 3 lots, each bay labelled occupied/empty | bay-level occupancy classifier, bay polygons | https://web.inf.ufpr.br/vri/databases/parking-lot-database/ (also on Roboflow Public) |
| **CNRPark+EXT** | 150k bay patches, 9 cameras, weather changes | occupancy robustness to rain, shadows, night | http://cnrpark.it/ |
| VisDrone | drone video with vehicle tracks | tracking from above | https://github.com/VisDrone/VisDrone-Dataset |
| UAVDT / DroneVehicle | drone video; RGB + infrared | night / thermal lot monitoring | search "UAVDT dataset", "DroneVehicle dataset" |

Converting axis-aligned labels (CARPK, VisDrone) to OBB: parked cars are aligned with the bays, so axis-aligned boxes rotated by the bay angle work. Ultralytics also documents a DOTA → YOLO-OBB converter.

## 3. Ground-level vehicles and traffic (gate cameras)

COCO (inside `yolo11s.pt`) is enough for detecting cars at a gate. For harder conditions:

| Dataset | Content | Link |
|---------|---------|------|
| UA-DETRAC | 10 h of traffic-camera video, 140k frames, night and rain | https://detrac-db.rit.albany.edu/ (mirrors on Kaggle) |
| MIO-TCD | 650k surveillance-camera crops, 11 vehicle classes | https://tcd.miovision.com/ |
| BDD100K | 100k driving videos, varied weather | https://www.bdd100k.com/ |

## 4. Cross-camera re-identification (do last)

| Dataset | Content | Link |
|---------|---------|------|
| **CityFlow / CityFlow-ReID** (AI City Challenge) | multi-camera, multi-target vehicle tracking across 40+ cameras, the closest public match to "track a car across the site" | https://www.aicitychallenge.org/ |
| VeRi-776 | 50k images of 776 vehicles from 20 cameras | https://github.com/JDAI-CV/VeRidataset |
| VehicleID / VERI-Wild | large-scale vehicle re-ID | search by name |

## 5. Test footage for demos

**The synthetic twin in this repo** (`python tools/make_demo_videos.py`) is the only footage where the entry, lot and exit cameras watch the *same* cars with known ground truth. Use it for the end-to-end story and the accuracy numbers.

Real footage, free for commercial use (Pexels licence). Download on your machine:

| Clip | Camera role it simulates |
|------|--------------------------|
| [Aerial view of crowded car parking lot](https://www.pexels.com/video/aerial-view-of-crowded-car-parking-lot-33610407/) | lot (overhead), many parked cars, good for the OBB model |
| [An aerial view of a parking lot with cars](https://www.pexels.com/video/an-aerial-view-of-a-parking-lot-with-cars-28125710/) | lot (overhead) |
| [Drone shot of cars parked in the parking lot](https://www.pexels.com/video/drone-shot-of-cars-parked-in-the-parking-lot-5972218/) | lot (overhead) |
| [Drone footage of cars parked and driving in a mall parking lot](https://www.pexels.com/video/drone-footage-of-cars-parked-and-driving-in-parking-lot-of-the-mall-5607778/) | lot with moving cars: parked / unparked transitions |
| [Aerial footage of a parking lot at night](https://www.pexels.com/video/aerial-footage-of-a-parking-lot-during-night-time-9782492/) | night stress test |
| Pexels searches: [license plate](https://www.pexels.com/search/videos/license%20plate/), [parking lot](https://www.pexels.com/search/videos/parking%20lot/), [garage](https://www.pexels.com/search/videos/garage/) | gate cameras: pick clips where the plate is at least ~130 px wide |
| Pixabay search: [car enters parking](https://pixabay.com/videos/search/car%20enters%20parking/) | gate cameras |
| Ultralytics sample: `https://github.com/ultralytics/assets/releases/download/v0.0.0/solution_ci_parking_demo.mp4` | small overhead parking clip used in Ultralytics CI |

Tip: for a gate clip, run `tools/draw_zones.py` on it, set `direction` to the way cars move across your line, and use the stock `yolo11s.pt`. Real cars are what COCO was trained on.

## 6. Why the twin is labelled "fine-tuned"

Stock weights fail on the *rendered* overhead view: COCO says "cell phone" (exactly your C3 finding) and DOTA-OBB says "ship". `tools/make_training_data.py` renders five random episodes (different cars, colours, bays and timings from the demo clip) with free, exact labels. `tools/train_twin.py` fine-tunes YOLO11n-OBB on them in about ten minutes on a CPU, reaching mAP50 0.995 on held-out episodes. The same three commands with real labelled frames are how you adapt the lot model to a client camera.
