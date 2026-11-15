# SoccerNet GameState — Progress Notes

## Pipeline Overview

```
bbox_detector → reid → track → pitch → calibration → [jersey_number_detect] → tracklet_agg → team → team_side
```

The pipeline runs on the SoccerNet Game State dataset (valid split).  
State can be saved/loaded as `.pklz` to avoid re-running expensive modules.

---

## 1. Jersey Number Recognition (JNR)

### V1 — TrackletsOCR (`tracklets_ocr.py`)

- **Approach**: for each tracklet, collect all detection crops → run OCR (EasyOCR) → vote for the most frequent digit string
- **Config**: `configs/modules/jersey_number_detect/tracklets_ocr.yaml`
- **Strengths**: fast, no GPU-heavy model, works on any visible side
- **Weaknesses**: OCR on small low-res crops fails frequently; no distinction between front/back of jersey; false positives from noise, numbers on shirts/ads

### V2 — TrackletsJNRv2 (`tracklets_jnr_v2.py` + `tracklets_jnr_v2_module.py`)

- **Approach**:
  1. YOLO pose model detects keypoints to locate the player's back region
  2. Crops the back-of-jersey ROI using shoulder/hip keypoints
  3. Runs Qwen-VL (vision-language model) to read the number
  4. Votes across frames per tracklet
- **Config**: `configs/modules/jersey_number_detect/tracklets_jnr_v2.yaml`
- **Strengths**: dramatically better accuracy; targets the back number specifically; robust to lighting/blur via VLM
- **Weaknesses**: slow (Qwen-VL is ~2–3s/frame on GPU); only works when back is visible; fails on very small/distant players (no debug image in `jnr_v2_debug/`)
- **Key insight found**: tracklets with no images in `jnr_v2_debug/` have no reliable jersey assignment — these should be excluded from any jersey-based evaluation

### Key Finding: Jersey-Based Eval is Ambiguous
Both teams can have players with the same jersey number (e.g., both teams have a #11 and #17).  
Any evaluation that links tracklets to GT via jersey number alone will produce false matches.  
→ Use tracklet IoU matching against GT bboxes instead (see `eval_team_by_tracklet.py`).

---

## 2. Team Clustering (`team` module)

**File**: `sn_gamestate/team/tracklet_team_clustering_api.py`

### What Changed

**Before**: KMeans on raw per-tracklet mean ReID embeddings (256-dim).  
Baseline result: ~74% team accuracy (jersey-number eval, 3 videos).

**After**:
1. **Color features added in ReID** (`prtreid_api.py`):
   - HSV histogram: 8 bins × 3 channels = 24 features, normalised to sum=1
   - Lab mean: 3 features normalised 0–1 (perceptually uniform, illumination-robust)
   - `embeddings_with_colors` = 256 (ReID) + 27 (color) = 283-dim vector
   - **Option A** (fallback): fixed 25–75% height crop — removes grass/shoes but includes some arm/skin pixels
   - **Option B** (BPBreID torso mask): uses body part attention mask index 1 (torso) to isolate jersey fabric. Resized to crop dimensions, threshold 0.25. Falls back to Option A if mask is empty or < 16 pixels.
   - Result: Option A and Option B gave identical results — the BPBreID torso mask at feature resolution is too diffuse after upsampling to reliably exclude arm/skin pixels.

2. **StandardScaler** applied before KMeans — prevents ReID dims (larger variance) from dominating color dims.

3. **Feature column selection**: uses `embeddings_with_colors` when available (non-zero color features), falls back to `embeddings`.

### Why Color Features Help
KMeans on ReID alone fails when players from different teams have similar poses/body shapes.  
HSV histogram gives KMeans direct jersey color information → cluster separation improved significantly.

### Remaining Limitations
The 25–75% crop still includes exposed arm/skin pixels in short-sleeve jerseys.  
Skin tone (hue ~22° in HSV) can dominate the histogram for distant/small players, misleading KMeans.

---

## 3. Team Side Labeling (`team_side` module)

**File**: `sn_gamestate/team/tracklet_team_side_labeling_api.py`

### What Changed

**Before**: simple cluster mean pitch_x comparison — whichever cluster had higher mean x was "right".  
Problem: a single cluster's mean can be skewed by a few outlier frames; SNGS-022 had only 5.1m separation.

**After** (priority chain):
1. **Tracklet vote**: each tracklet votes once based on its own mean `pitch_x` (not per-frame). Threshold: 60% of tracklets must agree. Each tracklet counts once regardless of length.
2. **Mean-X fallback**: original approach, used only if tracklet vote is inconclusive.
3. **GK direct assignment**: goalkeepers are never in team_cluster (excluded from KMeans). They're assigned directly via `pitch_x` — GKs with `|pitch_x| > 30m` are near their goal, giving a reliable geometric anchor.

### Evaluation Methodology

**Old eval** (`eval_team_side.py`): linked predicted tracklets to GT via jersey number → unreliable when:
- JNR assigned wrong jersey (short/distant tracklets)
- Both teams share the same jersey number
- Referee misclassified as player

**New eval** (`eval_team_by_tracklet.py`): IoU matching between predicted bboxes and GT bboxes per frame.
- `IOU_THRESHOLD = 0.3`
- `MIN_FRAMES = 5` — ignores ghost/short tracklets (< 5 detections)
- `MATCH_RATIO_MIN = 0.3` — predicted tracklet must match GT in at least 30% of its frames
- `PURITY_MIN = 0.8` — at least 80% of matched frames must go to the same GT track

Usage:
```bash
python eval_team_by_tracklet.py <path_to_state.pklz>
```

### Results (3 videos: SNGS-021, 022, 023)

| Eval method | Videos | Score |
|---|---|---|
| Jersey-number eval (old) | 021+022+023 | 86% (43/50) |
| IoU tracklet eval (new) | 021+022+023 | 100% (152/152 clean tracklets) |

The 14% gap was entirely due to eval artifacts, not real team_side failures.

### Key Finding: ID Switch Masking Failures

**SNGS-023, tid=6** — visible in the video as "wrong team color on jersey=11 (white)":
- Frames 1–363: tracker correctly follows jersey=11 (team right, white jersey)
- Frame 364: **ID switch** — tracker reassigns tid=6 to jersey=17 (team left, blue jersey)
- Frames 364–750: tracker follows jersey=17

Consequence:
- Mean embedding of tid=6 = mix of two players from opposite teams → KMeans assigns cluster "left"
- First 363 frames appear incorrectly colored in the video
- **Root cause**: tracker failure (ID switch), not team_side

Detection: purity filter catches this case (purity = 382/736 = 0.52 < 0.80 threshold) → excluded from eval.

How to confirm an ID switch in data:
```python
# Sort detections by frame, compute best IoU match to GT per frame
# If best GT track_id changes sharply at a specific frame → ID switch
```

The crops visually confirmed the switch:
- Early frames: white jersey players in bbox
- Late frames: blue jersey players in bbox

### Limitations of Current Eval
The `PURITY_MIN` filter correctly excludes ambiguous cases but means **the 100% is over the evaluable subset**.  
True system accuracy is lower when including ID-switch tracklets.  
To improve: fix at the tracker level (stronger ReID to prevent ID switches), or detect and split impure tracklets before clustering.

---

## 4. Files Reference

### Active Pipeline Files

| File | Purpose |
|---|---|
| `sn_gamestate/reid/prtreid_api.py` | ReID + color feature extraction |
| `sn_gamestate/team/tracklet_team_clustering_api.py` | KMeans team clustering |
| `sn_gamestate/team/tracklet_team_side_labeling_api.py` | Left/right assignment |
| `sn_gamestate/jersey/tracklets_ocr.py` | JNR v1 (OCR-based) |
| `sn_gamestate/jersey/tracklets_jnr_v2.py` | JNR v2 core logic |
| `sn_gamestate/jersey/tracklets_jnr_v2_module.py` | JNR v2 pipeline module |
| `sn_gamestate/jersey/tracklets_utils.py` | Shared crop/voting utilities |
| `eval_team_by_tracklet.py` | Team eval via IoU matching (no jersey dependency) |

### Config Files

| Config | Module |
|---|---|
| `configs/modules/jersey_number_detect/tracklets_ocr.yaml` | JNR v1 |
| `configs/modules/jersey_number_detect/tracklets_jnr_v2.yaml` | JNR v2 |
| `configs/modules/reid/prtreid.yaml` | ReID (BPBreID) |
| `configs/modules/team/kmeans_embeddings.yaml` | Team clustering |
| `configs/modules/team_side/mean_position.yaml` | Team side labeling |
| `configs/soccernet.yaml` | Main pipeline config |

### 20-Video Evaluation Run

Config ready in `soccernet.yaml`:
- Pipeline: `bbox_detector → reid → track → pitch → calibration → tracklet_agg → team → team_side`
- Videos: SNGS-021–024, 028–033, 045–050, 081–084
- No JNR (excluded for speed)
- State saved to: `states/sn-gamestate-20vid-team.pklz`

Run:
```bash
cd /home/joao/soccernet/sn-gamestate
python -m tracklab.main -cn soccernet
python eval_team_by_tracklet.py outputs/sn-gamestate/<date>/<time>/states/sn-gamestate-20vid-team.pklz
```
