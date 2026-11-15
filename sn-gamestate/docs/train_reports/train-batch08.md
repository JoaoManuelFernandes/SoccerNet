# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-07 07:41
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-09-05/19-47-13/states/train-batch08.pklz`

**Videos**: SNGS-100, SNGS-101, SNGS-102
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-100
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 17 | 7 | 22 | 2 | 46 | **37.0%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       12        34           5       260    0.98
       27        24          14        63    1.00
       43        14          65       108    1.00
       58        14          16       404    0.99
       67        62          52       318    1.00
       72        31           5       225    1.00
       75         4           9        46    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 54 | 0 | 2 | 54 | **100.0%** |

#### ID Switches detected: 2
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  17  frames=  74  purity=0.80
    primary : gt_tid=13 jersey=4 team=right (59fr)
    others  : gt_tid=8 jersey=14 team=right (15fr)
    predicted: jersey=None  team=right
  pred_tid=  46  frames=   6  purity=0.60
    primary : gt_tid=11 jersey=20 team=right (3fr)
    others  : gt_tid=8 jersey=14 team=right (2fr)
    predicted: jersey=None  team=right
```

### SNGS-101
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 15 | 1 | 22 | 2 | 38 | **39.5%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       13        33           6       207     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 47 | 0 | 2 | 47 | **100.0%** |

#### ID Switches detected: 2
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   5  frames= 259  purity=0.59
    primary : gt_tid=4 jersey=27 team=left (154fr)
    others  : gt_tid=13 jersey=None team=right (105fr)
    predicted: jersey=None  team=left
  pred_tid=  19  frames= 129  purity=0.66
    primary : gt_tid=4 jersey=27 team=left (76fr)
    others  : gt_tid=7 jersey=13 team=left (40fr)
    predicted: jersey=None  team=left
```

### SNGS-102
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 5 | 2 | 25 | 0 | 32 | **15.6%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       10        10          14       312     1.0
       24         7           1       563     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 46 | 0 | 1 | 46 | **100.0%** |

#### ID Switches detected: 1
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  32  frames=  45  purity=0.74
    primary : gt_tid=14 jersey=None team=right (31fr)
    others  : gt_tid=3 jersey=14 team=left (11fr)
    predicted: jersey=None  team=right
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 37 | 10 | 69 | 4 | 116 | **31.9%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 147 | 0 | 5 | 147 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **5** (across 3 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-100 | 17 | 74 | 0.8 | gt_tid=13 jersey=4 team=right (59fr) | right |
| SNGS-100 | 46 | 6 | 0.6 | gt_tid=11 jersey=20 team=right (3fr) | right |
| SNGS-101 | 5 | 259 | 0.59 | gt_tid=4 jersey=27 team=left (154fr) | left |
| SNGS-101 | 19 | 129 | 0.66 | gt_tid=4 jersey=27 team=left (76fr) | left |
| SNGS-102 | 32 | 45 | 0.74 | gt_tid=14 jersey=None team=right (31fr) | right |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.