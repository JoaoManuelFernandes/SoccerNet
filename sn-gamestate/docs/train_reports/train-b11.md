# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-08 17:59
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-09-08/16-57-31/states/train-b11.pklz`

**Videos**: SNGS-109, SNGS-110, SNGS-111
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-109
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 15 | 5 | 32 | 4 | 52 | **28.8%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       32         6           5       245     1.0
       61         9           5        65     1.0
       69        31           5        82     1.0
       97        33           5       137     1.0
      102         7           1        65     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 57 | 0 | 5 | 57 | **100.0%** |

#### ID Switches detected: 5
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  14  frames= 197  purity=0.72
    primary : gt_tid=12 jersey=33 team=left (142fr)
    others  : gt_tid=14 jersey=None team=left (43fr) | gt_tid=13 jersey=31 team=left (8fr)
    predicted: jersey=5  team=left
  pred_tid=  28  frames=   8  purity=0.62
    primary : gt_tid=14 jersey=None team=left (5fr)
    others  : gt_tid=21 jersey=9 team=right (1fr) | gt_tid=7 jersey=6 team=right (1fr)
    predicted: jersey=nan  team=left
  pred_tid=  31  frames=  16  purity=0.60
    primary : gt_tid=7 jersey=6 team=right (9fr)
    others  : gt_tid=6 jersey=3 team=right (3fr) | gt_tid=5 jersey=24 team=right (3fr)
    predicted: jersey=nan  team=right
  pred_tid=  47  frames=  47  purity=0.60
    primary : gt_tid=6 jersey=3 team=right (28fr)
    others  : gt_tid=3 jersey=7 team=right (19fr)
    predicted: jersey=nan  team=right
  pred_tid=  52  frames= 136  purity=0.56
    primary : gt_tid=3 jersey=7 team=right (76fr)
    others  : gt_tid=14 jersey=None team=left (54fr) | gt_tid=13 jersey=31 team=left (5fr)
    predicted: jersey=5  team=left
```

### SNGS-110
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 12 | 2 | 39 | 4 | 53 | **22.6%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        1        21          24       641    0.85
       83         9          16       236    0.97
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 57 | 0 | 4 | 57 | **100.0%** |

#### ID Switches detected: 4
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   4  frames= 155  purity=0.71
    primary : gt_tid=13 jersey=4 team=left (110fr)
    others  : gt_tid=12 jersey=None team=left (43fr) | gt_tid=6 jersey=6 team=right (2fr)
    predicted: jersey=nan  team=left
  pred_tid=   6  frames= 422  purity=0.69
    primary : gt_tid=19 jersey=33 team=left (289fr)
    others  : gt_tid=12 jersey=None team=left (126fr) | gt_tid=13 jersey=4 team=left (3fr)
    predicted: jersey=33  team=left
  pred_tid=  30  frames=  36  purity=0.69
    primary : gt_tid=13 jersey=4 team=left (25fr)
    others  : gt_tid=12 jersey=None team=left (6fr) | gt_tid=5 jersey=9 team=right (3fr)
    predicted: jersey=nan  team=left
  pred_tid=  43  frames=   6  purity=0.67
    primary : gt_tid=6 jersey=6 team=right (4fr)
    others  : gt_tid=4 jersey=3 team=right (2fr)
    predicted: jersey=nan  team=right
```

### SNGS-111
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 13 | 5 | 12 | 1 | 30 | **43.3%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       19        10           5       620    1.00
       29        34          14        89    0.98
       41         9           5       474    1.00
       62        16          17       282    1.00
       68        14           5       111    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 0 | 42 | 1 | 42 | **0.0%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
        1    left     right       112    1.00
        2   right      left       415    0.84
        3   right      left        55    1.00
        4    left     right       141    1.00
        5    left     right       743    1.00
        6    left     right       345    1.00
        8   right      left        86    1.00
       10   right      left       300    1.00
       11    left     right       137    1.00
       12    left     right       126    0.98
```

#### ID Switches detected: 1
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   7  frames=  80  purity=0.65
    primary : gt_tid=4 jersey=27 team=left (51fr)
    others  : gt_tid=3 jersey=None team=None (28fr)
    predicted: jersey=None  team=right
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 40 | 12 | 83 | 9 | 135 | **29.6%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 114 | 42 | 10 | 156 | **73.1%** |

### ID Switches Summary
Total ID switches detected: **10** (across 3 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-109 | 14 | 197 | 0.72 | gt_tid=12 jersey=33 team=left (142fr) | left |
| SNGS-109 | 28 | 8 | 0.62 | gt_tid=14 jersey=None team=left (5fr) | left |
| SNGS-109 | 31 | 16 | 0.6 | gt_tid=7 jersey=6 team=right (9fr) | right |
| SNGS-109 | 47 | 47 | 0.6 | gt_tid=6 jersey=3 team=right (28fr) | right |
| SNGS-109 | 52 | 136 | 0.56 | gt_tid=3 jersey=7 team=right (76fr) | left |
| SNGS-110 | 4 | 155 | 0.71 | gt_tid=13 jersey=4 team=left (110fr) | left |
| SNGS-110 | 6 | 422 | 0.69 | gt_tid=19 jersey=33 team=left (289fr) | left |
| SNGS-110 | 30 | 36 | 0.69 | gt_tid=13 jersey=4 team=left (25fr) | left |
| SNGS-110 | 43 | 6 | 0.67 | gt_tid=6 jersey=6 team=right (4fr) | right |
| SNGS-111 | 7 | 80 | 0.65 | gt_tid=4 jersey=27 team=left (51fr) | right |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.