# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-09 08:57
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-09-09/07-55-05/states/train-b15.pklz`

**Videos**: SNGS-156, SNGS-157, SNGS-158
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-156
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 12 | 9 | 21 | 6 | 42 | **28.6%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        7        29           5       170    0.99
        8        40           1       140    1.00
        9        30           6       130    1.00
       12        22           1       137    0.96
       32        10           5       496    1.00
       38        11          99       151    0.99
       43        29           5       503    1.00
       54         6           5       183    1.00
       60        24          26       160    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 43 | 0 | 6 | 43 | **100.0%** |

#### ID Switches detected: 6
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   2  frames= 487  purity=0.56
    primary : gt_tid=1 jersey=14 team=right (266fr)
    others  : gt_tid=9 jersey=28 team=left (211fr)
    predicted: jersey=14  team=right
  pred_tid=   6  frames= 395  purity=0.53
    primary : gt_tid=13 jersey=5 team=left (210fr)
    others  : gt_tid=11 jersey=10 team=left (178fr) | gt_tid=12 jersey=29 team=left (6fr)
    predicted: jersey=23  team=left
  pred_tid=  18  frames= 216  purity=0.51
    primary : gt_tid=9 jersey=28 team=left (111fr)
    others  : gt_tid=4 jersey=10 team=right (103fr) | gt_tid=17 jersey=11 team=left (1fr)
    predicted: jersey=33  team=left
  pred_tid=  36  frames=  29  purity=0.55
    primary : gt_tid=12 jersey=29 team=left (16fr)
    others  : gt_tid=19 jersey=None team=left (6fr) | gt_tid=13 jersey=5 team=left (5fr)
    predicted: jersey=29  team=left
  pred_tid=  44  frames=  17  purity=0.69
    primary : gt_tid=5 jersey=9 team=right (11fr)
    others  : gt_tid=3 jersey=55 team=right (5fr)
    predicted: jersey=nan  team=right
  pred_tid=  47  frames=   9  purity=0.50
    primary : gt_tid=5 jersey=9 team=right (4fr)
    others  : gt_tid=17 jersey=11 team=left (3fr) | gt_tid=21 jersey=None team=None (1fr)
    predicted: jersey=nan  team=left
```

### SNGS-157
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 17 | 7 | 13 | 2 | 37 | **45.9%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        4        29           5       341    1.00
       19        30          39       153    0.90
       21        10           5       147    0.88
       39        10           5        62    1.00
       42         3           6        63    0.98
       74         5           6       299    1.00
      127        11           1       143    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 51 | 0 | 2 | 51 | **100.0%** |

#### ID Switches detected: 2
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  20  frames= 391  purity=0.56
    primary : gt_tid=12 jersey=6 team=left (220fr)
    others  : gt_tid=7 jersey=29 team=left (171fr)
    predicted: jersey=5  team=left
  pred_tid=  87  frames= 299  purity=0.53
    primary : gt_tid=5 jersey=14 team=right (157fr)
    others  : gt_tid=7 jersey=29 team=left (139fr) | gt_tid=4 jersey=11 team=left (3fr)
    predicted: jersey=14  team=left
```

### SNGS-158
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 15 | 6 | 19 | 0 | 40 | **37.5%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        6        10           5       242    1.00
       17        30          37       282    1.00
       24        14           5       179    1.00
       34        14          16       269    0.99
       71         6           5       133    0.99
       97         6           5        84    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 52 | 0 | 1 | 52 | **100.0%** |

#### ID Switches detected: 2
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  30  frames=   9  purity=0.78
    primary : gt_tid=16 jersey=None team=left (7fr)
    others  : gt_tid=22 jersey=25 team=left (1fr) | gt_tid=23 jersey=32 team=left (1fr)
    predicted: jersey=None  team=left
  pred_tid=  31  frames=   6  purity=0.33
    primary : gt_tid=2 jersey=None team=None (2fr)
    others  : gt_tid=15 jersey=55 team=right (2fr) | gt_tid=21 jersey=9 team=left (1fr)
    predicted: jersey=None  team=left
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 44 | 22 | 53 | 8 | 119 | **37.0%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 146 | 0 | 9 | 146 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **10** (across 3 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-156 | 2 | 487 | 0.56 | gt_tid=1 jersey=14 team=right (266fr) | right |
| SNGS-156 | 6 | 395 | 0.53 | gt_tid=13 jersey=5 team=left (210fr) | left |
| SNGS-156 | 18 | 216 | 0.51 | gt_tid=9 jersey=28 team=left (111fr) | left |
| SNGS-156 | 36 | 29 | 0.55 | gt_tid=12 jersey=29 team=left (16fr) | left |
| SNGS-156 | 44 | 17 | 0.69 | gt_tid=5 jersey=9 team=right (11fr) | right |
| SNGS-156 | 47 | 9 | 0.5 | gt_tid=5 jersey=9 team=right (4fr) | left |
| SNGS-157 | 20 | 391 | 0.56 | gt_tid=12 jersey=6 team=left (220fr) | left |
| SNGS-157 | 87 | 299 | 0.53 | gt_tid=5 jersey=14 team=right (157fr) | left |
| SNGS-158 | 30 | 9 | 0.78 | gt_tid=16 jersey=None team=left (7fr) | left |
| SNGS-158 | 31 | 6 | 0.33 | gt_tid=2 jersey=None team=None (2fr) | left |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.