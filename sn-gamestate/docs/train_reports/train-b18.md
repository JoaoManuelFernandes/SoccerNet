# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-09 12:33
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-09-09/11-16-47/states/train-b18.pklz`

**Videos**: SNGS-165, SNGS-166, SNGS-167
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-165
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 11 | 2 | 7 | 2 | 20 | **55.0%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       86         8          25        77     1.0
       94        10          16         8     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 37 | 0 | 2 | 37 | **100.0%** |

#### ID Switches detected: 2
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  10  frames= 469  purity=0.56
    primary : gt_tid=4 jersey=23 team=left (264fr)
    others  : gt_tid=17 jersey=10 team=left (199fr) | gt_tid=7 jersey=29 team=right (6fr)
    predicted: jersey=5  team=left
  pred_tid=  22  frames= 322  purity=0.51
    primary : gt_tid=17 jersey=10 team=left (163fr)
    others  : gt_tid=4 jersey=23 team=left (153fr) | gt_tid=14 jersey=8 team=left (6fr)
    predicted: jersey=nan  team=left
```

### SNGS-166
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 7 | 5 | 5 | 1 | 17 | **41.2%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       12        10           4       339    0.99
        3        10           4       714    0.96
       35         4           6        49    1.00
       39        28          56        12    1.00
       47        22          14       171    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 36 | 0 | 1 | 36 | **100.0%** |

#### ID Switches detected: 1
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  55  frames=  61  purity=0.72
    primary : gt_tid=23 jersey=4 team=left (44fr)
    others  : gt_tid=2 jersey=None team=None (17fr)
    predicted: jersey=None  team=left
```

### SNGS-167
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 21 | 5 | 26 | 7 | 52 | **40.4%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       12        11           5       287    0.81
       18         9           6       250    0.98
       19         5          15       108    0.95
       30        28          38       278    1.00
       54         8           3        10    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 52 | 0 | 7 | 52 | **100.0%** |

#### ID Switches detected: 7
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   1  frames= 276  purity=0.80
    primary : gt_tid=15 jersey=8 team=right (220fr)
    others  : gt_tid=12 jersey=9 team=right (50fr) | gt_tid=16 jersey=10 team=right (5fr)
    predicted: jersey=5  team=right
  pred_tid=  15  frames= 274  purity=0.71
    primary : gt_tid=5 jersey=30 team=right (194fr)
    others  : gt_tid=21 jersey=None team=right (64fr) | gt_tid=19 jersey=5 team=right (6fr)
    predicted: jersey=22  team=right
  pred_tid=  16  frames= 284  purity=0.79
    primary : gt_tid=17 jersey=14 team=left (225fr)
    others  : gt_tid=18 jersey=25 team=left (53fr) | gt_tid=10 jersey=10 team=left (5fr)
    predicted: jersey=14  team=left
  pred_tid=  33  frames= 154  purity=0.75
    primary : gt_tid=22 jersey=11 team=right (116fr)
    others  : gt_tid=14 jersey=22 team=right (38fr)
    predicted: jersey=11  team=right
  pred_tid=  41  frames=  93  purity=0.74
    primary : gt_tid=23 jersey=3 team=left (69fr)
    others  : gt_tid=4 jersey=8 team=left (17fr) | gt_tid=5 jersey=30 team=right (7fr)
    predicted: jersey=3  team=left
  pred_tid=  32  frames= 154  purity=0.74
    primary : gt_tid=13 jersey=21 team=right (112fr)
    others  : gt_tid=15 jersey=8 team=right (37fr) | gt_tid=14 jersey=22 team=right (1fr)
    predicted: jersey=21  team=right
  pred_tid=  84  frames=  32  purity=0.75
    primary : gt_tid=16 jersey=10 team=right (24fr)
    others  : gt_tid=18 jersey=25 team=left (4fr) | gt_tid=1 jersey=26 team=left (4fr)
    predicted: jersey=nan  team=right
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 39 | 12 | 38 | 10 | 89 | **43.8%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 125 | 0 | 10 | 125 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **10** (across 3 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-165 | 10 | 469 | 0.56 | gt_tid=4 jersey=23 team=left (264fr) | left |
| SNGS-165 | 22 | 322 | 0.51 | gt_tid=17 jersey=10 team=left (163fr) | left |
| SNGS-166 | 55 | 61 | 0.72 | gt_tid=23 jersey=4 team=left (44fr) | left |
| SNGS-167 | 1 | 276 | 0.8 | gt_tid=15 jersey=8 team=right (220fr) | right |
| SNGS-167 | 15 | 274 | 0.71 | gt_tid=5 jersey=30 team=right (194fr) | right |
| SNGS-167 | 16 | 284 | 0.79 | gt_tid=17 jersey=14 team=left (225fr) | left |
| SNGS-167 | 33 | 154 | 0.75 | gt_tid=22 jersey=11 team=right (116fr) | right |
| SNGS-167 | 41 | 93 | 0.74 | gt_tid=23 jersey=3 team=left (69fr) | left |
| SNGS-167 | 32 | 154 | 0.74 | gt_tid=13 jersey=21 team=right (112fr) | right |
| SNGS-167 | 84 | 32 | 0.75 | gt_tid=16 jersey=10 team=right (24fr) | right |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.