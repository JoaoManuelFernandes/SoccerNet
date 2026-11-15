# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-08 20:20
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-09-08/19-10-42/states/train-b13.pklz`

**Videos**: SNGS-115, SNGS-151, SNGS-152
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-115
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 19 | 7 | 16 | 4 | 42 | **45.2%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        6        36           6       136    1.00
       31         9           6        28    1.00
       34         9          65        91    1.00
       47         9           3        16    0.81
       55         9           5       345    0.99
       59        35          56       240    1.00
       65        20           2       239    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 42 | 0 | 4 | 42 | **100.0%** |

#### ID Switches detected: 4
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   8  frames= 378  purity=0.58
    primary : gt_tid=6 jersey=36 team=left (219fr)
    others  : gt_tid=5 jersey=35 team=left (159fr)
    predicted: jersey=nan  team=left
  pred_tid=  26  frames= 523  purity=0.72
    primary : gt_tid=21 jersey=27 team=right (375fr)
    others  : gt_tid=19 jersey=36 team=right (142fr) | gt_tid=18 jersey=14 team=left (4fr)
    predicted: jersey=36  team=right
  pred_tid=  32  frames= 518  purity=0.60
    primary : gt_tid=8 jersey=8 team=right (309fr)
    others  : gt_tid=12 jersey=3 team=right (208fr) | gt_tid=9 jersey=14 team=right (1fr)
    predicted: jersey=5  team=right
  pred_tid=  41  frames=  95  purity=0.57
    primary : gt_tid=15 jersey=9 team=left (54fr)
    others  : gt_tid=5 jersey=35 team=left (41fr)
    predicted: jersey=5  team=left
```

### SNGS-151
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 10 | 7 | 11 | 2 | 28 | **35.7%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        6         6           5       171    1.00
       20         6           5        82    0.99
       67        30          10       212    1.00
       77        26           5       209    1.00
       81        10           5       207    1.00
       87         6           5       205    1.00
       98        14           5        79    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 40 | 0 | 2 | 40 | **100.0%** |

#### ID Switches detected: 2
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   2  frames= 503  purity=0.66
    primary : gt_tid=2 jersey=30 team=left (331fr)
    others  : gt_tid=12 jersey=6 team=right (172fr)
    predicted: jersey=30  team=left
  pred_tid=   9  frames= 451  purity=0.52
    primary : gt_tid=3 jersey=9 team=left (235fr)
    others  : gt_tid=1 jersey=10 team=left (201fr) | gt_tid=12 jersey=6 team=right (15fr)
    predicted: jersey=9  team=left
```

### SNGS-152
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 12 | 8 | 23 | 7 | 43 | **27.9%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        9        10           5        95    0.95
       16        25           5       237    0.93
       23         9           5       686    1.00
       29        55           5       328    0.86
       37         9           5       342    0.98
       48        29           5       185    1.00
       51        10           6       148    1.00
       58        40           1        94    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 46 | 0 | 7 | 46 | **100.0%** |

#### ID Switches detected: 7
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   7  frames= 466  purity=0.59
    primary : gt_tid=20 jersey=30 team=left (276fr)
    others  : gt_tid=4 jersey=10 team=left (169fr) | gt_tid=21 jersey=5 team=left (17fr)
    predicted: jersey=30  team=left
  pred_tid=  13  frames= 422  purity=0.78
    primary : gt_tid=16 jersey=22 team=left (329fr)
    others  : gt_tid=20 jersey=30 team=left (93fr)
    predicted: jersey=nan  team=left
  pred_tid=  18  frames= 292  purity=0.43
    primary : gt_tid=18 jersey=40 team=left (127fr)
    others  : gt_tid=20 jersey=30 team=left (125fr) | gt_tid=19 jersey=29 team=left (23fr)
    predicted: jersey=5  team=left
  pred_tid=  26  frames= 544  purity=0.72
    primary : gt_tid=10 jersey=10 team=right (393fr)
    others  : gt_tid=6 jersey=20 team=right (151fr)
    predicted: jersey=5  team=right
  pred_tid=  28  frames= 250  purity=0.76
    primary : gt_tid=21 jersey=5 team=left (191fr)
    others  : gt_tid=20 jersey=30 team=left (42fr) | gt_tid=19 jersey=29 team=left (16fr)
    predicted: jersey=29  team=left
  pred_tid=  30  frames= 536  purity=0.60
    primary : gt_tid=18 jersey=40 team=left (320fr)
    others  : gt_tid=16 jersey=22 team=left (216fr)
    predicted: jersey=5  team=left
  pred_tid=  34  frames= 150  purity=0.53
    primary : gt_tid=6 jersey=20 team=right (80fr)
    others  : gt_tid=5 jersey=None team=right (66fr) | gt_tid=19 jersey=29 team=left (4fr)
    predicted: jersey=nan  team=right
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 41 | 22 | 50 | 13 | 113 | **36.3%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 128 | 0 | 13 | 128 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **13** (across 3 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-115 | 8 | 378 | 0.58 | gt_tid=6 jersey=36 team=left (219fr) | left |
| SNGS-115 | 26 | 523 | 0.72 | gt_tid=21 jersey=27 team=right (375fr) | right |
| SNGS-115 | 32 | 518 | 0.6 | gt_tid=8 jersey=8 team=right (309fr) | right |
| SNGS-115 | 41 | 95 | 0.57 | gt_tid=15 jersey=9 team=left (54fr) | left |
| SNGS-151 | 2 | 503 | 0.66 | gt_tid=2 jersey=30 team=left (331fr) | left |
| SNGS-151 | 9 | 451 | 0.52 | gt_tid=3 jersey=9 team=left (235fr) | left |
| SNGS-152 | 7 | 466 | 0.59 | gt_tid=20 jersey=30 team=left (276fr) | left |
| SNGS-152 | 13 | 422 | 0.78 | gt_tid=16 jersey=22 team=left (329fr) | left |
| SNGS-152 | 18 | 292 | 0.43 | gt_tid=18 jersey=40 team=left (127fr) | left |
| SNGS-152 | 26 | 544 | 0.72 | gt_tid=10 jersey=10 team=right (393fr) | right |
| SNGS-152 | 28 | 250 | 0.76 | gt_tid=21 jersey=5 team=left (191fr) | left |
| SNGS-152 | 30 | 536 | 0.6 | gt_tid=18 jersey=40 team=left (320fr) | left |
| SNGS-152 | 34 | 150 | 0.53 | gt_tid=6 jersey=20 team=right (80fr) | right |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.