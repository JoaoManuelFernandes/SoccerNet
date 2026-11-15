# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-07-08 10:25
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-07-08/09-48-08/states/final-TB8.pklz`

**Videos**: SNGS-197, SNGS-198, SNGS-199, SNGS-200
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-197
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 12 | 3 | 13 | 1 | 28 | **42.9%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        4        62          34       739     1.0
       62        10           5        19     1.0
       72        10          11        13     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 38 | 0 | 1 | 38 | **100.0%** |

#### ID Switches detected: 1
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  31  frames=   5  purity=0.60
    primary : gt_tid=10 jersey=33 team=left (3fr)
    others  : gt_tid=16 jersey=None team=right (2fr)
    predicted: jersey=None  team=left
```

### SNGS-198
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 17 | 6 | 30 | 7 | 53 | **32.1%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        1        14          15       206    1.00
       12        14          16       311    0.97
       17        22           5       491    1.00
       25        31          22       297    1.00
       31        33          32        97    1.00
       49        14           5       175    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 53 | 0 | 7 | 53 | **100.0%** |

#### ID Switches detected: 8
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   9  frames= 438  purity=0.54
    primary : gt_tid=14 jersey=3 team=left (235fr)
    others  : gt_tid=10 jersey=5 team=left (198fr) | gt_tid=12 jersey=34 team=right (3fr)
    predicted: jersey=5  team=left
  pred_tid=  21  frames= 126  purity=0.63
    primary : gt_tid=17 jersey=17 team=left (79fr)
    others  : gt_tid=16 jersey=31 team=left (39fr) | gt_tid=19 jersey=9 team=right (8fr)
    predicted: jersey=None  team=left
  pred_tid=  24  frames= 296  purity=0.72
    primary : gt_tid=19 jersey=9 team=right (211fr)
    others  : gt_tid=12 jersey=34 team=right (69fr) | gt_tid=14 jersey=3 team=left (13fr)
    predicted: jersey=None  team=right
  pred_tid=  30  frames=   7  purity=0.71
    primary : gt_tid=4 jersey=27 team=left (5fr)
    others  : gt_tid=9 jersey=14 team=left (1fr) | gt_tid=10 jersey=5 team=left (1fr)
    predicted: jersey=None  team=left
  pred_tid=  34  frames=  10  purity=0.60
    primary : gt_tid=17 jersey=17 team=left (6fr)
    others  : gt_tid=19 jersey=9 team=right (4fr)
    predicted: jersey=None  team=right
  pred_tid=  52  frames=   9  purity=0.56
    primary : gt_tid=20 jersey=None team=left (5fr)
    others  : gt_tid=9 jersey=14 team=left (3fr) | gt_tid=14 jersey=3 team=left (1fr)
    predicted: jersey=None  team=left
  pred_tid=  53  frames=  19  purity=0.60
    primary : gt_tid=11 jersey=15 team=left (9fr)
    others  : gt_tid=9 jersey=14 team=left (6fr)
    predicted: jersey=None  team=left
  pred_tid=  67  frames=   8  purity=0.50
    primary : gt_tid=5 jersey=24 team=right (4fr)
    others  : gt_tid=11 jersey=15 team=left (4fr)
    predicted: jersey=None  team=right
```

### SNGS-199
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 2 | 5 | 23 | 4 | 30 | **6.7%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        6        27           5       388    0.99
       12         3           5       644    0.99
       15         7           5       384    1.00
       63        30           5       191    0.98
       80         7           1       222    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 62 | 0 | 12 | 62 | **100.0%** |

#### ID Switches detected: 12
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   7  frames= 379  purity=0.66
    primary : gt_tid=12 jersey=24 team=right (250fr)
    others  : gt_tid=20 jersey=None team=left (129fr)
    predicted: jersey=None  team=right
  pred_tid=  11  frames= 361  purity=0.80
    primary : gt_tid=8 jersey=None team=right (288fr)
    others  : gt_tid=19 jersey=None team=left (41fr) | gt_tid=10 jersey=None team=right (25fr)
    predicted: jersey=5  team=right
  pred_tid=  21  frames= 257  purity=0.41
    primary : gt_tid=12 jersey=24 team=right (106fr)
    others  : gt_tid=20 jersey=None team=left (101fr) | gt_tid=19 jersey=None team=left (25fr)
    predicted: jersey=None  team=left
  pred_tid=  25  frames=  28  purity=0.54
    primary : gt_tid=18 jersey=None team=left (15fr)
    others  : gt_tid=20 jersey=None team=left (7fr) | gt_tid=13 jersey=62 team=right (3fr)
    predicted: jersey=None  team=left
  pred_tid=  24  frames= 259  purity=0.59
    primary : gt_tid=9 jersey=None team=right (154fr)
    others  : gt_tid=7 jersey=30 team=right (92fr) | gt_tid=13 jersey=62 team=right (13fr)
    predicted: jersey=None  team=right
  pred_tid=  36  frames= 101  purity=0.59
    primary : gt_tid=18 jersey=None team=left (60fr)
    others  : gt_tid=19 jersey=None team=left (36fr) | gt_tid=9 jersey=None team=right (4fr)
    predicted: jersey=6  team=left
  pred_tid=  40  frames= 360  purity=0.74
    primary : gt_tid=16 jersey=14 team=left (265fr)
    others  : gt_tid=13 jersey=62 team=right (88fr) | gt_tid=14 jersey=None team=left (3fr)
    predicted: jersey=18  team=left
  pred_tid=  43  frames= 367  purity=0.57
    primary : gt_tid=20 jersey=None team=left (210fr)
    others  : gt_tid=21 jersey=None team=None (157fr)
    predicted: jersey=None  team=left
  pred_tid=  61  frames=  99  purity=0.61
    primary : gt_tid=4 jersey=None team=right (60fr)
    others  : gt_tid=18 jersey=None team=left (39fr)
    predicted: jersey=None  team=right
  pred_tid=  69  frames= 154  purity=0.50
    primary : gt_tid=8 jersey=None team=right (77fr)
    others  : gt_tid=9 jersey=None team=right (75fr) | gt_tid=19 jersey=None team=left (2fr)
    predicted: jersey=None  team=right
  pred_tid= 102  frames= 116  purity=0.58
    primary : gt_tid=16 jersey=14 team=left (66fr)
    others  : gt_tid=19 jersey=None team=left (48fr)
    predicted: jersey=None  team=left
  pred_tid= 108  frames=  12  purity=0.50
    primary : gt_tid=6 jersey=None team=right (6fr)
    others  : gt_tid=19 jersey=None team=left (6fr)
    predicted: jersey=None  team=left
```

### SNGS-200
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 18 | 3 | 17 | 2 | 38 | **47.4%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        5         3          13       329    1.00
       10        33          12        66    0.98
       45         3           5       367    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 38 | 0 | 2 | 38 | **100.0%** |

#### ID Switches detected: 2
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   4  frames= 241  purity=0.78
    primary : gt_tid=7 jersey=7 team=right (188fr)
    others  : gt_tid=6 jersey=16 team=right (53fr)
    predicted: jersey=7  team=right
  pred_tid=  15  frames= 263  purity=0.71
    primary : gt_tid=6 jersey=16 team=right (187fr)
    others  : gt_tid=7 jersey=7 team=right (76fr)
    predicted: jersey=16  team=right
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 49 | 17 | 83 | 14 | 149 | **32.9%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 191 | 0 | 22 | 191 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **23** (across 4 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-197 | 31 | 5 | 0.6 | gt_tid=10 jersey=33 team=left (3fr) | left |
| SNGS-198 | 9 | 438 | 0.54 | gt_tid=14 jersey=3 team=left (235fr) | left |
| SNGS-198 | 21 | 126 | 0.63 | gt_tid=17 jersey=17 team=left (79fr) | left |
| SNGS-198 | 24 | 296 | 0.72 | gt_tid=19 jersey=9 team=right (211fr) | right |
| SNGS-198 | 30 | 7 | 0.71 | gt_tid=4 jersey=27 team=left (5fr) | left |
| SNGS-198 | 34 | 10 | 0.6 | gt_tid=17 jersey=17 team=left (6fr) | right |
| SNGS-198 | 52 | 9 | 0.56 | gt_tid=20 jersey=None team=left (5fr) | left |
| SNGS-198 | 53 | 19 | 0.6 | gt_tid=11 jersey=15 team=left (9fr) | left |
| SNGS-198 | 67 | 8 | 0.5 | gt_tid=5 jersey=24 team=right (4fr) | right |
| SNGS-199 | 7 | 379 | 0.66 | gt_tid=12 jersey=24 team=right (250fr) | right |
| SNGS-199 | 11 | 361 | 0.8 | gt_tid=8 jersey=None team=right (288fr) | right |
| SNGS-199 | 21 | 257 | 0.41 | gt_tid=12 jersey=24 team=right (106fr) | left |
| SNGS-199 | 25 | 28 | 0.54 | gt_tid=18 jersey=None team=left (15fr) | left |
| SNGS-199 | 24 | 259 | 0.59 | gt_tid=9 jersey=None team=right (154fr) | right |
| SNGS-199 | 36 | 101 | 0.59 | gt_tid=18 jersey=None team=left (60fr) | left |
| SNGS-199 | 40 | 360 | 0.74 | gt_tid=16 jersey=14 team=left (265fr) | left |
| SNGS-199 | 43 | 367 | 0.57 | gt_tid=20 jersey=None team=left (210fr) | left |
| SNGS-199 | 61 | 99 | 0.61 | gt_tid=4 jersey=None team=right (60fr) | right |
| SNGS-199 | 69 | 154 | 0.5 | gt_tid=8 jersey=None team=right (77fr) | right |
| SNGS-199 | 102 | 116 | 0.58 | gt_tid=16 jersey=14 team=left (66fr) | left |
| SNGS-199 | 108 | 12 | 0.5 | gt_tid=6 jersey=None team=right (6fr) | left |
| SNGS-200 | 4 | 241 | 0.78 | gt_tid=7 jersey=7 team=right (188fr) | right |
| SNGS-200 | 15 | 263 | 0.71 | gt_tid=6 jersey=16 team=right (187fr) | right |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.