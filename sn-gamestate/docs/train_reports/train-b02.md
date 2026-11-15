# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-07 21:40
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-09-07/20-29-44/states/train-b02.pklz`

**Videos**: SNGS-063, SNGS-064, SNGS-065
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-063
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 10 | 5 | 26 | 9 | 41 | **24.4%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        4         8           6       184    1.00
       11         4           5       161    0.99
       14        30           1       449    0.82
       30        14          46       370    1.00
       43        17          11       294    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 50 | 0 | 9 | 50 | **100.0%** |

#### ID Switches detected: 9
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   6  frames= 284  purity=0.56
    primary : gt_tid=2 jersey=24 team=left (158fr)
    others  : gt_tid=6 jersey=4 team=left (125fr)
    predicted: jersey=4  team=left
  pred_tid=  10  frames= 129  purity=0.53
    primary : gt_tid=15 jersey=44 team=right (69fr)
    others  : gt_tid=5 jersey=None team=left (60fr)
    predicted: jersey=nan  team=right
  pred_tid=  15  frames= 234  purity=0.69
    primary : gt_tid=11 jersey=25 team=left (161fr)
    others  : gt_tid=16 jersey=36 team=right (73fr)
    predicted: jersey=nan  team=left
  pred_tid=  19  frames=  57  purity=0.61
    primary : gt_tid=15 jersey=44 team=right (35fr)
    others  : gt_tid=12 jersey=30 team=left (18fr) | gt_tid=13 jersey=10 team=right (4fr)
    predicted: jersey=nan  team=right
  pred_tid=  21  frames=  13  purity=0.69
    primary : gt_tid=12 jersey=30 team=left (9fr)
    others  : gt_tid=15 jersey=44 team=right (4fr)
    predicted: jersey=nan  team=left
  pred_tid=  36  frames= 362  purity=0.70
    primary : gt_tid=22 jersey=15 team=right (253fr)
    others  : gt_tid=20 jersey=8 team=right (107fr) | gt_tid=12 jersey=30 team=left (1fr)
    predicted: jersey=8  team=right
  pred_tid=  38  frames= 379  purity=0.55
    primary : gt_tid=23 jersey=5 team=right (207fr)
    others  : gt_tid=12 jersey=30 team=left (159fr) | gt_tid=20 jersey=8 team=right (8fr)
    predicted: jersey=5  team=right
  pred_tid=  42  frames= 394  purity=0.79
    primary : gt_tid=16 jersey=36 team=right (311fr)
    others  : gt_tid=17 jersey=14 team=right (83fr)
    predicted: jersey=nan  team=right
  pred_tid=  59  frames=   5  purity=0.75
    primary : gt_tid=22 jersey=15 team=right (3fr)
    others  : gt_tid=17 jersey=14 team=right (1fr)
    predicted: jersey=nan  team=right
```

### SNGS-064
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 10 | 7 | 19 | 12 | 36 | **27.8%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        7         8           6       418    1.00
       10        28          56       419    1.00
       11         4           9       238    1.00
       19        17          11       411    0.95
       28        27          77        23    1.00
       35        14          16        86    1.00
       40        15           6        26    0.96
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 40 | 3 | 12 | 43 | **93.0%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
       28   right      left        23    1.00
       32   right      left         5    1.00
       40   right      left        26    0.96
```

#### ID Switches detected: 12
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   1  frames= 272  purity=0.40
    primary : gt_tid=2 jersey=4 team=left (109fr)
    others  : gt_tid=19 jersey=None team=right (93fr) | gt_tid=11 jersey=5 team=right (50fr)
    predicted: jersey=5  team=right
  pred_tid=   2  frames= 335  purity=0.54
    primary : gt_tid=4 jersey=8 team=left (180fr)
    others  : gt_tid=6 jersey=10 team=left (129fr) | gt_tid=18 jersey=27 team=right (19fr)
    predicted: jersey=8  team=left
  pred_tid=   8  frames= 359  purity=0.68
    primary : gt_tid=1 jersey=24 team=left (245fr)
    others  : gt_tid=11 jersey=5 team=right (114fr)
    predicted: jersey=nan  team=left
  pred_tid=  16  frames= 640  purity=0.68
    primary : gt_tid=7 jersey=25 team=left (434fr)
    others  : gt_tid=9 jersey=30 team=left (187fr) | gt_tid=18 jersey=27 team=right (9fr)
    predicted: jersey=25  team=left
  pred_tid=  18  frames= 534  purity=0.72
    primary : gt_tid=12 jersey=44 team=right (381fr)
    others  : gt_tid=18 jersey=27 team=right (146fr) | gt_tid=7 jersey=25 team=left (3fr)
    predicted: jersey=44  team=right
  pred_tid=  26  frames= 619  purity=0.59
    primary : gt_tid=1 jersey=24 team=left (365fr)
    others  : gt_tid=11 jersey=5 team=right (253fr)
    predicted: jersey=5  team=left
  pred_tid=  37  frames= 348  purity=0.66
    primary : gt_tid=6 jersey=10 team=left (225fr)
    others  : gt_tid=18 jersey=27 team=right (113fr) | gt_tid=7 jersey=25 team=left (1fr)
    predicted: jersey=27  team=left
  pred_tid=  45  frames=  16  purity=0.62
    primary : gt_tid=2 jersey=4 team=left (10fr)
    others  : gt_tid=5 jersey=17 team=left (2fr) | gt_tid=6 jersey=10 team=left (2fr)
    predicted: jersey=nan  team=left
  pred_tid=  50  frames=  15  purity=0.53
    primary : gt_tid=13 jersey=14 team=right (8fr)
    others  : gt_tid=4 jersey=8 team=left (4fr) | gt_tid=7 jersey=25 team=left (2fr)
    predicted: jersey=nan  team=left
  pred_tid=  52  frames=  20  purity=0.70
    primary : gt_tid=18 jersey=27 team=right (14fr)
    others  : gt_tid=2 jersey=4 team=left (4fr) | gt_tid=3 jersey=3 team=left (2fr)
    predicted: jersey=nan  team=right
  pred_tid=  59  frames=  13  purity=0.69
    primary : gt_tid=15 jersey=15 team=right (9fr)
    others  : gt_tid=10 jersey=None team=left (4fr)
    predicted: jersey=nan  team=right
  pred_tid=  63  frames=  41  purity=0.71
    primary : gt_tid=7 jersey=25 team=left (29fr)
    others  : gt_tid=3 jersey=3 team=left (9fr) | gt_tid=4 jersey=8 team=left (3fr)
    predicted: jersey=nan  team=left
```

### SNGS-065
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 13 | 6 | 16 | 2 | 35 | **37.1%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        5         4           5       255    0.99
       10        17          15       341    1.00
       23        15          30       454    0.85
       44        10          16       139    0.99
       37        24           5       210    1.00
       56         4          54       163    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 42 | 0 | 3 | 42 | **100.0%** |

#### ID Switches detected: 3
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  21  frames= 225  purity=0.71
    primary : gt_tid=4 jersey=None team=left (160fr)
    others  : gt_tid=16 jersey=14 team=right (65fr)
    predicted: jersey=nan  team=left
  pred_tid=  30  frames= 395  purity=0.64
    primary : gt_tid=16 jersey=14 team=right (246fr)
    others  : gt_tid=14 jersey=8 team=right (140fr) | gt_tid=12 jersey=5 team=right (1fr)
    predicted: jersey=14  team=right
  pred_tid=  33  frames= 238  purity=0.63
    primary : gt_tid=12 jersey=5 team=right (150fr)
    others  : gt_tid=16 jersey=14 team=right (87fr) | gt_tid=4 jersey=None team=left (1fr)
    predicted: jersey=nan  team=right
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 33 | 18 | 61 | 23 | 112 | **29.5%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 132 | 3 | 24 | 135 | **97.8%** |

### ID Switches Summary
Total ID switches detected: **24** (across 3 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-063 | 6 | 284 | 0.56 | gt_tid=2 jersey=24 team=left (158fr) | left |
| SNGS-063 | 10 | 129 | 0.53 | gt_tid=15 jersey=44 team=right (69fr) | right |
| SNGS-063 | 15 | 234 | 0.69 | gt_tid=11 jersey=25 team=left (161fr) | left |
| SNGS-063 | 19 | 57 | 0.61 | gt_tid=15 jersey=44 team=right (35fr) | right |
| SNGS-063 | 21 | 13 | 0.69 | gt_tid=12 jersey=30 team=left (9fr) | left |
| SNGS-063 | 36 | 362 | 0.7 | gt_tid=22 jersey=15 team=right (253fr) | right |
| SNGS-063 | 38 | 379 | 0.55 | gt_tid=23 jersey=5 team=right (207fr) | right |
| SNGS-063 | 42 | 394 | 0.79 | gt_tid=16 jersey=36 team=right (311fr) | right |
| SNGS-063 | 59 | 5 | 0.75 | gt_tid=22 jersey=15 team=right (3fr) | right |
| SNGS-064 | 1 | 272 | 0.4 | gt_tid=2 jersey=4 team=left (109fr) | right |
| SNGS-064 | 2 | 335 | 0.54 | gt_tid=4 jersey=8 team=left (180fr) | left |
| SNGS-064 | 8 | 359 | 0.68 | gt_tid=1 jersey=24 team=left (245fr) | left |
| SNGS-064 | 16 | 640 | 0.68 | gt_tid=7 jersey=25 team=left (434fr) | left |
| SNGS-064 | 18 | 534 | 0.72 | gt_tid=12 jersey=44 team=right (381fr) | right |
| SNGS-064 | 26 | 619 | 0.59 | gt_tid=1 jersey=24 team=left (365fr) | left |
| SNGS-064 | 37 | 348 | 0.66 | gt_tid=6 jersey=10 team=left (225fr) | left |
| SNGS-064 | 45 | 16 | 0.62 | gt_tid=2 jersey=4 team=left (10fr) | left |
| SNGS-064 | 50 | 15 | 0.53 | gt_tid=13 jersey=14 team=right (8fr) | left |
| SNGS-064 | 52 | 20 | 0.7 | gt_tid=18 jersey=27 team=right (14fr) | right |
| SNGS-064 | 59 | 13 | 0.69 | gt_tid=15 jersey=15 team=right (9fr) | right |
| SNGS-064 | 63 | 41 | 0.71 | gt_tid=7 jersey=25 team=left (29fr) | left |
| SNGS-065 | 21 | 225 | 0.71 | gt_tid=4 jersey=None team=left (160fr) | left |
| SNGS-065 | 30 | 395 | 0.64 | gt_tid=16 jersey=14 team=right (246fr) | right |
| SNGS-065 | 33 | 238 | 0.63 | gt_tid=12 jersey=5 team=right (150fr) | right |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.