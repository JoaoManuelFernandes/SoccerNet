# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-07-07 19:15
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-07-07/18-32-53/states/final-TB1.pklz`

**Videos**: SNGS-118, SNGS-119, SNGS-120, SNGS-121, SNGS-122
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-118
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 16 | 3 | 8 | 0 | 27 | **59.3%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        5        11           5        83     1.0
        6        25           5       118     1.0
       29        10          15        30     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 40 | 0 | 0 | 40 | **100.0%** |

### SNGS-119
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 16 | 1 | 6 | 1 | 23 | **69.6%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       41        69          59       295     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 26 | 0 | 1 | 26 | **100.0%** |

#### ID Switches detected: 1
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   7  frames=  13  purity=0.55
    primary : gt_tid=11 jersey=4 team=left (6fr)
    others  : gt_tid=8 jersey=None team=left (3fr) | gt_tid=9 jersey=44 team=left (2fr)
    predicted: jersey=None  team=left
```

### SNGS-120
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 17 | 6 | 16 | 3 | 39 | **43.6%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        2        25           5       173    1.00
        5        93           5       134    0.99
       22         4          19       205    1.00
       48        25           5       115    1.00
       76        20           6       131    1.00
       78         4           5       107    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 48 | 0 | 4 | 48 | **100.0%** |

#### ID Switches detected: 4
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  18  frames= 429  purity=0.72
    primary : gt_tid=9 jersey=None team=right (307fr)
    others  : gt_tid=19 jersey=8 team=right (116fr) | gt_tid=3 jersey=34 team=right (4fr)
    predicted: jersey=5  team=right
  pred_tid=  49  frames= 127  purity=0.64
    primary : gt_tid=6 jersey=14 team=left (81fr)
    others  : gt_tid=27 jersey=20 team=right (45fr)
    predicted: jersey=6  team=left
  pred_tid=  58  frames= 237  purity=0.52
    primary : gt_tid=17 jersey=4 team=left (123fr)
    others  : gt_tid=6 jersey=14 team=left (109fr) | gt_tid=5 jersey=25 team=right (3fr)
    predicted: jersey=14  team=left
  pred_tid=  70  frames= 220  purity=0.56
    primary : gt_tid=5 jersey=25 team=right (124fr)
    others  : gt_tid=12 jersey=10 team=right (53fr) | gt_tid=6 jersey=14 team=left (43fr)
    predicted: jersey=14  team=right
```

### SNGS-121
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 17 | 8 | 12 | 2 | 37 | **45.9%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        1        33          73       143    1.00
        8        34           5       383    1.00
       13        60           5       342    1.00
       14        33          37       266    1.00
       15        14           6       158    0.95
       30        33           5       117    1.00
       31        31           5       321    1.00
       46        14           6       321    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 38 | 0 | 2 | 38 | **100.0%** |

#### ID Switches detected: 2
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   3  frames= 620  purity=0.75
    primary : gt_tid=2 jersey=8 team=left (468fr)
    others  : gt_tid=7 jersey=4 team=left (148fr) | gt_tid=8 jersey=25 team=right (4fr)
    predicted: jersey=8  team=left
  pred_tid=  33  frames= 282  purity=0.65
    primary : gt_tid=3 jersey=9 team=right (182fr)
    others  : gt_tid=2 jersey=8 team=left (96fr)
    predicted: jersey=5  team=right
```

### SNGS-122
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 11 | 6 | 12 | 1 | 29 | **37.9%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        5        10           5       244    1.00
       18         4           5       465    1.00
       67         9           5       113    0.92
       75        59           1       269    1.00
       79        10           1        43    1.00
       89        36          15        57    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 58 | 0 | 4 | 58 | **100.0%** |

#### ID Switches detected: 4
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  11  frames= 101  purity=0.55
    primary : gt_tid=6 jersey=None team=right (56fr)
    others  : gt_tid=9 jersey=None team=right (24fr) | gt_tid=8 jersey=31 team=right (18fr)
    predicted: jersey=5  team=right
  pred_tid=  23  frames=  13  purity=0.67
    primary : gt_tid=9 jersey=None team=right (8fr)
    others  : gt_tid=4 jersey=None team=right (4fr)
    predicted: jersey=None  team=right
  pred_tid=  28  frames=  24  purity=0.67
    primary : gt_tid=10 jersey=None team=right (16fr)
    others  : gt_tid=8 jersey=31 team=right (3fr) | gt_tid=4 jersey=None team=right (3fr)
    predicted: jersey=None  team=right
  pred_tid=  66  frames=  81  purity=0.66
    primary : gt_tid=2 jersey=10 team=right (53fr)
    others  : gt_tid=3 jersey=36 team=right (27fr)
    predicted: jersey=15  team=right
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 77 | 24 | 54 | 7 | 155 | **49.7%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 210 | 0 | 11 | 210 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **11** (across 4 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-119 | 7 | 13 | 0.55 | gt_tid=11 jersey=4 team=left (6fr) | left |
| SNGS-120 | 18 | 429 | 0.72 | gt_tid=9 jersey=None team=right (307fr) | right |
| SNGS-120 | 49 | 127 | 0.64 | gt_tid=6 jersey=14 team=left (81fr) | left |
| SNGS-120 | 58 | 237 | 0.52 | gt_tid=17 jersey=4 team=left (123fr) | left |
| SNGS-120 | 70 | 220 | 0.56 | gt_tid=5 jersey=25 team=right (124fr) | right |
| SNGS-121 | 3 | 620 | 0.75 | gt_tid=2 jersey=8 team=left (468fr) | left |
| SNGS-121 | 33 | 282 | 0.65 | gt_tid=3 jersey=9 team=right (182fr) | right |
| SNGS-122 | 11 | 101 | 0.55 | gt_tid=6 jersey=None team=right (56fr) | right |
| SNGS-122 | 23 | 13 | 0.67 | gt_tid=9 jersey=None team=right (8fr) | right |
| SNGS-122 | 28 | 24 | 0.67 | gt_tid=10 jersey=None team=right (16fr) | right |
| SNGS-122 | 66 | 81 | 0.66 | gt_tid=2 jersey=10 team=right (53fr) | right |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.