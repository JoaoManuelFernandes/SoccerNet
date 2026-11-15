# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-07-04 22:28
**State file**: `/home/joao/soccernet/outputs/sn-gamestate/2026-07-02/15-08-54/states/gshota-valid-V02.pklz`

**Videos**: SNGS-024, SNGS-025, SNGS-026
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-024
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 14 | 2 | 10 | 3 | 26 | **53.8%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       10        11          42       245     1.0
       45        42           5       140     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 41 | 0 | 3 | 41 | **100.0%** |

#### ID Switches detected: 3
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  20  frames= 141  purity=0.72
    primary : gt_tid=19 jersey=28 team=right (100fr)
    others  : gt_tid=10 jersey=11 team=left (37fr) | gt_tid=21 jersey=None team=left (1fr)
    predicted: jersey=23  team=right
  pred_tid=  33  frames= 210  purity=0.49
    primary : gt_tid=13 jersey=42 team=left (103fr)
    others  : gt_tid=22 jersey=None team=right (95fr) | gt_tid=10 jersey=11 team=left (8fr)
    predicted: jersey=42  team=left
  pred_tid=  47  frames= 144  purity=0.76
    primary : gt_tid=7 jersey=3 team=left (109fr)
    others  : gt_tid=11 jersey=None team=right (34fr)
    predicted: jersey=5  team=left
```

### SNGS-025
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 17 | 5 | 20 | 9 | 42 | **40.5%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       21        34           6       284    1.00
       24        23           4        10    1.00
       25        23           5        14    0.86
       39        15           5        32    1.00
       79        23          63       161    0.89
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 27 | 17 | 9 | 44 | **61.4%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
        3   right      left       582    0.99
        7   right      left        64    0.94
        9   right      left       574    0.99
       11   right      left       750    1.00
       14   right      left       193    0.98
       19   right      left        77    1.00
       20   right      left       205    0.96
       24   right      left        10    1.00
       25   right      left        14    0.86
       52   right      left        79    0.96
```

#### ID Switches detected: 9
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   4  frames= 425  purity=0.60
    primary : gt_tid=11 jersey=17 team=right (253fr)
    others  : gt_tid=1 jersey=11 team=right (171fr)
    predicted: jersey=11  team=left
  pred_tid=   8  frames= 278  purity=0.55
    primary : gt_tid=9 jersey=11 team=left (154fr)
    others  : gt_tid=11 jersey=17 team=right (124fr)
    predicted: jersey=11  team=left
  pred_tid=  10  frames= 353  purity=0.72
    primary : gt_tid=8 jersey=5 team=left (255fr)
    others  : gt_tid=9 jersey=11 team=left (91fr) | gt_tid=7 jersey=23 team=right (3fr)
    predicted: jersey=5  team=left
  pred_tid=  18  frames= 176  purity=0.76
    primary : gt_tid=11 jersey=17 team=right (133fr)
    others  : gt_tid=15 jersey=15 team=left (42fr)
    predicted: jersey=17  team=left
  pred_tid=  22  frames=  53  purity=0.77
    primary : gt_tid=16 jersey=19 team=left (41fr)
    others  : gt_tid=15 jersey=15 team=left (12fr)
    predicted: jersey=None  team=left
  pred_tid=  30  frames=  15  purity=0.40
    primary : gt_tid=9 jersey=11 team=left (6fr)
    others  : gt_tid=14 jersey=13 team=left (5fr) | gt_tid=13 jersey=22 team=right (3fr)
    predicted: jersey=11  team=left
  pred_tid=  40  frames= 433  purity=0.56
    primary : gt_tid=16 jersey=19 team=left (243fr)
    others  : gt_tid=14 jersey=13 team=left (186fr) | gt_tid=18 jersey=34 team=left (1fr)
    predicted: jersey=19  team=left
  pred_tid=  65  frames=  23  purity=0.78
    primary : gt_tid=15 jersey=15 team=left (18fr)
    others  : gt_tid=16 jersey=19 team=left (5fr)
    predicted: jersey=None  team=left
  pred_tid=  66  frames=  83  purity=0.67
    primary : gt_tid=8 jersey=5 team=left (56fr)
    others  : gt_tid=9 jersey=11 team=left (23fr) | gt_tid=18 jersey=34 team=left (3fr)
    predicted: jersey=None  team=left
```

### SNGS-026
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 15 | 6 | 13 | 1 | 34 | **44.1%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        1        23           5       215    0.99
        4         5          65       211    0.94
        7        19           6       184    1.00
       17        14          19       195    1.00
       25        13           5        79    1.00
       48        14           1       258    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 22 | 17 | 1 | 39 | **56.4%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
        3    left     right       171    1.00
        4    left     right       211    0.94
        6    left     right       267    1.00
        7    left     right       184    1.00
        8    left     right       268    1.00
        9    left     right       411    1.00
       16    left     right        52    1.00
       19    left     right       154    1.00
       23    left     right        89    1.00
       24    left     right       562    1.00
```

#### ID Switches detected: 1
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  31  frames=   5  purity=0.40
    primary : gt_tid=14 jersey=14 team=right (2fr)
    others  : gt_tid=20 jersey=11 team=left (2fr) | gt_tid=9 jersey=23 team=right (1fr)
    predicted: jersey=None  team=right
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 46 | 13 | 43 | 13 | 102 | **45.1%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 90 | 34 | 13 | 124 | **72.6%** |

### ID Switches Summary
Total ID switches detected: **13** (across 3 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-024 | 20 | 141 | 0.72 | gt_tid=19 jersey=28 team=right (100fr) | right |
| SNGS-024 | 33 | 210 | 0.49 | gt_tid=13 jersey=42 team=left (103fr) | left |
| SNGS-024 | 47 | 144 | 0.76 | gt_tid=7 jersey=3 team=left (109fr) | left |
| SNGS-025 | 4 | 425 | 0.6 | gt_tid=11 jersey=17 team=right (253fr) | left |
| SNGS-025 | 8 | 278 | 0.55 | gt_tid=9 jersey=11 team=left (154fr) | left |
| SNGS-025 | 10 | 353 | 0.72 | gt_tid=8 jersey=5 team=left (255fr) | left |
| SNGS-025 | 18 | 176 | 0.76 | gt_tid=11 jersey=17 team=right (133fr) | left |
| SNGS-025 | 22 | 53 | 0.77 | gt_tid=16 jersey=19 team=left (41fr) | left |
| SNGS-025 | 30 | 15 | 0.4 | gt_tid=9 jersey=11 team=left (6fr) | left |
| SNGS-025 | 40 | 433 | 0.56 | gt_tid=16 jersey=19 team=left (243fr) | left |
| SNGS-025 | 65 | 23 | 0.78 | gt_tid=15 jersey=15 team=left (18fr) | left |
| SNGS-025 | 66 | 83 | 0.67 | gt_tid=8 jersey=5 team=left (56fr) | left |
| SNGS-026 | 31 | 5 | 0.4 | gt_tid=14 jersey=14 team=right (2fr) | right |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.