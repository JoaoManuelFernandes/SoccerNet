# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-07-07 20:38
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-07-07/19-58-57/states/final-TB3.pklz`

**Videos**: SNGS-132, SNGS-133, SNGS-134, SNGS-135, SNGS-136
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-132
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 7 | 7 | 14 | 2 | 28 | **25.0%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        4         7          56       290    1.00
        6        33          53       288    0.88
       11        11          10       448    0.98
       13        14          65       209    1.00
       32        55           5       141    1.00
       51        15           5       130    1.00
       52        14          16       116    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 44 | 0 | 2 | 44 | **100.0%** |

#### ID Switches detected: 2
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  28  frames= 164  purity=0.52
    primary : gt_tid=13 jersey=7 team=right (85fr)
    others  : gt_tid=19 jersey=None team=right (75fr) | gt_tid=22 jersey=5 team=left (3fr)
    predicted: jersey=None  team=right
  pred_tid=  30  frames=   5  purity=0.60
    primary : gt_tid=11 jersey=26 team=right (3fr)
    others  : gt_tid=19 jersey=None team=right (1fr) | gt_tid=5 jersey=11 team=left (1fr)
    predicted: jersey=None  team=right
```

### SNGS-133
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 21 | 6 | 8 | 1 | 35 | **60.0%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        1        36          26        55    1.00
       23        55          59        53    0.96
       52        14           6       373    1.00
       58         7           5       471    1.00
       90        33           6        47    0.96
       97        10           5       178    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 47 | 0 | 2 | 47 | **100.0%** |

#### ID Switches detected: 2
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  39  frames=   6  purity=0.67
    primary : gt_tid=9 jersey=None team=left (4fr)
    others  : gt_tid=15 jersey=26 team=right (1fr) | gt_tid=10 jersey=7 team=right (1fr)
    predicted: jersey=None  team=left
  pred_tid=  89  frames=   6  purity=0.67
    primary : gt_tid=3 jersey=33 team=left (4fr)
    others  : gt_tid=8 jersey=None team=None (2fr)
    predicted: jersey=None  team=left
```

### SNGS-134
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 15 | 2 | 8 | 1 | 25 | **60.0%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        3        33           5       570     1.0
       93        25           5        87     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 44 | 0 | 1 | 44 | **100.0%** |

#### ID Switches detected: 1
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  41  frames=   7  purity=0.71
    primary : gt_tid=2 jersey=14 team=left (5fr)
    others  : gt_tid=4 jersey=14 team=right (2fr)
    predicted: jersey=None  team=left
```

### SNGS-135
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 21 | 9 | 33 | 12 | 63 | **33.3%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       32        43           3        69    0.86
       30        10          12        27    1.00
       35        26           5        40    1.00
       37        36           5       129    0.98
       41        15           5       296    0.99
       50        25          56        26    0.96
       66        36           3        48    1.00
       80        26           6       245    1.00
      105        50           6       198    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 63 | 0 | 12 | 63 | **100.0%** |

#### ID Switches detected: 12
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   2  frames= 525  purity=0.55
    primary : gt_tid=14 jersey=33 team=left (285fr)
    others  : gt_tid=11 jersey=11 team=left (232fr)
    predicted: jersey=33  team=left
  pred_tid=  10  frames= 349  purity=0.45
    primary : gt_tid=18 jersey=5 team=left (156fr)
    others  : gt_tid=17 jersey=50 team=left (156fr) | gt_tid=20 jersey=36 team=left (35fr)
    predicted: jersey=5  team=left
  pred_tid=  19  frames= 124  purity=0.45
    primary : gt_tid=12 jersey=43 team=left (54fr)
    others  : gt_tid=14 jersey=33 team=left (43fr) | gt_tid=11 jersey=11 team=left (24fr)
    predicted: jersey=None  team=left
  pred_tid=  20  frames=   6  purity=0.50
    primary : gt_tid=20 jersey=36 team=left (3fr)
    others  : gt_tid=18 jersey=5 team=left (3fr)
    predicted: jersey=None  team=left
  pred_tid=  40  frames=   5  purity=0.75
    primary : gt_tid=19 jersey=15 team=left (3fr)
    others  : gt_tid=1 jersey=7 team=right (1fr)
    predicted: jersey=None  team=left
  pred_tid=  43  frames=  35  purity=0.68
    primary : gt_tid=2 jersey=9 team=right (17fr)
    others  : gt_tid=1 jersey=7 team=right (6fr) | gt_tid=11 jersey=11 team=left (2fr)
    predicted: jersey=None  team=right
  pred_tid=  53  frames=   7  purity=0.67
    primary : gt_tid=15 jersey=14 team=left (4fr)
    others  : gt_tid=13 jersey=10 team=left (2fr)
    predicted: jersey=None  team=left
  pred_tid=  57  frames=   5  purity=0.60
    primary : gt_tid=2 jersey=9 team=right (3fr)
    others  : gt_tid=9 jersey=25 team=right (2fr)
    predicted: jersey=None  team=right
  pred_tid=  61  frames=  10  purity=0.78
    primary : gt_tid=1 jersey=7 team=right (7fr)
    others  : gt_tid=15 jersey=14 team=left (1fr) | gt_tid=11 jersey=11 team=left (1fr)
    predicted: jersey=None  team=right
  pred_tid=  73  frames= 150  purity=0.52
    primary : gt_tid=8 jersey=14 team=right (77fr)
    others  : gt_tid=14 jersey=33 team=left (68fr) | gt_tid=2 jersey=9 team=right (2fr)
    predicted: jersey=None  team=right
  pred_tid= 112  frames=  22  purity=0.77
    primary : gt_tid=11 jersey=11 team=left (17fr)
    others  : gt_tid=12 jersey=43 team=left (5fr)
    predicted: jersey=None  team=left
  pred_tid= 121  frames=  20  purity=0.75
    primary : gt_tid=12 jersey=43 team=left (15fr)
    others  : gt_tid=24 jersey=23 team=right (5fr)
    predicted: jersey=None  team=left
```

### SNGS-136
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 16 | 3 | 23 | 7 | 42 | **38.1%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       37        17          11        57    0.95
       61        15           5       139    1.00
       78         7           5        33    0.94
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 48 | 0 | 7 | 48 | **100.0%** |

#### ID Switches detected: 7
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   6  frames= 223  purity=0.56
    primary : gt_tid=11 jersey=3 team=left (124fr)
    others  : gt_tid=9 jersey=17 team=left (98fr) | gt_tid=17 jersey=50 team=left (1fr)
    predicted: jersey=11  team=left
  pred_tid=  10  frames= 111  purity=0.78
    primary : gt_tid=7 jersey=7 team=right (87fr)
    others  : gt_tid=6 jersey=None team=right (23fr) | gt_tid=2 jersey=26 team=right (1fr)
    predicted: jersey=7  team=right
  pred_tid=  11  frames= 408  purity=0.71
    primary : gt_tid=22 jersey=5 team=left (288fr)
    others  : gt_tid=5 jersey=4 team=right (95fr) | gt_tid=16 jersey=33 team=left (13fr)
    predicted: jersey=5  team=left
  pred_tid=  17  frames=  31  purity=0.61
    primary : gt_tid=21 jersey=9 team=right (19fr)
    others  : gt_tid=15 jersey=14 team=right (12fr)
    predicted: jersey=None  team=right
  pred_tid=  22  frames=   6  purity=0.60
    primary : gt_tid=23 jersey=15 team=left (3fr)
    others  : gt_tid=13 jersey=None team=None (1fr) | gt_tid=21 jersey=9 team=right (1fr)
    predicted: jersey=None  team=left
  pred_tid=  41  frames=  29  purity=0.76
    primary : gt_tid=22 jersey=5 team=left (22fr)
    others  : gt_tid=21 jersey=9 team=right (7fr)
    predicted: jersey=None  team=left
  pred_tid=  46  frames=  14  purity=0.50
    primary : gt_tid=21 jersey=9 team=right (7fr)
    others  : gt_tid=22 jersey=5 team=left (6fr) | gt_tid=16 jersey=33 team=left (1fr)
    predicted: jersey=None  team=left
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 80 | 27 | 86 | 23 | 193 | **41.5%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 246 | 0 | 24 | 246 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **24** (across 5 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-132 | 28 | 164 | 0.52 | gt_tid=13 jersey=7 team=right (85fr) | right |
| SNGS-132 | 30 | 5 | 0.6 | gt_tid=11 jersey=26 team=right (3fr) | right |
| SNGS-133 | 39 | 6 | 0.67 | gt_tid=9 jersey=None team=left (4fr) | left |
| SNGS-133 | 89 | 6 | 0.67 | gt_tid=3 jersey=33 team=left (4fr) | left |
| SNGS-134 | 41 | 7 | 0.71 | gt_tid=2 jersey=14 team=left (5fr) | left |
| SNGS-135 | 2 | 525 | 0.55 | gt_tid=14 jersey=33 team=left (285fr) | left |
| SNGS-135 | 10 | 349 | 0.45 | gt_tid=18 jersey=5 team=left (156fr) | left |
| SNGS-135 | 19 | 124 | 0.45 | gt_tid=12 jersey=43 team=left (54fr) | left |
| SNGS-135 | 20 | 6 | 0.5 | gt_tid=20 jersey=36 team=left (3fr) | left |
| SNGS-135 | 40 | 5 | 0.75 | gt_tid=19 jersey=15 team=left (3fr) | left |
| SNGS-135 | 43 | 35 | 0.68 | gt_tid=2 jersey=9 team=right (17fr) | right |
| SNGS-135 | 53 | 7 | 0.67 | gt_tid=15 jersey=14 team=left (4fr) | left |
| SNGS-135 | 57 | 5 | 0.6 | gt_tid=2 jersey=9 team=right (3fr) | right |
| SNGS-135 | 61 | 10 | 0.78 | gt_tid=1 jersey=7 team=right (7fr) | right |
| SNGS-135 | 73 | 150 | 0.52 | gt_tid=8 jersey=14 team=right (77fr) | right |
| SNGS-135 | 112 | 22 | 0.77 | gt_tid=11 jersey=11 team=left (17fr) | left |
| SNGS-135 | 121 | 20 | 0.75 | gt_tid=12 jersey=43 team=left (15fr) | left |
| SNGS-136 | 6 | 223 | 0.56 | gt_tid=11 jersey=3 team=left (124fr) | left |
| SNGS-136 | 10 | 111 | 0.78 | gt_tid=7 jersey=7 team=right (87fr) | right |
| SNGS-136 | 11 | 408 | 0.71 | gt_tid=22 jersey=5 team=left (288fr) | left |
| SNGS-136 | 17 | 31 | 0.61 | gt_tid=21 jersey=9 team=right (19fr) | right |
| SNGS-136 | 22 | 6 | 0.6 | gt_tid=23 jersey=15 team=left (3fr) | left |
| SNGS-136 | 41 | 29 | 0.76 | gt_tid=22 jersey=5 team=left (22fr) | left |
| SNGS-136 | 46 | 14 | 0.5 | gt_tid=21 jersey=9 team=right (7fr) | left |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.