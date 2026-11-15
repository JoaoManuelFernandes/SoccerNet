# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-07-08 08:18
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-07-08/07-43-42/states/final-TB5.pklz`

**Videos**: SNGS-142, SNGS-145, SNGS-146, SNGS-147, SNGS-148
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-142
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 15 | 6 | 33 | 17 | 54 | **27.8%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       27        93          83       207    0.90
       47        33          53       155    0.93
       53        14          36        45    0.84
       61         7           5       129    0.94
       78        20           5       113    0.93
      121        14          12       104    0.99
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 54 | 0 | 17 | 54 | **100.0%** |

#### ID Switches detected: 18
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   7  frames= 100  purity=0.39
    primary : gt_tid=16 jersey=25 team=left (39fr)
    others  : gt_tid=4 jersey=14 team=right (37fr) | gt_tid=17 jersey=55 team=left (21fr)
    predicted: jersey=None  team=left
  pred_tid=  14  frames= 122  purity=0.40
    primary : gt_tid=4 jersey=14 team=right (48fr)
    others  : gt_tid=7 jersey=5 team=right (41fr) | gt_tid=8 jersey=15 team=right (25fr)
    predicted: jersey=33  team=right
  pred_tid=  15  frames=   7  purity=0.71
    primary : gt_tid=16 jersey=25 team=left (5fr)
    others  : gt_tid=20 jersey=14 team=left (1fr) | gt_tid=18 jersey=23 team=left (1fr)
    predicted: jersey=None  team=left
  pred_tid=  25  frames= 115  purity=0.40
    primary : gt_tid=11 jersey=7 team=left (46fr)
    others  : gt_tid=17 jersey=55 team=left (36fr) | gt_tid=18 jersey=23 team=left (23fr)
    predicted: jersey=23  team=left
  pred_tid=  26  frames= 228  purity=0.57
    primary : gt_tid=11 jersey=7 team=left (130fr)
    others  : gt_tid=18 jersey=23 team=left (94fr) | gt_tid=6 jersey=36 team=right (3fr)
    predicted: jersey=7  team=left
  pred_tid=  29  frames= 143  purity=0.80
    primary : gt_tid=15 jersey=10 team=left (114fr)
    others  : gt_tid=16 jersey=25 team=left (11fr) | gt_tid=12 jersey=20 team=left (10fr)
    predicted: jersey=10  team=left
  pred_tid=  35  frames=  54  purity=0.52
    primary : gt_tid=15 jersey=10 team=left (28fr)
    others  : gt_tid=16 jersey=25 team=left (19fr) | gt_tid=12 jersey=20 team=left (5fr)
    predicted: jersey=None  team=left
  pred_tid=  39  frames= 234  purity=0.60
    primary : gt_tid=8 jersey=15 team=right (140fr)
    others  : gt_tid=12 jersey=20 team=left (88fr) | gt_tid=15 jersey=10 team=left (3fr)
    predicted: jersey=20  team=right
  pred_tid=  49  frames=   5  purity=0.40
    primary : gt_tid=6 jersey=36 team=right (2fr)
    others  : gt_tid=4 jersey=14 team=right (2fr) | gt_tid=23 jersey=93 team=right (1fr)
    predicted: jersey=None  team=right
  pred_tid=  50  frames= 222  purity=0.37
    primary : gt_tid=16 jersey=25 team=left (81fr)
    others  : gt_tid=15 jersey=10 team=left (81fr) | gt_tid=17 jersey=55 team=left (51fr)
    predicted: jersey=25  team=left
  pred_tid=  67  frames=   5  purity=0.60
    primary : gt_tid=15 jersey=10 team=left (3fr)
    others  : gt_tid=20 jersey=14 team=left (2fr)
    predicted: jersey=None  team=left
  pred_tid=  69  frames=  21  purity=0.71
    primary : gt_tid=8 jersey=15 team=right (15fr)
    others  : gt_tid=3 jersey=None team=None (2fr) | gt_tid=4 jersey=14 team=right (1fr)
    predicted: jersey=5  team=right
  pred_tid=  71  frames=   8  purity=0.67
    primary : gt_tid=18 jersey=23 team=left (2fr)
    others  : gt_tid=20 jersey=14 team=left (1fr)
    predicted: jersey=None  team=left
  pred_tid=  72  frames=  82  purity=0.62
    primary : gt_tid=16 jersey=25 team=left (51fr)
    others  : gt_tid=17 jersey=55 team=left (13fr) | gt_tid=15 jersey=10 team=left (13fr)
    predicted: jersey=None  team=left
  pred_tid=  73  frames=  10  purity=0.57
    primary : gt_tid=15 jersey=10 team=left (4fr)
    others  : gt_tid=11 jersey=7 team=left (3fr)
    predicted: jersey=None  team=left
  pred_tid=  74  frames=   9  purity=0.44
    primary : gt_tid=4 jersey=14 team=right (4fr)
    others  : gt_tid=15 jersey=10 team=left (3fr) | gt_tid=8 jersey=15 team=right (1fr)
    predicted: jersey=None  team=right
  pred_tid=  81  frames=   8  purity=0.62
    primary : gt_tid=18 jersey=23 team=left (5fr)
    others  : gt_tid=17 jersey=55 team=left (2fr) | gt_tid=5 jersey=33 team=right (1fr)
    predicted: jersey=None  team=left
  pred_tid= 100  frames=  10  purity=0.78
    primary : gt_tid=13 jersey=26 team=left (7fr)
    others  : gt_tid=11 jersey=7 team=left (2fr)
    predicted: jersey=None  team=left
```

### SNGS-145
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 13 | 3 | 42 | 10 | 58 | **22.4%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       16        15           5       284    0.83
      141        25           5       109    1.00
      142        23           5        79    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 59 | 0 | 10 | 59 | **100.0%** |

#### ID Switches detected: 11
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   9  frames= 200  purity=0.72
    primary : gt_tid=17 jersey=7 team=left (144fr)
    others  : gt_tid=14 jersey=25 team=left (38fr) | gt_tid=13 jersey=55 team=left (18fr)
    predicted: jersey=25  team=left
  pred_tid=  10  frames= 102  purity=0.54
    primary : gt_tid=11 jersey=10 team=left (55fr)
    others  : gt_tid=12 jersey=9 team=left (46fr) | gt_tid=13 jersey=55 team=left (1fr)
    predicted: jersey=9  team=left
  pred_tid=  29  frames=  95  purity=0.74
    primary : gt_tid=9 jersey=16 team=left (70fr)
    others  : gt_tid=13 jersey=55 team=left (19fr) | gt_tid=11 jersey=10 team=left (5fr)
    predicted: jersey=None  team=left
  pred_tid=  35  frames=  70  purity=0.53
    primary : gt_tid=13 jersey=55 team=left (37fr)
    others  : gt_tid=17 jersey=7 team=left (18fr) | gt_tid=14 jersey=25 team=left (12fr)
    predicted: jersey=None  team=left
  pred_tid=  40  frames=   9  purity=0.67
    primary : gt_tid=13 jersey=55 team=left (6fr)
    others  : gt_tid=17 jersey=7 team=left (3fr)
    predicted: jersey=None  team=left
  pred_tid=  42  frames=  25  purity=0.42
    primary : gt_tid=11 jersey=10 team=left (10fr)
    others  : gt_tid=5 jersey=33 team=right (7fr) | gt_tid=13 jersey=55 team=left (5fr)
    predicted: jersey=None  team=left
  pred_tid=  45  frames=  29  purity=0.71
    primary : gt_tid=5 jersey=33 team=right (20fr)
    others  : gt_tid=25 jersey=None team=right (4fr) | gt_tid=17 jersey=7 team=left (2fr)
    predicted: jersey=None  team=right
  pred_tid=  58  frames=   6  purity=0.50
    primary : gt_tid=8 jersey=26 team=left (3fr)
    others  : gt_tid=13 jersey=55 team=left (1fr) | gt_tid=17 jersey=7 team=left (1fr)
    predicted: jersey=None  team=left
  pred_tid= 110  frames=   5  purity=0.60
    primary : gt_tid=25 jersey=None team=right (3fr)
    others  : gt_tid=7 jersey=15 team=right (2fr)
    predicted: jersey=None  team=right
  pred_tid= 121  frames=   9  purity=0.78
    primary : gt_tid=16 jersey=23 team=left (7fr)
    others  : gt_tid=23 jersey=50 team=right (2fr)
    predicted: jersey=29  team=left
  pred_tid= 160  frames=   7  purity=0.43
    primary : gt_tid=14 jersey=25 team=left (3fr)
    others  : gt_tid=11 jersey=10 team=left (2fr) | gt_tid=9 jersey=16 team=left (2fr)
    predicted: jersey=None  team=left
```

### SNGS-146
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 14 | 1 | 4 | 6 | 19 | **73.7%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        9        26          67       201     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 27 | 0 | 6 | 27 | **100.0%** |

#### ID Switches detected: 6
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   5  frames= 184  purity=0.59
    primary : gt_tid=12 jersey=7 team=left (108fr)
    others  : gt_tid=1 jersey=14 team=right (76fr)
    predicted: jersey=7  team=left
  pred_tid=   6  frames= 300  purity=0.75
    primary : gt_tid=18 jersey=25 team=left (225fr)
    others  : gt_tid=13 jersey=None team=left (75fr)
    predicted: jersey=25  team=left
  pred_tid=  19  frames=   7  purity=0.71
    primary : gt_tid=1 jersey=14 team=right (5fr)
    others  : gt_tid=12 jersey=7 team=left (2fr)
    predicted: jersey=None  team=right
  pred_tid=  42  frames= 243  purity=0.60
    primary : gt_tid=16 jersey=20 team=left (141fr)
    others  : gt_tid=11 jersey=26 team=left (95fr)
    predicted: jersey=26  team=left
  pred_tid=  52  frames=  12  purity=0.50
    primary : gt_tid=1 jersey=14 team=right (6fr)
    others  : gt_tid=7 jersey=4 team=right (3fr) | gt_tid=18 jersey=25 team=left (2fr)
    predicted: jersey=None  team=right
  pred_tid=  53  frames=  18  purity=0.72
    primary : gt_tid=11 jersey=26 team=left (13fr)
    others  : gt_tid=19 jersey=55 team=left (5fr)
    predicted: jersey=None  team=left
```

### SNGS-147
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 11 | 4 | 9 | 1 | 24 | **45.8%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        3        44           4        75    1.00
       75        20          44       408    0.88
      127        14          44        42    1.00
       84         4           5        91    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 51 | 0 | 2 | 51 | **100.0%** |

#### ID Switches detected: 2
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  18  frames=   6  purity=0.67
    primary : gt_tid=12 jersey=27 team=left (4fr)
    others  : gt_tid=9 jersey=None team=left (2fr)
    predicted: jersey=None  team=left
  pred_tid=  92  frames=   7  purity=0.71
    primary : gt_tid=9 jersey=None team=left (5fr)
    others  : gt_tid=4 jersey=44 team=right (2fr)
    predicted: jersey=None  team=left
```

### SNGS-148
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 25 | 4 | 15 | 0 | 44 | **56.8%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        3         9          44        63     1.0
        5        14          44        47     1.0
       34         9          44        40     1.0
       52         9          99       114     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 48 | 0 | 0 | 48 | **100.0%** |

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 78 | 18 | 103 | 34 | 199 | **39.2%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 239 | 0 | 35 | 239 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **37** (across 4 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-142 | 7 | 100 | 0.39 | gt_tid=16 jersey=25 team=left (39fr) | left |
| SNGS-142 | 14 | 122 | 0.4 | gt_tid=4 jersey=14 team=right (48fr) | right |
| SNGS-142 | 15 | 7 | 0.71 | gt_tid=16 jersey=25 team=left (5fr) | left |
| SNGS-142 | 25 | 115 | 0.4 | gt_tid=11 jersey=7 team=left (46fr) | left |
| SNGS-142 | 26 | 228 | 0.57 | gt_tid=11 jersey=7 team=left (130fr) | left |
| SNGS-142 | 29 | 143 | 0.8 | gt_tid=15 jersey=10 team=left (114fr) | left |
| SNGS-142 | 35 | 54 | 0.52 | gt_tid=15 jersey=10 team=left (28fr) | left |
| SNGS-142 | 39 | 234 | 0.6 | gt_tid=8 jersey=15 team=right (140fr) | right |
| SNGS-142 | 49 | 5 | 0.4 | gt_tid=6 jersey=36 team=right (2fr) | right |
| SNGS-142 | 50 | 222 | 0.37 | gt_tid=16 jersey=25 team=left (81fr) | left |
| SNGS-142 | 67 | 5 | 0.6 | gt_tid=15 jersey=10 team=left (3fr) | left |
| SNGS-142 | 69 | 21 | 0.71 | gt_tid=8 jersey=15 team=right (15fr) | right |
| SNGS-142 | 71 | 8 | 0.67 | gt_tid=18 jersey=23 team=left (2fr) | left |
| SNGS-142 | 72 | 82 | 0.62 | gt_tid=16 jersey=25 team=left (51fr) | left |
| SNGS-142 | 73 | 10 | 0.57 | gt_tid=15 jersey=10 team=left (4fr) | left |
| SNGS-142 | 74 | 9 | 0.44 | gt_tid=4 jersey=14 team=right (4fr) | right |
| SNGS-142 | 81 | 8 | 0.62 | gt_tid=18 jersey=23 team=left (5fr) | left |
| SNGS-142 | 100 | 10 | 0.78 | gt_tid=13 jersey=26 team=left (7fr) | left |
| SNGS-145 | 9 | 200 | 0.72 | gt_tid=17 jersey=7 team=left (144fr) | left |
| SNGS-145 | 10 | 102 | 0.54 | gt_tid=11 jersey=10 team=left (55fr) | left |
| SNGS-145 | 29 | 95 | 0.74 | gt_tid=9 jersey=16 team=left (70fr) | left |
| SNGS-145 | 35 | 70 | 0.53 | gt_tid=13 jersey=55 team=left (37fr) | left |
| SNGS-145 | 40 | 9 | 0.67 | gt_tid=13 jersey=55 team=left (6fr) | left |
| SNGS-145 | 42 | 25 | 0.42 | gt_tid=11 jersey=10 team=left (10fr) | left |
| SNGS-145 | 45 | 29 | 0.71 | gt_tid=5 jersey=33 team=right (20fr) | right |
| SNGS-145 | 58 | 6 | 0.5 | gt_tid=8 jersey=26 team=left (3fr) | left |
| SNGS-145 | 110 | 5 | 0.6 | gt_tid=25 jersey=None team=right (3fr) | right |
| SNGS-145 | 121 | 9 | 0.78 | gt_tid=16 jersey=23 team=left (7fr) | left |
| SNGS-145 | 160 | 7 | 0.43 | gt_tid=14 jersey=25 team=left (3fr) | left |
| SNGS-146 | 5 | 184 | 0.59 | gt_tid=12 jersey=7 team=left (108fr) | left |
| SNGS-146 | 6 | 300 | 0.75 | gt_tid=18 jersey=25 team=left (225fr) | left |
| SNGS-146 | 19 | 7 | 0.71 | gt_tid=1 jersey=14 team=right (5fr) | right |
| SNGS-146 | 42 | 243 | 0.6 | gt_tid=16 jersey=20 team=left (141fr) | left |
| SNGS-146 | 52 | 12 | 0.5 | gt_tid=1 jersey=14 team=right (6fr) | right |
| SNGS-146 | 53 | 18 | 0.72 | gt_tid=11 jersey=26 team=left (13fr) | left |
| SNGS-147 | 18 | 6 | 0.67 | gt_tid=12 jersey=27 team=left (4fr) | left |
| SNGS-147 | 92 | 7 | 0.71 | gt_tid=9 jersey=None team=left (5fr) | left |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.