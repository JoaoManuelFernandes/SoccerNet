# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-07 07:41
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-09-05/20-47-53/states/train-batch09.pklz`

**Videos**: SNGS-103, SNGS-104, SNGS-105
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-103
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 22 | 4 | 30 | 8 | 56 | **39.3%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       22         6           5       121    0.99
       33         6           5        45    1.00
       44        20           5       170    0.89
       62        31          33        63    0.98
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 59 | 0 | 8 | 59 | **100.0%** |

#### ID Switches detected: 8
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   4  frames= 140  purity=0.65
    primary : gt_tid=4 jersey=4 team=right (91fr)
    others  : gt_tid=1 jersey=14 team=right (45fr) | gt_tid=15 jersey=14 team=left (4fr)
    predicted: jersey=4  team=right
  pred_tid=  13  frames=  73  purity=0.38
    primary : gt_tid=15 jersey=14 team=left (28fr)
    others  : gt_tid=3 jersey=20 team=right (24fr) | gt_tid=1 jersey=14 team=right (14fr)
    predicted: jersey=20  team=right
  pred_tid=  18  frames= 330  purity=0.51
    primary : gt_tid=21 jersey=32 team=left (169fr)
    others  : gt_tid=23 jersey=36 team=left (134fr) | gt_tid=19 jersey=6 team=left (27fr)
    predicted: jersey=36  team=left
  pred_tid=  19  frames= 451  purity=0.63
    primary : gt_tid=13 jersey=24 team=left (283fr)
    others  : gt_tid=12 jersey=33 team=left (165fr) | gt_tid=16 jersey=7 team=left (2fr)
    predicted: jersey=24  team=left
  pred_tid=  26  frames=  11  purity=0.73
    primary : gt_tid=21 jersey=32 team=left (8fr)
    others  : gt_tid=17 jersey=62 team=left (3fr)
    predicted: jersey=nan  team=left
  pred_tid=  34  frames= 192  purity=0.69
    primary : gt_tid=23 jersey=36 team=left (132fr)
    others  : gt_tid=21 jersey=32 team=left (60fr)
    predicted: jersey=nan  team=left
  pred_tid=  51  frames=   6  purity=0.67
    primary : gt_tid=3 jersey=20 team=right (2fr)
    others  : gt_tid=1 jersey=14 team=right (1fr)
    predicted: jersey=nan  team=right
  pred_tid=  78  frames=  41  purity=0.76
    primary : gt_tid=19 jersey=6 team=left (31fr)
    others  : gt_tid=23 jersey=36 team=left (10fr)
    predicted: jersey=nan  team=left
```

### SNGS-104
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 14 | 3 | 22 | 3 | 39 | **35.9%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       30        31           4       495    0.81
       43        36           5        75    1.00
       47        10           5        97    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 41 | 0 | 3 | 41 | **100.0%** |

#### ID Switches detected: 3
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  32  frames= 335  purity=0.66
    primary : gt_tid=18 jersey=7 team=left (222fr)
    others  : gt_tid=20 jersey=9 team=left (97fr) | gt_tid=9 jersey=14 team=right (16fr)
    predicted: jersey=7  team=left
  pred_tid=  40  frames=  23  purity=0.74
    primary : gt_tid=5 jersey=4 team=right (17fr)
    others  : gt_tid=2 jersey=10 team=right (4fr) | gt_tid=7 jersey=31 team=right (2fr)
    predicted: jersey=nan  team=right
  pred_tid=  46  frames=  26  purity=0.77
    primary : gt_tid=20 jersey=9 team=left (20fr)
    others  : gt_tid=5 jersey=4 team=right (3fr) | gt_tid=19 jersey=33 team=left (3fr)
    predicted: jersey=nan  team=left
```

### SNGS-105
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 19 | 2 | 36 | 3 | 57 | **33.3%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        5        16           5       133     1.0
       72        62           5       338     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 60 | 0 | 3 | 60 | **100.0%** |

#### ID Switches detected: 3
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   4  frames= 518  purity=0.68
    primary : gt_tid=8 jersey=24 team=left (350fr)
    others  : gt_tid=20 jersey=9 team=left (164fr) | gt_tid=17 jersey=31 team=right (4fr)
    predicted: jersey=24  team=left
  pred_tid=  36  frames=   7  purity=0.57
    primary : gt_tid=19 jersey=33 team=left (4fr)
    others  : gt_tid=18 jersey=4 team=right (3fr)
    predicted: jersey=nan  team=left
  pred_tid=  45  frames= 602  purity=0.67
    primary : gt_tid=23 jersey=14 team=right (406fr)
    others  : gt_tid=16 jersey=7 team=left (196fr)
    predicted: jersey=14  team=right
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 55 | 9 | 88 | 14 | 152 | **36.2%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 160 | 0 | 14 | 160 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **14** (across 3 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-103 | 4 | 140 | 0.65 | gt_tid=4 jersey=4 team=right (91fr) | right |
| SNGS-103 | 13 | 73 | 0.38 | gt_tid=15 jersey=14 team=left (28fr) | right |
| SNGS-103 | 18 | 330 | 0.51 | gt_tid=21 jersey=32 team=left (169fr) | left |
| SNGS-103 | 19 | 451 | 0.63 | gt_tid=13 jersey=24 team=left (283fr) | left |
| SNGS-103 | 26 | 11 | 0.73 | gt_tid=21 jersey=32 team=left (8fr) | left |
| SNGS-103 | 34 | 192 | 0.69 | gt_tid=23 jersey=36 team=left (132fr) | left |
| SNGS-103 | 51 | 6 | 0.67 | gt_tid=3 jersey=20 team=right (2fr) | right |
| SNGS-103 | 78 | 41 | 0.76 | gt_tid=19 jersey=6 team=left (31fr) | left |
| SNGS-104 | 32 | 335 | 0.66 | gt_tid=18 jersey=7 team=left (222fr) | left |
| SNGS-104 | 40 | 23 | 0.74 | gt_tid=5 jersey=4 team=right (17fr) | right |
| SNGS-104 | 46 | 26 | 0.77 | gt_tid=20 jersey=9 team=left (20fr) | left |
| SNGS-105 | 4 | 518 | 0.68 | gt_tid=8 jersey=24 team=left (350fr) | left |
| SNGS-105 | 36 | 7 | 0.57 | gt_tid=19 jersey=33 team=left (4fr) | left |
| SNGS-105 | 45 | 602 | 0.67 | gt_tid=23 jersey=14 team=right (406fr) | right |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.