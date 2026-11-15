# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-07 07:40
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-09-05/11-11-48/states/train-batch01.pklz`

**Videos**: SNGS-060, SNGS-061, SNGS-062
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-060
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 9 | 4 | 23 | 3 | 36 | **25.0%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        1        25           5       515    0.94
       11         4           5       607    1.00
       24         4           5       156    1.00
       62         8           5       145    0.99
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 55 | 0 | 3 | 55 | **100.0%** |

#### ID Switches detected: 3
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   8  frames= 384  purity=0.69
    primary : gt_tid=3 jersey=27 team=right (264fr)
    others  : gt_tid=19 jersey=None team=left (120fr)
    predicted: jersey=27  team=right
  pred_tid=  49  frames=  98  purity=0.60
    primary : gt_tid=16 jersey=8 team=left (59fr)
    others  : gt_tid=4 jersey=10 team=right (39fr)
    predicted: jersey=nan  team=left
  pred_tid=  73  frames=  12  purity=0.73
    primary : gt_tid=21 jersey=4 team=left (8fr)
    others  : gt_tid=3 jersey=27 team=right (3fr)
    predicted: jersey=nan  team=right
```

### SNGS-061
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 17 | 8 | 17 | 3 | 42 | **40.5%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       16        14           1       224    1.00
       20        44          46       100    1.00
       31        25          53       211    1.00
       33        15           6        48    1.00
       59        17          15        77    1.00
       67         9           5       258    0.94
       74        44          46       156    1.00
       81        30          22        76    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 47 | 0 | 3 | 47 | **100.0%** |

#### ID Switches detected: 3
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  36  frames= 303  purity=0.53
    primary : gt_tid=23 jersey=9 team=left (160fr)
    others  : gt_tid=15 jersey=14 team=right (82fr) | gt_tid=22 jersey=5 team=right (59fr)
    predicted: jersey=9  team=left
  pred_tid=  37  frames= 301  purity=0.53
    primary : gt_tid=22 jersey=5 team=right (161fr)
    others  : gt_tid=23 jersey=9 team=left (138fr) | gt_tid=17 jersey=50 team=right (2fr)
    predicted: jersey=5  team=right
  pred_tid=  44  frames= 438  purity=0.59
    primary : gt_tid=20 jersey=36 team=right (256fr)
    others  : gt_tid=10 jersey=4 team=right (180fr) | gt_tid=14 jersey=17 team=left (1fr)
    predicted: jersey=6  team=right
```

### SNGS-062
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 14 | 6 | 13 | 1 | 33 | **42.4%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       11        25           5       743    0.99
       15         4           6       326    1.00
       16        11          17       595    1.00
       26         3           5       150    1.00
       31         9           5       405    1.00
       54        11          17       123    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 42 | 0 | 1 | 42 | **100.0%** |

#### ID Switches detected: 1
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  34  frames=  18  purity=0.39
    primary : gt_tid=21 jersey=3 team=left (7fr)
    others  : gt_tid=11 jersey=14 team=right (6fr) | gt_tid=15 jersey=11 team=left (5fr)
    predicted: jersey=None  team=left
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 40 | 18 | 53 | 7 | 111 | **36.0%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 144 | 0 | 7 | 144 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **7** (across 3 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-060 | 8 | 384 | 0.69 | gt_tid=3 jersey=27 team=right (264fr) | right |
| SNGS-060 | 49 | 98 | 0.6 | gt_tid=16 jersey=8 team=left (59fr) | left |
| SNGS-060 | 73 | 12 | 0.73 | gt_tid=21 jersey=4 team=left (8fr) | right |
| SNGS-061 | 36 | 303 | 0.53 | gt_tid=23 jersey=9 team=left (160fr) | left |
| SNGS-061 | 37 | 301 | 0.53 | gt_tid=22 jersey=5 team=right (161fr) | right |
| SNGS-061 | 44 | 438 | 0.59 | gt_tid=20 jersey=36 team=right (256fr) | right |
| SNGS-062 | 34 | 18 | 0.39 | gt_tid=21 jersey=3 team=left (7fr) | left |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.