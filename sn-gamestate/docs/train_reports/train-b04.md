# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-08 10:07
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-09-08/08-53-55/states/train-b04.pklz`

**Videos**: SNGS-069, SNGS-070, SNGS-071
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-069
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 10 | 5 | 13 | 3 | 28 | **35.7%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       11        38          74       731    1.00
       25        28          26        75    1.00
       28        10           5       118    0.99
       53         4          14       115    1.00
       55        50           6        88    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 16 | 14 | 3 | 30 | **53.3%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
        5    left     right       750    1.00
        6    left     right       298    1.00
        8    left     right       514    0.97
       12    left     right       299    1.00
       15    left     right       204    1.00
       16    left     right       139    1.00
       19    left     right       489    1.00
       21    left     right       561    0.99
       23    left     right        23    1.00
       28    left     right       118    0.99
```

#### ID Switches detected: 3
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  13  frames= 668  purity=0.66
    primary : gt_tid=1 jersey=15 team=left (438fr)
    others  : gt_tid=4 jersey=8 team=left (228fr)
    predicted: jersey=15  team=right
  pred_tid=  14  frames= 368  purity=0.52
    primary : gt_tid=9 jersey=27 team=left (192fr)
    others  : gt_tid=8 jersey=10 team=left (176fr)
    predicted: jersey=10  team=right
  pred_tid=  17  frames= 336  purity=0.57
    primary : gt_tid=9 jersey=27 team=left (188fr)
    others  : gt_tid=6 jersey=4 team=left (143fr)
    predicted: jersey=nan  team=right
```

### SNGS-070
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 17 | 4 | 21 | 4 | 42 | **40.5%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       21        17           5       510     1.0
       23         4           5       465     1.0
       31         3           5       257     1.0
       51         4           5       181     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 46 | 0 | 4 | 46 | **100.0%** |

#### ID Switches detected: 4
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  30  frames=   5  purity=0.60
    primary : gt_tid=10 jersey=50 team=left (3fr)
    others  : gt_tid=16 jersey=None team=right (1fr) | gt_tid=14 jersey=14 team=left (1fr)
    predicted: jersey=None  team=left
  pred_tid=  34  frames=   6  purity=0.67
    primary : gt_tid=5 jersey=30 team=right (4fr)
    others  : gt_tid=2 jersey=None team=None (1fr) | gt_tid=11 jersey=5 team=left (1fr)
    predicted: jersey=None  team=right
  pred_tid=  46  frames=  37  purity=0.70
    primary : gt_tid=11 jersey=5 team=left (26fr)
    others  : gt_tid=10 jersey=50 team=left (11fr)
    predicted: jersey=None  team=left
  pred_tid=  55  frames=  15  purity=0.60
    primary : gt_tid=11 jersey=5 team=left (9fr)
    others  : gt_tid=2 jersey=None team=None (6fr)
    predicted: jersey=None  team=left
```

### SNGS-071
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 24 | 9 | 9 | 12 | 42 | **57.1%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       19         9           5       651    0.98
       22         3           5        75    0.87
       29         4          14        72    0.99
       30        11          17       543    0.99
       31        10           5       204    0.96
       33         8           5       131    0.95
       34        15           8        35    0.86
       65        44           4        98    1.00
       67        44           4        16    0.94
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 42 | 0 | 12 | 42 | **100.0%** |

#### ID Switches detected: 12
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   9  frames= 118  purity=0.80
    primary : gt_tid=21 jersey=11 team=right (94fr)
    others  : gt_tid=6 jersey=15 team=left (15fr) | gt_tid=15 jersey=3 team=right (7fr)
    predicted: jersey=5  team=right
  pred_tid=  10  frames=  76  purity=0.45
    primary : gt_tid=9 jersey=36 team=left (34fr)
    others  : gt_tid=6 jersey=15 team=left (25fr) | gt_tid=22 jersey=4 team=right (5fr)
    predicted: jersey=15  team=left
  pred_tid=  14  frames= 117  purity=0.61
    primary : gt_tid=8 jersey=14 team=left (71fr)
    others  : gt_tid=6 jersey=15 team=left (31fr) | gt_tid=13 jersey=9 team=right (14fr)
    predicted: jersey=15  team=left
  pred_tid=  15  frames=  54  purity=0.69
    primary : gt_tid=16 jersey=8 team=right (37fr)
    others  : gt_tid=9 jersey=36 team=left (13fr) | gt_tid=6 jersey=15 team=left (2fr)
    predicted: jersey=nan  team=right
  pred_tid=  17  frames= 341  purity=0.63
    primary : gt_tid=15 jersey=3 team=right (215fr)
    others  : gt_tid=17 jersey=10 team=right (112fr) | gt_tid=5 jersey=5 team=left (7fr)
    predicted: jersey=3  team=right
  pred_tid=  20  frames=  40  purity=0.53
    primary : gt_tid=22 jersey=4 team=right (20fr)
    others  : gt_tid=15 jersey=3 team=right (13fr) | gt_tid=14 jersey=25 team=right (4fr)
    predicted: jersey=3  team=right
  pred_tid=  25  frames= 209  purity=0.54
    primary : gt_tid=15 jersey=3 team=right (112fr)
    others  : gt_tid=7 jersey=44 team=left (74fr) | gt_tid=14 jersey=25 team=right (12fr)
    predicted: jersey=25  team=left
  pred_tid=  26  frames= 522  purity=0.73
    primary : gt_tid=22 jersey=4 team=right (382fr)
    others  : gt_tid=3 jersey=27 team=left (86fr) | gt_tid=21 jersey=11 team=right (23fr)
    predicted: jersey=4  team=right
  pred_tid=  36  frames=   5  purity=0.40
    primary : gt_tid=7 jersey=44 team=left (2fr)
    others  : gt_tid=21 jersey=11 team=right (1fr) | gt_tid=22 jersey=4 team=right (1fr)
    predicted: jersey=nan  team=left
  pred_tid=  53  frames=  38  purity=0.55
    primary : gt_tid=3 jersey=27 team=left (21fr)
    others  : gt_tid=14 jersey=25 team=right (16fr) | gt_tid=22 jersey=4 team=right (1fr)
    predicted: jersey=nan  team=left
  pred_tid=  54  frames=  11  purity=0.73
    primary : gt_tid=15 jersey=3 team=right (8fr)
    others  : gt_tid=7 jersey=44 team=left (3fr)
    predicted: jersey=nan  team=right
  pred_tid=  61  frames=   8  purity=0.75
    primary : gt_tid=5 jersey=5 team=left (6fr)
    others  : gt_tid=3 jersey=27 team=left (2fr)
    predicted: jersey=nan  team=left
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 51 | 18 | 43 | 19 | 112 | **45.5%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 104 | 14 | 19 | 118 | **88.1%** |

### ID Switches Summary
Total ID switches detected: **19** (across 3 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-069 | 13 | 668 | 0.66 | gt_tid=1 jersey=15 team=left (438fr) | right |
| SNGS-069 | 14 | 368 | 0.52 | gt_tid=9 jersey=27 team=left (192fr) | right |
| SNGS-069 | 17 | 336 | 0.57 | gt_tid=9 jersey=27 team=left (188fr) | right |
| SNGS-070 | 30 | 5 | 0.6 | gt_tid=10 jersey=50 team=left (3fr) | left |
| SNGS-070 | 34 | 6 | 0.67 | gt_tid=5 jersey=30 team=right (4fr) | right |
| SNGS-070 | 46 | 37 | 0.7 | gt_tid=11 jersey=5 team=left (26fr) | left |
| SNGS-070 | 55 | 15 | 0.6 | gt_tid=11 jersey=5 team=left (9fr) | left |
| SNGS-071 | 9 | 118 | 0.8 | gt_tid=21 jersey=11 team=right (94fr) | right |
| SNGS-071 | 10 | 76 | 0.45 | gt_tid=9 jersey=36 team=left (34fr) | left |
| SNGS-071 | 14 | 117 | 0.61 | gt_tid=8 jersey=14 team=left (71fr) | left |
| SNGS-071 | 15 | 54 | 0.69 | gt_tid=16 jersey=8 team=right (37fr) | right |
| SNGS-071 | 17 | 341 | 0.63 | gt_tid=15 jersey=3 team=right (215fr) | right |
| SNGS-071 | 20 | 40 | 0.53 | gt_tid=22 jersey=4 team=right (20fr) | right |
| SNGS-071 | 25 | 209 | 0.54 | gt_tid=15 jersey=3 team=right (112fr) | left |
| SNGS-071 | 26 | 522 | 0.73 | gt_tid=22 jersey=4 team=right (382fr) | right |
| SNGS-071 | 36 | 5 | 0.4 | gt_tid=7 jersey=44 team=left (2fr) | left |
| SNGS-071 | 53 | 38 | 0.55 | gt_tid=3 jersey=27 team=left (21fr) | left |
| SNGS-071 | 54 | 11 | 0.73 | gt_tid=15 jersey=3 team=right (8fr) | right |
| SNGS-071 | 61 | 8 | 0.75 | gt_tid=5 jersey=5 team=left (6fr) | left |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.