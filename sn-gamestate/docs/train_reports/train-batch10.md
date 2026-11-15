# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-07 07:42
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-09-05/22-02-29/states/train-batch10.pklz`

**Videos**: SNGS-106, SNGS-107, SNGS-108
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-106
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 4 | 11 | 18 | 9 | 33 | **12.1%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        5        62           5       416    0.96
       17         9           5       454    0.99
       18        36           5       163    1.00
       30        33          12       388    0.91
       57        20           5       110    1.00
       68         4           5       158    0.99
       71        36           5       116    0.99
       72         6           5        96    1.00
       75        62           6       125    0.99
       78        14           6        48    0.96
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 0 | 45 | 9 | 45 | **0.0%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
        1   right      left        29    1.00
        2    left     right        99    0.99
        3    left     right       713    1.00
        5    left     right       416    0.96
        7   right      left       472    1.00
       10    left     right         6    1.00
       11   right      left        21    1.00
       12    left     right        27    1.00
       13    left     right       562    1.00
       16   right      left       152    0.94
```

#### ID Switches detected: 9
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   4  frames= 389  purity=0.58
    primary : gt_tid=9 jersey=20 team=right (225fr)
    others  : gt_tid=14 jersey=10 team=right (142fr) | gt_tid=17 jersey=62 team=left (18fr)
    predicted: jersey=10  team=left
  pred_tid=   9  frames=  23  purity=0.48
    primary : gt_tid=15 jersey=9 team=right (11fr)
    others  : gt_tid=16 jersey=36 team=left (7fr) | gt_tid=17 jersey=62 team=left (3fr)
    predicted: jersey=nan  team=left
  pred_tid=  14  frames= 393  purity=0.66
    primary : gt_tid=3 jersey=6 team=left (260fr)
    others  : gt_tid=2 jersey=3 team=left (130fr) | gt_tid=21 jersey=33 team=left (2fr)
    predicted: jersey=6  team=right
  pred_tid=  32  frames= 198  purity=0.67
    primary : gt_tid=5 jersey=9 team=left (133fr)
    others  : gt_tid=2 jersey=3 team=left (52fr) | gt_tid=3 jersey=6 team=left (10fr)
    predicted: jersey=nan  team=right
  pred_tid=  40  frames=  84  purity=0.71
    primary : gt_tid=21 jersey=33 team=left (60fr)
    others  : gt_tid=20 jersey=None team=left (24fr)
    predicted: jersey=nan  team=right
  pred_tid=  50  frames=  36  purity=0.58
    primary : gt_tid=21 jersey=33 team=left (21fr)
    others  : gt_tid=20 jersey=None team=left (13fr) | gt_tid=14 jersey=10 team=right (2fr)
    predicted: jersey=nan  team=right
  pred_tid=  56  frames=   6  purity=0.40
    primary : gt_tid=24 jersey=14 team=right (2fr)
    others  : gt_tid=20 jersey=None team=left (2fr) | gt_tid=14 jersey=10 team=right (1fr)
    predicted: jersey=nan  team=left
  pred_tid=  63  frames=   7  purity=0.57
    primary : gt_tid=2 jersey=3 team=left (4fr)
    others  : gt_tid=23 jersey=None team=right (3fr)
    predicted: jersey=nan  team=left
  pred_tid=  74  frames=  24  purity=0.71
    primary : gt_tid=2 jersey=3 team=left (17fr)
    others  : gt_tid=6 jersey=None team=left (7fr)
    predicted: jersey=nan  team=right
```

### SNGS-107
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 7 | 2 | 36 | 3 | 45 | **15.6%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        8        34          15       344     1.0
       45        16          65        66     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 54 | 0 | 3 | 54 | **100.0%** |

#### ID Switches detected: 3
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  72  frames=   8  purity=0.50
    primary : gt_tid=2 jersey=31 team=left (4fr)
    others  : gt_tid=7 jersey=34 team=left (2fr) | gt_tid=14 jersey=9 team=right (2fr)
    predicted: jersey=nan  team=left
  pred_tid=  76  frames=  34  purity=0.76
    primary : gt_tid=14 jersey=9 team=right (26fr)
    others  : gt_tid=7 jersey=34 team=left (8fr)
    predicted: jersey=nan  team=right
  pred_tid=  77  frames=  39  purity=0.77
    primary : gt_tid=2 jersey=31 team=left (30fr)
    others  : gt_tid=3 jersey=4 team=left (9fr)
    predicted: jersey=31  team=left
```

### SNGS-108
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 16 | 9 | 21 | 3 | 46 | **34.8%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        2        16           1        19    1.00
       20         9           5       192    0.99
       36        14           6       200    0.98
       45        14          16        32    0.97
       46        31           5       150    1.00
       53         9          19       310    0.99
       58        14           6       145    0.99
       61        20          21       210    0.99
       64        27          21         7    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 53 | 0 | 3 | 53 | **100.0%** |

#### ID Switches detected: 3
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  34  frames=  72  purity=0.75
    primary : gt_tid=21 jersey=7 team=right (53fr)
    others  : gt_tid=19 jersey=20 team=left (18fr)
    predicted: jersey=nan  team=right
  pred_tid=  39  frames= 473  purity=0.66
    primary : gt_tid=14 jersey=24 team=right (313fr)
    others  : gt_tid=17 jersey=9 team=right (160fr)
    predicted: jersey=nan  team=right
  pred_tid=  50  frames= 359  purity=0.37
    primary : gt_tid=25 jersey=4 team=left (134fr)
    others  : gt_tid=22 jersey=33 team=right (124fr) | gt_tid=24 jersey=31 team=left (100fr)
    predicted: jersey=31  team=left
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 27 | 22 | 75 | 15 | 124 | **21.8%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 107 | 45 | 15 | 152 | **70.4%** |

### ID Switches Summary
Total ID switches detected: **15** (across 3 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-106 | 4 | 389 | 0.58 | gt_tid=9 jersey=20 team=right (225fr) | left |
| SNGS-106 | 9 | 23 | 0.48 | gt_tid=15 jersey=9 team=right (11fr) | left |
| SNGS-106 | 14 | 393 | 0.66 | gt_tid=3 jersey=6 team=left (260fr) | right |
| SNGS-106 | 32 | 198 | 0.67 | gt_tid=5 jersey=9 team=left (133fr) | right |
| SNGS-106 | 40 | 84 | 0.71 | gt_tid=21 jersey=33 team=left (60fr) | right |
| SNGS-106 | 50 | 36 | 0.58 | gt_tid=21 jersey=33 team=left (21fr) | right |
| SNGS-106 | 56 | 6 | 0.4 | gt_tid=24 jersey=14 team=right (2fr) | left |
| SNGS-106 | 63 | 7 | 0.57 | gt_tid=2 jersey=3 team=left (4fr) | left |
| SNGS-106 | 74 | 24 | 0.71 | gt_tid=2 jersey=3 team=left (17fr) | right |
| SNGS-107 | 72 | 8 | 0.5 | gt_tid=2 jersey=31 team=left (4fr) | left |
| SNGS-107 | 76 | 34 | 0.76 | gt_tid=14 jersey=9 team=right (26fr) | right |
| SNGS-107 | 77 | 39 | 0.77 | gt_tid=2 jersey=31 team=left (30fr) | left |
| SNGS-108 | 34 | 72 | 0.75 | gt_tid=21 jersey=7 team=right (53fr) | right |
| SNGS-108 | 39 | 473 | 0.66 | gt_tid=14 jersey=24 team=right (313fr) | right |
| SNGS-108 | 50 | 359 | 0.37 | gt_tid=25 jersey=4 team=left (134fr) | left |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.