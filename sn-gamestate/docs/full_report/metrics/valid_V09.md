# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-07-04 22:43
**State file**: `/home/joao/soccernet/outputs/sn-gamestate/2026-07-03/18-23-55/states/gshota-valid-V09.pklz`

**Videos**: SNGS-045, SNGS-046, SNGS-047
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-045
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 5 | 8 | 23 | 1 | 36 | **13.9%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       14        11           5       176    1.00
       15        43          63       147    1.00
       34        15           6       142    1.00
       43        36           5       157    1.00
       64        43          45       220    1.00
       82        14          17        27    1.00
       89        14           5       258    1.00
      115        15           5        68    0.99
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 26 | 12 | 1 | 38 | **68.4%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
       20    left     right       151    1.00
       22    left     right       145    1.00
       29    left     right        65    1.00
       44    left     right       105    0.95
       78    left     right       116    1.00
       82    left     right        27    1.00
       85    left     right        14    1.00
      102    left     right        76    1.00
      115    left     right        68    0.99
      119    left     right        21    0.90
```

#### ID Switches detected: 1
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid= 124  frames=   6  purity=0.67
    primary : gt_tid=5 jersey=8 team=left (4fr)
    others  : gt_tid=6 jersey=14 team=left (2fr)
    predicted: jersey=None  team=right
```

### SNGS-046
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 6 | 2 | 18 | 0 | 26 | **23.1%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       56         6           8       219     1.0
       66         4          14       108     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 2 | 26 | 0 | 28 | **7.1%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
        3    left     right       749    1.00
        5    left     right       533    1.00
        6    left     right       432    1.00
        9    left     right         8    1.00
       11    left     right        83    0.98
       16    left     right       376    1.00
       18    left     right       338    1.00
       19    left     right       723    1.00
       20    left     right        87    1.00
       24    left     right        25    1.00
```

#### ID Switches detected: 1
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  71  frames= 157  purity=0.55
    primary : gt_tid=24 jersey=None team=left (86fr)
    others  : gt_tid=6 jersey=5 team=left (39fr) | gt_tid=7 jersey=15 team=left (32fr)
    predicted: jersey=None  team=right
```

### SNGS-047
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 4 | 4 | 18 | 3 | 26 | **15.4%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       34        36          35       579    1.00
       35        14          16       385    1.00
       67        15           5       188    1.00
       68         8           5        73    0.99
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 3 | 23 | 3 | 26 | **11.5%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
        4    left     right        97    1.00
        5    left     right        42    1.00
        7    left     right       175    0.99
        8    left     right       122    1.00
       13    left     right       137    1.00
       14    left     right       729    1.00
       15    left     right        45    1.00
       21    left     right        54    0.98
       26    left     right        10    0.90
       30    left     right         8    0.88
```

#### ID Switches detected: 4
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  11  frames= 480  purity=0.33
    primary : gt_tid=15 jersey=16 team=left (157fr)
    others  : gt_tid=16 jersey=5 team=left (123fr) | gt_tid=14 jersey=8 team=left (110fr)
    predicted: jersey=None  team=right
  pred_tid=  16  frames=  75  purity=0.63
    primary : gt_tid=9 jersey=None team=None (47fr)
    others  : gt_tid=5 jersey=None team=right (28fr)
    predicted: jersey=None  team=left
  pred_tid=  33  frames= 427  purity=0.58
    primary : gt_tid=14 jersey=8 team=left (247fr)
    others  : gt_tid=15 jersey=16 team=left (177fr) | gt_tid=17 jersey=14 team=left (3fr)
    predicted: jersey=1  team=right
  pred_tid=  41  frames=   7  purity=0.57
    primary : gt_tid=17 jersey=14 team=left (4fr)
    others  : gt_tid=15 jersey=16 team=left (2fr) | gt_tid=14 jersey=8 team=left (1fr)
    predicted: jersey=None  team=right
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 15 | 14 | 59 | 4 | 88 | **17.0%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 31 | 61 | 4 | 92 | **33.7%** |

### ID Switches Summary
Total ID switches detected: **6** (across 3 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-045 | 124 | 6 | 0.67 | gt_tid=5 jersey=8 team=left (4fr) | right |
| SNGS-046 | 71 | 157 | 0.55 | gt_tid=24 jersey=None team=left (86fr) | right |
| SNGS-047 | 11 | 480 | 0.33 | gt_tid=15 jersey=16 team=left (157fr) | right |
| SNGS-047 | 16 | 75 | 0.63 | gt_tid=9 jersey=None team=None (47fr) | left |
| SNGS-047 | 33 | 427 | 0.58 | gt_tid=14 jersey=8 team=left (247fr) | right |
| SNGS-047 | 41 | 7 | 0.57 | gt_tid=17 jersey=14 team=left (4fr) | right |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.