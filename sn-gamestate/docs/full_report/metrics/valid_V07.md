# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-07-04 22:39
**State file**: `/home/joao/soccernet/outputs/sn-gamestate/2026-07-03/08-42-54/states/gshota-valid-V07.pklz`

**Videos**: SNGS-039, SNGS-040, SNGS-041
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-039
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 5 | 5 | 22 | 1 | 32 | **15.6%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        5        36          35       726     1.0
        9        11           5       127     1.0
       22        93          56       319     1.0
       46        15           5       158     1.0
       52         8           6       189     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 0 | 32 | 1 | 32 | **0.0%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
        2    left     right       151    1.00
        3    left     right       338    1.00
        4    left     right       350    0.97
        5    left     right       726    1.00
        9    left     right       127    1.00
       11    left     right       449    1.00
       12    left     right       429    1.00
       14    left     right       252    0.97
       22    left     right       319    1.00
       24    left     right       217    1.00
```

#### ID Switches detected: 1
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  68  frames=   6  purity=0.67
    primary : gt_tid=15 jersey=17 team=right (4fr)
    others  : gt_tid=13 jersey=5 team=left (2fr)
    predicted: jersey=None  team=right
```

### SNGS-040
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 3 | 1 | 31 | 8 | 35 | **8.6%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        4        93          33       270     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 32 | 3 | 8 | 35 | **91.4%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
       17    left     right         9    0.89
       67    left     right         5    1.00
       75    left     right         6    0.83
```

#### ID Switches detected: 9
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  19  frames=  13  purity=0.38
    primary : gt_tid=21 jersey=None team=None (5fr)
    others  : gt_tid=16 jersey=36 team=left (4fr) | gt_tid=18 jersey=5 team=left (2fr)
    predicted: jersey=None  team=left
  pred_tid=  25  frames= 269  purity=0.66
    primary : gt_tid=17 jersey=43 team=left (178fr)
    others  : gt_tid=13 jersey=None team=right (91fr)
    predicted: jersey=56  team=left
  pred_tid=  27  frames=   6  purity=0.33
    primary : gt_tid=18 jersey=5 team=left (2fr)
    others  : gt_tid=16 jersey=36 team=left (1fr) | gt_tid=6 jersey=5 team=right (1fr)
    predicted: jersey=None  team=left
  pred_tid=  33  frames=  28  purity=0.61
    primary : gt_tid=16 jersey=36 team=left (17fr)
    others  : gt_tid=15 jersey=8 team=left (7fr) | gt_tid=14 jersey=15 team=left (4fr)
    predicted: jersey=None  team=left
  pred_tid=  34  frames=   9  purity=0.56
    primary : gt_tid=17 jersey=43 team=left (5fr)
    others  : gt_tid=2 jersey=99 team=right (2fr) | gt_tid=14 jersey=15 team=left (1fr)
    predicted: jersey=None  team=left
  pred_tid=  39  frames=   8  purity=0.75
    primary : gt_tid=16 jersey=36 team=left (6fr)
    others  : gt_tid=15 jersey=8 team=left (2fr)
    predicted: jersey=None  team=left
  pred_tid=  48  frames=   8  purity=0.62
    primary : gt_tid=18 jersey=5 team=left (5fr)
    others  : gt_tid=4 jersey=23 team=right (1fr) | gt_tid=13 jersey=None team=right (1fr)
    predicted: jersey=None  team=left
  pred_tid=  72  frames=  14  purity=0.64
    primary : gt_tid=19 jersey=93 team=left (9fr)
    others  : gt_tid=21 jersey=None team=None (5fr)
    predicted: jersey=None  team=left
  pred_tid=  99  frames= 247  purity=0.52
    primary : gt_tid=17 jersey=43 team=left (128fr)
    others  : gt_tid=9 jersey=43 team=right (119fr)
    predicted: jersey=None  team=left
```

### SNGS-041
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 5 | 2 | 10 | 0 | 17 | **29.4%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       26        14          16        64     1.0
       41        43          56       228     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 3 | 14 | 0 | 17 | **17.6%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
        2    left     right       750     1.0
        5    left     right       631     1.0
        9    left     right       158     1.0
       10    left     right       172     1.0
       17    left     right       396     1.0
       27    left     right       300     1.0
       30    left     right       391     1.0
       31    left     right       424     1.0
       37    left     right        23     1.0
       41    left     right       228     1.0
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 13 | 8 | 63 | 9 | 84 | **15.5%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 35 | 49 | 9 | 84 | **41.7%** |

### ID Switches Summary
Total ID switches detected: **10** (across 2 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-039 | 68 | 6 | 0.67 | gt_tid=15 jersey=17 team=right (4fr) | right |
| SNGS-040 | 19 | 13 | 0.38 | gt_tid=21 jersey=None team=None (5fr) | left |
| SNGS-040 | 25 | 269 | 0.66 | gt_tid=17 jersey=43 team=left (178fr) | left |
| SNGS-040 | 27 | 6 | 0.33 | gt_tid=18 jersey=5 team=left (2fr) | left |
| SNGS-040 | 33 | 28 | 0.61 | gt_tid=16 jersey=36 team=left (17fr) | left |
| SNGS-040 | 34 | 9 | 0.56 | gt_tid=17 jersey=43 team=left (5fr) | left |
| SNGS-040 | 39 | 8 | 0.75 | gt_tid=16 jersey=36 team=left (6fr) | left |
| SNGS-040 | 48 | 8 | 0.62 | gt_tid=18 jersey=5 team=left (5fr) | left |
| SNGS-040 | 72 | 14 | 0.64 | gt_tid=19 jersey=93 team=left (9fr) | left |
| SNGS-040 | 99 | 247 | 0.52 | gt_tid=17 jersey=43 team=left (128fr) | left |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.