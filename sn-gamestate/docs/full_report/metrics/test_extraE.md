# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-07-04 22:23
**State file**: `/home/joao/soccernet/outputs/sn-gamestate/2026-07-01/19-35-27/states/gshota-test-extraE.pklz`

**Videos**: SNGS-190, SNGS-191
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-190
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 12 | 6 | 11 | 4 | 29 | **41.4%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        3        62          52       745    1.00
        7        24          26       115    0.92
       20         9           5        96    1.00
       39        34           5        60    1.00
       48        36           5       291    1.00
       49        24           2       370    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 0 | 38 | 4 | 38 | **0.0%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
        1    left     right       734    1.00
        2   right      left       240    1.00
        3    left     right       745    1.00
        4   right      left        95    1.00
        5   right      left       141    1.00
        7    left     right       115    0.92
        9   right      left       142    1.00
       10   right      left       153    0.99
       11   right      left        41    1.00
       12    left     right        95    1.00
```

#### ID Switches detected: 5
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   8  frames=  82  purity=0.66
    primary : gt_tid=13 jersey=22 team=right (54fr)
    others  : gt_tid=16 jersey=None team=None (19fr) | gt_tid=5 jersey=24 team=left (5fr)
    predicted: jersey=56  team=left
  pred_tid=  17  frames= 680  purity=0.72
    primary : gt_tid=4 jersey=30 team=left (491fr)
    others  : gt_tid=3 jersey=16 team=left (186fr) | gt_tid=23 jersey=36 team=left (2fr)
    predicted: jersey=30  team=right
  pred_tid=  19  frames= 277  purity=0.75
    primary : gt_tid=16 jersey=None team=None (205fr)
    others  : gt_tid=18 jersey=31 team=right (66fr) | gt_tid=11 jersey=10 team=right (2fr)
    predicted: jersey=5  team=left
  pred_tid=  54  frames=   5  purity=0.60
    primary : gt_tid=4 jersey=30 team=left (3fr)
    others  : gt_tid=3 jersey=16 team=left (2fr)
    predicted: jersey=20  team=right
  pred_tid=  57  frames=  59  purity=0.75
    primary : gt_tid=8 jersey=15 team=right (44fr)
    others  : gt_tid=4 jersey=30 team=left (13fr) | gt_tid=21 jersey=34 team=left (1fr)
    predicted: jersey=None  team=left
```

### SNGS-191
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 14 | 10 | 12 | 3 | 36 | **38.9%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        4         3           5       173    1.00
        8        22          56       479    0.99
       17         9           5       384    1.00
       18         2          36       473    1.00
       24         9          14       118    0.99
       28        36          56        65    1.00
       36        22          50       239    1.00
       46        17           1       170    1.00
       47         7          34       163    0.99
       48        24          34       157    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 38 | 0 | 3 | 38 | **100.0%** |

#### ID Switches detected: 3
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   9  frames= 331  purity=0.67
    primary : gt_tid=18 jersey=17 team=right (219fr)
    others  : gt_tid=10 jersey=15 team=right (102fr) | gt_tid=9 jersey=3 team=right (3fr)
    predicted: jersey=17  team=right
  pred_tid=  26  frames= 225  purity=0.66
    primary : gt_tid=10 jersey=15 team=right (147fr)
    others  : gt_tid=5 jersey=10 team=right (77fr)
    predicted: jersey=None  team=right
  pred_tid=  34  frames=  54  purity=0.80
    primary : gt_tid=2 jersey=14 team=left (43fr)
    others  : gt_tid=7 jersey=9 team=left (8fr) | gt_tid=4 jersey=14 team=right (2fr)
    predicted: jersey=None  team=left
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 26 | 16 | 23 | 7 | 65 | **40.0%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 38 | 38 | 7 | 76 | **50.0%** |

### ID Switches Summary
Total ID switches detected: **8** (across 2 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-190 | 8 | 82 | 0.66 | gt_tid=13 jersey=22 team=right (54fr) | left |
| SNGS-190 | 17 | 680 | 0.72 | gt_tid=4 jersey=30 team=left (491fr) | right |
| SNGS-190 | 19 | 277 | 0.75 | gt_tid=16 jersey=None team=None (205fr) | left |
| SNGS-190 | 54 | 5 | 0.6 | gt_tid=4 jersey=30 team=left (3fr) | right |
| SNGS-190 | 57 | 59 | 0.75 | gt_tid=8 jersey=15 team=right (44fr) | left |
| SNGS-191 | 9 | 331 | 0.67 | gt_tid=18 jersey=17 team=right (219fr) | right |
| SNGS-191 | 26 | 225 | 0.66 | gt_tid=10 jersey=15 team=right (147fr) | right |
| SNGS-191 | 34 | 54 | 0.8 | gt_tid=2 jersey=14 team=left (43fr) | left |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.