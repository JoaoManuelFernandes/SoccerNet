# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-07-08 11:07
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-07-08/10-26-02/states/final-TEA.pklz`

**Videos**: SNGS-116, SNGS-117
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-116
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 20 | 7 | 11 | 8 | 38 | **52.6%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        1        93          33       197    0.99
       23        36           6       120    0.99
       24        10           5       211    1.00
       26        34           6       281    1.00
       31        31           5       375    0.88
       47        20          15        26    1.00
       83         9          99        59    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 47 | 0 | 8 | 47 | **100.0%** |

#### ID Switches detected: 8
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   4  frames= 242  purity=0.67
    primary : gt_tid=7 jersey=34 team=right (162fr)
    others  : gt_tid=18 jersey=8 team=right (68fr) | gt_tid=17 jersey=36 team=left (6fr)
    predicted: jersey=34  team=right
  pred_tid=   5  frames= 333  purity=0.71
    primary : gt_tid=14 jersey=33 team=left (237fr)
    others  : gt_tid=6 jersey=11 team=left (93fr) | gt_tid=11 jersey=10 team=right (2fr)
    predicted: jersey=11  team=left
  pred_tid=  11  frames= 106  purity=0.65
    primary : gt_tid=11 jersey=10 team=right (69fr)
    others  : gt_tid=7 jersey=34 team=right (25fr) | gt_tid=16 jersey=36 team=right (12fr)
    predicted: jersey=5  team=right
  pred_tid=  13  frames= 231  purity=0.54
    primary : gt_tid=13 jersey=14 team=right (124fr)
    others  : gt_tid=18 jersey=8 team=right (71fr) | gt_tid=12 jersey=31 team=right (34fr)
    predicted: jersey=5  team=right
  pred_tid=  16  frames= 296  purity=0.57
    primary : gt_tid=16 jersey=36 team=right (168fr)
    others  : gt_tid=19 jersey=None team=right (124fr) | gt_tid=6 jersey=11 team=left (4fr)
    predicted: jersey=56  team=right
  pred_tid=  27  frames= 217  purity=0.74
    primary : gt_tid=15 jersey=8 team=left (160fr)
    others  : gt_tid=19 jersey=None team=right (57fr)
    predicted: jersey=8  team=left
  pred_tid=  37  frames=   6  purity=0.50
    primary : gt_tid=14 jersey=33 team=left (3fr)
    others  : gt_tid=8 jersey=44 team=left (2fr) | gt_tid=18 jersey=8 team=right (1fr)
    predicted: jersey=None  team=left
  pred_tid=  91  frames=   6  purity=0.67
    primary : gt_tid=9 jersey=33 team=right (4fr)
    others  : gt_tid=21 jersey=None team=right (2fr)
    predicted: jersey=None  team=right
```

### SNGS-117
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 16 | 8 | 4 | 1 | 28 | **57.1%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        9        10          11       174    1.00
       11        44           4       305    0.98
       13        14          26       104    1.00
       26        31          15       147    1.00
       32        38          36       400    1.00
       39         9           5       346    0.99
       57        36          19        76    1.00
       62        53           3       199    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 38 | 0 | 1 | 38 | **100.0%** |

#### ID Switches detected: 1
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  28  frames=   9  purity=0.75
    primary : gt_tid=7 jersey=31 team=right (3fr)
    others  : gt_tid=9 jersey=44 team=left (1fr)
    predicted: jersey=None  team=right
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 36 | 15 | 15 | 9 | 66 | **54.5%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 85 | 0 | 9 | 85 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **9** (across 2 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-116 | 4 | 242 | 0.67 | gt_tid=7 jersey=34 team=right (162fr) | right |
| SNGS-116 | 5 | 333 | 0.71 | gt_tid=14 jersey=33 team=left (237fr) | left |
| SNGS-116 | 11 | 106 | 0.65 | gt_tid=11 jersey=10 team=right (69fr) | right |
| SNGS-116 | 13 | 231 | 0.54 | gt_tid=13 jersey=14 team=right (124fr) | right |
| SNGS-116 | 16 | 296 | 0.57 | gt_tid=16 jersey=36 team=right (168fr) | right |
| SNGS-116 | 27 | 217 | 0.74 | gt_tid=15 jersey=8 team=left (160fr) | left |
| SNGS-116 | 37 | 6 | 0.5 | gt_tid=14 jersey=33 team=left (3fr) | left |
| SNGS-116 | 91 | 6 | 0.67 | gt_tid=9 jersey=33 team=right (4fr) | right |
| SNGS-117 | 28 | 9 | 0.75 | gt_tid=7 jersey=31 team=right (3fr) | right |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.