# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-07-04 22:59
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-07-04/17-41-28/states/gshota-valid-V16.pklz`

**Videos**: SNGS-088, SNGS-089, SNGS-090
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-088
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 12 | 4 | 16 | 3 | 32 | **37.5%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        1        44           5        79     1.0
       35        36          56       490     1.0
       51         5          65        89     1.0
       60        19          99        96     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 40 | 0 | 3 | 40 | **100.0%** |

#### ID Switches detected: 3
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  19  frames= 290  purity=0.74
    primary : gt_tid=19 jersey=27 team=left (212fr)
    others  : gt_tid=16 jersey=None team=left (76fr)
    predicted: jersey=None  team=left
  pred_tid=  31  frames=  95  purity=0.59
    primary : gt_tid=23 jersey=27 team=right (56fr)
    others  : gt_tid=24 jersey=5 team=left (39fr)
    predicted: jersey=None  team=right
  pred_tid=  67  frames=   7  purity=0.57
    primary : gt_tid=13 jersey=20 team=left (4fr)
    others  : gt_tid=2 jersey=8 team=right (3fr)
    predicted: jersey=None  team=right
```

### SNGS-089
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 17 | 5 | 22 | 2 | 44 | **38.6%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       11        44           5       148    1.00
       12        27          17       141    1.00
       31        20          10       309    1.00
       33        15           5       297    1.00
       42        44          27        86    0.92
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 49 | 0 | 2 | 49 | **100.0%** |

#### ID Switches detected: 2
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   4  frames= 593  purity=0.68
    primary : gt_tid=19 jersey=13 team=left (402fr)
    others  : gt_tid=9 jersey=4 team=right (183fr) | gt_tid=16 jersey=19 team=left (8fr)
    predicted: jersey=13  team=left
  pred_tid=  67  frames=   5  purity=0.60
    primary : gt_tid=13 jersey=44 team=right (3fr)
    others  : gt_tid=16 jersey=19 team=left (2fr)
    predicted: jersey=None  team=right
```

### SNGS-090
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 19 | 2 | 22 | 3 | 43 | **44.2%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       31        43          53        90    1.00
       36        14           5        87    0.99
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 43 | 0 | 3 | 43 | **100.0%** |

#### ID Switches detected: 3
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  17  frames= 345  purity=0.51
    primary : gt_tid=9 jersey=19 team=left (175fr)
    others  : gt_tid=1 jersey=50 team=right (167fr) | gt_tid=3 jersey=8 team=right (3fr)
    predicted: jersey=None  team=right
  pred_tid=  32  frames= 224  purity=0.68
    primary : gt_tid=20 jersey=13 team=left (151fr)
    others  : gt_tid=25 jersey=17 team=left (70fr)
    predicted: jersey=None  team=left
  pred_tid=  61  frames=   7  purity=0.67
    primary : gt_tid=3 jersey=8 team=right (4fr)
    others  : gt_tid=10 jersey=20 team=left (2fr)
    predicted: jersey=None  team=right
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 48 | 11 | 60 | 8 | 119 | **40.3%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 132 | 0 | 8 | 132 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **8** (across 3 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-088 | 19 | 290 | 0.74 | gt_tid=19 jersey=27 team=left (212fr) | left |
| SNGS-088 | 31 | 95 | 0.59 | gt_tid=23 jersey=27 team=right (56fr) | right |
| SNGS-088 | 67 | 7 | 0.57 | gt_tid=13 jersey=20 team=left (4fr) | right |
| SNGS-089 | 4 | 593 | 0.68 | gt_tid=19 jersey=13 team=left (402fr) | left |
| SNGS-089 | 67 | 5 | 0.6 | gt_tid=13 jersey=44 team=right (3fr) | right |
| SNGS-090 | 17 | 345 | 0.51 | gt_tid=9 jersey=19 team=left (175fr) | right |
| SNGS-090 | 32 | 224 | 0.68 | gt_tid=20 jersey=13 team=left (151fr) | left |
| SNGS-090 | 61 | 7 | 0.67 | gt_tid=3 jersey=8 team=right (4fr) | right |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.