# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-07-08 12:02
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-07-08/11-07-53/states/final-TEB.pklz`

**Videos**: SNGS-127, SNGS-128
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-127
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 7 | 2 | 1 | 0 | 10 | **70.0%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       13        44           4       329     1.0
       23        44           6         7     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 22 | 0 | 0 | 22 | **100.0%** |

### SNGS-128
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 17 | 5 | 11 | 1 | 33 | **51.5%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       20        16           5       651    0.99
       27        27           5       156    0.99
       34        34           5       608    1.00
       49         9           5        76    1.00
       47        14           5       338    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 37 | 0 | 2 | 37 | **100.0%** |

#### ID Switches detected: 2
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  14  frames=  25  purity=0.60
    primary : gt_tid=10 jersey=None team=right (15fr)
    others  : gt_tid=20 jersey=27 team=right (10fr)
    predicted: jersey=None  team=right
  pred_tid=  33  frames=   7  purity=0.71
    primary : gt_tid=22 jersey=14 team=left (5fr)
    others  : gt_tid=24 jersey=31 team=left (2fr)
    predicted: jersey=None  team=left
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 24 | 7 | 12 | 1 | 43 | **55.8%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 59 | 0 | 2 | 59 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **2** (across 1 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-128 | 14 | 25 | 0.6 | gt_tid=10 jersey=None team=right (15fr) | right |
| SNGS-128 | 33 | 7 | 0.71 | gt_tid=22 jersey=14 team=left (5fr) | left |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.