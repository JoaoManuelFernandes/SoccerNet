# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-09 13:42
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-09-09/12-34-08/states/train-b19.pklz`

**Videos**: SNGS-168, SNGS-169, SNGS-170
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-168
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 10 | 1 | 25 | 0 | 36 | **27.8%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       16        19           5       605     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 50 | 0 | 0 | 50 | **100.0%** |

### SNGS-169
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 15 | 4 | 29 | 2 | 48 | **31.2%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        1        26          25       163    1.00
        4        21           5        75    1.00
        6        10           7       314    1.00
      112        25           5        30    0.93
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 61 | 0 | 2 | 61 | **100.0%** |

#### ID Switches detected: 2
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  61  frames=   7  purity=0.71
    primary : gt_tid=22 jersey=19 team=right (5fr)
    others  : gt_tid=11 jersey=5 team=right (1fr) | gt_tid=8 jersey=21 team=right (1fr)
    predicted: jersey=None  team=right
  pred_tid= 124  frames=  12  purity=0.58
    primary : gt_tid=24 jersey=55 team=left (7fr)
    others  : gt_tid=25 jersey=25 team=left (3fr) | gt_tid=22 jersey=19 team=right (2fr)
    predicted: jersey=None  team=left
```

### SNGS-170
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 11 | 5 | 30 | 2 | 46 | **23.9%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        2        28          25       129    1.00
       22        28           5       255    0.99
       37         8           5       109    1.00
       68        19           5       558    0.99
       74        28           5       191    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 57 | 0 | 3 | 57 | **100.0%** |

#### ID Switches detected: 4
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  27  frames=   7  purity=0.40
    primary : gt_tid=19 jersey=None team=left (2fr)
    others  : gt_tid=22 jersey=23 team=left (2fr) | gt_tid=23 jersey=8 team=left (1fr)
    predicted: jersey=nan  team=left
  pred_tid=  56  frames= 451  purity=0.65
    primary : gt_tid=18 jersey=55 team=left (294fr)
    others  : gt_tid=20 jersey=8 team=right (155fr) | gt_tid=13 jersey=17 team=left (2fr)
    predicted: jersey=5  team=left
  pred_tid=  78  frames=   9  purity=0.56
    primary : gt_tid=19 jersey=None team=left (5fr)
    others  : gt_tid=18 jersey=55 team=left (2fr) | gt_tid=17 jersey=4 team=left (2fr)
    predicted: jersey=nan  team=left
  pred_tid=  84  frames= 339  purity=0.77
    primary : gt_tid=18 jersey=55 team=left (261fr)
    others  : gt_tid=19 jersey=None team=left (70fr) | gt_tid=17 jersey=4 team=left (6fr)
    predicted: jersey=5  team=left
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 36 | 10 | 84 | 4 | 130 | **27.7%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 168 | 0 | 5 | 168 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **6** (across 2 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-169 | 61 | 7 | 0.71 | gt_tid=22 jersey=19 team=right (5fr) | right |
| SNGS-169 | 124 | 12 | 0.58 | gt_tid=24 jersey=55 team=left (7fr) | left |
| SNGS-170 | 27 | 7 | 0.4 | gt_tid=19 jersey=None team=left (2fr) | left |
| SNGS-170 | 56 | 451 | 0.65 | gt_tid=18 jersey=55 team=left (294fr) | left |
| SNGS-170 | 78 | 9 | 0.56 | gt_tid=19 jersey=None team=left (5fr) | left |
| SNGS-170 | 84 | 339 | 0.77 | gt_tid=18 jersey=55 team=left (261fr) | left |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.