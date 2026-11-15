# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-07-08 13:31
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-07-08/12-51-48/states/final-TED.pklz`

**Videos**: SNGS-143, SNGS-144
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-143
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 21 | 5 | 20 | 3 | 46 | **45.7%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        4        93           3       194    0.99
       16        11           5        35    1.00
       20        14          16       121    0.99
       26        36          33        57    1.00
       69        14           1        37    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 49 | 0 | 3 | 49 | **100.0%** |

#### ID Switches detected: 3
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  19  frames=   6  purity=0.67
    primary : gt_tid=5 jersey=14 team=left (4fr)
    others  : gt_tid=2 jersey=16 team=left (2fr)
    predicted: jersey=None  team=left
  pred_tid=  62  frames= 120  purity=0.70
    primary : gt_tid=9 jersey=33 team=right (84fr)
    others  : gt_tid=8 jersey=14 team=right (36fr)
    predicted: jersey=5  team=right
  pred_tid=  90  frames=   7  purity=0.71
    primary : gt_tid=9 jersey=33 team=right (5fr)
    others  : gt_tid=16 jersey=9 team=left (2fr)
    predicted: jersey=None  team=right
```

### SNGS-144
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 27 | 4 | 16 | 1 | 47 | **57.4%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       16        93           5       138     1.0
       24         7           5        11     1.0
       61        36           5       191     1.0
       81        93           5       174     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 47 | 0 | 1 | 47 | **100.0%** |

#### ID Switches detected: 1
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  17  frames= 289  purity=0.60
    primary : gt_tid=7 jersey=15 team=right (174fr)
    others  : gt_tid=8 jersey=36 team=right (114fr)
    predicted: jersey=15  team=right
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 48 | 9 | 36 | 4 | 93 | **51.6%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 96 | 0 | 4 | 96 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **4** (across 2 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-143 | 19 | 6 | 0.67 | gt_tid=5 jersey=14 team=left (4fr) | left |
| SNGS-143 | 62 | 120 | 0.7 | gt_tid=9 jersey=33 team=right (84fr) | right |
| SNGS-143 | 90 | 7 | 0.71 | gt_tid=9 jersey=33 team=right (5fr) | right |
| SNGS-144 | 17 | 289 | 0.6 | gt_tid=7 jersey=15 team=right (174fr) | right |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.