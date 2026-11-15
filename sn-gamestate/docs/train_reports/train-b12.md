# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-08 19:10
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-09-08/17-59-45/states/train-b12.pklz`

**Videos**: SNGS-112, SNGS-113, SNGS-114
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-112
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 11 | 2 | 23 | 2 | 36 | **30.6%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        8        31          65       188     1.0
       29        31           5       141     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 41 | 0 | 2 | 41 | **100.0%** |

#### ID Switches detected: 2
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   1  frames= 217  purity=0.56
    primary : gt_tid=8 jersey=9 team=right (122fr)
    others  : gt_tid=10 jersey=25 team=right (94fr)
    predicted: jersey=9  team=right
  pred_tid=  73  frames=   7  purity=0.57
    primary : gt_tid=19 jersey=14 team=right (4fr)
    others  : gt_tid=22 jersey=8 team=left (3fr)
    predicted: jersey=nan  team=left
```

### SNGS-113
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 10 | 1 | 10 | 0 | 21 | **47.6%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       52        24           5       174     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 0 | 46 | 0 | 46 | **0.0%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
        1    left     right       566    1.00
        2    left     right       750    1.00
        3    left     right       208    1.00
        4   right      left       154    1.00
        5    left     right       750    1.00
        6   right      left       412    1.00
        7   right      left       464    1.00
        8   right      left       746    1.00
       10    left     right       313    0.99
       11   right      left        99    0.96
```

### SNGS-114
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 12 | 3 | 14 | 1 | 29 | **41.4%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        3         7           5       447     1.0
        5        33           5        14     1.0
       17         4           5       151     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 39 | 0 | 1 | 39 | **100.0%** |

#### ID Switches detected: 1
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  33  frames= 251  purity=0.67
    primary : gt_tid=2 jersey=31 team=left (169fr)
    others  : gt_tid=22 jersey=9 team=left (82fr)
    predicted: jersey=9  team=left
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 33 | 6 | 47 | 3 | 86 | **38.4%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 80 | 46 | 3 | 126 | **63.5%** |

### ID Switches Summary
Total ID switches detected: **3** (across 2 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-112 | 1 | 217 | 0.56 | gt_tid=8 jersey=9 team=right (122fr) | right |
| SNGS-112 | 73 | 7 | 0.57 | gt_tid=19 jersey=14 team=right (4fr) | left |
| SNGS-114 | 33 | 251 | 0.67 | gt_tid=2 jersey=31 team=left (169fr) | left |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.