# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-08 21:26
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-09-08/20-21-00/states/train-b14.pklz`

**Videos**: SNGS-153, SNGS-154, SNGS-155
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-153
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 17 | 5 | 22 | 1 | 44 | **38.6%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       27        10           6       129    0.99
       38        30           5       397    1.00
       41         6           5       122    1.00
       43        20           5        42    0.98
       52        25           5        66    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 49 | 0 | 1 | 49 | **100.0%** |

#### ID Switches detected: 1
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  70  frames=  22  purity=0.71
    primary : gt_tid=17 jersey=23 team=right (15fr)
    others  : gt_tid=20 jersey=None team=left (6fr)
    predicted: jersey=None  team=right
```

### SNGS-154
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 23 | 8 | 31 | 11 | 62 | **37.1%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       28        22           5        41    0.85
       36        26           5       154    0.99
       66        22           5       142    0.98
       72        25           5       248    0.97
       80        40           5       223    0.97
      103         6           5       198    1.00
      111        30           3       161    0.99
      122         5          19        17    0.82
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 72 | 0 | 11 | 72 | **100.0%** |

#### ID Switches detected: 11
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   5  frames= 232  purity=0.52
    primary : gt_tid=7 jersey=29 team=left (120fr)
    others  : gt_tid=5 jersey=10 team=left (104fr) | gt_tid=11 jersey=9 team=right (5fr)
    predicted: jersey=10  team=left
  pred_tid=  14  frames= 236  purity=0.71
    primary : gt_tid=9 jersey=5 team=left (166fr)
    others  : gt_tid=2 jersey=40 team=left (49fr) | gt_tid=13 jersey=20 team=right (12fr)
    predicted: jersey=20  team=left
  pred_tid=  20  frames=   6  purity=0.50
    primary : gt_tid=1 jersey=11 team=left (3fr)
    others  : gt_tid=2 jersey=40 team=left (2fr) | gt_tid=6 jersey=9 team=left (1fr)
    predicted: jersey=nan  team=left
  pred_tid=  33  frames=   7  purity=0.43
    primary : gt_tid=8 jersey=22 team=left (3fr)
    others  : gt_tid=7 jersey=29 team=left (1fr) | gt_tid=20 jersey=25 team=right (1fr)
    predicted: jersey=nan  team=left
  pred_tid=  60  frames=   6  purity=0.60
    primary : gt_tid=22 jersey=26 team=right (3fr)
    others  : gt_tid=6 jersey=9 team=left (1fr) | gt_tid=15 jersey=6 team=right (1fr)
    predicted: jersey=nan  team=right
  pred_tid=  62  frames=   9  purity=0.78
    primary : gt_tid=1 jersey=11 team=left (7fr)
    others  : gt_tid=4 jersey=None team=None (1fr) | gt_tid=13 jersey=20 team=right (1fr)
    predicted: jersey=nan  team=left
  pred_tid=  69  frames=   5  purity=0.60
    primary : gt_tid=15 jersey=6 team=right (3fr)
    others  : gt_tid=22 jersey=26 team=right (2fr)
    predicted: jersey=nan  team=right
  pred_tid=  96  frames=   6  purity=0.50
    primary : gt_tid=1 jersey=11 team=left (3fr)
    others  : gt_tid=13 jersey=20 team=right (2fr) | gt_tid=12 jersey=55 team=right (1fr)
    predicted: jersey=nan  team=right
  pred_tid= 108  frames=  97  purity=0.77
    primary : gt_tid=15 jersey=6 team=right (75fr)
    others  : gt_tid=22 jersey=26 team=right (22fr)
    predicted: jersey=nan  team=right
  pred_tid= 124  frames=  10  purity=0.60
    primary : gt_tid=11 jersey=9 team=right (6fr)
    others  : gt_tid=13 jersey=20 team=right (2fr) | gt_tid=7 jersey=29 team=left (1fr)
    predicted: jersey=nan  team=right
  pred_tid= 130  frames=  14  purity=0.79
    primary : gt_tid=7 jersey=29 team=left (11fr)
    others  : gt_tid=13 jersey=20 team=right (3fr)
    predicted: jersey=nan  team=left
```

### SNGS-155
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 17 | 4 | 9 | 0 | 30 | **56.7%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        3        40           5       482    1.00
       14        29           5       742    1.00
       18        26          25       314    1.00
       24         9           5       487    0.99
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 40 | 0 | 0 | 40 | **100.0%** |

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 57 | 17 | 62 | 12 | 136 | **41.9%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 161 | 0 | 12 | 161 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **12** (across 2 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-153 | 70 | 22 | 0.71 | gt_tid=17 jersey=23 team=right (15fr) | right |
| SNGS-154 | 5 | 232 | 0.52 | gt_tid=7 jersey=29 team=left (120fr) | left |
| SNGS-154 | 14 | 236 | 0.71 | gt_tid=9 jersey=5 team=left (166fr) | left |
| SNGS-154 | 20 | 6 | 0.5 | gt_tid=1 jersey=11 team=left (3fr) | left |
| SNGS-154 | 33 | 7 | 0.43 | gt_tid=8 jersey=22 team=left (3fr) | left |
| SNGS-154 | 60 | 6 | 0.6 | gt_tid=22 jersey=26 team=right (3fr) | right |
| SNGS-154 | 62 | 9 | 0.78 | gt_tid=1 jersey=11 team=left (7fr) | left |
| SNGS-154 | 69 | 5 | 0.6 | gt_tid=15 jersey=6 team=right (3fr) | right |
| SNGS-154 | 96 | 6 | 0.5 | gt_tid=1 jersey=11 team=left (3fr) | right |
| SNGS-154 | 108 | 97 | 0.77 | gt_tid=15 jersey=6 team=right (75fr) | right |
| SNGS-154 | 124 | 10 | 0.6 | gt_tid=11 jersey=9 team=right (6fr) | right |
| SNGS-154 | 130 | 14 | 0.79 | gt_tid=7 jersey=29 team=left (11fr) | left |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.