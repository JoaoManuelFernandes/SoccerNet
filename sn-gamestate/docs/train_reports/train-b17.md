# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-09 11:16
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-09-09/10-01-32/states/train-b17.pklz`

**Videos**: SNGS-162, SNGS-163, SNGS-164
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-162
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 14 | 6 | 26 | 0 | 46 | **30.4%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       24        10           5       340    0.84
       42        26           5        83    1.00
       64        26           5        79    1.00
      136        10           5       159    1.00
      143        28          10       126    1.00
      152         9           5       108    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 80 | 0 | 7 | 80 | **100.0%** |

#### ID Switches detected: 7
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   9  frames= 373  purity=0.54
    primary : gt_tid=22 jersey=None team=right (200fr)
    others  : gt_tid=13 jersey=9 team=right (169fr) | gt_tid=20 jersey=None team=right (1fr)
    predicted: jersey=None  team=right
  pred_tid=  15  frames=   6  purity=0.50
    primary : gt_tid=21 jersey=None team=right (3fr)
    others  : gt_tid=22 jersey=None team=right (3fr)
    predicted: jersey=None  team=right
  pred_tid=  18  frames=  24  purity=0.58
    primary : gt_tid=20 jersey=None team=right (14fr)
    others  : gt_tid=21 jersey=None team=right (10fr)
    predicted: jersey=None  team=right
  pred_tid=  25  frames=  13  purity=0.77
    primary : gt_tid=21 jersey=None team=right (10fr)
    others  : gt_tid=9 jersey=10 team=left (1fr) | gt_tid=19 jersey=None team=right (1fr)
    predicted: jersey=None  team=right
  pred_tid=  34  frames=   5  purity=0.50
    primary : gt_tid=15 jersey=None team=right (2fr)
    others  : gt_tid=8 jersey=55 team=left (2fr)
    predicted: jersey=None  team=right
  pred_tid=  41  frames=  12  purity=0.58
    primary : gt_tid=21 jersey=None team=right (7fr)
    others  : gt_tid=19 jersey=None team=right (5fr)
    predicted: jersey=None  team=right
  pred_tid= 124  frames=   5  purity=0.50
    primary : gt_tid=20 jersey=None team=right (2fr)
    others  : gt_tid=22 jersey=None team=right (2fr)
    predicted: jersey=None  team=right
```

### SNGS-163
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 21 | 5 | 25 | 4 | 51 | **41.2%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        7         9           5       150    0.97
        9        55          56       282    0.98
       90         6           8       183    0.99
       93        30           6       166    0.99
      103         9           5       114    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 51 | 0 | 4 | 51 | **100.0%** |

#### ID Switches detected: 4
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  29  frames=   6  purity=0.75
    primary : gt_tid=11 jersey=25 team=left (3fr)
    others  : gt_tid=10 jersey=26 team=left (1fr)
    predicted: jersey=None  team=left
  pred_tid=  59  frames=   7  purity=0.71
    primary : gt_tid=13 jersey=8 team=left (5fr)
    others  : gt_tid=14 jersey=3 team=left (2fr)
    predicted: jersey=None  team=left
  pred_tid=  60  frames=   6  purity=0.40
    primary : gt_tid=8 jersey=6 team=right (2fr)
    others  : gt_tid=10 jersey=26 team=left (1fr) | gt_tid=4 jersey=9 team=right (1fr)
    predicted: jersey=None  team=right
  pred_tid=  78  frames= 145  purity=0.50
    primary : gt_tid=13 jersey=8 team=left (73fr)
    others  : gt_tid=14 jersey=3 team=left (72fr)
    predicted: jersey=None  team=left
```

### SNGS-164
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 18 | 5 | 20 | 1 | 43 | **41.9%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        9        10           5       159    1.00
       33        29           5        74    0.95
       51        10           5        45    1.00
       53        29           5        52    1.00
       75        29           5       231    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 57 | 0 | 1 | 57 | **100.0%** |

#### ID Switches detected: 1
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  61  frames= 139  purity=0.79
    primary : gt_tid=20 jersey=29 team=right (110fr)
    others  : gt_tid=8 jersey=26 team=left (29fr)
    predicted: jersey=None  team=right
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 53 | 16 | 71 | 5 | 140 | **37.9%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 188 | 0 | 12 | 188 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **12** (across 3 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-162 | 9 | 373 | 0.54 | gt_tid=22 jersey=None team=right (200fr) | right |
| SNGS-162 | 15 | 6 | 0.5 | gt_tid=21 jersey=None team=right (3fr) | right |
| SNGS-162 | 18 | 24 | 0.58 | gt_tid=20 jersey=None team=right (14fr) | right |
| SNGS-162 | 25 | 13 | 0.77 | gt_tid=21 jersey=None team=right (10fr) | right |
| SNGS-162 | 34 | 5 | 0.5 | gt_tid=15 jersey=None team=right (2fr) | right |
| SNGS-162 | 41 | 12 | 0.58 | gt_tid=21 jersey=None team=right (7fr) | right |
| SNGS-162 | 124 | 5 | 0.5 | gt_tid=20 jersey=None team=right (2fr) | right |
| SNGS-163 | 29 | 6 | 0.75 | gt_tid=11 jersey=25 team=left (3fr) | left |
| SNGS-163 | 59 | 7 | 0.71 | gt_tid=13 jersey=8 team=left (5fr) | left |
| SNGS-163 | 60 | 6 | 0.4 | gt_tid=8 jersey=6 team=right (2fr) | right |
| SNGS-163 | 78 | 145 | 0.5 | gt_tid=13 jersey=8 team=left (73fr) | left |
| SNGS-164 | 61 | 139 | 0.79 | gt_tid=20 jersey=29 team=right (110fr) | right |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.