# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-09 10:01
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-09-09/08-57-48/states/train-b16.pklz`

**Videos**: SNGS-159, SNGS-160, SNGS-161
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-159
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 16 | 7 | 49 | 3 | 72 | **22.2%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        8         5          14       418    0.87
       29        29           9       105    0.95
       80        22           5       367    0.99
       81        29           5       362    1.00
       86        14           5       346    1.00
       90         9           5       246    1.00
      108        26           5       139    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 74 | 0 | 3 | 74 | **100.0%** |

#### ID Switches detected: 4
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  18  frames=   7  purity=0.67
    primary : gt_tid=4 jersey=5 team=left (4fr)
    others  : gt_tid=12 jersey=14 team=right (2fr)
    predicted: jersey=14  team=right
  pred_tid=  27  frames=  87  purity=0.76
    primary : gt_tid=15 jersey=10 team=right (66fr)
    others  : gt_tid=23 jersey=None team=right (21fr)
    predicted: jersey=10  team=right
  pred_tid=  40  frames=   5  purity=0.50
    primary : gt_tid=7 jersey=29 team=left (1fr)
    others  : gt_tid=6 jersey=23 team=left (1fr)
    predicted: jersey=nan  team=left
  pred_tid=  57  frames= 101  purity=0.64
    primary : gt_tid=14 jersey=22 team=right (63fr)
    others  : gt_tid=23 jersey=None team=right (35fr)
    predicted: jersey=nan  team=right
```

### SNGS-160
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 13 | 2 | 5 | 1 | 20 | **65.0%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       34        29          38       130    0.95
       39        23          25       192    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 36 | 0 | 1 | 36 | **100.0%** |

#### ID Switches detected: 1
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  46  frames=   5  purity=0.60
    primary : gt_tid=2 jersey=9 team=left (3fr)
    others  : gt_tid=21 jersey=22 team=right (2fr)
    predicted: jersey=None  team=left
```

### SNGS-161
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 17 | 5 | 26 | 3 | 48 | **35.4%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        3        20           5        24    1.00
       10        29           5       260    1.00
       34         9           5       427    0.99
       44        56          55       109    1.00
       47         3           5       356    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 48 | 0 | 3 | 48 | **100.0%** |

#### ID Switches detected: 3
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   8  frames=   5  purity=0.75
    primary : gt_tid=19 jersey=25 team=left (3fr)
    others  : gt_tid=18 jersey=20 team=left (1fr)
    predicted: jersey=None  team=left
  pred_tid=  15  frames= 119  purity=0.53
    primary : gt_tid=15 jersey=6 team=left (63fr)
    others  : gt_tid=24 jersey=56 team=left (56fr)
    predicted: jersey=None  team=left
  pred_tid=  56  frames=   9  purity=0.56
    primary : gt_tid=24 jersey=56 team=left (5fr)
    others  : gt_tid=13 jersey=23 team=left (4fr)
    predicted: jersey=None  team=left
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 46 | 14 | 80 | 7 | 140 | **32.9%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 158 | 0 | 7 | 158 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **8** (across 3 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-159 | 18 | 7 | 0.67 | gt_tid=4 jersey=5 team=left (4fr) | right |
| SNGS-159 | 27 | 87 | 0.76 | gt_tid=15 jersey=10 team=right (66fr) | right |
| SNGS-159 | 40 | 5 | 0.5 | gt_tid=7 jersey=29 team=left (1fr) | left |
| SNGS-159 | 57 | 101 | 0.64 | gt_tid=14 jersey=22 team=right (63fr) | right |
| SNGS-160 | 46 | 5 | 0.6 | gt_tid=2 jersey=9 team=left (3fr) | left |
| SNGS-161 | 8 | 5 | 0.75 | gt_tid=19 jersey=25 team=left (3fr) | left |
| SNGS-161 | 15 | 119 | 0.53 | gt_tid=15 jersey=6 team=left (63fr) | left |
| SNGS-161 | 56 | 9 | 0.56 | gt_tid=24 jersey=56 team=left (5fr) | left |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.