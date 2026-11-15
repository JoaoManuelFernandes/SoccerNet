# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-07-04 22:26
**State file**: `/home/joao/soccernet/outputs/sn-gamestate/2026-07-02/07-54-35/states/gshota-valid-V01.pklz`

**Videos**: SNGS-021, SNGS-022, SNGS-023
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-021
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 10 | 4 | 9 | 1 | 23 | **43.5%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       29        19           5       461     1.0
       42        10          56        27     1.0
       73        13           5       217     1.0
       76        14          28       143     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 51 | 0 | 2 | 51 | **100.0%** |

#### ID Switches detected: 2
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   8  frames= 480  purity=0.62
    primary : gt_tid=14 jersey=None team=left (298fr)
    others  : gt_tid=15 jersey=None team=left (182fr)
    predicted: jersey=5  team=left
  pred_tid=  14  frames= 234  purity=0.65
    primary : gt_tid=5 jersey=7 team=right (152fr)
    others  : gt_tid=6 jersey=None team=right (82fr)
    predicted: jersey=7  team=right
```

### SNGS-022
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 16 | 7 | 19 | 3 | 42 | **38.1%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        6        13           5        13    1.00
       36        10           1       282    0.98
       40        17           5       249    0.99
       55        10           5        96    1.00
       61        15           5       363    1.00
       69        11           5       171    0.99
       80        14          54       121    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 50 | 0 | 3 | 50 | **100.0%** |

#### ID Switches detected: 3
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  33  frames=   7  purity=0.50
    primary : gt_tid=9 jersey=3 team=left (3fr)
    others  : gt_tid=15 jersey=15 team=left (2fr) | gt_tid=5 jersey=22 team=right (1fr)
    predicted: jersey=None  team=left
  pred_tid=  42  frames=  15  purity=0.47
    primary : gt_tid=22 jersey=8 team=right (7fr)
    others  : gt_tid=11 jersey=10 team=left (3fr) | gt_tid=14 jersey=11 team=left (3fr)
    predicted: jersey=None  team=right
  pred_tid=  81  frames=   7  purity=0.71
    primary : gt_tid=14 jersey=11 team=left (5fr)
    others  : gt_tid=20 jersey=14 team=right (2fr)
    predicted: jersey=None  team=left
```

### SNGS-023
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 14 | 3 | 20 | 7 | 37 | **37.8%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       16        23           1       480    1.00
       25         4           5       302    0.99
       42        14           5       174    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 39 | 0 | 7 | 39 | **100.0%** |

#### ID Switches detected: 7
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   6  frames= 736  purity=0.52
    primary : gt_tid=1 jersey=17 team=left (382fr)
    others  : gt_tid=14 jersey=11 team=right (353fr) | gt_tid=11 jersey=3 team=left (1fr)
    predicted: jersey=17  team=left
  pred_tid=  14  frames= 323  purity=0.80
    primary : gt_tid=18 jersey=9 team=right (257fr)
    others  : gt_tid=16 jersey=10 team=right (59fr) | gt_tid=6 jersey=14 team=left (5fr)
    predicted: jersey=None  team=right
  pred_tid=  17  frames= 520  purity=0.54
    primary : gt_tid=18 jersey=9 team=right (283fr)
    others  : gt_tid=10 jersey=11 team=left (223fr) | gt_tid=6 jersey=14 team=left (10fr)
    predicted: jersey=5  team=right
  pred_tid=  35  frames=  28  purity=0.75
    primary : gt_tid=19 jersey=23 team=right (21fr)
    others  : gt_tid=6 jersey=14 team=left (4fr) | gt_tid=16 jersey=10 team=right (3fr)
    predicted: jersey=None  team=right
  pred_tid=  38  frames= 203  purity=0.55
    primary : gt_tid=20 jersey=4 team=right (112fr)
    others  : gt_tid=9 jersey=15 team=left (91fr)
    predicted: jersey=15  team=left
  pred_tid=  39  frames=  19  purity=0.79
    primary : gt_tid=16 jersey=10 team=right (15fr)
    others  : gt_tid=18 jersey=9 team=right (3fr) | gt_tid=6 jersey=14 team=left (1fr)
    predicted: jersey=None  team=right
  pred_tid=  46  frames= 150  purity=0.78
    primary : gt_tid=9 jersey=15 team=left (116fr)
    others  : gt_tid=20 jersey=4 team=right (32fr)
    predicted: jersey=5  team=left
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 40 | 14 | 48 | 11 | 102 | **39.2%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 140 | 0 | 12 | 140 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **12** (across 3 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-021 | 8 | 480 | 0.62 | gt_tid=14 jersey=None team=left (298fr) | left |
| SNGS-021 | 14 | 234 | 0.65 | gt_tid=5 jersey=7 team=right (152fr) | right |
| SNGS-022 | 33 | 7 | 0.5 | gt_tid=9 jersey=3 team=left (3fr) | left |
| SNGS-022 | 42 | 15 | 0.47 | gt_tid=22 jersey=8 team=right (7fr) | right |
| SNGS-022 | 81 | 7 | 0.71 | gt_tid=14 jersey=11 team=left (5fr) | left |
| SNGS-023 | 6 | 736 | 0.52 | gt_tid=1 jersey=17 team=left (382fr) | left |
| SNGS-023 | 14 | 323 | 0.8 | gt_tid=18 jersey=9 team=right (257fr) | right |
| SNGS-023 | 17 | 520 | 0.54 | gt_tid=18 jersey=9 team=right (283fr) | right |
| SNGS-023 | 35 | 28 | 0.75 | gt_tid=19 jersey=23 team=right (21fr) | right |
| SNGS-023 | 38 | 203 | 0.55 | gt_tid=20 jersey=4 team=right (112fr) | left |
| SNGS-023 | 39 | 19 | 0.79 | gt_tid=16 jersey=10 team=right (15fr) | right |
| SNGS-023 | 46 | 150 | 0.78 | gt_tid=9 jersey=15 team=left (116fr) | left |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.