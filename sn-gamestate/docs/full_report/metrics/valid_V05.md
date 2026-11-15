# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-07-04 22:34
**State file**: `/home/joao/soccernet/outputs/sn-gamestate/2026-07-02/20-13-05/states/gshota-valid-V05.pklz`

**Videos**: SNGS-033, SNGS-034, SNGS-035
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-033
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 8 | 3 | 14 | 0 | 25 | **32.0%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       67        26           6       118    0.96
       98        26          25       199    0.98
      126        20           5       156    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 42 | 11 | 2 | 53 | **79.2%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
        6    left     right        85    0.82
       14   right      left        32    1.00
       36    left     right        25    0.96
       41    left     right        19    1.00
       86   right      left        51    1.00
       89    left     right        11    1.00
      115    left     right       171    0.98
      114    left     right        31    0.97
      104    left     right       166    1.00
      127    left     right       128    1.00
```

#### ID Switches detected: 2
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  18  frames= 109  purity=0.76
    primary : gt_tid=11 jersey=None team=left (83fr)
    others  : gt_tid=23 jersey=8 team=left (24fr) | gt_tid=15 jersey=42 team=right (1fr)
    predicted: jersey=None  team=left
  pred_tid=  83  frames= 418  purity=0.58
    primary : gt_tid=6 jersey=None team=right (236fr)
    others  : gt_tid=12 jersey=None team=right (174fr)
    predicted: jersey=5  team=right
```

### SNGS-034
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 22 | 3 | 13 | 10 | 38 | **57.9%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       19        17          20        15    0.87
       43        10          11        89    0.98
       50         8           5       126    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 37 | 1 | 10 | 38 | **97.4%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
       19    left     right        15    0.87
```

#### ID Switches detected: 10
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   5  frames= 457  purity=0.55
    primary : gt_tid=8 jersey=42 team=right (251fr)
    others  : gt_tid=6 jersey=15 team=right (164fr) | gt_tid=15 jersey=17 team=left (39fr)
    predicted: jersey=42  team=right
  pred_tid=  10  frames=  67  purity=0.79
    primary : gt_tid=8 jersey=42 team=right (53fr)
    others  : gt_tid=9 jersey=19 team=right (9fr) | gt_tid=17 jersey=None team=None (4fr)
    predicted: jersey=None  team=right
  pred_tid=  13  frames= 529  purity=0.68
    primary : gt_tid=16 jersey=26 team=left (358fr)
    others  : gt_tid=12 jersey=23 team=left (168fr) | gt_tid=11 jersey=11 team=left (3fr)
    predicted: jersey=23  team=left
  pred_tid=  25  frames= 200  purity=0.55
    primary : gt_tid=4 jersey=3 team=right (110fr)
    others  : gt_tid=9 jersey=19 team=right (60fr) | gt_tid=8 jersey=42 team=right (27fr)
    predicted: jersey=5  team=right
  pred_tid=  26  frames= 362  purity=0.64
    primary : gt_tid=20 jersey=10 team=right (230fr)
    others  : gt_tid=13 jersey=22 team=left (128fr) | gt_tid=11 jersey=11 team=left (2fr)
    predicted: jersey=22  team=left
  pred_tid=  28  frames=  58  purity=0.64
    primary : gt_tid=8 jersey=42 team=right (37fr)
    others  : gt_tid=9 jersey=19 team=right (21fr)
    predicted: jersey=None  team=right
  pred_tid=  33  frames=  92  purity=0.77
    primary : gt_tid=9 jersey=19 team=right (71fr)
    others  : gt_tid=6 jersey=15 team=right (16fr) | gt_tid=20 jersey=10 team=right (3fr)
    predicted: jersey=None  team=right
  pred_tid=  35  frames=  92  purity=0.75
    primary : gt_tid=7 jersey=17 team=right (69fr)
    others  : gt_tid=20 jersey=10 team=right (10fr) | gt_tid=9 jersey=19 team=right (10fr)
    predicted: jersey=17  team=right
  pred_tid=  37  frames= 235  purity=0.63
    primary : gt_tid=8 jersey=42 team=right (147fr)
    others  : gt_tid=15 jersey=17 team=left (86fr) | gt_tid=7 jersey=17 team=right (1fr)
    predicted: jersey=17  team=right
  pred_tid=  44  frames= 141  purity=0.45
    primary : gt_tid=9 jersey=19 team=right (64fr)
    others  : gt_tid=20 jersey=10 team=right (64fr) | gt_tid=6 jersey=15 team=right (13fr)
    predicted: jersey=5  team=right
```

### SNGS-035
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 14 | 2 | 17 | 0 | 33 | **42.4%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        4         6          26       369    1.00
       86         5          19        75    0.99
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 53 | 4 | 0 | 57 | **93.0%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
       29    left     right        35    1.00
       35    left     right        16    1.00
       56    left     right       357    1.00
       86    left     right        75    0.99
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 44 | 8 | 44 | 10 | 96 | **45.8%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 132 | 16 | 12 | 148 | **89.2%** |

### ID Switches Summary
Total ID switches detected: **12** (across 2 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-033 | 18 | 109 | 0.76 | gt_tid=11 jersey=None team=left (83fr) | left |
| SNGS-033 | 83 | 418 | 0.58 | gt_tid=6 jersey=None team=right (236fr) | right |
| SNGS-034 | 5 | 457 | 0.55 | gt_tid=8 jersey=42 team=right (251fr) | right |
| SNGS-034 | 10 | 67 | 0.79 | gt_tid=8 jersey=42 team=right (53fr) | right |
| SNGS-034 | 13 | 529 | 0.68 | gt_tid=16 jersey=26 team=left (358fr) | left |
| SNGS-034 | 25 | 200 | 0.55 | gt_tid=4 jersey=3 team=right (110fr) | right |
| SNGS-034 | 26 | 362 | 0.64 | gt_tid=20 jersey=10 team=right (230fr) | left |
| SNGS-034 | 28 | 58 | 0.64 | gt_tid=8 jersey=42 team=right (37fr) | right |
| SNGS-034 | 33 | 92 | 0.77 | gt_tid=9 jersey=19 team=right (71fr) | right |
| SNGS-034 | 35 | 92 | 0.75 | gt_tid=7 jersey=17 team=right (69fr) | right |
| SNGS-034 | 37 | 235 | 0.63 | gt_tid=8 jersey=42 team=right (147fr) | right |
| SNGS-034 | 44 | 141 | 0.45 | gt_tid=9 jersey=19 team=right (64fr) | right |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.