# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-07-04 22:52
**State file**: `/home/joao/soccernet/outputs/sn-gamestate/2026-07-04/10-14-25/states/gshota-valid-V13.pklz`

**Videos**: SNGS-057, SNGS-058, SNGS-081
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-057
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 12 | 2 | 19 | 0 | 33 | **36.4%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       57        15           5        81     1.0
       75        15           5        73     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 15 | 18 | 0 | 33 | **45.5%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
        2   right      left       283    1.00
        5   right      left       571    0.81
        6   right      left       399    1.00
       10   right      left        90    1.00
       14   right      left       286    1.00
       16   right      left        13    0.92
       20   right      left        90    0.99
       21   right      left       101    0.97
       31   right      left        17    0.94
       42   right      left        17    1.00
```

### SNGS-058
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 8 | 5 | 17 | 3 | 30 | **26.7%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        5        27          25       174    1.00
       35        27           3       177    0.99
       47        15           5       202    0.98
       90        43           5        95    1.00
      136        33           6        67    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 17 | 13 | 3 | 30 | **56.7%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
        7   right      left       157    1.00
        9   right      left        53    1.00
       17   right      left        90    0.98
       19   right      left        10    1.00
       47   right      left       202    0.98
       58   right      left        59    1.00
       62   right      left       167    0.97
       71   right      left        18    1.00
       80   right      left       217    1.00
       81   right      left         5    1.00
```

#### ID Switches detected: 3
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  10  frames=  12  purity=0.75
    primary : gt_tid=8 jersey=14 team=right (9fr)
    others  : gt_tid=12 jersey=32 team=right (2fr) | gt_tid=15 jersey=33 team=right (1fr)
    predicted: jersey=None  team=left
  pred_tid=  18  frames=  19  purity=0.44
    primary : gt_tid=16 jersey=43 team=right (7fr)
    others  : gt_tid=18 jersey=20 team=left (6fr) | gt_tid=23 jersey=3 team=right (1fr)
    predicted: jersey=None  team=right
  pred_tid=  31  frames=   5  purity=0.60
    primary : gt_tid=5 jersey=11 team=right (3fr)
    others  : gt_tid=6 jersey=27 team=right (2fr)
    predicted: jersey=None  team=left
```

### SNGS-081
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 26 | 8 | 20 | 3 | 54 | **48.1%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        1        27           5        23    1.00
       20        15          16       209    1.00
       37         3           1        96    1.00
       55        10           5        23    1.00
       59         4           6       244    1.00
       62        10           5        46    0.98
       82        59          69       170    1.00
       88        20          35        10    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 9 | 45 | 3 | 54 | **16.7%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
        1    left     right        23    1.00
        2   right      left       422    1.00
        5    left     right        57    1.00
        6   right      left       306    1.00
        8   right      left        59    0.91
       10    left     right        92    1.00
       13    left     right       180    0.99
       14    left     right       285    1.00
       16   right      left       345    1.00
       12    left     right        49    0.96
```

#### ID Switches detected: 3
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   4  frames= 569  purity=0.56
    primary : gt_tid=5 jersey=13 team=right (320fr)
    others  : gt_tid=6 jersey=44 team=left (249fr)
    predicted: jersey=13  team=left
  pred_tid=   9  frames= 164  purity=0.79
    primary : gt_tid=8 jersey=3 team=right (130fr)
    others  : gt_tid=10 jersey=5 team=right (34fr)
    predicted: jersey=None  team=left
  pred_tid=  75  frames= 111  purity=0.55
    primary : gt_tid=8 jersey=3 team=right (61fr)
    others  : gt_tid=1 jersey=27 team=left (50fr)
    predicted: jersey=27  team=right
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 46 | 15 | 56 | 6 | 117 | **39.3%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 41 | 76 | 6 | 117 | **35.0%** |

### ID Switches Summary
Total ID switches detected: **6** (across 2 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-058 | 10 | 12 | 0.75 | gt_tid=8 jersey=14 team=right (9fr) | left |
| SNGS-058 | 18 | 19 | 0.44 | gt_tid=16 jersey=43 team=right (7fr) | right |
| SNGS-058 | 31 | 5 | 0.6 | gt_tid=5 jersey=11 team=right (3fr) | left |
| SNGS-081 | 4 | 569 | 0.56 | gt_tid=5 jersey=13 team=right (320fr) | left |
| SNGS-081 | 9 | 164 | 0.79 | gt_tid=8 jersey=3 team=right (130fr) | left |
| SNGS-081 | 75 | 111 | 0.55 | gt_tid=8 jersey=3 team=right (61fr) | right |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.