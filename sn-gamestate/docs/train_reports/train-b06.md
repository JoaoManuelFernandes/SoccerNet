# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-08 12:45
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-09-08/11-23-11/states/train-b06.pklz`

**Videos**: SNGS-075, SNGS-076, SNGS-077
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-075
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 17 | 8 | 15 | 11 | 40 | **42.5%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        9        24          25       488    1.00
       15         9          14       601    0.91
       21         8           5        63    0.97
       28         3           5       376    0.93
       40        25           5       394    0.96
       48        15           5        98    1.00
       54         8           1        92    1.00
       55        28           1       353    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 43 | 1 | 11 | 44 | **97.7%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
       14    left     right       189     1.0
```

#### ID Switches detected: 11
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   7  frames= 729  purity=0.79
    primary : gt_tid=7 jersey=14 team=left (577fr)
    others  : gt_tid=14 jersey=11 team=right (151fr) | gt_tid=12 jersey=29 team=right (1fr)
    predicted: jersey=14  team=left
  pred_tid=  25  frames=  79  purity=0.66
    primary : gt_tid=13 jersey=25 team=right (52fr)
    others  : gt_tid=5 jersey=27 team=left (25fr) | gt_tid=19 jersey=8 team=right (1fr)
    predicted: jersey=27  team=left
  pred_tid=  26  frames=  45  purity=0.76
    primary : gt_tid=14 jersey=11 team=right (34fr)
    others  : gt_tid=18 jersey=3 team=right (8fr) | gt_tid=13 jersey=25 team=right (3fr)
    predicted: jersey=11  team=right
  pred_tid=  30  frames= 247  purity=0.56
    primary : gt_tid=17 jersey=6 team=right (137fr)
    others  : gt_tid=2 jersey=5 team=left (102fr) | gt_tid=20 jersey=28 team=right (7fr)
    predicted: jersey=5  team=left
  pred_tid=  31  frames=  23  purity=0.74
    primary : gt_tid=5 jersey=27 team=left (17fr)
    others  : gt_tid=2 jersey=5 team=left (4fr) | gt_tid=6 jersey=36 team=left (1fr)
    predicted: jersey=nan  team=left
  pred_tid=  32  frames=  61  purity=0.79
    primary : gt_tid=14 jersey=11 team=right (48fr)
    others  : gt_tid=16 jersey=4 team=right (12fr) | gt_tid=1 jersey=15 team=left (1fr)
    predicted: jersey=nan  team=right
  pred_tid=  36  frames=   8  purity=0.38
    primary : gt_tid=5 jersey=27 team=left (3fr)
    others  : gt_tid=2 jersey=5 team=left (3fr) | gt_tid=19 jersey=8 team=right (1fr)
    predicted: jersey=nan  team=left
  pred_tid=  41  frames= 405  purity=0.63
    primary : gt_tid=5 jersey=27 team=left (254fr)
    others  : gt_tid=6 jersey=36 team=left (135fr) | gt_tid=2 jersey=5 team=left (13fr)
    predicted: jersey=nan  team=left
  pred_tid=  45  frames=  27  purity=0.69
    primary : gt_tid=6 jersey=36 team=left (18fr)
    others  : gt_tid=13 jersey=25 team=right (6fr) | gt_tid=18 jersey=3 team=right (2fr)
    predicted: jersey=nan  team=left
  pred_tid=  66  frames=  35  purity=0.74
    primary : gt_tid=18 jersey=3 team=right (26fr)
    others  : gt_tid=16 jersey=4 team=right (8fr) | gt_tid=17 jersey=6 team=right (1fr)
    predicted: jersey=nan  team=right
  pred_tid=  73  frames= 120  purity=0.56
    primary : gt_tid=21 jersey=9 team=right (65fr)
    others  : gt_tid=15 jersey=24 team=right (51fr) | gt_tid=18 jersey=3 team=right (1fr)
    predicted: jersey=nan  team=right
```

### SNGS-076
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 5 | 3 | 7 | 0 | 15 | **33.3%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       14        24           5       321    1.00
       28        25           5       360    1.00
       40        25           5       136    0.99
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 30 | 0 | 1 | 30 | **100.0%** |

#### ID Switches detected: 2
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  29  frames=   6  purity=0.67
    primary : gt_tid=15 jersey=None team=right (4fr)
    others  : gt_tid=3 jersey=28 team=right (2fr)
    predicted: jersey=None  team=right
  pred_tid=  26  frames= 268  purity=0.67
    primary : gt_tid=4 jersey=24 team=right (10fr)
    others  : gt_tid=10 jersey=None team=None (5fr)
    predicted: jersey=None  team=left
```

### SNGS-077
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 8 | 6 | 11 | 0 | 25 | **32.0%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        6         6          56       525    1.00
       23        11           5       539    1.00
       33        36          23       372    0.98
       36         8          18        98    1.00
       38         9           5       391    1.00
       50        11          17        40    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 38 | 1 | 0 | 39 | **97.4%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
       42    left     right         5     1.0
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 30 | 17 | 33 | 11 | 80 | **37.5%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 111 | 2 | 12 | 113 | **98.2%** |

### ID Switches Summary
Total ID switches detected: **13** (across 2 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-075 | 7 | 729 | 0.79 | gt_tid=7 jersey=14 team=left (577fr) | left |
| SNGS-075 | 25 | 79 | 0.66 | gt_tid=13 jersey=25 team=right (52fr) | left |
| SNGS-075 | 26 | 45 | 0.76 | gt_tid=14 jersey=11 team=right (34fr) | right |
| SNGS-075 | 30 | 247 | 0.56 | gt_tid=17 jersey=6 team=right (137fr) | left |
| SNGS-075 | 31 | 23 | 0.74 | gt_tid=5 jersey=27 team=left (17fr) | left |
| SNGS-075 | 32 | 61 | 0.79 | gt_tid=14 jersey=11 team=right (48fr) | right |
| SNGS-075 | 36 | 8 | 0.38 | gt_tid=5 jersey=27 team=left (3fr) | left |
| SNGS-075 | 41 | 405 | 0.63 | gt_tid=5 jersey=27 team=left (254fr) | left |
| SNGS-075 | 45 | 27 | 0.69 | gt_tid=6 jersey=36 team=left (18fr) | left |
| SNGS-075 | 66 | 35 | 0.74 | gt_tid=18 jersey=3 team=right (26fr) | right |
| SNGS-075 | 73 | 120 | 0.56 | gt_tid=21 jersey=9 team=right (65fr) | right |
| SNGS-076 | 29 | 6 | 0.67 | gt_tid=15 jersey=None team=right (4fr) | right |
| SNGS-076 | 26 | 268 | 0.67 | gt_tid=4 jersey=24 team=right (10fr) | left |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.