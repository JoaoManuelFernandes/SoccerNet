# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-08 08:53
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-09-08/07-44-04/states/train-b03.pklz`

**Videos**: SNGS-066, SNGS-067, SNGS-068
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-066
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 13 | 7 | 38 | 10 | 58 | **22.4%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       11        25          53       162    0.96
       15        24          15       708    1.00
       20         9           5       267    1.00
       61        44           8       249    1.00
       68        11           1       360    0.99
       76        25           5       244    1.00
       84        27          21        43    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 59 | 0 | 10 | 59 | **100.0%** |

#### ID Switches detected: 10
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  10  frames= 257  purity=0.67
    primary : gt_tid=20 jersey=10 team=left (171fr)
    others  : gt_tid=14 jersey=11 team=left (78fr) | gt_tid=19 jersey=9 team=left (4fr)
    predicted: jersey=nan  team=left
  pred_tid=  13  frames= 184  purity=0.54
    primary : gt_tid=6 jersey=5 team=right (100fr)
    others  : gt_tid=3 jersey=27 team=right (64fr) | gt_tid=18 jersey=8 team=left (14fr)
    predicted: jersey=5  team=right
  pred_tid=  18  frames= 253  purity=0.73
    primary : gt_tid=16 jersey=3 team=left (183fr)
    others  : gt_tid=7 jersey=15 team=right (52fr) | gt_tid=17 jersey=25 team=left (10fr)
    predicted: jersey=55  team=left
  pred_tid=  19  frames= 200  purity=0.72
    primary : gt_tid=6 jersey=5 team=right (144fr)
    others  : gt_tid=15 jersey=None team=left (55fr) | gt_tid=4 jersey=8 team=right (1fr)
    predicted: jersey=5  team=right
  pred_tid=  24  frames= 508  purity=0.78
    primary : gt_tid=18 jersey=8 team=left (395fr)
    others  : gt_tid=14 jersey=11 team=left (94fr) | gt_tid=15 jersey=None team=left (14fr)
    predicted: jersey=8  team=left
  pred_tid=  25  frames= 130  purity=0.65
    primary : gt_tid=17 jersey=25 team=left (85fr)
    others  : gt_tid=7 jersey=15 team=right (30fr) | gt_tid=18 jersey=8 team=left (12fr)
    predicted: jersey=15  team=left
  pred_tid=  34  frames=  13  purity=0.77
    primary : gt_tid=3 jersey=27 team=right (10fr)
    others  : gt_tid=21 jersey=13 team=left (2fr) | gt_tid=7 jersey=15 team=right (1fr)
    predicted: jersey=nan  team=right
  pred_tid=  46  frames=  10  purity=0.50
    primary : gt_tid=18 jersey=8 team=left (5fr)
    others  : gt_tid=14 jersey=11 team=left (5fr)
    predicted: jersey=nan  team=left
  pred_tid=  51  frames=   8  purity=0.62
    primary : gt_tid=6 jersey=5 team=right (5fr)
    others  : gt_tid=19 jersey=9 team=left (2fr) | gt_tid=18 jersey=8 team=left (1fr)
    predicted: jersey=nan  team=left
  pred_tid=  69  frames=  11  purity=0.73
    primary : gt_tid=8 jersey=36 team=right (8fr)
    others  : gt_tid=4 jersey=8 team=right (2fr) | gt_tid=17 jersey=25 team=left (1fr)
    predicted: jersey=nan  team=right
```

### SNGS-067
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 4 | 1 | 14 | 3 | 19 | **21.1%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       22        50           5       248     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 47 | 0 | 14 | 47 | **100.0%** |

#### ID Switches detected: 14
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   2  frames= 642  purity=0.76
    primary : gt_tid=12 jersey=11 team=left (488fr)
    others  : gt_tid=13 jersey=8 team=left (153fr) | gt_tid=4 jersey=44 team=right (1fr)
    predicted: jersey=8  team=left
  pred_tid=   3  frames= 429  purity=0.52
    primary : gt_tid=6 jersey=10 team=right (221fr)
    others  : gt_tid=14 jersey=None team=left (205fr) | gt_tid=13 jersey=8 team=left (2fr)
    predicted: jersey=5  team=right
  pred_tid=  13  frames= 153  purity=0.79
    primary : gt_tid=10 jersey=None team=right (121fr)
    others  : gt_tid=15 jersey=None team=left (32fr)
    predicted: jersey=nan  team=right
  pred_tid=  15  frames=   5  purity=0.60
    primary : gt_tid=17 jersey=None team=left (3fr)
    others  : gt_tid=8 jersey=None team=right (2fr)
    predicted: jersey=nan  team=left
  pred_tid=  19  frames= 523  purity=0.53
    primary : gt_tid=13 jersey=8 team=left (275fr)
    others  : gt_tid=12 jersey=11 team=left (248fr)
    predicted: jersey=11  team=left
  pred_tid=  23  frames=  17  purity=0.71
    primary : gt_tid=18 jersey=None team=left (12fr)
    others  : gt_tid=2 jersey=None team=right (5fr)
    predicted: jersey=nan  team=left
  pred_tid=  27  frames= 221  purity=0.76
    primary : gt_tid=14 jersey=None team=left (169fr)
    others  : gt_tid=15 jersey=None team=left (52fr)
    predicted: jersey=nan  team=left
  pred_tid=  32  frames=  19  purity=0.47
    primary : gt_tid=18 jersey=None team=left (9fr)
    others  : gt_tid=14 jersey=None team=left (3fr) | gt_tid=10 jersey=None team=right (2fr)
    predicted: jersey=nan  team=left
  pred_tid=  33  frames=   7  purity=0.71
    primary : gt_tid=18 jersey=None team=left (5fr)
    others  : gt_tid=16 jersey=None team=left (1fr) | gt_tid=15 jersey=None team=left (1fr)
    predicted: jersey=nan  team=left
  pred_tid=  37  frames=   5  purity=0.40
    primary : gt_tid=18 jersey=None team=left (2fr)
    others  : gt_tid=4 jersey=44 team=right (1fr) | gt_tid=7 jersey=14 team=right (1fr)
    predicted: jersey=nan  team=right
  pred_tid=  39  frames=  29  purity=0.55
    primary : gt_tid=10 jersey=None team=right (16fr)
    others  : gt_tid=17 jersey=None team=left (13fr)
    predicted: jersey=nan  team=right
  pred_tid=  60  frames=   7  purity=0.71
    primary : gt_tid=15 jersey=None team=left (5fr)
    others  : gt_tid=13 jersey=8 team=left (2fr)
    predicted: jersey=nan  team=left
  pred_tid=  62  frames=  38  purity=0.53
    primary : gt_tid=14 jersey=None team=left (20fr)
    others  : gt_tid=10 jersey=None team=right (16fr) | gt_tid=9 jersey=None team=right (2fr)
    predicted: jersey=nan  team=right
  pred_tid=  65  frames=  16  purity=0.50
    primary : gt_tid=9 jersey=None team=right (8fr)
    others  : gt_tid=17 jersey=None team=left (8fr)
    predicted: jersey=nan  team=right
```

### SNGS-068
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 10 | 4 | 23 | 1 | 37 | **27.0%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        6        30           4       733    0.95
        9        10           1       497    0.99
       11         8           5       381    0.99
       68        10          14        11    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 43 | 2 | 1 | 45 | **95.6%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
       22   right      left         5     1.0
       65   right      left         6     1.0
```

#### ID Switches detected: 1
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  33  frames= 356  purity=0.56
    primary : gt_tid=18 jersey=8 team=left (201fr)
    others  : gt_tid=22 jersey=4 team=left (155fr)
    predicted: jersey=4  team=left
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 27 | 12 | 75 | 14 | 114 | **23.7%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 149 | 2 | 25 | 151 | **98.7%** |

### ID Switches Summary
Total ID switches detected: **25** (across 3 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-066 | 10 | 257 | 0.67 | gt_tid=20 jersey=10 team=left (171fr) | left |
| SNGS-066 | 13 | 184 | 0.54 | gt_tid=6 jersey=5 team=right (100fr) | right |
| SNGS-066 | 18 | 253 | 0.73 | gt_tid=16 jersey=3 team=left (183fr) | left |
| SNGS-066 | 19 | 200 | 0.72 | gt_tid=6 jersey=5 team=right (144fr) | right |
| SNGS-066 | 24 | 508 | 0.78 | gt_tid=18 jersey=8 team=left (395fr) | left |
| SNGS-066 | 25 | 130 | 0.65 | gt_tid=17 jersey=25 team=left (85fr) | left |
| SNGS-066 | 34 | 13 | 0.77 | gt_tid=3 jersey=27 team=right (10fr) | right |
| SNGS-066 | 46 | 10 | 0.5 | gt_tid=18 jersey=8 team=left (5fr) | left |
| SNGS-066 | 51 | 8 | 0.62 | gt_tid=6 jersey=5 team=right (5fr) | left |
| SNGS-066 | 69 | 11 | 0.73 | gt_tid=8 jersey=36 team=right (8fr) | right |
| SNGS-067 | 2 | 642 | 0.76 | gt_tid=12 jersey=11 team=left (488fr) | left |
| SNGS-067 | 3 | 429 | 0.52 | gt_tid=6 jersey=10 team=right (221fr) | right |
| SNGS-067 | 13 | 153 | 0.79 | gt_tid=10 jersey=None team=right (121fr) | right |
| SNGS-067 | 15 | 5 | 0.6 | gt_tid=17 jersey=None team=left (3fr) | left |
| SNGS-067 | 19 | 523 | 0.53 | gt_tid=13 jersey=8 team=left (275fr) | left |
| SNGS-067 | 23 | 17 | 0.71 | gt_tid=18 jersey=None team=left (12fr) | left |
| SNGS-067 | 27 | 221 | 0.76 | gt_tid=14 jersey=None team=left (169fr) | left |
| SNGS-067 | 32 | 19 | 0.47 | gt_tid=18 jersey=None team=left (9fr) | left |
| SNGS-067 | 33 | 7 | 0.71 | gt_tid=18 jersey=None team=left (5fr) | left |
| SNGS-067 | 37 | 5 | 0.4 | gt_tid=18 jersey=None team=left (2fr) | right |
| SNGS-067 | 39 | 29 | 0.55 | gt_tid=10 jersey=None team=right (16fr) | right |
| SNGS-067 | 60 | 7 | 0.71 | gt_tid=15 jersey=None team=left (5fr) | left |
| SNGS-067 | 62 | 38 | 0.53 | gt_tid=14 jersey=None team=left (20fr) | right |
| SNGS-067 | 65 | 16 | 0.5 | gt_tid=9 jersey=None team=right (8fr) | right |
| SNGS-068 | 33 | 356 | 0.56 | gt_tid=18 jersey=8 team=left (201fr) | left |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.