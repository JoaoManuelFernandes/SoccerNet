# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-07-04 22:20
**State file**: `/home/joao/soccernet/outputs/sn-gamestate/2026-07-01/10-33-17/states/gshota-test-extraC.pklz`

**Videos**: SNGS-130, SNGS-131
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-130
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 18 | 2 | 11 | 0 | 31 | **58.1%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       12         9           6       231    1.00
       34        36          35       502    0.99
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 36 | 0 | 0 | 36 | **100.0%** |

### SNGS-131
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 10 | 5 | 11 | 6 | 26 | **38.5%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       37        31           6        77    0.99
       43        44           6        90    0.99
       52        14          16       261    1.00
       75        50           5        95    1.00
       76        21          16         5    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 17 | 22 | 10 | 39 | **43.6%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
        5   right      left       388    1.00
        6   right      left       174    1.00
        9   right      left       132    1.00
       17   right      left       146    1.00
       27   right      left       251    1.00
       29   right      left       112    0.93
       31   right      left         5    1.00
       33   right      left        41    1.00
       34   right      left       176    1.00
       36   right      left       145    1.00
```

#### ID Switches detected: 10
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=   8  frames= 693  purity=0.57
    primary : gt_tid=12 jersey=31 team=right (392fr)
    others  : gt_tid=13 jersey=16 team=right (301fr)
    predicted: jersey=31  team=left
  pred_tid=  10  frames= 316  purity=0.50
    primary : gt_tid=16 jersey=14 team=right (159fr)
    others  : gt_tid=14 jersey=None team=right (155fr) | gt_tid=17 jersey=None team=right (2fr)
    predicted: jersey=5  team=left
  pred_tid=  12  frames= 115  purity=0.76
    primary : gt_tid=6 jersey=27 team=left (87fr)
    others  : gt_tid=8 jersey=14 team=left (28fr)
    predicted: jersey=None  team=left
  pred_tid=  13  frames= 157  purity=0.52
    primary : gt_tid=9 jersey=None team=left (82fr)
    others  : gt_tid=7 jersey=44 team=left (75fr)
    predicted: jersey=None  team=left
  pred_tid=  15  frames= 245  purity=0.76
    primary : gt_tid=9 jersey=None team=left (187fr)
    others  : gt_tid=8 jersey=14 team=left (58fr)
    predicted: jersey=None  team=left
  pred_tid=  16  frames=  65  purity=0.45
    primary : gt_tid=15 jersey=None team=right (29fr)
    others  : gt_tid=16 jersey=14 team=right (20fr) | gt_tid=18 jersey=None team=right (15fr)
    predicted: jersey=None  team=left
  pred_tid=  19  frames= 114  purity=0.39
    primary : gt_tid=15 jersey=None team=right (45fr)
    others  : gt_tid=18 jersey=None team=right (36fr) | gt_tid=16 jersey=14 team=right (33fr)
    predicted: jersey=None  team=left
  pred_tid=  30  frames=  19  purity=0.68
    primary : gt_tid=12 jersey=31 team=right (13fr)
    others  : gt_tid=13 jersey=16 team=right (4fr) | gt_tid=6 jersey=27 team=left (2fr)
    predicted: jersey=None  team=left
  pred_tid=  48  frames=   9  purity=0.67
    primary : gt_tid=10 jersey=21 team=right (6fr)
    others  : gt_tid=5 jersey=10 team=left (3fr)
    predicted: jersey=None  team=left
  pred_tid=  54  frames=  10  purity=0.50
    primary : gt_tid=16 jersey=14 team=right (5fr)
    others  : gt_tid=15 jersey=None team=right (3fr) | gt_tid=7 jersey=44 team=left (2fr)
    predicted: jersey=None  team=left
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 28 | 7 | 22 | 6 | 57 | **49.1%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 53 | 22 | 10 | 75 | **70.7%** |

### ID Switches Summary
Total ID switches detected: **10** (across 1 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-131 | 8 | 693 | 0.57 | gt_tid=12 jersey=31 team=right (392fr) | left |
| SNGS-131 | 10 | 316 | 0.5 | gt_tid=16 jersey=14 team=right (159fr) | left |
| SNGS-131 | 12 | 115 | 0.76 | gt_tid=6 jersey=27 team=left (87fr) | left |
| SNGS-131 | 13 | 157 | 0.52 | gt_tid=9 jersey=None team=left (82fr) | left |
| SNGS-131 | 15 | 245 | 0.76 | gt_tid=9 jersey=None team=left (187fr) | left |
| SNGS-131 | 16 | 65 | 0.45 | gt_tid=15 jersey=None team=right (29fr) | left |
| SNGS-131 | 19 | 114 | 0.39 | gt_tid=15 jersey=None team=right (45fr) | left |
| SNGS-131 | 30 | 19 | 0.68 | gt_tid=12 jersey=31 team=right (13fr) | left |
| SNGS-131 | 48 | 9 | 0.67 | gt_tid=10 jersey=21 team=right (6fr) | left |
| SNGS-131 | 54 | 10 | 0.5 | gt_tid=16 jersey=14 team=right (5fr) | left |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.