# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-07 07:41
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-09-05/16-05-21/states/train-batch05.pklz`

**Videos**: SNGS-072, SNGS-073, SNGS-074
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-072
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 18 | 5 | 20 | 4 | 43 | **41.9%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        2        17          11       746    0.98
        8        25           5       325    1.00
       13        28          39       162    1.00
       19        36           5       639    1.00
       34        14          16        60    0.98
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 42 | 1 | 4 | 43 | **97.7%** |

Wrong team predictions:
```
 pred_tid gt_team pred_team  n_frames  purity
       68   right      left         5     1.0
```

#### ID Switches detected: 4
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  22  frames=  21  purity=0.52
    primary : gt_tid=18 jersey=36 team=left (11fr)
    others  : gt_tid=16 jersey=10 team=right (10fr)
    predicted: jersey=5  team=left
  pred_tid=  27  frames= 359  purity=0.57
    primary : gt_tid=14 jersey=3 team=right (204fr)
    others  : gt_tid=12 jersey=4 team=right (153fr) | gt_tid=11 jersey=27 team=left (2fr)
    predicted: jersey=3  team=right
  pred_tid=  35  frames=   7  purity=0.57
    primary : gt_tid=12 jersey=4 team=right (4fr)
    others  : gt_tid=14 jersey=3 team=right (3fr)
    predicted: jersey=nan  team=right
  pred_tid=  43  frames=   6  purity=0.67
    primary : gt_tid=11 jersey=27 team=left (4fr)
    others  : gt_tid=14 jersey=3 team=right (2fr)
    predicted: jersey=nan  team=left
```

### SNGS-073
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 7 | 1 | 22 | 4 | 30 | **23.3%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       43        29          25       260    0.96
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 52 | 0 | 4 | 52 | **100.0%** |

#### ID Switches detected: 4
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  32  frames=   9  purity=0.75
    primary : gt_tid=20 jersey=29 team=right (6fr)
    others  : gt_tid=11 jersey=8 team=right (2fr)
    predicted: jersey=None  team=right
  pred_tid=  34  frames=   7  purity=0.60
    primary : gt_tid=3 jersey=14 team=left (3fr)
    others  : gt_tid=20 jersey=29 team=right (2fr)
    predicted: jersey=None  team=right
  pred_tid=  35  frames=   6  purity=0.33
    primary : gt_tid=23 jersey=10 team=right (2fr)
    others  : gt_tid=3 jersey=14 team=left (2fr) | gt_tid=13 jersey=4 team=right (2fr)
    predicted: jersey=None  team=right
  pred_tid=  52  frames=   6  purity=0.33
    primary : gt_tid=1 jersey=27 team=left (2fr)
    others  : gt_tid=2 jersey=None team=left (2fr) | gt_tid=6 jersey=4 team=left (1fr)
    predicted: jersey=None  team=left
```

### SNGS-074
- JNR column present: ✗  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 0 | 0 | 36 | 5 | 36 | **0.0%** |

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 37 | 0 | 5 | 37 | **100.0%** |

#### ID Switches detected: 5
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  18  frames= 622  purity=0.69
    primary : gt_tid=12 jersey=4 team=left (427fr)
    others  : gt_tid=8 jersey=29 team=right (195fr)
    predicted: jersey=None  team=left
  pred_tid=  22  frames=   7  purity=0.57
    primary : gt_tid=5 jersey=50 team=left (4fr)
    others  : gt_tid=3 jersey=None team=left (3fr)
    predicted: jersey=None  team=left
  pred_tid=  27  frames=   5  purity=0.60
    primary : gt_tid=13 jersey=17 team=right (3fr)
    others  : gt_tid=17 jersey=10 team=left (1fr) | gt_tid=4 jersey=25 team=right (1fr)
    predicted: jersey=None  team=right
  pred_tid=  40  frames=  76  purity=0.77
    primary : gt_tid=19 jersey=27 team=left (57fr)
    others  : gt_tid=18 jersey=4 team=right (17fr)
    predicted: jersey=None  team=left
  pred_tid=  43  frames= 194  purity=0.51
    primary : gt_tid=21 jersey=3 team=right (95fr)
    others  : gt_tid=14 jersey=28 team=right (91fr)
    predicted: jersey=None  team=right
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 25 | 6 | 78 | 13 | 109 | **22.9%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 131 | 1 | 13 | 132 | **99.2%** |

### ID Switches Summary
Total ID switches detected: **13** (across 3 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-072 | 22 | 21 | 0.52 | gt_tid=18 jersey=36 team=left (11fr) | left |
| SNGS-072 | 27 | 359 | 0.57 | gt_tid=14 jersey=3 team=right (204fr) | right |
| SNGS-072 | 35 | 7 | 0.57 | gt_tid=12 jersey=4 team=right (4fr) | right |
| SNGS-072 | 43 | 6 | 0.67 | gt_tid=11 jersey=27 team=left (4fr) | left |
| SNGS-073 | 32 | 9 | 0.75 | gt_tid=20 jersey=29 team=right (6fr) | right |
| SNGS-073 | 34 | 7 | 0.6 | gt_tid=3 jersey=14 team=left (3fr) | right |
| SNGS-073 | 35 | 6 | 0.33 | gt_tid=23 jersey=10 team=right (2fr) | right |
| SNGS-073 | 52 | 6 | 0.33 | gt_tid=1 jersey=27 team=left (2fr) | left |
| SNGS-074 | 18 | 622 | 0.69 | gt_tid=12 jersey=4 team=left (427fr) | left |
| SNGS-074 | 22 | 7 | 0.57 | gt_tid=5 jersey=50 team=left (4fr) | left |
| SNGS-074 | 27 | 5 | 0.6 | gt_tid=13 jersey=17 team=right (3fr) | right |
| SNGS-074 | 40 | 76 | 0.77 | gt_tid=19 jersey=27 team=left (57fr) | left |
| SNGS-074 | 43 | 194 | 0.51 | gt_tid=21 jersey=3 team=right (95fr) | right |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.