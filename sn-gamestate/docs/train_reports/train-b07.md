# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-08 13:45
**State file**: `/home/joao/soccernet/sn-gamestate/outputs/sn-gamestate/2026-09-08/12-45-26/states/train-b07.pklz`

**Videos**: SNGS-097, SNGS-098, SNGS-099
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-097
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 16 | 0 | 18 | 0 | 34 | **47.1%** |

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 34 | 0 | 0 | 34 | **100.0%** |

### SNGS-098
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 18 | 2 | 12 | 3 | 32 | **56.2%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
       45        31           5       204     1.0
       59        36          56        40     1.0
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 34 | 0 | 3 | 34 | **100.0%** |

#### ID Switches detected: 3
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  17  frames= 412  purity=0.51
    primary : gt_tid=12 jersey=16 team=right (209fr)
    others  : gt_tid=26 jersey=24 team=right (203fr)
    predicted: jersey=24  team=right
  pred_tid=  33  frames= 307  purity=0.63
    primary : gt_tid=17 jersey=33 team=left (192fr)
    others  : gt_tid=11 jersey=27 team=left (115fr)
    predicted: jersey=nan  team=left
  pred_tid=  42  frames= 258  purity=0.52
    primary : gt_tid=11 jersey=27 team=left (133fr)
    others  : gt_tid=17 jersey=33 team=left (125fr)
    predicted: jersey=15  team=left
```

### SNGS-099
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 17 | 2 | 15 | 2 | 34 | **50.0%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        1        34           4       260    0.88
        2        14           5       242    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 42 | 0 | 2 | 42 | **100.0%** |

#### ID Switches detected: 2
*(purity < 0.8 — tracklets tracking 2+ GT players; excluded from JNR+team eval)*

```
  pred_tid=  18  frames= 416  purity=0.60
    primary : gt_tid=12 jersey=10 team=right (251fr)
    others  : gt_tid=8 jersey=33 team=right (164fr)
    predicted: jersey=33  team=right
  pred_tid=  49  frames= 320  purity=0.63
    primary : gt_tid=23 jersey=36 team=left (202fr)
    others  : gt_tid=21 jersey=27 team=left (118fr)
    predicted: jersey=nan  team=left
```

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 51 | 4 | 45 | 5 | 100 | **51.0%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 110 | 0 | 5 | 110 | **100.0%** |

### ID Switches Summary
Total ID switches detected: **5** (across 2 videos)

| Video | pred_tid | frames | purity | primary GT | predicted team |
|-------|----------|--------|--------|------------|----------------|
| SNGS-098 | 17 | 412 | 0.51 | gt_tid=12 jersey=16 team=right (209fr) | right |
| SNGS-098 | 33 | 307 | 0.63 | gt_tid=17 jersey=33 team=left (192fr) | left |
| SNGS-098 | 42 | 258 | 0.52 | gt_tid=11 jersey=27 team=left (133fr) | left |
| SNGS-099 | 18 | 416 | 0.6 | gt_tid=12 jersey=10 team=right (251fr) | right |
| SNGS-099 | 49 | 320 | 0.63 | gt_tid=23 jersey=36 team=left (202fr) | left |

> **Note**: ID switches are tracklets where the tracker jumps between two different GT players
> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,
> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.