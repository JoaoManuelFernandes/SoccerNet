# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-13 17:33
**State file**: `/home/joao/soccernet/baseline_out/states/SNGS-118.pklz`

**Videos**: SNGS-118
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-118
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 16 | 5 | 6 | 0 | 27 | **59.3%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        5        11           6        83    1.00
        6        25           5       118    1.00
       13         4          34       278    0.95
       29        10           6        30    1.00
       65         8          65        69    1.00
```

#### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 40 | 0 | 0 | 40 | **100.0%** |

---
## Global Summary

### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 16 | 5 | 6 | 0 | 27 | **59.3%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 40 | 0 | 0 | 40 | **100.0%** |
