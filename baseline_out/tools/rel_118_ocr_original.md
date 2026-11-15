# Pipeline Report — JNR & Team/Team-side
**Generated**: 2026-09-13 17:34
**State file**: `/home/joao/soccernet/baseline_out/states_ocr/SNGS-118.pklz`

**Videos**: SNGS-118
**Eval filters**: min_frames=5, match_ratio≥0.3, iou≥0.3, purity≥0.8 (team only)

---
## Per-video Results

### SNGS-118
- JNR column present: ✓  |  Team column present: ✓

#### JNR
| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|---------------|-------------------|-------------|-------------|
| 12 | 3 | 12 | 0 | 27 | **44.4%** |

Wrong JNR predictions:
```
 pred_tid gt_jersey pred_jersey  n_frames  purity
        1        93          19       726     1.0
       18        11          17       653     1.0
       69         8          11       233     1.0
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
| 12 | 3 | 12 | 0 | 27 | **44.4%** |

### Team / Team-side
| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |
|---------|-------|-------------------|-------------|-------------|
| 40 | 0 | 0 | 40 | **100.0%** |
