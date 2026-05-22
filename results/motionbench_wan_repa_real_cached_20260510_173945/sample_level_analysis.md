# MotionBench Sample-Level Analysis

- n: 96
- text low-margin p35 threshold: 0.2454

## 7.1 Correct-Set Overlap

### Both-correct count matrix

| model | text | pixel | flow | raw_wan | morepa | moft | trd |
|---|---:|---:|---:|---:|---:|---:|---:|
| text | 44 | 43 | 39 | 42 | 43 | 41 | 34 |
| pixel | 43 | 45 | 39 | 43 | 43 | 40 | 35 |
| flow | 39 | 39 | 43 | 40 | 40 | 39 | 32 |
| raw_wan | 42 | 43 | 40 | 46 | 43 | 41 | 35 |
| morepa | 43 | 43 | 40 | 43 | 46 | 41 | 35 |
| moft | 41 | 40 | 39 | 41 | 41 | 42 | 32 |
| trd | 34 | 35 | 32 | 35 | 35 | 32 | 37 |

### Raw Wan complementarity

| condition | count/prob |
|---|---:|
| text wrong, raw Wan correct | 4 |
| pixel wrong, raw Wan correct | 3 |
| flow wrong, raw Wan correct | 6 |
| text+pixel+flow all wrong | 46 |
| text+pixel+flow all wrong, raw Wan correct | 1 |
| P(raw Wan correct | all three wrong) | 0.0217 |
| 4-way chance | 0.2500 |

## 7.2 Accuracy By Question Type

| Question type | text | pixel | flow | raw Wan | MoREPA | MOFT | TRD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Action Order | 0.3333 (8/24) | 0.3750 (9/24) | 0.3333 (8/24) | 0.3750 (9/24) | 0.3750 (9/24) | 0.2917 (7/24) | 0.3750 (9/24) |
| Motion Recognition | 0.4583 (11/24) | 0.4167 (10/24) | 0.4167 (10/24) | 0.5000 (12/24) | 0.5000 (12/24) | 0.4583 (11/24) | 0.2917 (7/24) |
| Motion-related Objects | 0.5417 (13/24) | 0.5417 (13/24) | 0.5417 (13/24) | 0.5000 (12/24) | 0.5000 (12/24) | 0.5000 (12/24) | 0.4167 (10/24) |
| Repetition Count | 0.5000 (12/24) | 0.5417 (13/24) | 0.5000 (12/24) | 0.5417 (13/24) | 0.5417 (13/24) | 0.5000 (12/24) | 0.4583 (11/24) |

### Raw Wan unique gains by type

| Question type | text wrong raw correct | pixel wrong raw correct | flow wrong raw correct | T/P/F all wrong raw correct | P(raw correct given T/P/F wrong) |
|---|---:|---:|---:|---:|---:|
| Action Order | 2 | 1 | 3 | 0/12 | 0.0000 |
| Motion Recognition | 1 | 2 | 2 | 1/13 | 0.0769 |
| Motion-related Objects | 0 | 0 | 0 | 0/10 | 0.0000 |
| Repetition Count | 1 | 0 | 1 | 0/11 | 0.0000 |

## 7.3 Redefined Hard Subsets

| Subset | n | text | pixel | flow | raw Wan | MoREPA | MOFT | TRD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_text_hard_wrong_or_low_margin | 63 | 0.1746 | 0.1905 | 0.2063 | 0.2222 | 0.2063 | 0.1587 | 0.1587 |
| A1_text_wrong_only | 52 | 0.0000 | 0.0385 | 0.0769 | 0.0769 | 0.0577 | 0.0192 | 0.0577 |
| E_QA_hard_text_low_margin_only | 34 | 0.3235 | 0.3529 | 0.3824 | 0.3824 | 0.3529 | 0.2941 | 0.2647 |
| B_appearance_hard_pixel_first_and_avg_wrong | 52 | 0.0192 | 0.0577 | 0.0769 | 0.0962 | 0.0769 | 0.0192 | 0.0577 |
| B2_appearance_hard_pixel_rawwan_controls_wrong | 48 | 0.0000 | 0.0000 | 0.0625 | 0.0417 | 0.0417 | 0.0000 | 0.0417 |
| C_temporal_sensitive_rawwan_pred_changes | 8 | 0.3750 | 0.2500 | 0.3750 | 0.6250 | 0.3750 | 0.2500 | 0.2500 |
| C2_temporal_sensitive_any_video_pred_changes | 28 | 0.3214 | 0.3571 | 0.2500 | 0.3929 | 0.3571 | 0.2500 | 0.2857 |
| D_wan_favorable_top35_motion_teacher | 34 | 0.3529 | 0.3824 | 0.4118 | 0.4118 | 0.3824 | 0.3529 | 0.4118 |
| base_all_wrong_text_pixel_flow | 46 | 0.0000 | 0.0000 | 0.0000 | 0.0217 | 0.0217 | 0.0000 | 0.0217 |

### Subset definitions

- A_text_hard_wrong_or_low_margin: text wrong OR text margin <= p35 (0.2454)
- A1_text_wrong_only: text wrong
- B_appearance_hard_pixel_first_and_avg_wrong: pixel first-frame-only wrong AND pixel time-average wrong
- B2_appearance_hard_pixel_rawwan_controls_wrong: pixel and raw Wan first-frame/time-average controls all wrong
- C_temporal_sensitive_rawwan_pred_changes: raw Wan normal prediction differs from shuffled or reversed prediction
- C2_temporal_sensitive_any_video_pred_changes: pixel/flow/raw Wan normal prediction differs from shuffled or reversed prediction
- D_wan_favorable_top35_motion_teacher: top 35% cached motion-teacher residual score = z(energy_mean)+z(energy_max)+0.5*z(periodicity)+z(shift_mag_mean)
- E_QA_hard_text_low_margin_only: text margin <= p35 (0.2454)

## Temporal Control Accuracies

| transform | text | pixel | flow | raw Wan |
|---|---:|---:|---:|---:|
| normal | 0.4583 | 0.4688 | 0.4479 | 0.4792 |
| shuffled | 0.4583 | 0.4792 | 0.4792 | 0.4375 |
| reversed | 0.4583 | 0.4792 | 0.5000 | 0.4688 |
| first_frame | 0.4583 | 0.4583 | 0.4583 | 0.4688 |
| time_average | 0.4583 | 0.4479 | 0.4792 | 0.4688 |
