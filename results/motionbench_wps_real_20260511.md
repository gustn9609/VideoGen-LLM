# MotionBench WPS Experiment

## Overall

| Model | Acc | Temporal-sensitive acc | Correct/total | Shuffle gap | Reverse gap |
|---|---:|---:|---:|---:|---:|
| text | 0.4583 | 0.3750 | 44/96 | 0.0000 | 0.0000 |
| pixel | 0.4688 | 0.2500 | 45/96 | -0.0104 | -0.0104 |
| flow | 0.4479 | 0.3750 | 43/96 | -0.0312 | -0.0521 |
| raw_wan | 0.4792 | 0.6250 | 46/96 | 0.0417 | 0.0104 |
| text_pixel_flow | 0.4583 | 0.2500 | 44/96 | 0.0000 | 0.0000 |
| wan_wps | 0.2500 | 0.6250 | 24/96 | 0.0312 | 0.0312 |
| wan_raw_plus_wps | 0.4375 | 0.3750 | 42/96 | 0.0312 | 0.0312 |
| pixel_wps | 0.2604 | 0.3750 | 25/96 | -0.0208 | 0.0104 |
| flow_wps | 0.3750 | 0.5000 | 36/96 | 0.1042 | 0.1562 |
| text_pixel_flow_wps | 0.4583 | 0.3750 | 44/96 | 0.0625 | -0.0104 |
| score_final_tpf_wps | 0.4792 | 0.3750 | 46/96 | 0.0312 | 0.0312 |
| score_final_tpf_raw_wan_wps | 0.4375 | 0.2500 | 42/96 | 0.0208 | 0.0104 |
| wps_gate_switch | 0.4375 | 0.3750 | 42/96 | 0.0000 | 0.0000 |

## Question Type
| Model | Action Order | Motion Recognition | Motion-related Objects | Repetition Count |
|---|---:|---:|---:|---:|
| text | 0.3333 | 0.4583 | 0.5417 | 0.5000 |
| pixel | 0.3750 | 0.4167 | 0.5417 | 0.5417 |
| flow | 0.3333 | 0.4167 | 0.5417 | 0.5000 |
| raw_wan | 0.3750 | 0.5000 | 0.5000 | 0.5417 |
| text_pixel_flow | 0.3750 | 0.4167 | 0.5417 | 0.5000 |
| wan_wps | 0.3333 | 0.1667 | 0.1667 | 0.3333 |
| wan_raw_plus_wps | 0.2917 | 0.4583 | 0.4583 | 0.5417 |
| pixel_wps | 0.1667 | 0.2500 | 0.1667 | 0.4583 |
| flow_wps | 0.3750 | 0.4583 | 0.2500 | 0.4167 |
| text_pixel_flow_wps | 0.4167 | 0.4583 | 0.4583 | 0.5000 |
| score_final_tpf_wps | 0.3333 | 0.5417 | 0.5417 | 0.5000 |
| score_final_tpf_raw_wan_wps | 0.2917 | 0.5000 | 0.4583 | 0.5000 |
| wps_gate_switch | 0.2917 | 0.4167 | 0.5833 | 0.4583 |

## WPS Gate
| Gate | Value |
|---|---:|
| gate_coverage | 0.2083 |
| base_accuracy | 0.4583 |
| wps_accuracy | 0.2500 |
| switch_accuracy | 0.4375 |
| switch_helps | 4 |
| switch_hurts | 6 |

## Success Criteria
| Criterion | Result | Pass |
|---|---|---:|
| WPS > raw Wan overall | 0.2500 vs 0.4792 | False |
| temporal-sensitive WPS > text+pixel+flow by >=5%p | 0.6250 vs 0.2500 | True |
| held-out WPS gate > raw Wan everywhere | 0.4375 vs 0.4792 | False |
| WPS normal-shuffle/reverse gap > raw Wan | shuffle 0.0312 vs 0.0417; reverse 0.0312 vs 0.0104 | False |
