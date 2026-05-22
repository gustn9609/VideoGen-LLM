# MotionBench Residual Score Ensemble

| Model | Overall | Hard subset | Action Order | Repetition Count | Motion Recognition | Motion Objects | Correct/total |
|---|---:|---:|---:|---:|---:|---:|---:|
| text | 0.4583 | 0.1746 | 0.3333 | 0.5000 | 0.4583 | 0.5417 | 44/96 |
| pixel | 0.4688 | 0.2063 | 0.3750 | 0.5417 | 0.4167 | 0.5417 | 45/96 |
| flow | 0.4479 | 0.2063 | 0.3333 | 0.5000 | 0.4167 | 0.5417 | 43/96 |
| raw_wan | 0.4792 | 0.2222 | 0.3750 | 0.5417 | 0.5000 | 0.5000 | 46/96 |
| structured | 0.4375 | 0.2063 | 0.2917 | 0.4583 | 0.4583 | 0.5417 | 42/96 |
| dynamics | 0.3542 | 0.2063 | 0.2500 | 0.4583 | 0.2917 | 0.4167 | 34/96 |
| residual_dyn | 0.4062 | 0.2381 | 0.4167 | 0.4583 | 0.3333 | 0.4167 | 39/96 |
| relation | 0.4583 | 0.2063 | 0.3333 | 0.5000 | 0.4583 | 0.5417 | 44/96 |
| equivariance | 0.2917 | 0.1905 | 0.1667 | 0.3750 | 0.4167 | 0.2083 | 28/96 |
| multi_target | 0.3125 | 0.2222 | 0.2083 | 0.3750 | 0.3333 | 0.3333 | 30/96 |
| calibrated_residual_ensemble | 0.4271 | 0.2381 | 0.4583 | 0.5000 | 0.3333 | 0.4167 | 41/96 |

## Best Weight Sweep

| rank | weights | acc | hard acc | correct/total |
|---:|---|---:|---:|---:|
| 1 | text=0, pixel=0, flow=0, raw_wan=0, structured=0, dynamics=0.25, residual_dyn=0.25, relation=2, equivariance=1, multi_target=0.25 | 0.4896 | 0.2698 | 47/96 |
| 2 | text=0, pixel=0, flow=0, raw_wan=0, structured=0, dynamics=0.5, residual_dyn=0.25, relation=2, equivariance=1, multi_target=0 | 0.4896 | 0.2698 | 47/96 |
| 3 | text=0, pixel=0, flow=0, raw_wan=0, structured=0.5, dynamics=0, residual_dyn=0.25, relation=1, equivariance=0, multi_target=0.25 | 0.4896 | 0.2222 | 47/96 |
| 4 | text=0, pixel=0, flow=0, raw_wan=0, structured=1, dynamics=0, residual_dyn=0.5, relation=2, equivariance=0, multi_target=0.5 | 0.4896 | 0.2222 | 47/96 |
| 5 | text=0, pixel=0, flow=0, raw_wan=0, structured=1, dynamics=0.25, residual_dyn=0.5, relation=2, equivariance=0, multi_target=0.25 | 0.4896 | 0.2222 | 47/96 |
| 6 | text=0, pixel=0, flow=0, raw_wan=0.25, structured=0.25, dynamics=0, residual_dyn=0.25, relation=0.25, equivariance=0, multi_target=0 | 0.4896 | 0.2222 | 47/96 |
| 7 | text=0, pixel=0, flow=0, raw_wan=0.25, structured=0.25, dynamics=0, residual_dyn=0.25, relation=0.5, equivariance=0, multi_target=0 | 0.4896 | 0.2222 | 47/96 |
| 8 | text=0, pixel=0, flow=0, raw_wan=0.25, structured=0.25, dynamics=0, residual_dyn=0.25, relation=0.5, equivariance=0, multi_target=0.25 | 0.4896 | 0.2222 | 47/96 |
| 9 | text=0, pixel=0, flow=0, raw_wan=0.25, structured=0.25, dynamics=0.25, residual_dyn=0.25, relation=0.5, equivariance=0, multi_target=0 | 0.4896 | 0.2222 | 47/96 |
| 10 | text=0, pixel=0, flow=0, raw_wan=0, structured=0, dynamics=0, residual_dyn=0, relation=0.5, equivariance=0.25, multi_target=0 | 0.4792 | 0.2540 | 46/96 |
| 11 | text=0, pixel=0, flow=0, raw_wan=0, structured=0, dynamics=0, residual_dyn=0, relation=1, equivariance=0.5, multi_target=0 | 0.4792 | 0.2540 | 46/96 |
| 12 | text=0, pixel=0, flow=0, raw_wan=0, structured=0, dynamics=0, residual_dyn=0, relation=2, equivariance=1, multi_target=0 | 0.4792 | 0.2540 | 46/96 |
| 13 | text=0, pixel=0, flow=0, raw_wan=0, structured=0, dynamics=0, residual_dyn=0, relation=2, equivariance=1, multi_target=0.25 | 0.4792 | 0.2540 | 46/96 |
| 14 | text=0, pixel=0, flow=0, raw_wan=0, structured=0, dynamics=0.25, residual_dyn=0, relation=2, equivariance=1, multi_target=0 | 0.4792 | 0.2540 | 46/96 |
| 15 | text=0, pixel=0, flow=0, raw_wan=0, structured=0, dynamics=0.25, residual_dyn=0.25, relation=1, equivariance=0.5, multi_target=0 | 0.4792 | 0.2540 | 46/96 |
| 16 | text=0, pixel=0, flow=0, raw_wan=0, structured=0, dynamics=0.25, residual_dyn=0.25, relation=2, equivariance=1, multi_target=0 | 0.4792 | 0.2540 | 46/96 |
| 17 | text=0, pixel=0, flow=0, raw_wan=0, structured=0, dynamics=0.5, residual_dyn=0.5, relation=2, equivariance=1, multi_target=0 | 0.4792 | 0.2540 | 46/96 |
| 18 | text=0, pixel=0, flow=0, raw_wan=0, structured=0.25, dynamics=0, residual_dyn=0, relation=2, equivariance=1, multi_target=0.25 | 0.4792 | 0.2540 | 46/96 |
| 19 | text=0, pixel=0, flow=0, raw_wan=0, structured=0.25, dynamics=0, residual_dyn=0, relation=2, equivariance=1, multi_target=0.5 | 0.4792 | 0.2540 | 46/96 |
| 20 | text=0, pixel=0, flow=0, raw_wan=0, structured=0.25, dynamics=0.25, residual_dyn=0, relation=2, equivariance=1, multi_target=0 | 0.4792 | 0.2540 | 46/96 |

## Complementarity

- dynamics_only_vs_equivariance: 12
- dynamics_only_vs_flow: 4
- dynamics_only_vs_multi_target: 12
- dynamics_only_vs_pixel: 7
- dynamics_only_vs_raw_wan: 5
- dynamics_only_vs_relation: 7
- dynamics_only_vs_residual_dyn: 9
- dynamics_only_vs_structured: 7
- dynamics_only_vs_text: 6
- equivariance_only_vs_dynamics: 6
- equivariance_only_vs_flow: 4
- equivariance_only_vs_multi_target: 5
- equivariance_only_vs_pixel: 4
- equivariance_only_vs_raw_wan: 3
- equivariance_only_vs_relation: 3
- equivariance_only_vs_residual_dyn: 7
- equivariance_only_vs_structured: 5
- equivariance_only_vs_text: 4
- flow_only_vs_dynamics: 13
- flow_only_vs_equivariance: 19
- flow_only_vs_multi_target: 18
- flow_only_vs_pixel: 4
- flow_only_vs_raw_wan: 3
- flow_only_vs_relation: 5
- flow_only_vs_residual_dyn: 14
- flow_only_vs_structured: 9
- flow_only_vs_text: 4
- multi_target_only_vs_dynamics: 8
- multi_target_only_vs_equivariance: 7
- multi_target_only_vs_flow: 5
- multi_target_only_vs_pixel: 7
- multi_target_only_vs_raw_wan: 6
- multi_target_only_vs_relation: 7
- multi_target_only_vs_residual_dyn: 10
- multi_target_only_vs_structured: 8
- multi_target_only_vs_text: 6
- oracle_union_accuracy: 0.6666666666666666
- oracle_union_correct: 64
- oracle_union_total: 96
- pixel_only_vs_dynamics: 18
- pixel_only_vs_equivariance: 21
- pixel_only_vs_flow: 6
- pixel_only_vs_multi_target: 22
- pixel_only_vs_raw_wan: 2
- pixel_only_vs_relation: 2
- pixel_only_vs_residual_dyn: 13
- pixel_only_vs_structured: 7
- pixel_only_vs_text: 2
- raw_wan_only_vs_dynamics: 17
- raw_wan_only_vs_equivariance: 21
- raw_wan_only_vs_flow: 6
- raw_wan_only_vs_multi_target: 22
- raw_wan_only_vs_pixel: 3
- raw_wan_only_vs_relation: 3
- raw_wan_only_vs_residual_dyn: 15
- raw_wan_only_vs_structured: 8
- raw_wan_only_vs_text: 4
- relation_only_vs_dynamics: 17
- relation_only_vs_equivariance: 19
- relation_only_vs_flow: 6
- relation_only_vs_multi_target: 21
- relation_only_vs_pixel: 1
- relation_only_vs_raw_wan: 1
- relation_only_vs_residual_dyn: 13
- relation_only_vs_structured: 7
- relation_only_vs_text: 2
- residual_dyn_only_vs_dynamics: 14
- residual_dyn_only_vs_equivariance: 18
- residual_dyn_only_vs_flow: 10
- residual_dyn_only_vs_multi_target: 19
- residual_dyn_only_vs_pixel: 7
- residual_dyn_only_vs_raw_wan: 8
- residual_dyn_only_vs_relation: 8
- residual_dyn_only_vs_structured: 8
- residual_dyn_only_vs_text: 7
- structured_only_vs_dynamics: 15
- structured_only_vs_equivariance: 19
- structured_only_vs_flow: 8
- structured_only_vs_multi_target: 20
- structured_only_vs_pixel: 4
- structured_only_vs_raw_wan: 4
- structured_only_vs_relation: 5
- structured_only_vs_residual_dyn: 11
- structured_only_vs_text: 4
- text_only_vs_dynamics: 16
- text_only_vs_equivariance: 20
- text_only_vs_flow: 5
- text_only_vs_multi_target: 20
- text_only_vs_pixel: 1
- text_only_vs_raw_wan: 2
- text_only_vs_relation: 2
- text_only_vs_residual_dyn: 12
- text_only_vs_structured: 6
- text_pixel_flow_wrong_dynamics_right: 3
- text_pixel_flow_wrong_equivariance_right: 2
- text_pixel_flow_wrong_multi_target_right: 4
- text_pixel_flow_wrong_raw_wan_right: 1
- text_pixel_flow_wrong_relation_right: 0
- text_pixel_flow_wrong_residual_dyn_right: 6
- text_pixel_flow_wrong_structured_right: 2
