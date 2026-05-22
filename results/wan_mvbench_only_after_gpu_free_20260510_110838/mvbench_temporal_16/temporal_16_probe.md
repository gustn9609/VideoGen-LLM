# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | wan_vae_grid_4x4 | hash | logistic | 0.3333 | 0.0431 | 0.3076 | 0.3590 | 428/1284 | 2.7001 |
| none | pixel_grid_sequence | hash | logistic | 0.3310 | 0.0462 | 0.3037 | 0.3590 | 425/1284 | 2.7516 |
| none | flow_grid_sequence | hash | logistic | 0.3170 | 0.0374 | 0.2913 | 0.3435 | 407/1284 | 2.6936 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | wan_vae_grid_4x4 | action_antonym | 0.5833 | 42/72 |
| none | wan_vae_grid_4x4 | action_count | 0.3611 | 26/72 |
| none | wan_vae_grid_4x4 | action_localization | 0.4583 | 33/72 |
| none | wan_vae_grid_4x4 | action_prediction | 0.1667 | 12/72 |
| none | wan_vae_grid_4x4 | action_sequence | 0.2917 | 21/72 |
| none | wan_vae_grid_4x4 | character_order | 0.3810 | 24/63 |
| none | wan_vae_grid_4x4 | counterfactual_inference | 0.2029 | 14/69 |
| none | wan_vae_grid_4x4 | egocentric_navigation | 0.1528 | 11/72 |
| none | wan_vae_grid_4x4 | fine_grained_action | 0.2917 | 21/72 |
| none | wan_vae_grid_4x4 | moving_attribute | 0.3611 | 26/72 |
| none | wan_vae_grid_4x4 | moving_count | 0.2500 | 18/72 |
| none | wan_vae_grid_4x4 | moving_direction | 0.4028 | 29/72 |
| none | wan_vae_grid_4x4 | object_existence | 0.4861 | 35/72 |
| none | wan_vae_grid_4x4 | object_interaction | 0.1528 | 11/72 |
| none | wan_vae_grid_4x4 | object_shuffle | 0.3472 | 25/72 |
| none | wan_vae_grid_4x4 | scene_transition | 0.4583 | 33/72 |
| none | wan_vae_grid_4x4 | state_change | 0.4167 | 30/72 |
| none | wan_vae_grid_4x4 | unexpected_action | 0.2361 | 17/72 |
| none | pixel_grid_sequence | action_antonym | 0.6250 | 45/72 |
| none | pixel_grid_sequence | action_count | 0.3472 | 25/72 |
| none | pixel_grid_sequence | action_localization | 0.2500 | 18/72 |
| none | pixel_grid_sequence | action_prediction | 0.2083 | 15/72 |
| none | pixel_grid_sequence | action_sequence | 0.3333 | 24/72 |
| none | pixel_grid_sequence | character_order | 0.4921 | 31/63 |
| none | pixel_grid_sequence | counterfactual_inference | 0.1739 | 12/69 |
| none | pixel_grid_sequence | egocentric_navigation | 0.1806 | 13/72 |
| none | pixel_grid_sequence | fine_grained_action | 0.2917 | 21/72 |
| none | pixel_grid_sequence | moving_attribute | 0.1528 | 11/72 |
| none | pixel_grid_sequence | moving_count | 0.2500 | 18/72 |
| none | pixel_grid_sequence | moving_direction | 0.3472 | 25/72 |
| none | pixel_grid_sequence | object_existence | 0.3611 | 26/72 |
| none | pixel_grid_sequence | object_interaction | 0.2222 | 16/72 |
| none | pixel_grid_sequence | object_shuffle | 0.5417 | 39/72 |
| none | pixel_grid_sequence | scene_transition | 0.5417 | 39/72 |
| none | pixel_grid_sequence | state_change | 0.3750 | 27/72 |
| none | pixel_grid_sequence | unexpected_action | 0.2778 | 20/72 |
| none | flow_grid_sequence | action_antonym | 0.5694 | 41/72 |
| none | flow_grid_sequence | action_count | 0.2917 | 21/72 |
| none | flow_grid_sequence | action_localization | 0.4306 | 31/72 |
| none | flow_grid_sequence | action_prediction | 0.2639 | 19/72 |
| none | flow_grid_sequence | action_sequence | 0.2639 | 19/72 |
| none | flow_grid_sequence | character_order | 0.4762 | 30/63 |
| none | flow_grid_sequence | counterfactual_inference | 0.1884 | 13/69 |
| none | flow_grid_sequence | egocentric_navigation | 0.1944 | 14/72 |
| none | flow_grid_sequence | fine_grained_action | 0.1806 | 13/72 |
| none | flow_grid_sequence | moving_attribute | 0.1667 | 12/72 |
| none | flow_grid_sequence | moving_count | 0.2500 | 18/72 |
| none | flow_grid_sequence | moving_direction | 0.3194 | 23/72 |
| none | flow_grid_sequence | object_existence | 0.4306 | 31/72 |
| none | flow_grid_sequence | object_interaction | 0.1806 | 13/72 |
| none | flow_grid_sequence | object_shuffle | 0.3056 | 22/72 |
| none | flow_grid_sequence | scene_transition | 0.5278 | 38/72 |
| none | flow_grid_sequence | state_change | 0.4583 | 33/72 |
| none | flow_grid_sequence | unexpected_action | 0.2222 | 16/72 |
