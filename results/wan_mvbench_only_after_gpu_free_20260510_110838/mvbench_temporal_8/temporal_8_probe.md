# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | wan_vae_grid_4x4 | hash | logistic | 0.3123 | 0.0397 | 0.2874 | 0.3372 | 401/1284 | 2.4553 |
| none | pixel_grid_sequence | hash | logistic | 0.3217 | 0.0438 | 0.2967 | 0.3466 | 413/1284 | 2.6905 |
| none | flow_grid_sequence | hash | logistic | 0.2773 | 0.0452 | 0.2539 | 0.3022 | 356/1284 | 2.8148 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | wan_vae_grid_4x4 | action_antonym | 0.5833 | 42/72 |
| none | wan_vae_grid_4x4 | action_count | 0.3611 | 26/72 |
| none | wan_vae_grid_4x4 | action_localization | 0.3750 | 27/72 |
| none | wan_vae_grid_4x4 | action_prediction | 0.2083 | 15/72 |
| none | wan_vae_grid_4x4 | action_sequence | 0.1944 | 14/72 |
| none | wan_vae_grid_4x4 | character_order | 0.4127 | 26/63 |
| none | wan_vae_grid_4x4 | counterfactual_inference | 0.1449 | 10/69 |
| none | wan_vae_grid_4x4 | egocentric_navigation | 0.2361 | 17/72 |
| none | wan_vae_grid_4x4 | fine_grained_action | 0.2639 | 19/72 |
| none | wan_vae_grid_4x4 | moving_attribute | 0.2639 | 19/72 |
| none | wan_vae_grid_4x4 | moving_count | 0.2500 | 18/72 |
| none | wan_vae_grid_4x4 | moving_direction | 0.3889 | 28/72 |
| none | wan_vae_grid_4x4 | object_existence | 0.3056 | 22/72 |
| none | wan_vae_grid_4x4 | object_interaction | 0.1111 | 8/72 |
| none | wan_vae_grid_4x4 | object_shuffle | 0.3750 | 27/72 |
| none | wan_vae_grid_4x4 | scene_transition | 0.5000 | 36/72 |
| none | wan_vae_grid_4x4 | state_change | 0.4444 | 32/72 |
| none | wan_vae_grid_4x4 | unexpected_action | 0.2083 | 15/72 |
| none | pixel_grid_sequence | action_antonym | 0.5694 | 41/72 |
| none | pixel_grid_sequence | action_count | 0.3333 | 24/72 |
| none | pixel_grid_sequence | action_localization | 0.1944 | 14/72 |
| none | pixel_grid_sequence | action_prediction | 0.2639 | 19/72 |
| none | pixel_grid_sequence | action_sequence | 0.2500 | 18/72 |
| none | pixel_grid_sequence | character_order | 0.4127 | 26/63 |
| none | pixel_grid_sequence | counterfactual_inference | 0.1594 | 11/69 |
| none | pixel_grid_sequence | egocentric_navigation | 0.3056 | 22/72 |
| none | pixel_grid_sequence | fine_grained_action | 0.2222 | 16/72 |
| none | pixel_grid_sequence | moving_attribute | 0.2083 | 15/72 |
| none | pixel_grid_sequence | moving_count | 0.2500 | 18/72 |
| none | pixel_grid_sequence | moving_direction | 0.3472 | 25/72 |
| none | pixel_grid_sequence | object_existence | 0.5139 | 37/72 |
| none | pixel_grid_sequence | object_interaction | 0.2222 | 16/72 |
| none | pixel_grid_sequence | object_shuffle | 0.4444 | 32/72 |
| none | pixel_grid_sequence | scene_transition | 0.4583 | 33/72 |
| none | pixel_grid_sequence | state_change | 0.5278 | 38/72 |
| none | pixel_grid_sequence | unexpected_action | 0.1111 | 8/72 |
| none | flow_grid_sequence | action_antonym | 0.4306 | 31/72 |
| none | flow_grid_sequence | action_count | 0.2917 | 21/72 |
| none | flow_grid_sequence | action_localization | 0.3056 | 22/72 |
| none | flow_grid_sequence | action_prediction | 0.1806 | 13/72 |
| none | flow_grid_sequence | action_sequence | 0.2222 | 16/72 |
| none | flow_grid_sequence | character_order | 0.2381 | 15/63 |
| none | flow_grid_sequence | counterfactual_inference | 0.1884 | 13/69 |
| none | flow_grid_sequence | egocentric_navigation | 0.1528 | 11/72 |
| none | flow_grid_sequence | fine_grained_action | 0.1389 | 10/72 |
| none | flow_grid_sequence | moving_attribute | 0.1944 | 14/72 |
| none | flow_grid_sequence | moving_count | 0.2500 | 18/72 |
| none | flow_grid_sequence | moving_direction | 0.2917 | 21/72 |
| none | flow_grid_sequence | object_existence | 0.5417 | 39/72 |
| none | flow_grid_sequence | object_interaction | 0.2917 | 21/72 |
| none | flow_grid_sequence | object_shuffle | 0.2083 | 15/72 |
| none | flow_grid_sequence | scene_transition | 0.4444 | 32/72 |
| none | flow_grid_sequence | state_change | 0.3889 | 28/72 |
| none | flow_grid_sequence | unexpected_action | 0.2222 | 16/72 |
