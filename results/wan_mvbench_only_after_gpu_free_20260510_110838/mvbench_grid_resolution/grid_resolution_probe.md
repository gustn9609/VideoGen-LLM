# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | wan_vae_grid_1x1 | hash | logistic | 0.3411 | 0.0304 | 0.3162 | 0.3668 | 438/1284 | 2.7936 |
| none | wan_vae_grid_2x2 | hash | logistic | 0.3239 | 0.0481 | 0.2991 | 0.3489 | 416/1284 | 2.8328 |
| none | wan_vae_grid_4x4 | hash | logistic | 0.3536 | 0.0568 | 0.3271 | 0.3808 | 454/1284 | 2.6055 |
| none | wan_vae_grid_8x8 | hash | logistic | 0.3552 | 0.0441 | 0.3310 | 0.3816 | 456/1284 | 2.7078 |
| none | wan_vae_grid_16x16 | hash | logistic | 0.3395 | 0.0306 | 0.3138 | 0.3637 | 436/1284 | 2.7131 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | wan_vae_grid_1x1 | action_antonym | 0.5417 | 39/72 |
| none | wan_vae_grid_1x1 | action_count | 0.3056 | 22/72 |
| none | wan_vae_grid_1x1 | action_localization | 0.3472 | 25/72 |
| none | wan_vae_grid_1x1 | action_prediction | 0.3472 | 25/72 |
| none | wan_vae_grid_1x1 | action_sequence | 0.2222 | 16/72 |
| none | wan_vae_grid_1x1 | character_order | 0.3333 | 21/63 |
| none | wan_vae_grid_1x1 | counterfactual_inference | 0.2319 | 16/69 |
| none | wan_vae_grid_1x1 | egocentric_navigation | 0.3333 | 24/72 |
| none | wan_vae_grid_1x1 | fine_grained_action | 0.3750 | 27/72 |
| none | wan_vae_grid_1x1 | moving_attribute | 0.3889 | 28/72 |
| none | wan_vae_grid_1x1 | moving_count | 0.2500 | 18/72 |
| none | wan_vae_grid_1x1 | moving_direction | 0.3194 | 23/72 |
| none | wan_vae_grid_1x1 | object_existence | 0.4583 | 33/72 |
| none | wan_vae_grid_1x1 | object_interaction | 0.1944 | 14/72 |
| none | wan_vae_grid_1x1 | object_shuffle | 0.3056 | 22/72 |
| none | wan_vae_grid_1x1 | scene_transition | 0.5972 | 43/72 |
| none | wan_vae_grid_1x1 | state_change | 0.3611 | 26/72 |
| none | wan_vae_grid_1x1 | unexpected_action | 0.2222 | 16/72 |
| none | wan_vae_grid_2x2 | action_antonym | 0.5556 | 40/72 |
| none | wan_vae_grid_2x2 | action_count | 0.2778 | 20/72 |
| none | wan_vae_grid_2x2 | action_localization | 0.3333 | 24/72 |
| none | wan_vae_grid_2x2 | action_prediction | 0.2361 | 17/72 |
| none | wan_vae_grid_2x2 | action_sequence | 0.2083 | 15/72 |
| none | wan_vae_grid_2x2 | character_order | 0.3016 | 19/63 |
| none | wan_vae_grid_2x2 | counterfactual_inference | 0.2029 | 14/69 |
| none | wan_vae_grid_2x2 | egocentric_navigation | 0.2361 | 17/72 |
| none | wan_vae_grid_2x2 | fine_grained_action | 0.2222 | 16/72 |
| none | wan_vae_grid_2x2 | moving_attribute | 0.3889 | 28/72 |
| none | wan_vae_grid_2x2 | moving_count | 0.2500 | 18/72 |
| none | wan_vae_grid_2x2 | moving_direction | 0.3889 | 28/72 |
| none | wan_vae_grid_2x2 | object_existence | 0.3889 | 28/72 |
| none | wan_vae_grid_2x2 | object_interaction | 0.2361 | 17/72 |
| none | wan_vae_grid_2x2 | object_shuffle | 0.4167 | 30/72 |
| none | wan_vae_grid_2x2 | scene_transition | 0.6667 | 48/72 |
| none | wan_vae_grid_2x2 | state_change | 0.3889 | 28/72 |
| none | wan_vae_grid_2x2 | unexpected_action | 0.1250 | 9/72 |
| none | wan_vae_grid_4x4 | action_antonym | 0.5417 | 39/72 |
| none | wan_vae_grid_4x4 | action_count | 0.3472 | 25/72 |
| none | wan_vae_grid_4x4 | action_localization | 0.3750 | 27/72 |
| none | wan_vae_grid_4x4 | action_prediction | 0.3611 | 26/72 |
| none | wan_vae_grid_4x4 | action_sequence | 0.2778 | 20/72 |
| none | wan_vae_grid_4x4 | character_order | 0.3968 | 25/63 |
| none | wan_vae_grid_4x4 | counterfactual_inference | 0.1594 | 11/69 |
| none | wan_vae_grid_4x4 | egocentric_navigation | 0.3472 | 25/72 |
| none | wan_vae_grid_4x4 | fine_grained_action | 0.1944 | 14/72 |
| none | wan_vae_grid_4x4 | moving_attribute | 0.3194 | 23/72 |
| none | wan_vae_grid_4x4 | moving_count | 0.2500 | 18/72 |
| none | wan_vae_grid_4x4 | moving_direction | 0.3611 | 26/72 |
| none | wan_vae_grid_4x4 | object_existence | 0.4306 | 31/72 |
| none | wan_vae_grid_4x4 | object_interaction | 0.2917 | 21/72 |
| none | wan_vae_grid_4x4 | object_shuffle | 0.4306 | 31/72 |
| none | wan_vae_grid_4x4 | scene_transition | 0.5139 | 37/72 |
| none | wan_vae_grid_4x4 | state_change | 0.4444 | 32/72 |
| none | wan_vae_grid_4x4 | unexpected_action | 0.3194 | 23/72 |
| none | wan_vae_grid_8x8 | action_antonym | 0.5694 | 41/72 |
| none | wan_vae_grid_8x8 | action_count | 0.2778 | 20/72 |
| none | wan_vae_grid_8x8 | action_localization | 0.4028 | 29/72 |
| none | wan_vae_grid_8x8 | action_prediction | 0.2361 | 17/72 |
| none | wan_vae_grid_8x8 | action_sequence | 0.1667 | 12/72 |
| none | wan_vae_grid_8x8 | character_order | 0.4286 | 27/63 |
| none | wan_vae_grid_8x8 | counterfactual_inference | 0.2174 | 15/69 |
| none | wan_vae_grid_8x8 | egocentric_navigation | 0.3056 | 22/72 |
| none | wan_vae_grid_8x8 | fine_grained_action | 0.1944 | 14/72 |
| none | wan_vae_grid_8x8 | moving_attribute | 0.3750 | 27/72 |
| none | wan_vae_grid_8x8 | moving_count | 0.2500 | 18/72 |
| none | wan_vae_grid_8x8 | moving_direction | 0.5278 | 38/72 |
| none | wan_vae_grid_8x8 | object_existence | 0.5278 | 38/72 |
| none | wan_vae_grid_8x8 | object_interaction | 0.2639 | 19/72 |
| none | wan_vae_grid_8x8 | object_shuffle | 0.6111 | 44/72 |
| none | wan_vae_grid_8x8 | scene_transition | 0.4722 | 34/72 |
| none | wan_vae_grid_8x8 | state_change | 0.3750 | 27/72 |
| none | wan_vae_grid_8x8 | unexpected_action | 0.1944 | 14/72 |
| none | wan_vae_grid_16x16 | action_antonym | 0.5139 | 37/72 |
| none | wan_vae_grid_16x16 | action_count | 0.2361 | 17/72 |
| none | wan_vae_grid_16x16 | action_localization | 0.2917 | 21/72 |
| none | wan_vae_grid_16x16 | action_prediction | 0.3056 | 22/72 |
| none | wan_vae_grid_16x16 | action_sequence | 0.2500 | 18/72 |
| none | wan_vae_grid_16x16 | character_order | 0.4444 | 28/63 |
| none | wan_vae_grid_16x16 | counterfactual_inference | 0.1449 | 10/69 |
| none | wan_vae_grid_16x16 | egocentric_navigation | 0.2500 | 18/72 |
| none | wan_vae_grid_16x16 | fine_grained_action | 0.3194 | 23/72 |
| none | wan_vae_grid_16x16 | moving_attribute | 0.3889 | 28/72 |
| none | wan_vae_grid_16x16 | moving_count | 0.2500 | 18/72 |
| none | wan_vae_grid_16x16 | moving_direction | 0.3611 | 26/72 |
| none | wan_vae_grid_16x16 | object_existence | 0.5278 | 38/72 |
| none | wan_vae_grid_16x16 | object_interaction | 0.2639 | 19/72 |
| none | wan_vae_grid_16x16 | object_shuffle | 0.3611 | 26/72 |
| none | wan_vae_grid_16x16 | scene_transition | 0.5278 | 38/72 |
| none | wan_vae_grid_16x16 | state_change | 0.4167 | 30/72 |
| none | wan_vae_grid_16x16 | unexpected_action | 0.2639 | 19/72 |
