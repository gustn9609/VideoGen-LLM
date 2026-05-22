# Wan Feature Alternatives Toy Experiments

## Feature Probe Summary

| Task | Best feature | Best bal acc | raw Wan 1x1 | Wan dynamics | Structured compact | WPS | raw+WPS | Pixel dynamics |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| direction4 | pixel_delta_2x2 | 1.0000 | 0.8125 | 0.6667 | 0.7292 | 0.9583 | 0.9375 | 1.0000 |
| action_order | pixel_raw_1x1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| repetition_count | pixel_delta_2x2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9167 | 1.0000 | 1.0000 |
| same_first_last_path | pixel_raw_2x2 | 1.0000 | 0.7500 | 0.6250 | 1.0000 | 0.8333 | 0.7917 | 1.0000 |
| contact_causality | pixel_raw_2x2 | 1.0000 | 0.9444 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| camera_object_motion | pixel_raw_2x2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| shuffle_detection | pixel_structured_compact | 1.0000 | 1.0000 | 0.8333 | 0.7083 | 1.0000 | 1.0000 | 0.5417 |
| rhythm | wan_wps_signature | 0.9444 | 0.8611 | 0.8333 | 0.7222 | 0.9444 | 0.9444 | 0.7778 |

## WPS Gate

| Task | Coverage | Base | Raw Wan everywhere | Score ensemble | WPS gate switch | Helps | Hurts | Gate=1 base | Gate=1 Wan |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direction4 | 0.0000 | 1.0000 | 0.8125 | 0.9792 | 1.0000 | 0 | 0 | 0.0000 | 0.0000 |
| action_order | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0.0000 | 0.0000 |
| repetition_count | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0.0000 | 0.0000 |
| same_first_last_path | 0.0000 | 1.0000 | 0.7500 | 1.0000 | 1.0000 | 0 | 0 | 0.0000 | 0.0000 |
| contact_causality | 0.0000 | 1.0000 | 0.9444 | 1.0000 | 1.0000 | 0 | 0 | 0.0000 | 0.0000 |
| camera_object_motion | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0.0000 | 0.0000 |
| shuffle_detection | 0.2083 | 0.5417 | 1.0000 | 0.6250 | 0.7083 | 4 | 0 | 0.2000 | 1.0000 |
| rhythm | 0.0000 | 0.7778 | 0.8611 | 0.8056 | 0.7778 | 0 | 0 | 0.0000 | 0.0000 |

## Score Contrast

| Task | shuffle | reverse | timeavg | lowfps | first_repeat |
|---|---:|---:|---:|---:|---:|
| direction4 | 0.2500 | 1.0000 | 0.4167 | 0.7708 | 0.5000 |
| action_order | 0.9583 | 1.0000 | 0.6667 | 1.0000 | 1.0000 |
| repetition_count | 0.6389 | 0.1944 | 0.6667 | 0.7500 | 0.7778 |
| same_first_last_path | 0.7500 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| contact_causality | 0.5278 | 0.6667 | 0.6667 | 0.3333 | 0.6667 |
| camera_object_motion | 0.5833 | 0.6667 | 0.3333 | 0.6667 | 0.6389 |
| shuffle_detection | 0.7917 | 0.6250 | 0.5000 | 1.0000 | 0.5000 |
| rhythm | 0.6111 | 0.5833 | 0.6667 | 0.7222 | 0.6667 |

## CFG / Future Consistency Proxies

| Task | CFG alignment proxy | Future consistency proxy |
|---|---:|---:|
| direction4 | 0.7917 | 0.1875 |
| action_order | 1.0000 | 1.0000 |
| repetition_count | 1.0000 | 1.0000 |
| same_first_last_path | 0.6667 | 0.7083 |
| contact_causality | 1.0000 | 1.0000 |
| camera_object_motion | 1.0000 | 0.6111 |
| shuffle_detection | 0.9583 | 0.7500 |
| rhythm | 0.8333 | 0.6944 |

## Segment Localization

| Selector | Top-1 recall | Top-3 recall |
|---|---:|---:|
| uniform | 0.0000 | 0.5278 |
| pixel_motion | 0.0000 | 0.5278 |
| flow_motion | 0.0000 | 0.5278 |
| wan_temporal_sensitivity | 0.4722 | 0.5556 |

## Notes

- CFG and future-consistency entries are feature-space toy proxies, not Wan generation runs.
- All Wan rows use real Wan-VAE features when `feature_source=wan`.
