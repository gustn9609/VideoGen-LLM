# Wan Feature Alternatives Toy Experiments

## Feature Probe Summary

| Task | Best feature | Best bal acc | raw Wan 1x1 | Wan dynamics | Structured compact | WPS | raw+WPS | Pixel dynamics |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| direction4 | pixel_raw_4x4 | 1.0000 | nan | nan | nan | 0.5000 | 0.3750 | 1.0000 |
| action_order | pixel_raw_1x1 | 1.0000 | nan | nan | nan | 1.0000 | 1.0000 | 1.0000 |

## WPS Gate

| Task | Coverage | Base | Raw Wan everywhere | Score ensemble | WPS gate switch | Helps | Hurts | Gate=1 base | Gate=1 Wan |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direction4 | 0.0000 | 1.0000 | 0.5000 | 0.7500 | 1.0000 | 0 | 0 | 0.0000 | 0.0000 |
| action_order | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0.0000 | 0.0000 |

## Score Contrast

| Task | shuffle | reverse | timeavg | lowfps | first_repeat |
|---|---:|---:|---:|---:|---:|
| direction4 | 0.5000 | 0.5000 | 0.5000 | 0.0000 | 0.5000 |
| action_order | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 |

## CFG / Future Consistency Proxies

| Task | CFG alignment proxy | Future consistency proxy |
|---|---:|---:|
| direction4 | 0.3750 | 0.2500 |
| action_order | 1.0000 | 0.5000 |

## Segment Localization

| Selector | Top-1 recall | Top-3 recall |
|---|---:|---:|
| uniform | 0.0000 | 1.0000 |
| pixel_motion | 1.0000 | 1.0000 |
| flow_motion | 1.0000 | 1.0000 |
| wan_temporal_sensitivity | 1.0000 | 1.0000 |

## Notes

- CFG and future-consistency entries are feature-space toy proxies, not Wan generation runs.
- All Wan rows use real Wan-VAE features when `feature_source=wan`.
