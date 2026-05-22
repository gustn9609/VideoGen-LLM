# MotionBench Factorial Mini-Sweep

| Pool | Frames | Sampling | Acc | Text-only gain | Pixel gap | Flow gap | Normal-shuffle gap |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1x1 | 8 | all | 0.4418 | -0.0135 | -0.0030 | 0.0109 | 0.0002 |
| 1x1 | 8 | high-motion | 0.4381 | -0.0172 | 0.0002 | -0.0168 | 0.0068 |
| 1x1 | 8 | camera-comp | 0.4347 | -0.0205 | -0.0100 | -0.0065 | 0.0035 |
| 1x1 | 8 | high-motion+camera-comp | 0.4519 | -0.0033 | 0.0140 | 0.0074 | 0.0207 |
| 1x1 | 16 | all | 0.4346 | -0.0207 | -0.0240 | 0.0282 | 0.0068 |
| 1x1 | 16 | high-motion | 0.4381 | -0.0172 | -0.0070 | -0.0033 | -0.0002 |
| 1x1 | 16 | camera-comp | 0.4275 | -0.0277 | -0.0242 | -0.0032 | -0.0002 |
| 1x1 | 16 | high-motion+camera-comp | 0.4311 | -0.0242 | -0.0072 | -0.0109 | -0.0107 |
| 2x2 | 8 | all | 0.3754 | -0.0798 | -0.0346 | -0.0725 | 0.0105 |
| 2x2 | 8 | high-motion | 0.3686 | -0.0867 | -0.0275 | -0.0453 | 0.0039 |
| 2x2 | 8 | camera-comp | 0.3754 | -0.0798 | -0.0514 | -0.0447 | 0.0072 |
| 2x2 | 8 | high-motion+camera-comp | 0.3721 | -0.0832 | -0.0307 | -0.0835 | 0.0074 |
| 2x2 | 16 | all | 0.3993 | -0.0560 | -0.0104 | -0.0523 | 0.0275 |
| 2x2 | 16 | high-motion | 0.4028 | -0.0525 | -0.0209 | -0.0595 | 0.0137 |
| 2x2 | 16 | camera-comp | 0.3851 | -0.0702 | -0.0316 | -0.0595 | 0.0204 |
| 2x2 | 16 | high-motion+camera-comp | 0.3923 | -0.0630 | -0.0314 | -0.0628 | 0.0135 |
| 4x4 | 8 | all | 0.3679 | -0.0874 | 0.0214 | -0.0628 | -0.0075 |
| 4x4 | 8 | high-motion | 0.3649 | -0.0904 | 0.0011 | -0.0689 | -0.0246 |
| 4x4 | 8 | camera-comp | 0.3679 | -0.0874 | 0.0177 | -0.0695 | 0.0028 |
| 4x4 | 8 | high-motion+camera-comp | 0.3788 | -0.0765 | 0.0114 | -0.0553 | -0.0072 |
| 4x4 | 16 | all | 0.3609 | -0.0944 | 0.0625 | -0.0523 | -0.0067 |
| 4x4 | 16 | high-motion | 0.3198 | -0.1354 | -0.0168 | -0.0475 | -0.0656 |
| 4x4 | 16 | camera-comp | 0.3474 | -0.1079 | -0.0623 | -0.0560 | -0.0521 |
| 4x4 | 16 | high-motion+camera-comp | 0.3546 | -0.1007 | -0.0067 | -0.0863 | -0.0202 |

## Temporal Robustness

| Pool | Frames | Sampling | Normal-reverse gap | Normal-first-frame gap |
|---:|---:|---|---:|---:|
| 1x1 | 8 | all | 0.0172 | -0.0104 |
| 1x1 | 8 | high-motion | 0.0033 | -0.0107 |
| 1x1 | 8 | camera-comp | 0.0137 | -0.0174 |
| 1x1 | 8 | high-motion+camera-comp | 0.0137 | 0.0032 |
| 1x1 | 16 | all | 0.0002 | -0.0175 |
| 1x1 | 16 | high-motion | -0.0033 | -0.0246 |
| 1x1 | 16 | camera-comp | -0.0033 | -0.0246 |
| 1x1 | 16 | high-motion+camera-comp | -0.0104 | -0.0316 |
| 2x2 | 8 | all | -0.0102 | -0.0451 |
| 2x2 | 8 | high-motion | -0.0205 | -0.0486 |
| 2x2 | 8 | camera-comp | -0.0137 | -0.0451 |
| 2x2 | 8 | high-motion+camera-comp | -0.0205 | -0.0451 |
| 2x2 | 16 | all | 0.0209 | -0.0212 |
| 2x2 | 16 | high-motion | 0.0175 | -0.0247 |
| 2x2 | 16 | camera-comp | 0.0032 | -0.0354 |
| 2x2 | 16 | high-motion+camera-comp | -0.0002 | -0.0353 |
| 4x4 | 8 | all | -0.0072 | -0.0316 |
| 4x4 | 8 | high-motion | -0.0212 | -0.0384 |
| 4x4 | 8 | camera-comp | -0.0175 | -0.0316 |
| 4x4 | 8 | high-motion+camera-comp | -0.0109 | -0.0246 |
| 4x4 | 16 | all | -0.0386 | -0.0386 |
| 4x4 | 16 | high-motion | -0.0174 | -0.0800 |
| 4x4 | 16 | camera-comp | -0.0232 | -0.0521 |
| 4x4 | 16 | high-motion+camera-comp | -0.0167 | -0.0453 |

## Wan + Pixel/Flow Ensembles

| Pool | Frames | Sampling | Wan+Pixel | Wan+Flow | Wan+Pixel+Flow |
|---:|---:|---|---:|---:|---:|
| 1x1 | 8 | all | 0.4410 | 0.4306 | 0.4549 |
| 1x1 | 8 | high-motion | 0.4410 | 0.4306 | 0.4444 |
| 1x1 | 8 | camera-comp | 0.4410 | 0.4306 | 0.4410 |
| 1x1 | 8 | high-motion+camera-comp | 0.4479 | 0.4271 | 0.4340 |
| 1x1 | 16 | all | 0.4514 | 0.4340 | 0.4514 |
| 1x1 | 16 | high-motion | 0.4375 | 0.4618 | 0.4618 |
| 1x1 | 16 | camera-comp | 0.4410 | 0.4201 | 0.4410 |
| 1x1 | 16 | high-motion+camera-comp | 0.4410 | 0.4514 | 0.4583 |
| 2x2 | 8 | all | 0.4097 | 0.4132 | 0.4097 |
| 2x2 | 8 | high-motion | 0.3993 | 0.4028 | 0.4132 |
| 2x2 | 8 | camera-comp | 0.4062 | 0.3924 | 0.4132 |
| 2x2 | 8 | high-motion+camera-comp | 0.3889 | 0.4201 | 0.4236 |
| 2x2 | 16 | all | 0.3924 | 0.4271 | 0.4132 |
| 2x2 | 16 | high-motion | 0.3993 | 0.4514 | 0.4410 |
| 2x2 | 16 | camera-comp | 0.3854 | 0.4167 | 0.4132 |
| 2x2 | 16 | high-motion+camera-comp | 0.3854 | 0.4479 | 0.4340 |
| 4x4 | 8 | all | 0.3715 | 0.4132 | 0.4028 |
| 4x4 | 8 | high-motion | 0.3750 | 0.4271 | 0.4028 |
| 4x4 | 8 | camera-comp | 0.3785 | 0.4340 | 0.4062 |
| 4x4 | 8 | high-motion+camera-comp | 0.3646 | 0.4167 | 0.4132 |
| 4x4 | 16 | all | 0.3299 | 0.4201 | 0.3958 |
| 4x4 | 16 | high-motion | 0.3368 | 0.3507 | 0.3750 |
| 4x4 | 16 | camera-comp | 0.3889 | 0.3681 | 0.4236 |
| 4x4 | 16 | high-motion+camera-comp | 0.3264 | 0.4062 | 0.4062 |

## Per-Type Wan Incremental View

| Pool | Frames | Sampling | Type | Wan Acc | Text-only gain | Pixel gap | Flow gap |
|---:|---:|---|---|---:|---:|---:|---:|
| 1x1 | 8 | all | Action Order | 0.2778 | -0.0417 | -0.0139 | -0.0139 |
| 1x1 | 8 | all | Motion Recognition | 0.4583 | -0.0417 | 0.0139 | 0.0000 |
| 1x1 | 8 | all | Motion-related Objects | 0.5139 | -0.0278 | -0.0139 | 0.0000 |
| 1x1 | 8 | all | Repetition Count | 0.5139 | 0.0556 | 0.0000 | 0.0556 |
| 1x1 | 8 | high-motion | Action Order | 0.2778 | -0.0417 | 0.0000 | -0.0417 |
| 1x1 | 8 | high-motion | Motion Recognition | 0.4861 | -0.0139 | 0.0417 | 0.0000 |
| 1x1 | 8 | high-motion | Motion-related Objects | 0.4722 | -0.0694 | -0.0417 | -0.0694 |
| 1x1 | 8 | high-motion | Repetition Count | 0.5139 | 0.0556 | 0.0000 | 0.0417 |
| 1x1 | 8 | camera-comp | Action Order | 0.2917 | -0.0278 | 0.0139 | 0.0000 |
| 1x1 | 8 | camera-comp | Motion Recognition | 0.4444 | -0.0556 | -0.0139 | 0.0139 |
| 1x1 | 8 | camera-comp | Motion-related Objects | 0.4861 | -0.0556 | -0.0417 | -0.0556 |
| 1x1 | 8 | camera-comp | Repetition Count | 0.5139 | 0.0556 | 0.0000 | 0.0139 |
| 1x1 | 8 | high-motion+camera-comp | Action Order | 0.3056 | -0.0139 | 0.0278 | -0.0556 |
| 1x1 | 8 | high-motion+camera-comp | Motion Recognition | 0.5000 | 0.0000 | 0.0556 | 0.0833 |
| 1x1 | 8 | high-motion+camera-comp | Motion-related Objects | 0.4722 | -0.0694 | -0.0417 | -0.0417 |
| 1x1 | 8 | high-motion+camera-comp | Repetition Count | 0.5278 | 0.0694 | 0.0139 | 0.0417 |
| 1x1 | 16 | all | Action Order | 0.3056 | -0.0139 | -0.0417 | 0.0833 |
| 1x1 | 16 | all | Motion Recognition | 0.4167 | -0.0833 | -0.0278 | -0.0139 |
| 1x1 | 16 | all | Motion-related Objects | 0.5139 | -0.0278 | -0.0278 | -0.0139 |
| 1x1 | 16 | all | Repetition Count | 0.5000 | 0.0417 | 0.0000 | 0.0556 |
| 1x1 | 16 | high-motion | Action Order | 0.2917 | -0.0278 | -0.0278 | 0.0000 |
| 1x1 | 16 | high-motion | Motion Recognition | 0.4444 | -0.0556 | 0.0417 | -0.0417 |
| 1x1 | 16 | high-motion | Motion-related Objects | 0.5139 | -0.0278 | -0.0417 | -0.0139 |
| 1x1 | 16 | high-motion | Repetition Count | 0.5000 | 0.0417 | 0.0000 | 0.0417 |
| 1x1 | 16 | camera-comp | Action Order | 0.3056 | -0.0139 | -0.0417 | 0.0694 |
| 1x1 | 16 | camera-comp | Motion Recognition | 0.3889 | -0.1111 | -0.0417 | -0.0833 |
| 1x1 | 16 | camera-comp | Motion-related Objects | 0.5139 | -0.0278 | -0.0278 | -0.0139 |
| 1x1 | 16 | camera-comp | Repetition Count | 0.5000 | 0.0417 | 0.0139 | 0.0139 |
| 1x1 | 16 | high-motion+camera-comp | Action Order | 0.2778 | -0.0417 | -0.0417 | -0.0139 |
| 1x1 | 16 | high-motion+camera-comp | Motion Recognition | 0.4167 | -0.0833 | 0.0278 | -0.0417 |
| 1x1 | 16 | high-motion+camera-comp | Motion-related Objects | 0.5139 | -0.0278 | -0.0278 | -0.0139 |
| 1x1 | 16 | high-motion+camera-comp | Repetition Count | 0.5139 | 0.0556 | 0.0139 | 0.0278 |
| 2x2 | 8 | all | Action Order | 0.2361 | -0.0833 | -0.0417 | -0.0278 |
| 2x2 | 8 | all | Motion Recognition | 0.3333 | -0.1667 | -0.0972 | -0.2222 |
| 2x2 | 8 | all | Motion-related Objects | 0.4861 | -0.0556 | 0.0000 | -0.0694 |
| 2x2 | 8 | all | Repetition Count | 0.4444 | -0.0139 | 0.0000 | 0.0278 |
| 2x2 | 8 | high-motion | Action Order | 0.2361 | -0.0833 | -0.0278 | 0.0000 |
| 2x2 | 8 | high-motion | Motion Recognition | 0.3194 | -0.1806 | -0.0833 | -0.1389 |
| 2x2 | 8 | high-motion | Motion-related Objects | 0.4722 | -0.0694 | -0.0139 | -0.0278 |
| 2x2 | 8 | high-motion | Repetition Count | 0.4444 | -0.0139 | 0.0139 | -0.0139 |
| 2x2 | 8 | camera-comp | Action Order | 0.2361 | -0.0833 | -0.0694 | -0.0139 |
| 2x2 | 8 | camera-comp | Motion Recognition | 0.3333 | -0.1667 | -0.1250 | -0.1806 |
| 2x2 | 8 | camera-comp | Motion-related Objects | 0.4861 | -0.0556 | -0.0139 | -0.0278 |
| 2x2 | 8 | camera-comp | Repetition Count | 0.4444 | -0.0139 | 0.0000 | 0.0417 |
| 2x2 | 8 | high-motion+camera-comp | Action Order | 0.2361 | -0.0833 | -0.0417 | -0.0833 |
| 2x2 | 8 | high-motion+camera-comp | Motion Recognition | 0.3194 | -0.1806 | -0.0833 | -0.1528 |
| 2x2 | 8 | high-motion+camera-comp | Motion-related Objects | 0.4861 | -0.0556 | -0.0139 | -0.0694 |
| 2x2 | 8 | high-motion+camera-comp | Repetition Count | 0.4444 | -0.0139 | 0.0139 | -0.0278 |
| 2x2 | 16 | all | Action Order | 0.2500 | -0.0694 | -0.0694 | -0.0417 |
| 2x2 | 16 | all | Motion Recognition | 0.4444 | -0.0556 | 0.0417 | -0.0833 |
| 2x2 | 16 | all | Motion-related Objects | 0.4306 | -0.1111 | -0.0278 | -0.0833 |
| 2x2 | 16 | all | Repetition Count | 0.4722 | 0.0139 | 0.0139 | 0.0000 |
| 2x2 | 16 | high-motion | Action Order | 0.2500 | -0.0694 | -0.0972 | -0.1250 |
| 2x2 | 16 | high-motion | Motion Recognition | 0.4583 | -0.0417 | 0.0278 | -0.0694 |
| 2x2 | 16 | high-motion | Motion-related Objects | 0.4306 | -0.1111 | -0.0278 | -0.0556 |
| 2x2 | 16 | high-motion | Repetition Count | 0.4722 | 0.0139 | 0.0139 | 0.0139 |
| 2x2 | 16 | camera-comp | Action Order | 0.2083 | -0.1111 | -0.1250 | -0.0972 |
| 2x2 | 16 | camera-comp | Motion Recognition | 0.4444 | -0.0556 | 0.0278 | -0.0556 |
| 2x2 | 16 | camera-comp | Motion-related Objects | 0.4306 | -0.1111 | -0.0278 | -0.0833 |
| 2x2 | 16 | camera-comp | Repetition Count | 0.4583 | 0.0000 | 0.0000 | 0.0000 |
| 2x2 | 16 | high-motion+camera-comp | Action Order | 0.2361 | -0.0833 | -0.1111 | -0.1111 |
| 2x2 | 16 | high-motion+camera-comp | Motion Recognition | 0.4306 | -0.0694 | 0.0000 | -0.0972 |
| 2x2 | 16 | high-motion+camera-comp | Motion-related Objects | 0.4306 | -0.1111 | -0.0278 | -0.0278 |
| 2x2 | 16 | high-motion+camera-comp | Repetition Count | 0.4722 | 0.0139 | 0.0139 | -0.0139 |
| 4x4 | 8 | all | Action Order | 0.3194 | 0.0000 | 0.1250 | -0.0556 |
| 4x4 | 8 | all | Motion Recognition | 0.3750 | -0.1250 | 0.0139 | -0.1111 |
| 4x4 | 8 | all | Motion-related Objects | 0.3889 | -0.1528 | -0.0278 | -0.0417 |
| 4x4 | 8 | all | Repetition Count | 0.3889 | -0.0694 | -0.0278 | -0.0417 |
| 4x4 | 8 | high-motion | Action Order | 0.3056 | -0.0139 | 0.0694 | -0.1111 |
| 4x4 | 8 | high-motion | Motion Recognition | 0.4306 | -0.0694 | 0.0278 | -0.0417 |
| 4x4 | 8 | high-motion | Motion-related Objects | 0.3611 | -0.1806 | -0.0417 | -0.0694 |
| 4x4 | 8 | high-motion | Repetition Count | 0.3611 | -0.0972 | -0.0556 | -0.0556 |
| 4x4 | 8 | camera-comp | Action Order | 0.3194 | 0.0000 | 0.1111 | -0.0694 |
| 4x4 | 8 | camera-comp | Motion Recognition | 0.3611 | -0.1389 | 0.0000 | -0.0972 |
| 4x4 | 8 | camera-comp | Motion-related Objects | 0.3889 | -0.1528 | -0.0417 | -0.0556 |
| 4x4 | 8 | camera-comp | Repetition Count | 0.4028 | -0.0556 | 0.0000 | -0.0556 |
| 4x4 | 8 | high-motion+camera-comp | Action Order | 0.3611 | 0.0417 | 0.1250 | -0.0139 |
| 4x4 | 8 | high-motion+camera-comp | Motion Recognition | 0.3889 | -0.1111 | -0.0139 | -0.0833 |
| 4x4 | 8 | high-motion+camera-comp | Motion-related Objects | 0.4028 | -0.1389 | -0.0139 | -0.0417 |
| 4x4 | 8 | high-motion+camera-comp | Repetition Count | 0.3611 | -0.0972 | -0.0556 | -0.0833 |
| 4x4 | 16 | all | Action Order | 0.2778 | -0.0417 | 0.1389 | -0.0556 |
| 4x4 | 16 | all | Motion Recognition | 0.3889 | -0.1111 | 0.0278 | -0.1111 |
| 4x4 | 16 | all | Motion-related Objects | 0.3333 | -0.2083 | 0.0000 | -0.0556 |
| 4x4 | 16 | all | Repetition Count | 0.4444 | -0.0139 | 0.0833 | 0.0139 |
| 4x4 | 16 | high-motion | Action Order | 0.2361 | -0.0833 | -0.0278 | -0.0278 |
| 4x4 | 16 | high-motion | Motion Recognition | 0.3056 | -0.1944 | -0.0694 | -0.1667 |
| 4x4 | 16 | high-motion | Motion-related Objects | 0.3750 | -0.1667 | 0.0139 | 0.0139 |
| 4x4 | 16 | high-motion | Repetition Count | 0.3611 | -0.0972 | 0.0139 | -0.0139 |
| 4x4 | 16 | camera-comp | Action Order | 0.2361 | -0.0833 | 0.0000 | -0.0556 |
| 4x4 | 16 | camera-comp | Motion Recognition | 0.3333 | -0.1667 | -0.1944 | -0.1806 |
| 4x4 | 16 | camera-comp | Motion-related Objects | 0.4167 | -0.1250 | 0.0139 | 0.0694 |
| 4x4 | 16 | camera-comp | Repetition Count | 0.4028 | -0.0556 | -0.0694 | -0.0556 |
| 4x4 | 16 | high-motion+camera-comp | Action Order | 0.2639 | -0.0556 | -0.0278 | -0.1944 |
| 4x4 | 16 | high-motion+camera-comp | Motion Recognition | 0.3333 | -0.1667 | -0.0833 | -0.0972 |
| 4x4 | 16 | high-motion+camera-comp | Motion-related Objects | 0.3750 | -0.1667 | 0.0139 | -0.0694 |
| 4x4 | 16 | high-motion+camera-comp | Repetition Count | 0.4444 | -0.0139 | 0.0694 | 0.0139 |
