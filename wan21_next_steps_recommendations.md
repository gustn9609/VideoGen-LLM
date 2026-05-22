# Wan2.1 Low-FPS Augmentation 결과 기반 다음 실험 추천

## 0. 핵심 결론

현재 결과 기준으로 다음 단계의 핵심 질문은 더 이상 **“Wan feature가 motion을 담고 있는가?”**가 아닙니다. 이미 여러 toy temporal task에서 Wan-VAE feature, 특히 `wan_vae_grid_sequence`가 강한 motion/order/count 정보를 담고 있음이 확인되었습니다.

이제 집중해야 할 질문은 아래 세 가지입니다.

1. **Wan-VAE grid feature가 raw pixel / optical flow 같은 강한 baseline보다 진짜 나은가?**
2. **toy synthetic probe에서의 100% 성능이 real video / real VideoQA에서도 유지되는가?**
3. **VideoLLM에 붙였을 때 token budget 대비 실질적인 성능 향상이 나는가?**

현재 실험 결과만 놓고 보면:

- `wan_vae_grid_sequence`는 거의 모든 temporal toy task와 low-fps shift에서 가장 안전한 default feature입니다.
- `wan_vae_global_sequence`는 compact하지만 fps shift에 매우 취약합니다.
- low-fps augmentation은 `global_delta`, `global_sequence`, DiT token을 어느 정도 rescue하지만, direction task에서는 충분하지 않습니다.
- 따라서 다음 실험은 **grid-preserving Wan motion token을 중심으로, real benchmark transfer와 fair baseline 검증**으로 가는 것이 가장 좋습니다.

---

## 1. 현재 결과 요약

### 1.1 Low-FPS augmentation 주요 결과

| task | feature | full→low | full+low→low | gain |
|---|---:|---:|---:|---:|
| direction4 | `dit_l14_t900_token_mean` | 28.6% | 57.8% | +29.2 |
| direction4 | `wan_vae_global_delta` | 27.1% | 42.7% | +15.6 |
| direction4 | `wan_vae_global_sequence` | 25.0% | 28.1% | +3.1 |
| direction4 | `wan_vae_grid_sequence` | 100.0% | 100.0% | 0.0 |
| before_after_cycle | `wan_vae_global_delta` | 53.1% | 97.9% | +44.8 |
| before_after_cycle | `wan_vae_global_sequence` | 57.3% | 89.6% | +32.3 |
| before_after_cycle | `dit_l14_t900_token_mean` | 51.0% | 65.6% | +14.6 |
| before_after_cycle | `wan_vae_grid_sequence` | 100.0% | 100.0% | 0.0 |

### 1.2 해석

- `wan_vae_grid_sequence`는 augmentation 없이도 low-fps shift에 매우 강합니다.
- `wan_vae_global_sequence`는 full-fps에서는 강할 수 있지만, low-fps shift에서 무너집니다.
- `wan_vae_global_delta`는 before/after order에서는 low-fps augmentation으로 크게 회복됩니다.
- `dit_l14_t900_token_mean`은 direction에서 augmentation gain이 크지만, 절대 성능은 grid VAE보다 낮습니다.
- 따라서 **spatial grid를 보존하는 것이 low-fps robustness에 핵심**으로 보입니다.

---

## 2. 가장 먼저 해야 할 것: Wan 고유 이득을 분리하는 control 실험

현재 결과에서 가장 조심해야 할 점은 `wan_vae_grid_sequence`가 강하지만, 일부 task에서는 `pixel_grid_sequence`나 `pixel_grid_delta`도 거의 완벽하다는 점입니다.

즉 현 단계의 안전한 결론은:

> spatial grid를 보존하면 temporal task가 잘 풀린다.

아직 완전히 증명되지 않은 결론은:

> Wan latent라서 raw pixel / optical flow보다 더 잘 풀린다.

따라서 다음 control 실험이 중요합니다.

| Control | 목적 | 추천 비교 |
|---|---|---|
| 동일 token budget 비교 | VAE grid가 단순히 token 수가 많아서 좋은지 확인 | pixel grid, optical flow grid, VAE grid를 모두 8/16/32/64 token으로 압축 |
| 동일 classifier capacity | Ridge probe가 feature별 차원을 다르게 이용하는 문제 제거 | linear, MLP, same hidden dim |
| spatial resolution ablation | grid가 얼마나 coarse해져도 유지되는지 확인 | VAE grid 1×1, 2×2, 4×4, 8×8 |
| temporal length ablation | 5/9/17/33 frame에서 robust한지 확인 | full, low-fps, non-uniform low-fps |
| pixel shortcut control | 배경/색/위치 cue만으로 풀리는지 확인 | background-only, first+last only, time-average only |
| reversal/shuffle negative | feature가 temporal order를 진짜 보는지 확인 | normal vs reverse vs shuffle |

가장 중요한 비교는 다음입니다.

```text
A. raw pixel grid -> same resampler -> 16 motion tokens
B. optical flow grid -> same resampler -> 16 motion tokens
C. wan_vae_grid_sequence -> same resampler -> 16 motion tokens
D. wan_vae_global_sequence -> 5 temporal tokens
E. dit_l14_t900_token_mean -> 1 global token
```

여기서 C가 A/B보다 real benchmark나 OOD shift에서 좋아야 연구 주장이 강해집니다.

---

## 3. Toy task를 더 어렵게 만들기

현재 toy task는 좋은 sanity check였지만, 이미 여러 feature가 100%를 찍고 있어서 더 이상 정보량이 크지 않습니다. 다음 synthetic task는 **pixel shortcut을 어렵게 만들고, VideoLLM이 실제로 헷갈리는 motion reasoning**에 가까워야 합니다.

### 3.1 추천 synthetic stress tasks

| Task | 목적 |
|---|---|
| camera motion vs object motion | 픽셀 이동 방향과 실제 객체 이동 방향을 분리 |
| object crossing / occlusion | identity tracking 필요 |
| same first/last frame, different path | 시작/끝 프레임 shortcut 제거 |
| clockwise vs counter-clockwise cycle | global average나 endpoint cue 제거 |
| target-conditioned counting | distractor object가 움직일 때 특정 객체만 count |
| contact/causality | “A가 B를 밀어서 B가 움직였다” 같은 interaction reasoning |
| speed profile | 같은 총 이동거리지만 slow→fast vs fast→slow 구분 |
| temporal relation with distractors | “red object moves before blue object”처럼 question-conditioned temporal order 필요 |
| non-uniform low-fps | 5 keyframes 반복보다 현실적인 dropped/irregular frames |
| motion blur / compression / crop shift | real video degradation에 가까운 robustness 확인 |

### 3.2 가장 추천하는 stress task: same first/last frame, different path

예시:

```text
Class A: object moves left -> up -> right
Class B: object moves left -> down -> right

first frame: same
last frame: same
time-average: similar
정답: 중간 trajectory를 봐야 함
```

이 task에서 `wan_vae_grid_sequence`가 pixel grid보다 강하면, Wan latent가 단순 좌표 변화 이상의 motion prior를 담고 있다는 주장이 훨씬 강해집니다.

---

## 4. Real benchmark probing으로 넘어가기

다음으로는 반드시 real video로 가야 합니다. 지금 결과는 가능성을 충분히 보였지만, 논문화하려면 real benchmark transfer가 필요합니다.

추천 순서는 다음과 같습니다.

---

### 4.1 SSV2 selected subset

SSV2는 hand-object interaction과 fine-grained temporal motion이 많은 dataset이므로 Wan-VAE motion feature probing에 적합합니다.

처음부터 전체 174-class classification으로 가지 말고, pair/subset부터 시작하는 것이 좋습니다.

추천 subset:

```text
left vs right:
  pushing something from left to right
  pushing something from right to left

up vs down:
  moving something up
  moving something down

before/after-like:
  putting something onto something
  taking something off something

interaction:
  covering / uncovering
  putting into / taking out of
```

목표는 전체 SSV2 top-1 성능이 아니라:

> Wan feature가 direction / order / action-pair discrimination에서 CLIP / pixel / flow보다 나은지 확인하는 것

입니다.

---

### 4.2 MotionBench DEV

MotionBench는 fine-grained motion understanding 자체를 평가하는 benchmark이므로 현재 연구 방향과 가장 직접적으로 맞습니다.

여기서는 두 가지 실험을 추천합니다.

```text
1. frozen feature probe:
   video -> Wan feature -> small classifier -> MotionBench question/category

2. multiple-choice reranking:
   VideoLLM answer score + Wan motion score
```

MotionBench는 toy task에서 본 direction / order / count 결과를 real QA로 연결하기 좋습니다.

---

### 4.3 MVBench temporal-heavy tasks

MVBench에서는 전체 평균보다 task별 분석이 더 중요합니다.

우선순위 task:

```text
Moving Direction
Action Sequence
Action Prediction
Action Antonym
Object Shuffle
Moving Count
Fine-grained Action
Unexpected Action
Episodic Reasoning
```

좋은 결과 패턴은 다음입니다.

```text
static-heavy subset:
  Wan motion token ≈ baseline

temporal-heavy subset:
  Wan motion token > baseline
```

이 패턴이 나오면 논문 설득력이 큽니다.

---

### 4.4 Video-MME short/medium subset

Video-MME는 다양한 domain과 길이를 갖는 multiple-choice VideoQA benchmark이므로 generalization 확인에 좋습니다.

처음부터 long video 전체로 가기보다는 short/medium subset에서 시작하는 것이 안전합니다.

추천 비교:

```text
baseline VideoLLM
baseline + Wan motion tokens
baseline + Wan frame selector
baseline + Wan reranker
```

Video-MME는 motion-specific signal이 dilute될 수 있으므로, MotionBench/MVBench에서 효과를 확인한 뒤 확장하는 것이 좋습니다.

---

## 5. Dataset-facing feature extractor를 먼저 만들기

다음 단계의 실험 인프라는 benchmark용 feature extractor입니다.

### 5.1 추천 input / output format

```text
input:
  videos/
  metadata.jsonl

metadata.jsonl:
  {
    "video_id": "...",
    "path": "...",
    "question": "...",
    "choices": ["...", "...", "...", "..."],
    "answer": "A",
    "task": "moving_direction",
    "fps": ...,
    "duration": ...
  }

output:
  features/
    wan_vae_grid_sequence.h5
    wan_vae_global_sequence.h5
    wan_vae_global_delta.h5
    dit_l14_t900_token_mean.h5
    pixel_grid_sequence.h5
    flow_grid_sequence.h5
  metadata_aligned.jsonl
```

### 5.2 반드시 저장해야 하는 metadata

Feature만 저장하지 말고, 다음 metadata도 함께 저장해야 합니다.

```text
original_fps
sampled_frame_indices
sampled_timestamps
low_fps_mode
resize/crop parameters
vae latent shape
feature shape
Wan checkpoint name
feature extractor version
```

특히 low-fps 실험을 계속하려면 `sampled_timestamps`가 중요합니다. 단순히 5 keyframes를 17 frames로 반복하면 모델이 “반복된 frame” artifact를 학습할 수 있습니다. Real low-fps setting에서는 irregular timestamp를 고려해야 하므로, adapter에 `Δt embedding`을 넣는 실험도 추천합니다.

---

## 6. VideoLLM adapter 설계 추천

현재 결과 기준으로 production default는 `wan_vae_grid_sequence`입니다.

- `global_sequence`: compact ablation
- `global_delta`: before/after order용 auxiliary 가능성
- DiT token: compute budget이 있을 때 auxiliary

### 6.1 기본 구조

```text
video
  -> Wan-VAE encoder
  -> wan_vae_grid_sequence  [T', H', W', C]
  -> spatial-temporal resampler
  -> 16 or 32 Wan motion tokens
  -> VideoLLM visual token stream에 concat
  -> LLM
```

### 6.2 최소 ablation set

| Variant | 구성 | 목적 |
|---|---|---|
| A | VideoLLM baseline | 기준선 |
| B | + `wan_vae_global_sequence` | compact feature가 도움이 되는지 |
| C | + `wan_vae_grid_sequence` | main claim |
| D | + `wan_vae_grid_sequence` + low-fps aug | robustness |
| E | + grid + global_delta | compact temporal delta 추가 이득 |
| F | + grid + DiT l14/t900 token | DiT auxiliary 가치 확인 |
| G | + pixel_grid_sequence same tokens | Wan 고유성 control |
| H | + optical_flow_grid same tokens | motion baseline control |

### 6.3 Training setup

처음에는 최대한 고정된 setup으로 시작하는 것이 좋습니다.

```text
Wan-VAE: freeze
VideoLLM visual encoder: freeze
LLM: freeze or LoRA only
Trainable:
  - Wan feature projector
  - temporal/spatial resampler
  - optional gate
```

### 6.4 Resampler 후보

```text
1. Spatial average + temporal transformer
   가장 단순한 baseline

2. Perceiver resampler
   [T', H', W'] grid를 16/32 tokens로 압축

3. Factorized resampler
   spatial pooling -> temporal pooling을 분리
```

가장 추천하는 것은 **factorized resampler**입니다.

이유는 현재 실험에서 global pooling이 direction 정보를 날리는 경향이 있었기 때문입니다. 처음부터 완전 global token으로 압축하면 motion direction 정보가 사라질 수 있습니다.

추천 구조:

```text
wan_vae_grid_sequence
  -> per-frame spatial 4 or 8 tokens
  -> temporal attention
  -> final 16/32 motion tokens
```

---

## 7. Low-FPS augmentation은 계속 쓰되, 더 다양하게 만들기

현재 결과에서 low-fps augmentation은 특히 `before_after_cycle`에서 큰 효과가 있었습니다. 하지만 direction task에서는 compact/global feature가 충분히 회복되지 않았습니다.

따라서 adapter training에는 low-fps augmentation을 넣되, 단순한 한 가지 방식만 쓰면 안 됩니다.

### 7.1 추천 temporal augmentation

```text
1. uniform low-fps
   17 frames -> 5 frames -> repeat/interpolate

2. non-uniform frame drop
   random temporal gaps

3. speed jitter
   0.5x / 1x / 2x

4. random frame repeat
   cheap camera / streaming artifact simulation

5. motion blur
   fast action degradation

6. compression noise
   web video artifact

7. temporal crop shift
   action의 시작/끝이 잘리는 상황

8. reverse/shuffle as negative
   positive augmentation과 구분되는 temporal corruption
```

### 7.2 Consistency / contrastive loss

Positive pair:

```text
same video full-fps
same video low-fps
same video speed-jittered
```

Negative pair:

```text
reversed video
shuffled video
wrong direction counterfactual
```

Loss 예시:

```text
L = L_QA
  + λ1 * contrastive(full_motion_token, lowfps_motion_token)
  + λ2 * classification(aux_temporal_task)
  + λ3 * separation(normal, reverse/shuffle)
```

주의할 점은 모든 temporal 변형에 invariance를 주면 안 된다는 것입니다.

- low-fps / speed jitter에는 어느 정도 invariant해야 합니다.
- reverse / shuffle에는 sensitive해야 합니다.

---

## 8. DiT는 main feature보다 보조 실험으로 두기

현재 실험에서 `dit_l14_t900_token_mean`은 direction4에서 low-fps augmentation gain이 컸지만, 절대 성능은 VAE grid보다 낮았습니다.

다만 DiT를 버릴 필요는 없습니다. 지금까지는 `token_mean`만 썼기 때문에 direction 정보가 평균 pooling에서 사라졌을 가능성이 큽니다.

### 8.1 DiT 실험 1: token_mean 대신 spatial token 유지

```text
current:
  DiT hidden -> token_mean -> 1 token

next:
  DiT hidden -> reshape [T, H, W, C]
             -> spatial/temporal resampler
             -> 8/16 tokens
```

이게 `token_mean`보다 좋아지면, DiT가 약한 것이 아니라 pooling이 약했던 것입니다.

### 8.2 DiT 실험 2: layer/timestep ensemble

현재 best가 `layer14/timestep900`이면 다음은 주변을 더 촘촘히 봐야 합니다.

```text
layers: 10, 12, 14, 16, 18
timesteps: 700, 800, 900, 950
pooling: token_mean, temporal_mean, spatial_grid, temporal_delta
```

### 8.3 DiT 실험 3: question-conditioned feature

Zero-text denoising loss가 약했다면, DiT는 unconditional score보다 **question/candidate-conditioned representation**으로 쓰는 편이 낫습니다.

```text
prompt = question + candidate answer

video latent + noise + prompt
  -> DiT hidden
  -> candidate-specific motion token
  -> score or rerank
```

이 실험은 MotionBench/MVBench multiple-choice setting에서 특히 유용합니다.

---

## 9. Diffusion-loss reranking은 후순위

초기 실험에서 zero-text DiT denoising loss는 shuffle에는 약간 신호가 있었지만, reversal에는 unreliable했습니다.

따라서 지금 당장은 reranking보다 **Wan-VAE grid adapter**가 우선입니다.

다만 candidate-conditioned reranking은 별도 가능성이 있습니다.

```text
For each answer candidate a_i:

prompt_i = "Question: ... Answer: ..."

score_i =
  - denoising_loss(video | prompt_i)
  + VideoLLM_logprob(a_i)
```

추천 실험:

```text
Dataset:
  MotionBench DEV
  MVBench selected temporal tasks

Compare:
  VideoLLM only
  VideoLLM + Wan denoising rerank
  VideoLLM + Wan hidden-feature rerank
  VideoLLM + Wan motion adapter
```

기대치는 낮게 잡는 것이 좋습니다. Reranking은 main contribution보다는 **calibration / auxiliary score** 정도로 두는 편이 안전합니다.

---

## 10. Question-aware frame selector도 좋은 다음 주제

VideoLLM의 실전 문제는 모든 frame을 다 넣을 수 없다는 점입니다. Wan grid feature가 temporal signal을 잘 잡는다면, 이를 **frame selector**로 쓸 수 있습니다.

구성:

```text
video
  -> Wan-VAE grid_sequence
  -> question-conditioned scoring
  -> top-K temporal segments
  -> selected frames/features only VideoLLM에 입력
```

비교군:

```text
uniform sampling
CLIP text-frame similarity sampling
motion magnitude sampling
Wan-VAE motion-token sampling
Wan-DiT question-conditioned sampling
```

추천 metric:

```text
accuracy vs frame budget:
  8 frames
  16 frames
  32 frames
  64 frames

breakdown:
  temporal-heavy questions
  static-heavy questions
  subtitle/no-subtitle split
```

좋은 결과 패턴:

```text
static-heavy QA:
  Wan selector ≈ uniform

motion-heavy QA:
  Wan selector > uniform

long video:
  Wan selector > uniform by larger margin
```

---

## 11. 논문화 가능한 가장 강한 스토리

현재 결과에서 가장 좋아 보이는 논문 방향은 다음입니다.

> **Grid-preserved generative video latents are robust motion tokens for VideoLLMs, while globally pooled generative features collapse under frame-rate shift.**

제목 후보:

```text
Generative Motion Tokens:
Using Spatially Preserved Wan2.1 VAE Latents for Robust Fine-Grained VideoLLM Reasoning
```

핵심 claim:

1. Wan2.1 VAE latent는 motion direction, order, repetition count를 강하게 담고 있다.
2. 그러나 global pooling은 fps shift에서 쉽게 깨진다.
3. Spatial grid를 보존한 Wan-VAE sequence는 low-fps shift에서도 robust하다.
4. Low-fps augmentation은 compact/global features를 일부 rescue하지만, grid-preserving token을 대체하지 못한다.
5. Wan-VAE grid token을 VideoLLM adapter로 넣으면 real temporal QA에서 성능이 오른다.

이 스토리가 성립하려면 반드시 필요한 다음 증거는 세 개입니다.

```text
Evidence 1:
  real benchmark에서도 grid > global

Evidence 2:
  same token budget에서 Wan grid > pixel/flow grid

Evidence 3:
  VideoLLM adapter에서 motion-heavy QA subset 개선
```

---

## 12. 지금 바로 추천하는 실행 순서

### Step 1. Feature extractor 일반화

Benchmark용 extractor를 먼저 만듭니다.

```text
extract_wan_features.py

input:
  --video_jsonl
  --feature_types wan_vae_grid_sequence,wan_vae_global_sequence,wan_vae_global_delta,dit_l14_t900_token_mean,pixel_grid_sequence
  --num_frames 17
  --lowfps_modes none,uniform5,nonuniform5
  --output_h5

output:
  features.h5
  metadata.jsonl
```

---

### Step 2. Fair baseline suite 추가

현재 toy task에 아래 feature를 추가합니다.

```text
pixel_grid_sequence_same_token
pixel_grid_delta_same_token
optical_flow_grid_same_token
DINO/SigLIP frame grid if available
CLIP frame mean
VideoMAE/InternVideo feature if available
```

핵심은 **same token budget**입니다.

---

### Step 3. Harder synthetic stress set 제작

아래 5개를 먼저 추천합니다.

```text
same_first_last_different_path
camera_motion_vs_object_motion
crossing_objects_identity
target_object_count_with_distractor
contact_causality_push
```

여기서 grid feature가 계속 강하면 real benchmark 전에 논리적 근거가 단단해집니다.

---

### Step 4. SSV2 small subset probing

전체 학습이 아니라 selected class pair로 시작합니다.

```text
left/right
up/down
put/take
cover/uncover
push/pull
```

평가 setting:

```text
linear probe
MLP probe
few-shot 10/50/100 samples per class
full-fps -> low-fps
low-fps -> low-fps
full+low -> low-fps
```

---

### Step 5. MotionBench DEV probing

MotionBench는 fine-grained motion understanding 자체가 목적이므로 가장 직접적인 real QA test입니다.

할 일:

```text
1. VideoLLM baseline answer 수집
2. Wan feature-only small classifier/reranker
3. VideoLLM confidence + Wan score ensemble
4. question type별 breakdown
```

---

### Step 6. Minimal VideoLLM adapter

가장 작은 adapter부터 시작합니다.

```text
Wan-VAE grid_sequence
  -> factorized resampler
  -> 16 motion tokens
  -> VideoLLM visual tokens에 concat
```

처음 볼 ablation:

```text
baseline
+ global_sequence
+ grid_sequence
+ grid_sequence + lowfps aug
+ grid_sequence + DiT aux token
+ pixel_grid_sequence same token
```

---

### Step 7. Video-MME / MVBench로 확장

MotionBench/MVBench temporal subset에서 효과가 확인되면 Video-MME로 일반화합니다.

추천 순서:

```text
MVBench temporal-heavy tasks
  -> Video-MME short/medium
  -> Video-MME long / frame budget experiment
```

---

## 13. 하지 않는 게 좋은 것

지금 단계에서 비추천하는 방향도 명확합니다.

### 13.1 `wan_vae_global_sequence`를 main feature로 밀지 않기

Compact ablation으로는 가치가 있지만, direction low-fps에서 너무 취약합니다.

### 13.2 Zero-text DiT denoising loss reranker에 많은 compute를 쓰지 않기

이미 reversal에서 불안정한 신호가 나왔습니다. DiT를 쓰려면 question-conditioned hidden feature나 spatial token으로 다시 보는 것이 좋습니다.

### 13.3 Toy task 100%만으로 강한 claim을 하지 않기

Pixel grid도 잘 되는 task가 많으므로, 현 상태의 claim은 “Wan feature가 된다” 정도입니다. “Wan이 기존 feature보다 낫다”는 주장을 하려면 real benchmark와 fair baseline이 필요합니다.

### 13.4 Low-fps augmentation 하나만으로 robustness를 주장하지 않기

현재 low-fps simulation은 5 keyframes를 17 frames로 확장하는 특정 설정입니다. Non-uniform drop, speed jitter, frame repeat, blur, compression까지 확장해야 합니다.

---

## 14. 최종 추천 실험 묶음

가장 좋은 다음 실험 묶음은 다음입니다.

```text
1. Same-token fair baseline:
   Wan grid vs pixel grid vs optical flow grid

2. Hard synthetic stress:
   same first/last, camera motion, occlusion, distractor count

3. Real probing:
   SSV2 selected pairs -> MotionBench DEV -> MVBench temporal tasks

4. Minimal VideoLLM adapter:
   Wan-VAE grid_sequence -> factorized resampler -> 16/32 motion tokens

5. Robust training:
   low-fps + non-uniform fps + speed jitter + reverse/shuffle negatives

6. Auxiliary only:
   DiT l14/t900 spatial tokens or question-conditioned tokens
```

한 줄로 정리하면:

> 다음 목표는 **“Wan-VAE grid가 toy task에서 좋다”**를 넘어서, **“같은 token budget의 pixel/flow baseline보다 real temporal VideoQA에서 더 robust하다”**를 증명하는 것입니다.

---

## 15. Minimum viable next experiment

가장 작게 시작한다면 아래 조합을 추천합니다.

```text
Dataset:
  harder synthetic stress set 5개
  + SSV2 selected pair subset

Features:
  pixel_grid_same_token
  optical_flow_grid_same_token
  wan_vae_global_sequence
  wan_vae_grid_sequence
  dit_l14_t900_token_mean

Training:
  frozen feature
  same resampler
  linear probe + small MLP

Evaluation:
  in-distribution full fps
  full -> low fps
  full+low -> low fps
  non-uniform low fps
  reverse/shuffle negative

Success criterion:
  wan_vae_grid_sequence가 same-token pixel/flow baseline보다
  OOD temporal shift에서 더 안정적이어야 함.
```

이 결과가 좋으면 그 다음에 VideoLLM adapter로 넘어가는 것이 가장 안전합니다.
