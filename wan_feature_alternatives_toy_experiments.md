# Wan Feature를 Video Understanding에 더 잘 활용하기 위한 다음 아이디어

## 0. 현재까지의 상황 요약

지금까지의 결과를 기준으로 보면, **Wan feature를 일반적인 video understanding feature로 그대로 쓰는 방향은 한계가 뚜렷하다.** Controlled toy / low-fps 실험에서는 Wan-VAE가 direction, temporal order, repetition count, shuffle detection에서 강한 신호를 보였지만, real MotionBench 후보 재랭킹에서는 text / pixel / flow와의 보완성이 매우 제한적이었다.

특히 최근 correct-set overlap 결과는 중요하다.

```text
text+pixel+flow all wrong: 46
text+pixel+flow all wrong, raw Wan correct: 1
P(raw Wan correct | all three wrong): 1/46 = 2.17%
4-way chance: 25%
```

즉 raw Wan은 **base 모델들이 전부 못 푸는 샘플을 구해주는 universal residual feature가 아니다.**  
반면 temporal perturbation에 민감한 작은 subset에서는 raw Wan이 강하게 보였다.

```text
C raw-Wan temporal-sensitive subset:
  n = 8
  text   0.375
  pixel  0.250
  flow   0.375
  raw Wan 0.625

C2 any-video temporal-sensitive subset:
  n = 28
  text   0.3214
  pixel  0.3571
  flow   0.2500
  raw Wan 0.3929
```

따라서 이제 연구 방향은 다음처럼 바뀌어야 한다.

```text
기존 방향:
  Wan feature를 더 좋은 general video feature로 만들기

수정 방향:
  Wan이 잘 반응하는 temporal-sensitive 상황을 찾아내고,
  그 상황에서만 Wan의 motion / generation prior를 선택적으로 활용하기
```

---

## 1. 가장 추천하는 방향: Wan Perturbation Signature, WPS

### 1.1 핵심 아이디어

raw Wan feature 자체보다, **같은 비디오를 temporal corruption했을 때 Wan score / feature가 어떻게 바뀌는지**를 feature로 사용한다.

변형 예시:

```text
v_normal
v_shuffle
v_reverse
v_lowfps
v_timeavg
v_first_repeat
v_hflip
```

Wan feature 또는 candidate score를 각각 계산한 뒤, 절대값이 아니라 차이를 쓴다.

```text
WPS(v) = [
  score_normal - score_shuffle,
  score_normal - score_reverse,
  margin_normal - margin_shuffle,
  margin_normal - margin_reverse,
  JS(candidate_dist_normal, candidate_dist_shuffle),
  JS(candidate_dist_normal, candidate_dist_reverse),
  pred_changed_normal_shuffle,
  pred_changed_normal_reverse,
  ||h_normal - h_shuffle||,
  ||h_normal - h_timeavg||,
  cosine(h_normal, h_lowfps)
]
```

이걸 **Wan Perturbation Signature**라고 부를 수 있다.

### 1.2 왜 중요한가

지금까지의 결과는 다음을 말한다.

```text
raw Wan score 자체는 전체 QA에서 약하다.
하지만 temporal perturbation에 민감한 샘플에서는 raw Wan이 상대적으로 강하다.
```

따라서 “Wan score가 높은가?”보다 더 중요한 질문은 다음이다.

```text
Wan이 이 샘플에서 temporal corruption에 민감하게 반응하는가?
```

### 1.3 Toy 실험: Temporal Sensitivity Gate

목표:

```text
WPS가 raw Wan을 써야 하는 샘플을 찾아낼 수 있는지 확인한다.
```

구성:

```text
input:
  normal / shuffle / reverse / lowfps 변형이 있는 toy 또는 MotionBench subset

feature:
  WPS = normal-shuffle/reverse score gap, margin gap, prediction change, distribution divergence

target:
  이 샘플에서 raw Wan이 base보다 나은가?
```

비교:

```text
base only
raw Wan everywhere
base + raw Wan global ensemble
base + WPS gate + raw Wan
```

평가:

```text
gate coverage
gate=1 subset accuracy
gate=0 subset accuracy
overall accuracy
switch helps count
switch hurts count
nested CV accuracy
```

성공 기준:

```text
gate coverage: 10~30%
gate=1에서 raw Wan > base by 5%p 이상
held-out / nested CV에서 raw Wan everywhere보다 높음
```

---

## 2. Raw score 대신 Score Contrast 사용

### 2.1 기존 방식의 한계

기존 candidate reranker는 보통 다음 구조였다.

```text
score_i = f(WanFeature(video), Text(question, answer_i))
```

이 방식은 text prior와 appearance shortcut을 많이 탈 수 있다.

### 2.2 추천 방식

같은 candidate에 대해 normal video와 corrupted video의 score 차이를 사용한다.

```text
S_motion(a_i) = S_wan(v_normal, q, a_i) - S_wan(v_corrupted, q, a_i)
```

예:

```text
S_shuffle_contrast(a_i) = S_wan(v_normal, q, a_i) - S_wan(v_shuffle, q, a_i)
S_timeavg_contrast(a_i) = S_wan(v_normal, q, a_i) - S_wan(v_timeavg, q, a_i)
S_reverse_contrast(a_i) = S_wan(v_normal, q, a_i) - S_wan(v_reverse, q, a_i)
```

텍스트는 normal과 corrupted video에서 동일하기 때문에, text-only shortcut은 어느 정도 cancel될 수 있다.

### 2.3 Toy 실험: Candidate Score Equivariance

Task별로 perturbation 후 정답이 어떻게 바뀌어야 하는지 정의한다.

```text
Action Order:
  reverse하면 before/after 정답이 바뀌어야 함

Moving Direction:
  horizontal flip하면 left/right 정답이 바뀌어야 함

Repetition Count:
  low-fps나 speed jitter에도 정답은 유지되어야 함

Shuffle:
  order-sensitive question에서는 confidence가 떨어져야 함
```

평가:

```text
normal에서 정답 candidate score가 가장 높은가?
reverse/hflip에서 변환된 정답 candidate score가 가장 높은가?
shuffle/timeavg에서 score margin이 낮아지는가?
```

성공 기준:

```text
raw score보다 contrast score가 temporal-sensitive subset에서 높음
normal-shuffle / normal-reverse gap이 증가
text-only 대비 video incremental gain이 양수
```

---

## 3. VAE Latent Trajectory Feature

### 3.1 핵심 아이디어

지금까지는 VAE latent sequence 자체를 사용했다.

```text
z_1, z_2, ..., z_T
```

하지만 real benchmark에서는 dense grid보다 1x1 / 2x2 coarse pooling이 좋았다. 이는 spatial detail보다 **global temporal dynamics**가 더 유용할 수 있음을 시사한다.

따라서 latent 값 자체보다 시간 궤적의 형태를 feature로 사용한다.

### 3.2 Feature 후보

```text
velocity:
  Δz_t = z_t - z_{t-1}

acceleration:
  Δ²z_t = Δz_t - Δz_{t-1}

motion energy:
  ||Δz_t||

latent curvature:
  cosine(Δz_t, Δz_{t+1})

periodicity:
  autocorrelation(||Δz_t||)
  FFT peak of ||Δz_t||

directional asymmetry:
  mean(Δz_t) vs mean(reverse(Δz_t))

temporal entropy:
  entropy over motion-energy distribution
```

### 3.3 Toy 실험: Latent Dynamics Probe

Task:

```text
repetition count
slow-fast vs fast-slow
same first/last but different path
object moves then stops vs stops then moves
before/after cycle
```

비교:

```text
raw Wan 1x1
raw Wan 2x2
Δz only
Δ²z only
energy/autocorr feature
raw + dynamics
pixel dynamics
flow dynamics
```

성공 기준:

```text
Repetition Count / Action Order에서 raw Wan보다 dynamics feature가 높음
normal-shuffle gap 증가
temporal-sensitive subset에서 raw Wan 대비 improvement
```

---

## 4. Structured Compact Spatial Feature

### 4.1 문제의식

현재 결과상:

```text
1x1:
  안정적이지만 left/right, top/bottom 같은 방향 정보 손실

4x4 이상 dense grid:
  방향 정보는 있지만 real benchmark에서는 noisy하고 overfit 가능성 큼
```

따라서 중간 형태가 필요하다.

### 4.2 추천 feature

```text
1x1 global
1x2 left/right split
2x1 top/bottom split
2x2 quadrant
```

각 split에 대해 temporal dynamics를 계산한다.

```text
h_global = pool(z, 1x1)
h_lr     = pool(z, 1x2)
h_tb     = pool(z, 2x1)
h_quad   = pool(z, 2x2)

feature = [
  dynamics(h_global),
  dynamics(h_lr_left) - dynamics(h_lr_right),
  dynamics(h_tb_top) - dynamics(h_tb_bottom),
  quadrant transition statistics
]
```

### 4.3 Toy 실험: Minimal Spatial Direction Test

Task:

```text
same global motion magnitude, different direction:
  left -> right
  right -> left
  up -> down
  down -> up

variants:
  camera motion distractor
  object crossing
  same first/last different middle path
```

비교:

```text
1x1
1x2 / 2x1
2x2
4x4
pixel same pooling
flow same pooling
```

성공 기준:

```text
1x2 / 2x1이 1x1보다 direction task에서 개선
4x4보다 real subset에서 안정적
structured compact feature가 dense grid보다 낮은 variance
```

---

## 5. MOFT-lite: Hard Content Removal 대신 Motion Channel Selection

### 5.1 기존 MOFT-style 시도가 실패한 이유

이전 Wan-MOFT 결과는 raw Wan보다 낮았다.

가능한 이유:

```text
1. candidate rerank에는 motion뿐 아니라 object/scene/context cue도 필요함
2. content PCA 제거가 유용한 semantic cue까지 제거함
3. raw Wan best config가 이미 1x1 coarse라 content와 motion이 강하게 섞여 있음
4. first-frame/time-average/shuffle 기반 content basis가 너무 공격적임
```

### 5.2 추천 방식: Channel Selection

hard subtraction 대신 channel별 motion sensitivity를 계산한다.

```text
channel_score(c) =
  + corr(channel_c, normal_vs_shuffle_label)
  + corr(channel_c, flow_magnitude)
  + corr(channel_c, camera_comp_motion)
  + corr(channel_c, repetition_count_proxy)
  - corr(channel_c, first_frame_prediction)
  - corr(channel_c, text_only_score)
```

그리고 top-K motion-sensitive channel만 사용한다.

```text
h_selected = h_raw[:, topK_channels]
feature = [h_raw, h_selected, h_raw - h_selected]
```

중요한 점은 raw Wan을 버리지 않는 것이다.

```text
bad:
  h_motion = h_raw - content_projection

better:
  feature = raw Wan + motion-selected Wan channels
```

### 5.3 Toy 실험: Motion Channel Discovery

Train:

```text
toy normal/shuffle/reverse/count labels로 channel score 계산
```

Test:

```text
unseen toy
MotionBench temporal-sensitive subset
Action Order / Repetition Count subset
```

비교:

```text
raw Wan
PCA-subtracted MOFT
top-K motion channels
raw + top-K motion channels
```

성공 기준:

```text
shuffle/reverse AUC 증가
raw Wan보다 temporal-sensitive subset에서 개선
Motion-related Objects에서는 gate가 낮게 나와야 함
```

---

## 6. DiT Hidden 대신 CFG Motion Vector

### 6.1 기존 DiT 사용의 한계

지금까지 DiT는 주로 다음 방식으로 사용했다.

```text
DiT hidden token mean
DiT spatial tokens
zero-text denoising loss
QA-conditioned hidden
```

하지만 real QA에서는 큰 이득이 없었다.

### 6.2 새로운 아이디어

DiT의 더 흥미로운 신호는 hidden feature 자체가 아니라 **conditional과 unconditional의 차이**일 수 있다.

각 candidate answer에 대해:

```text
ε_uncond = DiT(z_t, empty_prompt)
ε_cond_i = DiT(z_t, prompt(q, a_i))

CFG_vector_i = ε_cond_i - ε_uncond
```

또는 hidden state에서도:

```text
h_cfg_i = h_cond_i - h_uncond
```

이건 “candidate text가 이 video latent에서 denoising direction을 어떻게 바꾸는가”를 나타낸다.

### 6.3 Score 후보

```text
score_i = cosine(CFG_vector_i, observed_motion_vector)
```

observed motion vector 후보:

```text
z_clean - z_noisy
predicted velocity target
Wan-VAE Δz
camera-compensated flow summary
```

### 6.4 Toy 실험: Candidate CFG Alignment

Candidates:

```text
object moves left
object moves right
person picks up before putting down
person puts down before picking up
```

Metric:

```text
correct candidate의 CFG_vector가 observed motion vector와 더 align되는가?
```

비교:

```text
DiT denoising loss
DiT hidden token mean
DiT CFG vector
VAE latent dynamics
```

성공 기준:

```text
CFG vector가 loss보다 candidate discrimination에서 좋음
reverse / hflip equivariance가 나타남
```

---

## 7. DiT / VAE Relation Descriptor

### 7.1 핵심 아이디어

학습형 TRD adapter는 실패했지만, relation 아이디어 자체는 버릴 필요가 없다. Adapter를 학습하지 말고, **training-free relation descriptor**로 사용한다.

DiT 또는 VAE token을 `h_i`라고 할 때:

```text
R[i, j] = cosine(h_i, h_j)
```

관계행렬 전체를 쓰지 말고 요약한다.

```text
temporal relation entropy
within-frame vs cross-frame similarity
near-frame similarity decay
forward/backward asymmetry
normal-shuffle relation distance
normal-reverse relation distance
```

### 7.2 Toy 실험: Relation Sensitivity

Task:

```text
normal vs shuffled
normal vs reversed
same first/last different path
collision vs no-collision
before/after order
```

비교:

```text
raw DiT token mean
DiT relation descriptor
VAE relation descriptor
pixel relation descriptor
flow relation descriptor
```

성공 기준:

```text
relation descriptor가 raw token보다 shuffle/reverse AUC 높음
WPS feature로 썼을 때 temporal gate 성능 개선
```

---

## 8. Denoising Trajectory Feature

### 8.1 핵심 아이디어

단일 timestep / layer만 쓰지 않고, 여러 timestep에서 DiT가 어떻게 반응하는지를 feature로 쓴다.

예:

```text
t = 950, 900, 800, 700, 500
```

각 timestep에서:

```text
predicted velocity
hidden norm
candidate CFG vector
normal-shuffle score gap
normal-reverse score gap
```

Trajectory feature:

```text
feature = [
  score_t950,
  score_t900,
  score_t700,
  score_t500,
  score_t950 - score_t500,
  max_t temporal_sensitivity(t),
  argmax_t temporal_sensitivity(t)
]
```

### 8.2 Toy 실험: Timestep Sensitivity Map

Task:

```text
direction
count
before/after
object interaction
camera motion
normal vs shuffle
normal vs reverse
```

평가:

```text
각 task에서 어느 timestep의 normal-shuffle/reverse gap이 가장 큰가?
multi-timestep feature가 single t=900보다 개선되는가?
```

성공 기준:

```text
task별 best timestep이 다르게 나타남
multi-timestep feature가 single t=900보다 개선
```

---

## 9. Wan을 Feature Extractor가 아니라 Future Consistency Judge로 쓰기

### 9.1 핵심 아이디어

feature가 약하다면, Wan의 본래 능력인 generation / prediction prior를 활용한다.

즉 Wan을 다음처럼 사용한다.

```text
관측된 앞부분 + candidate prompt -> 가능한 미래 예측
예측된 미래와 실제 미래가 얼마나 일치하는가?
```

### 9.2 Score 예시

```text
future_consistency_i =
  - distance(
      WanFuture(first_half, prompt_i),
      observed_second_half
    )
```

Candidate prompt 예:

```text
prompt_A = "the object moves to the left"
prompt_B = "the object moves to the right"
```

정답 prompt에서 observed future와 더 가까워야 한다.

### 9.3 Toy 실험: Future Prediction Consistency

Task:

```text
left vs right
slow-fast vs fast-slow
collision causes movement vs object moves spontaneously
before/after order
same first half, different future
```

비교:

```text
VAE raw feature
DiT candidate score
future consistency score
pixel/flow baseline
```

성공 기준:

```text
observed future가 correct prompt continuation과 더 가까움
wrong prompt continuation과 margin이 있음
```

---

## 10. Masked Middle / Inpainting Consistency

### 10.1 핵심 아이디어

전체 future generation이 비싸면, 중간 segment를 가리고 Wan/VACE/inpainting 계열로 consistency를 본다.

```text
video:
  [before] [masked middle] [after]

condition:
  before + after + candidate prompt

score:
  observed middle segment의 reconstruction / consistency score
```

### 10.2 Toy 실험: Middle Segment Consistency

Task:

```text
same first frame
same last frame
different middle path

Class A:
  object goes above obstacle

Class B:
  object goes below obstacle
```

평가:

```text
correct middle segment consistency > wrong middle consistency
first/last baseline은 chance
```

이 실험은 Wan이 단순 endpoint가 아니라 middle trajectory를 이해하는지 확인하는 데 좋다.

---

## 11. Wan-generated Counterfactual은 학습보다 검증에 먼저 쓰기

### 11.1 이유

Wan으로 만든 비디오를 바로 adapter 학습에 넣으면 generation artifact를 배울 위험이 있다. 따라서 처음에는 **검증용 toy benchmark**로 쓰는 것이 안전하다.

### 11.2 만들 counterfactual 예시

```text
same scene, same objects:
  left -> right
  right -> left

same first/last:
  path A
  path B

same object count:
  two repetitions
  three repetitions

same contact:
  A hits B
  A misses B

same events:
  open then pick
  pick then open
```

### 11.3 목적

```text
Wan feature/scorer가 자기 생성 데이터에서는 어떤 유형을 잘 보는가?
real video에서는 왜 transfer가 안 되는가?
```

가능한 원인 분석:

```text
compression
camera motion
small object
domain gap
prompt mismatch
annotation shortcut
```

---

## 12. Question-aware Frame / Segment Selector

### 12.1 핵심 아이디어

Wan이 직접 답을 고르는 데 약하다면, **VideoLLM에 넣을 frame / segment를 고르는 데만 사용**한다.

```text
video -> segments
for each segment:
  compute Wan temporal sensitivity
  compute motion energy
  compute normal-shuffle divergence

select top-K segments
feed selected frames to VideoLLM
```

### 12.2 Toy 실험: Segment Localization

Synthetic setup:

```text
video length: 64 frames
relevant event: frames 25~32
distractor motion: frames 5~12, 45~52
question:
  "How many times does the red object bounce?"
```

평가:

```text
Wan selector가 relevant segment를 top-K에 포함하는가?
uniform / flow magnitude / pixel motion selector보다 좋은가?
selected-frame QA accuracy가 오르는가?
```

성공 기준:

```text
top-1 / top-3 localization recall 개선
VideoLLM 또는 simple QA accuracy 개선
```

이 방향은 지금까지의 high-motion + camera-comp 결과와 잘 맞는다.

---

## 13. VAE Encoder Intermediate Feature 확인

### 13.1 문제의식

지금까지는 대부분 final VAE latent를 사용했다. 하지만 final latent는 reconstruction용으로 압축되어 있어 understanding에 필요한 intermediate motion cue가 사라졌을 수 있다.

### 13.2 뽑을 후보

```text
VAE encoder early feature:
  local texture / local motion

VAE encoder middle feature:
  object-level motion

VAE encoder late feature:
  compressed semantic / motion summary

VAE mean vs logvar:
  mean: content / motion code
  logvar: uncertainty / ambiguity signal

reconstruction residual:
  Wan이 어떤 motion을 reconstruct하기 어려워하는가?
```

### 13.3 Toy 실험: VAE Layer Probe

Features:

```text
final latent
encoder block early
encoder block middle
encoder block late
mean + logvar
reconstruction residual
```

Tasks:

```text
direction
repetition
same first/last different path
normal vs shuffle
real temporal-sensitive subset
```

성공 기준:

```text
intermediate feature가 final latent보다 temporal-sensitive subset에서 좋음
logvar / residual이 ambiguity 또는 hard sample detector로 유용함
```

---

## 14. Toy Benchmark 묶음 제안

아래 toy들은 “Wan이 understanding에 도움이 되는지”를 판단하기 좋다.

### A. Temporal Equivariance Toy

```text
Action Order:
  normal answer = A before B
  reversed answer = B before A

Direction:
  normal answer = left
  hflip answer = right

Count:
  speed jitter answer unchanged
  frame drop answer mostly unchanged

Shuffle:
  confidence should drop
```

평가:

```text
equivariance accuracy
confidence drop
candidate score swap accuracy
```

---

### B. Same First/Last, Different Middle Path

```text
Class 1:
  object goes above obstacle

Class 2:
  object goes below obstacle

first frame same
last frame same
time average similar
```

평가:

```text
first/last baseline chance
Wan middle consistency > baseline
```

---

### C. Contact Causality Toy

```text
A hits B -> B moves
A misses B -> B does not move
B moves before A contacts it
```

평가:

```text
Wan future consistency
DiT relation descriptor
VAE dynamics feature
```

---

### D. Repetition and Rhythm Toy

```text
2 taps
3 taps
4 taps

same total duration
same first/last

different rhythm:
  regular
  irregular
  bursty
```

평가:

```text
latent autocorrelation
FFT peak
motion energy count
```

---

### E. Camera Motion vs Object Motion Toy

```text
camera pans left, object static
camera static, object moves right
both camera and object move
camera-comp residual object motion
```

평가:

```text
raw Wan
camera-comp Wan
flow
Wan Perturbation Signature
```

---

### F. Segment Localization Toy

```text
long video with one relevant motion event
distractor motion elsewhere
question asks about relevant event
```

평가:

```text
top-K segment recall
selected-frame QA accuracy
```

---

## 15. 추천 실험 우선순위

### Step 1. Wan Perturbation Signature

```text
normal / shuffle / reverse / timeavg / lowfps에 대한
score gap, margin gap, distribution divergence, feature distance를 만든다.
```

비교:

```text
raw Wan
WPS
raw Wan + WPS
pixel/flow WPS
```

목표:

```text
temporal-sensitive gate 성능 개선
```

---

### Step 2. Latent Dynamics Feature

```text
z_t 대신 Δz, Δ²z, energy, autocorr, periodicity를 사용한다.
```

특히 다음 subset에서 확인한다.

```text
Repetition Count
Action Order
same-first-last toy
normal vs shuffle/reverse
```

---

### Step 3. Structured Compact Spatial Feature

```text
1x1 + 1x2 + 2x1 + 2x2
```

목표:

```text
1x1의 안정성 + direction cue를 절충한다.
```

---

### Step 4. DiT CFG Vector

```text
candidate prompt별 conditional - unconditional vector를 사용한다.
```

목표:

```text
DiT loss / hidden mean보다 candidate discrimination에 적합한지 확인한다.
```

---

### Step 5. Future / Masked Consistency

Wan을 feature extractor가 아니라 simulator / consistency judge로 사용한다.

```text
first half -> future consistency
before/after -> middle consistency
```

처음에는 toy에서만 확인한다.

---

## 16. Minimal Experiment Set

가장 작게 시작한다면 아래 5개만 먼저 실행한다.

```text
1. WPS gate:
   normal-shuffle/reverse score divergence로 raw Wan 사용 여부 결정

2. Latent dynamics:
   Δz, Δ²z, energy, autocorr feature 추가

3. Structured compact:
   1x1 + 1x2 + 2x1 + 2x2 feature 추가

4. CFG vector:
   candidate conditional - unconditional DiT vector 비교

5. Segment localization toy:
   Wan temporal sensitivity가 relevant event segment를 찾는지 평가
```

각 실험의 공통 report:

```text
raw accuracy
text-only gain
pixel/flow gap
normal-shuffle gap
normal-reverse gap
temporal-sensitive subset accuracy
gate coverage
switch helps / switch hurts
per-type breakdown
```

---

## 17. 최종 정리

지금 상황에서 가장 유망한 방향은 다음이다.

```text
1. raw Wan feature 자체를 더 잘 probe하기보다는,
   temporal perturbation에 대한 Wan의 반응을 feature로 만든다.

2. VAE latent 값이 아니라 latent trajectory:
   Δz, Δ²z, energy, autocorr, periodicity를 사용한다.

3. dense grid 대신 structured compact spatial:
   1x1 + 1x2 + 2x1 + 2x2를 사용한다.

4. DiT는 hidden token mean이 아니라
   candidate CFG vector, relation descriptor, multi-timestep sensitivity를 본다.

5. feature가 아니라 Wan의 생성 능력을 이용해
   future consistency / masked middle consistency / counterfactual consistency를 평가한다.
```

가장 먼저 할 단일 실험:

```text
Wan Perturbation Signature를 만들고,
normal vs shuffle/reverse score divergence가 raw Wan을 써야 하는 temporal-sensitive 샘플을
held-out에서 잘 찾아내는지 확인한다.
```

두 번째 실험:

```text
Wan-VAE latent trajectory에서 Δz, Δ²z, motion energy, autocorrelation을 뽑아서
Repetition Count / Action Order / same-first-last toy에서 raw Wan보다 나은지 확인한다.
```

이 두 개가 잘 나오면, Wan은 “좋은 일반 video feature”가 아니라 다음 역할로 정리할 수 있다.

```text
Temporal perturbation detector
Motion dynamics specialist
Candidate score contrast feature
Segment selector
Generative consistency judge
```

즉 최종 방향은:

```text
Wan as universal representation X
Wan as temporal-sensitive specialist O
```
