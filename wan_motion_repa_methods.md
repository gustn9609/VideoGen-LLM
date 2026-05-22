# Wan-Motion-REPA: Wan Feature를 VideoLLM Understanding에 REPA-style로 활용하는 방법

## 0. 핵심 요약

지금까지의 실험을 보면, **raw Wan feature를 그대로 VideoLLM에 붙이거나 candidate reranker로 쓰는 방식은 안정적인 real VideoQA gain을 만들기 어렵다.** Controlled toy에서는 Wan-VAE feature가 direction, order, shuffle, repetition count를 강하게 잡았지만, real MotionBench류 QA에서는 raw Wan, WPS, MoREPA/TRD, gate, ensemble이 모두 제한적인 gain에 그쳤다.

따라서 REPA처럼 쓰려면 방향을 바꿔야 한다.

```text
나쁜 방향:
  VideoLLM token ≈ raw Wan feature

좋은 방향:
  VideoLLM motion token ≈ transformed Wan temporal target
```

여기서 `transformed Wan temporal target`은 raw feature가 아니라 다음과 같은 조작된 target이다.

```text
1. Wan latent dynamics target
   h, Δh, Δ²h, motion energy, autocorrelation, periodicity

2. Residualized Wan target
   Wan target - projection(text/pixel/flow/first-frame)

3. Temporal relation target
   token value가 아니라 segment 간 similarity / relation matrix

4. Equivariant perturbation target
   normal vs reverse/shuffle/lowfps에 대한 Wan response vector

5. MOFT-lite motion-channel target
   hard content subtraction이 아니라 motion-sensitive channel selection

6. Structured compact target
   1x1 + 1x2 + 2x1 + 2x2 pooling으로 coarse stability와 direction cue 절충
```

최종적으로는 다음과 같은 학습 목표를 추천한다.

```text
L = L_QA
  + λ1 * Align(StudentMotion, Residual(WanDynamics))
  + λ2 * AlignRelation(StudentSegments, WanTemporalRelation)
  + λ3 * AlignEquivariance(StudentPerturbResponse, WanPerturbResponse)
```

이 문서에서는 이 방향을 **Wan-Motion-REPA**라고 부른다.

---

## 1. 왜 raw Wan feature alignment는 위험한가

REPA의 아이디어를 단순히 가져오면 다음과 같은 구조가 된다.

```text
VideoLLM visual token
  -> projector
  -> raw Wan feature에 alignment
```

하지만 지금까지의 실험 결과상 이 방식은 위험하다.

raw Wan feature에는 다음이 섞여 있다.

```text
- reconstruction-oriented latent signal
- appearance / scene / domain cue
- camera motion / compression artifact
- temporal coherence signal
- motion direction / order / count signal
- VideoQA answer decision과 직접 맞지 않는 latent direction
```

따라서 raw Wan feature에 그대로 align하면, VideoLLM adapter가 useful temporal signal뿐 아니라 noisy reconstruction signal과 shortcut까지 같이 배울 수 있다.

더 안전한 접근은 다음이다.

```text
Wan raw feature
  -> temporal/motion transformation
  -> residualization / relation / equivariance target
  -> VideoLLM motion adapter alignment
```

즉 **Wan을 좋은 feature extractor로 가정하지 말고, Wan feature 안에서 understanding에 필요한 motion component만 뽑아 target으로 만든다.**

---

## 2. 기본 구조: VideoLLM 쪽으로 뒤집은 REPA

원래 REPA는 diffusion model 내부 hidden state를 external pretrained representation에 align하는 방식이다. 여기서는 그 방향을 VideoLLM 쪽으로 뒤집는다.

```text
video
  ├─ VideoLLM visual encoder / motion adapter
  │     -> student motion token z_s
  │
  └─ frozen Wan feature extractor
        -> raw Wan feature h_wan
        -> target transformation T(h_wan)
        -> stop-gradient teacher target z_wan

loss:
  L = L_QA + λ * L_align(z_s, stopgrad(z_wan))
```

중요한 것은 `T(h_wan)`이다.  
`T`가 없으면 raw Wan alignment가 되고, 지금까지 결과상 그 방식은 실패 가능성이 높다.

---

## 3. 방법 1: Wan Residual REPA

### 3.1 아이디어

Wan feature가 text, pixel, flow, first-frame cue와 많이 겹친다면, 그 겹치는 부분을 제거한 뒤 남는 **Wan-only residual temporal signal**에 align한다.

```text
h_wan = transformed Wan target
b = [text feature, pixel feature, flow feature, first-frame feature]

h_wan_pred = Ridge(b -> h_wan)
h_wan_res  = h_wan - h_wan_pred
```

그리고 VideoLLM adapter는 `h_wan`이 아니라 `h_wan_res`에 align한다.

```text
L_res_repa = 1 - cosine(P(z_student), stopgrad(h_wan_res))
```

### 3.2 왜 필요한가

현재 real MotionBench류 결과에서는 Wan이 text/pixel/flow와 완전히 독립적인 보완 신호를 많이 제공하지 못했다. 따라서 raw Wan을 그대로 쓰면 이미 다른 feature로 설명되는 부분을 반복해서 배우게 된다.

Residual target은 다음을 강제한다.

```text
VideoLLM adapter가 text/pixel/flow/appearance로 설명되지 않는
Wan의 고유 temporal residual에 집중하게 만든다.
```

### 3.3 주의점

Residualization은 반드시 train fold 안에서만 fit해야 한다.

```text
올바른 방식:
  train fold에서 Ridge fit
  validation/test fold에는 train Ridge로 transform

잘못된 방식:
  전체 데이터로 Ridge fit 후 평가
```

전체 데이터로 residual을 만들면 leakage가 생긴다.

---

## 4. 방법 2: Wan Motion-Subspace REPA

### 4.1 아이디어

Wan raw feature 전체가 아니라, motion pseudo-task를 잘 풀도록 만든 subspace를 target으로 쓴다.

Wan feature에서 다음 pseudo-label을 예측하는 small head를 학습한다.

```text
- camera-compensated flow magnitude
- dominant motion direction
- motion energy over time
- repetition periodicity
- normal vs shuffle
- normal vs reverse
- temporal order label
```

그 head의 중간 embedding을 motion subspace로 사용한다.

```text
h_wan -> MotionHead -> m_wan

L_motion_repa = 1 - cosine(P(z_student), stopgrad(m_wan))
```

### 4.2 더 간단한 버전

별도 head를 학습하지 않고 hand-crafted motion target을 만들 수도 있다.

```text
m_wan = [
  mean ||Δh_t||,
  max ||Δh_t||,
  argmax_t ||Δh_t||,
  mean ||Δ²h_t||,
  autocorr(||Δh_t||),
  FFT peak(||Δh_t||),
  normal-shuffle feature distance,
  normal-reverse feature distance
]
```

이 방식은 특히 다음 task에 적합하다.

```text
- Repetition Count
- Action Order
- Rhythm
- Contact Causality
- Slow-fast / fast-slow dynamics
```

---

## 5. 방법 3: Relation REPA

### 5.1 아이디어

Raw token value를 직접 맞추지 말고, segment/token 간 temporal relation을 맞춘다.

```text
bad:
  student token_i ≈ Wan token_i

good:
  sim(student token_i, student token_j)
    ≈ sim(Wan token_i, Wan token_j)
```

### 5.2 Loss

비디오를 segment `s1...sN`으로 나눈 뒤, Wan target과 student token을 segment 단위로 만든다.

```text
R_wan[i, j] = cosine(T(h_wan_i), T(h_wan_j))
R_student[i, j] = cosine(P(z_student_i), P(z_student_j))

L_rel = || R_student - stopgrad(R_wan) ||²
```

여기서도 `T(h_wan)`은 raw feature가 아니라 motion-transformed feature가 좋다.

```text
T(h_wan_i) = [h_i, Δh_i, motion_energy_i, structured_spatial_i]
```

### 5.3 장점

Dense feature가 noisy할 때도 temporal relation은 남을 수 있다.

```text
- segment 간 유사도
- 변화량
- 반복 패턴
- normal vs reverse relation 차이
- normal vs shuffle relation 붕괴
```

따라서 relation target은 raw feature target보다 robust할 가능성이 있다.

---

## 6. 방법 4: Equivariant REPA

### 6.1 아이디어

WPS를 candidate answer score로 쓰면 real QA에서 실패했다. 하지만 WPS류 신호를 **representation regularizer**로 쓰는 것은 여전히 가능하다.

핵심은 이것이다.

```text
정답을 WPS로 고르지 말고,
VideoLLM motion token이 Wan처럼 temporal perturbation에 반응하도록 학습시킨다.
```

### 6.2 Perturbation set

```text
v        = normal video
v_rev    = reversed video
v_shuf   = shuffled video
v_lowfps = low-fps video
v_avg    = time-average / first-frame-repeat
v_hflip  = horizontal flip
```

Wan target response:

```text
d_wan_rev  = T(h_wan(v)) - T(h_wan(v_rev))
d_wan_shuf = T(h_wan(v)) - T(h_wan(v_shuf))
d_wan_low  = T(h_wan(v)) - T(h_wan(v_lowfps))
```

Student response:

```text
d_student_rev = P(z_student(v)) - P(z_student(v_rev))
```

Equivariance loss:

```text
L_equiv = || normalize(d_student_rev)
           - stopgrad(normalize(d_wan_rev)) ||²
```

### 6.3 왜 score가 아니라 loss로 써야 하는가

WPS score는 real QA에서 정답 후보와 잘 정렬되지 않았다. 하지만 perturbation response는 여전히 temporal sensitivity를 담을 수 있다.

따라서:

```text
사용하지 말 것:
  answer = argmax WPS(candidate)

사용할 것:
  student representation의 temporal response가 Wan response와 비슷해지도록 regularization
```

---

## 7. 방법 5: MOFT-lite REPA

### 7.1 기존 hard MOFT의 문제

Hard content subtraction은 raw Wan보다 성능을 떨어뜨릴 수 있다.

```text
h_motion = h_normal - Proj_content(h_normal)
```

문제는 useful context까지 제거될 수 있다는 점이다.

```text
- candidate QA에는 object/scene/context cue도 필요함
- content와 motion이 완전히 분리되지 않음
- 1x1 coarse feature에서는 content와 motion이 강하게 entangled될 수 있음
```

### 7.2 Soft motion-channel selection

각 Wan channel에 motion score를 부여한다.

```text
score(c) =
  + corr(h_c, normal_vs_shuffle)
  + corr(h_c, normal_vs_reverse)
  + corr(h_c, flow_magnitude)
  + corr(h_c, camera_comp_motion)
  + corr(h_c, repetition_proxy)
  - corr(h_c, first_frame_feature)
  - corr(h_c, text_only_score)
```

Top-K channel만 선택한다.

```text
h_motion = h_wan[:, topK_channels]
```

중요한 점은 raw Wan을 버리지 않는 것이다.

```text
target = concat(
  PCA(h_wan),
  PCA(h_motion),
  PCA(h_motion - mean_content)
)
```

그리고 student를 이 target에 align한다.

```text
L_moft_lite = 1 - cosine(P(z_student), stopgrad(target))
```

---

## 8. 방법 6: Structured Compact Wan Target

Real rerank에서는 dense grid보다 1x1/2x2 coarse pooling이 더 안정적이었다. 하지만 1x1은 direction/path 정보를 잃는다.

따라서 dense grid 대신 structured compact pooling을 쓴다.

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

이 target은 다음에 적합하다.

```text
- Moving Direction
- same first/last different path
- Action Order
- object crossing / occlusion
- path reasoning
```

---

## 9. Multi-target Wan-Motion-REPA

하나의 target만 쓰면 특정 task에만 맞을 가능성이 높다. 지금까지 결과상 feature별 역할이 달랐다.

```text
raw/coarse Wan:
  weak overall signal, temporal-sensitive subset에서 일부 신호

WPS:
  toy direction/rhythm에는 강하지만 real answer score로는 실패

structured compact:
  same first/last path에 강함

latent dynamics:
  contact causality에 강함

relation target:
  temporal order / segment relation에 적합
```

따라서 여러 target head를 둔다.

```text
T1 = coarse global target
T2 = structured spatial target
T3 = latent dynamics target
T4 = temporal relation target
T5 = perturbation response target
```

Student도 여러 head를 둔다.

```text
z1, z2, z3, z4, z5 = Adapter(video_tokens)

L =
  λ1 L_align(z1, T1)
+ λ2 L_align(z2, T2)
+ λ3 L_align(z3, T3)
+ λ4 L_rel(z4, T4)
+ λ5 L_equiv(z5, T5)
```

Downstream에서는 question type이나 learned gate가 어떤 head를 더 사용할지 결정한다.

```text
Action Order:
  relation + structured

Repetition Count:
  dynamics + perturbation

Motion Recognition:
  coarse + dynamics

Motion-related Objects:
  Wan target weight 낮게
```

---

## 10. VideoLLM 학습 구조

### 10.1 Adapter 위치

VideoLLM 전체를 바꾸기보다 작은 motion adapter를 추가한다.

```text
video frames
  -> existing VideoLLM visual encoder
  -> visual tokens
  -> temporal adapter / Perceiver / small transformer
  -> motion tokens z_student
  -> LLM input에 concat
```

Wan branch는 frozen이다.

```text
video
  -> frozen Wan-VAE / optional Wan-DiT
  -> transformed Wan target
  -> stop-gradient
```

### 10.2 Loss

```text
L_total =
  L_QA
  + λ_repa  * L_wan_repa
  + λ_rel   * L_relation
  + λ_equiv * L_equivariance
  + λ_ctr   * L_temporal_contrast
  + λ_reg   * L_preserve_base
```

각 항목의 의미:

```text
L_wan_repa:
  student token과 transformed Wan target의 cosine / SmoothL1 alignment

L_relation:
  segment relation matrix alignment

L_equivariance:
  normal vs reverse/shuffle/lowfps response alignment

L_temporal_contrast:
  normal-lowfps positive
  normal-shuffle/reverse negative

L_preserve_base:
  기존 VideoLLM output을 지나치게 망가뜨리지 않기 위한 KL/gate regularizer
```

### 10.3 추천 alignment 위치

```text
추천:
  visual encoder output
  temporal adapter output
  motion token output

비추천:
  LLM hidden 전체
  final answer logits 직접
```

Wan target은 language decision boundary와 완전히 정렬되어 있지 않기 때문에, LLM 내부 깊은 layer나 answer logit에 직접 강하게 걸면 위험하다.

---

## 11. Target Recipe v1: Dynamics-Relation Wan Target

가장 먼저 해볼 수 있는 작은 버전이다.

### 11.1 Wan feature 추출

```text
Input video:
  8 or 16 frames
  low resolution
  full frame
  optional high-motion segment

Wan feature:
  VAE latent
  1x1, 1x2, 2x1, 2x2 pooling
```

### 11.2 Temporal statistics

```text
h_t
Δh_t   = h_t - h_{t-1}
Δ²h_t  = Δh_t - Δh_{t-1}
energy_t = ||Δh_t||
R[i,j] = cosine(h_i, h_j)
```

### 11.3 Target vector

```text
T_wan = PCA_128(
  concat(
    mean(h_t),
    mean(Δh_t),
    mean(Δ²h_t),
    max_t energy_t,
    autocorr(energy),
    flatten_upper(R),
    D_normal_reverse,
    D_normal_shuffle
  )
)
```

PCA는 train set에서만 fit한다.

### 11.4 Alignment

```text
z_student = Adapter(VideoLLM visual tokens)

L = L_QA + λ * (1 - cosine(Project(z_student), stopgrad(T_wan)))
```

이것이 가장 단순한 **Wan-Motion-REPA v1**이다.

---

## 12. Target Recipe v2: Residualized Dynamics-Relation Target

v1이 raw Wan과 비슷하게 끝날 수 있으므로, 바로 v2도 준비하는 것이 좋다.

### 12.1 Base feature

```text
B = concat(
  text-only embedding,
  pixel feature,
  flow feature,
  first-frame feature,
  time-average feature
)
```

### 12.2 Residual target

```text
T_wan_raw  = Dynamics-Relation Wan Target
T_wan_pred = Ridge(B -> T_wan_raw)
T_wan_res  = T_wan_raw - T_wan_pred
```

### 12.3 Alignment

```text
L = L_QA
  + λ1 * Align(z_student_main, T_wan_raw)
  + λ2 * Align(z_student_res_head, T_wan_res)
```

`T_wan_res`는 Wan의 고유 temporal residual이다. 이 target이 가장 “feature를 조작해서라도 REPA처럼 쓰는 방법”에 가깝다.

---

## 13. Target Recipe v3: Relation-only Target

feature 값 자체가 계속 불안정하면, value를 버리고 relation만 쓴다.

```text
R_wan = temporal relation matrix from transformed Wan features
R_student = temporal relation matrix from student segment tokens

L = L_QA + λ * || R_student - stopgrad(R_wan) ||²
```

이 방식은 raw value가 noisy한 경우에도 temporal structure를 전달할 수 있다.

---

## 14. Target Recipe v4: Equivariance-only Target

WPS를 answer score로 쓰지 않고 perturbation response target으로만 사용한다.

```text
T_rev = normalize(T_wan(video) - T_wan(reverse(video)))
S_rev = normalize(Student(video) - Student(reverse(video)))

L_equiv_rev = || S_rev - stopgrad(T_rev) ||²
```

추가 perturbation:

```text
shuffle
lowfps
time-average
first-frame-repeat
horizontal flip
```

이 방식은 특히 다음 목표에 맞다.

```text
- temporal order sensitivity
- reversal sensitivity
- shuffle robustness
- low-fps invariance
- direction equivariance
```

---

## 15. Pseudo-code

### 15.1 Wan target extraction

```python
# pseudo-code

with torch.no_grad():
    h = wan_vae_features(video)              # [B, T, C] or [B, T, H, W, C]
    h_struct = structured_pool(h)            # 1x1, 1x2, 2x1, 2x2

    dh = h_struct[:, 1:] - h_struct[:, :-1]
    ddh = dh[:, 1:] - dh[:, :-1]

    energy = dh.norm(dim=-1)
    rel = cosine_relation(h_struct)

    target_raw = concat_stats(h_struct, dh, ddh, energy, rel)
    target_raw = pca_or_projection(target_raw)

    # optional residualization
    base_feat = concat(text_feat, pixel_feat, flow_feat, first_frame_feat)
    target_res = target_raw - ridge_predict(base_feat)
```

### 15.2 Student alignment

```python
video_tokens = videollm_visual_encoder(video)
z = motion_adapter(video_tokens)

loss_align = 1 - cosine(projector(z), stopgrad(target_res)).mean()
loss = loss_qa + lambda_align * loss_align
```

### 15.3 Relation loss

```python
z_seg = motion_adapter.segment_tokens(video_tokens)   # [B, S, D]
t_seg = wan_segment_targets(video)                    # [B, S, D]

R_s = normalize(z_seg) @ normalize(z_seg).transpose(-1, -2)
R_t = normalize(t_seg) @ normalize(t_seg).transpose(-1, -2)

loss_rel = ((R_s - R_t.detach()) ** 2).mean()
```

### 15.4 Equivariance loss

```python
z_n = motion_adapter(video_normal)
z_r = motion_adapter(video_reverse)

t_n = wan_target(video_normal)
t_r = wan_target(video_reverse)

loss_equiv = mse(
    normalize(z_n - z_r),
    normalize((t_n - t_r).detach())
)
```

---

## 16. 학습 데이터 구성

REPA-style auxiliary loss는 labeled QA 데이터만으로는 약할 수 있다. Unlabeled video에도 적용할 수 있도록 설계해야 한다.

추천 데이터 구성:

```text
1. unlabeled real videos
   L_repa, L_rel, L_equiv만 사용

2. generated/counterfactual videos
   temporal contrast loss 사용

3. MotionBench / MVBench / FAVOR-Bench labeled subset
   L_QA 사용

4. toy/stress data
   debugging and ablation
```

추천 훈련 순서:

```text
Stage 1:
  unlabeled video에서 Wan-Motion-REPA pretraining
  VideoLLM은 freeze
  adapter만 학습

Stage 2:
  temporal QA 데이터로 adapter fine-tuning
  L_QA + L_REPA 같이 사용

Stage 3:
  real benchmark에서 evaluation
```

---

## 17. Baseline과 성공 기준

### 17.1 반드시 비교해야 할 baseline

```text
QA-only adapter
QA + pixel/flow alignment
QA + VideoMAE/V-JEPA alignment
QA + raw Wan alignment
QA + transformed Wan dynamics alignment
QA + residualized Wan alignment
QA + relation-only Wan alignment
QA + equivariant Wan alignment
```

### 17.2 성공 기준

Raw Wan feature accuracy가 아니라, adapter 학습 결과로 평가해야 한다.

```text
1. QA-only adapter보다
   QA + Wan-Motion-REPA adapter가 temporal subset에서 개선

2. low-fps / reverse / shuffle robustness 개선

3. generated validation이 아니라 real temporal QA에서 개선

4. VideoLLM visual token의 temporal probing 성능 개선

5. same frame budget에서 pixel/flow auxiliary보다 개선
```

### 17.3 Stop rule

다음이 나오면 해당 branch는 중단한다.

```text
- raw Wan alignment와 차이 없음
- transformed target이 raw target보다 낮음
- residualized target이 temporal subset에서 gain 없음
- relation loss가 QA-only보다 불안정
- equivariance loss가 static/object QA를 크게 손상
```

---

## 18. Toy 실험 계획

### Toy A. Target quality probe

```text
feature:
  raw Wan
  transformed Wan
  residualized Wan
  relation Wan
  dynamics Wan

task:
  direction
  same first/last path
  contact causality
  rhythm
  shuffle/reverse
```

목표:

```text
transformed target > raw target
```

### Toy B. Student alignment probe

```text
student input:
  cheap pixel/flow/CLIP frame feature

train:
  align student to Wan target

eval:
  toy temporal tasks
```

목표:

```text
student + Wan target alignment > student without alignment
```

### Toy C. Negative control

```text
align to random projection of Wan
align to first-frame-only Wan
align to time-average Wan
```

목표:

```text
motion-transformed Wan target만 좋아야 함
```

### Toy D. Real-failure toy

real MotionBench에서 WPS가 실패한 이유를 재현한다.

```text
- text/object/appearance shortcut이 섞인 mixed-cue QA
- high-motion distractor
- target-conditioned direction/count
- answer-invariant perturbation
- same first/last + same time-average
```

목표:

```text
raw/WPS score가 왜 real QA에서 실패하는지 재현하고,
Wan-Motion-REPA target이 그 실패를 줄이는지 확인
```

---

## 19. 추천 실행 순서

### Step 1. Raw Wan alignment baseline

```text
VideoLLM adapter / small student
  + L_align(raw Wan 1x1/2x2)
```

이것은 기준선이다. 성능이 낮아도 필요하다.

### Step 2. Dynamics target alignment

```text
target:
  Δh, Δ²h, energy, autocorr, relation
```

현재 toy 결과와 가장 잘 맞는 저비용 target이다.

### Step 3. Residualized target alignment

```text
target:
  Wan target - projection(text/pixel/flow/first-frame)
```

Wan 고유 신호를 강제한다.

### Step 4. Relation target alignment

```text
target:
  temporal relation matrix
```

Raw value 대신 structure를 전달한다.

### Step 5. Equivariance alignment

```text
target:
  normal-reverse/shuffle response vector
```

WPS를 score가 아니라 loss로만 사용한다.

### Step 6. Multi-target adapter

```text
coarse + structured + dynamics + relation + equivariance
```

단일 target이 아니라 task별 expert로 사용한다.

---

## 20. 최종 추천 방식

하나만 고르면 다음을 추천한다.

```text
Residualized Dynamics-Relation Wan-REPA
```

구체적으로:

```text
1. Wan-VAE 1x1/2x2/structured compact feature 추출
2. Δh, Δ²h, energy, autocorr, temporal relation matrix 생성
3. text/pixel/flow/first-frame으로 설명되는 부분을 ridge로 제거
4. 남은 residual temporal target에 VideoLLM motion adapter를 align
5. QA loss와 같이 학습
6. WPS는 score가 아니라 equivariance regularizer로만 사용
```

최종 loss:

```text
L = L_QA
  + λ1 * Align(StudentMotion, Residual(WanDynamics))
  + λ2 * AlignRelation(StudentSegments, WanTemporalRelation)
  + λ3 * AlignEquivariance(StudentNormalReverse, WanNormalReverse)
```

---

## 21. 최종 주장 후보

이 방향이 성공하면 다음처럼 주장할 수 있다.

```text
Raw Wan features are weak as plug-and-play VideoQA features.
However, transformed Wan dynamics and temporal relations can serve as
REPA-style auxiliary targets for training VideoLLM motion adapters.
```

한국어로는:

```text
Wan feature는 그대로 붙이면 약하지만,
Wan feature에서 dynamics, relation, equivariance, residual temporal signal을 뽑아
VideoLLM motion adapter의 REPA-style target으로 쓰면 temporal understanding을 학습시키는 데 사용할 수 있다.
```

즉 Wan을 “좋은 feature extractor”로 파는 것이 아니라,
**VideoLLM adapter가 temporal representation을 배우도록 만드는 generative temporal alignment target**으로 쓰는 것이 핵심이다.

---

## 22. 참고 키워드

관련 아이디어를 찾을 때 사용할 키워드:

```text
REPA
Representation Alignment for Diffusion Transformers
VideoREPA
token-level relation distillation
MOFT
motion-aware diffusion feature
MoAlign
motion-centric subspace alignment
GenRec
video diffusion recognition
VideoLLM temporal adapter
temporal equivariance loss
motion residual representation
```
