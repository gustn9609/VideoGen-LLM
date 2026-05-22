# REPA류 연구를 참고한 Wan Feature의 Video Understanding 활용 전략

**작성일:** 2026-05-10  
**목표:** 지금까지의 Wan feature 실험 결과를 바탕으로, REPA / VideoREPA / MoAlign / MOFT / GenRec / REG 등 관련 연구 아이디어를 참고해 Wan feature를 VideoLLM 또는 VideoQA에 더 잘 활용하는 방법을 정리한다.

---

## 0. 한 줄 결론

지금까지의 결과를 보면 **raw Wan feature를 그대로 video understanding feature로 쓰는 방향은 약하다.** 대신 더 유망한 방향은 다음이다.

> **Wan feature를 motion-centric residual representation으로 재정렬하고, text-only / pixel / flow scorer가 못 푸는 hard temporal subset에서 보조 신호로 쓰는 것.**

추천하는 핵심 방법은 다음 조합이다.

```text
Wan 1x1/2x2 high-motion+camera-comp feature
  -> MOFT식 content removal
  -> REPA-style motion teacher alignment
  -> text-only scorer의 residual score로 사용
  -> hard temporal subset / Repetition Count / Action Order에서 평가
```

---

## 1. 현재까지 실험 결과에서 얻은 문제 정의

### 1.1 기존 실험의 핵심 패턴

초기 synthetic / toy / low-fps 실험에서는 Wan-VAE feature가 매우 강했다.

```text
Controlled toy/stress setting:
  - direction / order / repetition count에서 Wan-VAE grid sequence가 강함
  - low-fps shift에서도 grid sequence는 robust
  - DiT layer14/t900 token도 일부 motion signal 보유
```

하지만 real MotionBench candidate reranking에서는 다른 양상이 나왔다.

```text
Real MotionBench candidate rerank:
  - video-only Wan probe는 chance 근처
  - candidate text와 결합하면 45% 안팎까지 올라감
  - 하지만 text-only를 안정적으로 넘지 못함
  - pixel/flow baseline과 비슷하거나 더 약한 경우가 많음
  - best Wan 설정도 main feature라기보다는 weak auxiliary signal
```

최근 factorial mini-sweep 기준 가장 괜찮았던 Wan 조건은 다음이다.

```text
Wan-VAE 1x1, 8 frames, high-motion + camera compensation
  Wan Acc:        0.4519
  Text-only gain: -0.0033
  Pixel gap:      +0.0140
  Flow gap:       +0.0074
  Shuffle gap:    +0.0207
```

이 결과는 다음을 의미한다.

```text
Wan feature는 real MotionBench에서 main feature로는 약하다.
다만 1x1 / high-motion / camera-compensated 조건에서는
pixel/flow보다 약간 높고 shuffle gap도 있어 temporal auxiliary 후보로는 남길 가치가 있다.
```

### 1.2 현재 병목

현재 병목은 feature extraction 자체가 아니다.

```text
병목:
  1. raw Wan feature space가 QA/recognition 목적에 정렬되어 있지 않음
  2. text / appearance / metadata shortcut이 강함
  3. dense grid는 real video에서 noisy함
  4. 1x1 coarse feature는 안정적이지만 fine temporal/semantic reasoning에는 부족함
  5. VideoLLM에 바로 붙이기에는 incremental gain이 작음
```

따라서 연구 질문을 다음처럼 바꾸는 것이 좋다.

```text
기존 질문:
  Wan feature가 pixel/flow보다 좋은가?

수정된 질문:
  text/pixel/flow가 못 푸는 hard temporal subset에서,
  Wan feature를 motion-centric residual representation으로 정렬하면
  보완 신호를 줄 수 있는가?
```

---

## 2. REPA에서 가져올 핵심 아이디어

REPA는 diffusion transformer의 noisy hidden state를 pretrained visual encoder의 clean representation에 align하는 방식이다. 핵심은 다음이다.

```text
Diffusion hidden state가 representation learning에 바로 적합하지 않을 수 있으므로,
강한 pretrained visual representation을 target으로 삼아 intermediate hidden을 정렬한다.
```

이를 Wan feature에 적용하면 다음과 같다.

### 기존 방식

```text
Wan feature -> probe / reranker -> QA
```

### 추천 방식

```text
Wan feature
  -> small adapter
  -> video understanding teacher representation에 alignment
  -> motion residual scorer / VideoLLM adapter
```

즉, Wan feature를 “완성된 video understanding feature”로 보는 것이 아니라, **understanding-friendly space로 재투영해야 할 generative latent**로 보는 것이 핵심이다.

---

## 3. 가장 추천하는 방향: REPA-U / Wan-MoREPA

### 3.1 개념

```text
REPA-U: REPresentation Alignment for Video Understanding
Wan-MoREPA: Motion-centric REPA for Wan features
```

구조는 다음과 같다.

```text
video
  -> Wan-VAE / Wan-DiT feature
  -> trainable adapter gθ
  -> aligned motion-understanding representation
  -> candidate residual reranker / VideoLLM motion token
```

### 3.2 Loss 구성

```text
L =
  L_candidate_rank
  + λ_sem    * L_align(g(Wan), VideoSSL_teacher)
  + λ_motion * L_align(g(Wan), Motion_teacher)
  + λ_rel    * L_relation(g(Wan), Teacher)
  + λ_temp   * L_temporal_control
```

### 3.3 Teacher 후보

| Teacher | 역할 |
|---|---|
| V-JEPA / V-JEPA2 | high-level video understanding, physical dynamics, action semantics |
| VideoMAE | masked video representation, data-efficient video SSL |
| DINOv2 / SigLIP / CLIP image encoder | appearance / object / scene semantics |
| Optical flow / RAFT / CoTracker류 | low-level motion, trajectory, direction |
| VideoLLM candidate score | language-conditioned QA prior |
| Wan raw feature | generative motion prior 보존 |

핵심은 Wan feature를 단일 teacher에 맞추는 것이 아니라, **motion teacher + semantic teacher + relation teacher**를 역할별로 나누는 것이다.

---

## 4. VideoREPA / CREPA에서 가져올 것: token relation alignment

단순 feature regression은 위험하다.

```text
Naive alignment:
  g(Wan_token_i) ≈ Teacher_token_i
```

real video에서는 dense grid가 noisy했고, 1x1 coarse pooling이 더 안정적이었다. 따라서 patch/token 단위 raw feature를 직접 맞추기보다, **token 간 관계**를 맞추는 방식이 더 안전하다.

### 4.1 Relation alignment

```text
Wan segment tokens:
  h1, h2, ..., hN

Teacher segment tokens:
  y1, y2, ..., yN

R_wan[i, j] = cosine(g(hi), g(hj))
R_teacher[i, j] = cosine(yi, yj)

L_TRD = || R_wan - R_teacher ||²
```

### 4.2 왜 좋은가

```text
1. dense token 자체를 classifier에 넣지 않아 overfit을 줄임
2. temporal segment 간 관계를 보존함
3. frame selector / segment scorer와 잘 맞음
4. VideoREPA의 Token Relation Distillation 아이디어와 구조적으로 유사함
```

### 4.3 적용 위치

```text
video -> segments s1...sN
각 segment -> Wan 1x1/2x2 high-motion feature
각 segment -> Teacher video feature
Wan adapter가 teacher relation matrix를 맞추도록 학습
```

---

## 5. MoAlign에서 가져올 것: motion-only subspace

현재 real MotionBench probe는 text/appearance shortcut을 많이 탄다. 따라서 Wan feature를 semantic/object space에 맞추는 것보다, **motion-centric subspace**에 맞추는 것이 더 중요하다.

### 5.1 Motion teacher 만들기

간단한 pseudo target부터 시작할 수 있다.

```text
Motion pseudo-labels:
  - camera-compensated optical flow magnitude
  - dominant motion direction
  - residual object motion
  - track displacement statistics
  - repetition / periodicity score
  - high-motion segment index
```

이를 통해 motion teacher를 만든다.

```text
video feature from VideoMAE/V-JEPA/flow summary
  -> motion teacher mφ(video)
  -> z_motion
```

### 5.2 Wan adapter alignment

```text
h_wan = Wan-VAE 1x1/2x2 high-motion feature
z_wan = gθ(h_wan)

L_motion_align = 1 - cosine(z_wan, stopgrad(z_motion_teacher))
```

이렇게 하면 Wan feature가 object/appearance/text shortcut이 아니라 **motion residual**로 쓰이도록 유도할 수 있다.

---

## 6. MOFT에서 가져올 것: content removal + motion channel filtering

MOFT 계열 아이디어는 현재 상황에 매우 직접적으로 맞는다.

현재 raw Wan feature는 다음 문제가 있다.

```text
1. text-only를 넘지 못함
2. first-frame/time-average/shuffle control과 차이가 작음
3. appearance/content/domain cue가 섞여 있을 가능성이 큼
```

따라서 다음처럼 content component를 제거한다.

### 6.1 Content basis 만들기

```text
h_normal  = Wan(video)
h_static  = Wan(repeat_first_frame(video))
h_avg     = Wan(time_average_video(video))
h_shuffle = Wan(shuffled_video)

content_bank = [h_static, h_avg, h_shuffle]
content_basis = PCA(content_bank)
```

### 6.2 Motion residual feature

```text
h_motion = h_normal - Proj_content(h_normal)
```

### 6.3 Motion channel filtering

각 채널에 대해 다음 점수를 계산한다.

```text
score_motion(c) =
  corr(channel_c, flow_magnitude or motion_label)
  - corr(channel_c, first_frame_label or scene_label)
```

그리고 motion score가 높은 channel만 남긴다.

```text
h_wan_moft = h_motion[:, topK_motion_channels]
```

### 6.4 가장 먼저 적용할 config

```text
Wan-VAE 1x1, 8 frames, high-motion + camera compensation
```

이 config가 현재까지 가장 좋은 weak temporal auxiliary 후보이므로, MOFT식 content removal을 가장 먼저 적용할 가치가 있다.

---

## 7. GenRec에서 가져올 것: random-frame / masked-segment representation training

GenRec의 핵심은 video diffusion prior를 recognition에도 쓸 수 있도록 random-frame conditioning과 recognition objective를 결합하는 것이다.

이를 Wan feature adapter에 적용하면 다음과 같다.

```text
Wan feature를 그대로 probe하지 말고,
Wan feature 기반 student를 masked/random-frame prediction으로 pretrain한다.
```

### 7.1 Training task

```text
Input:
  일부 segment의 Wan feature

Targets:
  - masked segment의 teacher feature
  - temporal order label
  - flow summary
  - repetition count bin
  - candidate answer residual score
```

### 7.2 Loss

```text
L_masked = || student(Wan_visible_segments) - teacher(masked_segment) ||²
L_order  = CE(order_classifier(student_features), original/reverse/shuffle)
L_count  = CE(count_classifier(student_features), repetition_bin)
```

이 방법은 labeled MotionBench가 작더라도 unlabeled video를 활용할 수 있다는 장점이 있다.

---

## 8. REG에서 가져올 것: text token을 shortcut이 아니라 residual query로 쓰기

현재 candidate rerank에서 text-only가 너무 강하다. 단순히 text feature와 video feature를 concat하면 모델이 text prior만 사용할 수 있다.

### 8.1 피해야 할 구조

```text
score_i = f(concat(video_feature, text_feature_i))
```

이 구조는 text-only shortcut을 학습하기 쉽다.

### 8.2 추천 구조: frozen text score + video residual

```text
text_score_i = frozen_text_scorer(q, a_i)

motion_residual_i = scorer(
  CrossAttention(
    query = candidate_text_token_i,
    key/value = Wan_motion_tokens
  )
)

final_score_i = text_score_i + λ * motion_residual_i
```

### 8.3 Shortcut 방지

```text
1. text_score는 freeze
2. residual scorer는 text-only feature 단독 사용 금지
3. residual output 평균을 0 근처로 regularize
4. text-only wrong / low-margin sample에 더 높은 weight 부여
5. video token dropout / candidate dropout 적용
```

이 구조는 Wan을 main scorer가 아니라 **text scorer를 보정하는 residual signal**로 사용한다.

---

## 9. iREPA류 spatial-structure 결과를 해석할 때 주의할 점

REPA 관련 후속 연구들은 target representation의 spatial structure가 generation에 중요하다는 점을 강조한다. 하지만 너희 실험에서는 real MotionBench candidate rerank에서 dense grid보다 1x1/2x2가 더 좋았다.

이는 모순이 아니라 task 차이다.

```text
Generation:
  spatial structure가 중요

Real VideoQA reranking:
  dense spatial detail이 noisy할 수 있음
  coarse high-motion summary가 더 안정적일 수 있음
```

따라서 spatial structure 아이디어를 그대로 “dense grid를 classifier에 넣자”로 해석하면 안 된다.

추천은 다음이다.

```text
bad:
  Wan 16x16 dense tokens -> reranker

better:
  Wan dense tokens -> relation matrix / structural descriptor -> compact reranker
```

---

## 10. 추천 방법 5개

## 방법 A. Wan-MoREPA: Motion Representation Alignment

가장 추천하는 방향이다.

```text
Wan 1x1/2x2 high-motion feature
  -> adapter
  -> motion teacher subspace에 align
  -> candidate residual scorer
```

### Teacher target

```text
- camera-compensated optical flow summary
- dominant motion direction
- motion magnitude
- repetition periodicity
- V-JEPA / VideoMAE temporal feature
```

### Loss

```text
L =
  L_rank_residual
  + λ1 * cosine_align(Wan_adapter, motion_teacher)
  + λ2 * flow_summary_prediction
  + λ3 * normal-vs-shuffle contrastive loss
```

### 평가

```text
Text-only
Pixel / Flow
Raw Wan
Wan-MoREPA
Wan-MoREPA + Pixel / Flow ensemble
```

### 성공 기준

```text
Text+Pixel+Flow+Wan-MoREPA > Text+Pixel+Flow
Hard temporal subset에서 +3~5% 이상
Repetition Count / Action Order에서 gain
Shuffle gap 증가
```

---

## 방법 B. Wan-TRD: Token Relation Distillation

VideoREPA식 relation distillation이다.

```text
Wan segment tokens h1...hN
Teacher segment tokens y1...yN

R_wan[i,j] = cosine(g(hi), g(hj))
R_teacher[i,j] = cosine(yi, yj)

L_TRD = ||R_wan - R_teacher||²
```

추천 downstream:

```text
- segment-level candidate reranker
- question-aware frame selector
- hard temporal subset classifier
```

---

## 방법 C. Wan-MOFT: Content-free Motion Feature

MOFT식 content removal과 motion channel filtering이다.

```text
h_motion = h_normal - projection_to_content_basis(h_normal)
```

Content basis:

```text
- first-frame repeated video
- time-average video
- shuffled video
- static frames
```

Motion channel selection:

```text
keep channels where:
  normal-shuffle gap high
  flow correlation high
  first-frame predictability low
  answer/text shortcut correlation low
```

---

## 방법 D. Wan-GenRec: Masked Segment Prediction + Recognition

```text
Input:
  일부 segment의 Wan feature

Target:
  masked segment의 teacher feature
  temporal order
  flow summary
  candidate answer residual
```

Training tasks:

```text
1. masked segment representation prediction
2. original vs reverse vs shuffle
3. before/after order
4. repetition count bin
5. candidate answer residual ranking
```

---

## 방법 E. Question-aware Wan residual token

VideoLLM에 넣는다면 dense motion token보다 이 구조가 더 안전하다.

```text
Wan feature:
  1x1/2x2 high-motion segment tokens

Question/candidate:
  text tokens

Cross-attention:
  candidate text token queries Wan motion tokens

Output:
  1~4 residual motion tokens
```

LLM input:

```text
[original VideoLLM visual tokens]
[Wan residual motion tokens]
[question tokens]
```

학습:

```text
Freeze:
  - Wan
  - base VideoLLM visual encoder
  - base LLM initially

Train:
  - Wan adapter
  - question-aware cross-attention
  - small gate
  - optional LoRA
```

Gate는 반드시 넣는 것이 좋다.

```text
final_visual = original_visual + α(q) * Wan_motion_residual
```

이유는 Motion-related Objects처럼 Wan이 약한 type에서는 Wan residual을 꺼야 하기 때문이다.

---

## 11. 실험 우선순위

## Step 1. Wan-MOFT부터 적용

가장 싸고 빠른 실험이다.

```text
feature:
  Wan 1x1 8f high-motion+camera-comp

controls:
  first-frame
  time-average
  shuffled

method:
  content PCA 제거
  motion-channel filtering

eval:
  candidate rerank
  hard temporal subset
  normal-shuffle gap
```

성공 기준:

```text
text-only gain이 음수에서 0 이상으로 올라가는지
shuffle gap이 2% -> 5% 이상으로 커지는지
```

---

## Step 2. Motion teacher alignment

```text
teacher target:
  flow magnitude / dominant direction / periodicity / V-JEPA or VideoMAE feature

student:
  Wan 1x1/2x2 high-motion adapter

loss:
  motion alignment + candidate residual ranking
```

이게 REPA류 아이디어에 가장 가까운 실험이다.

---

## Step 3. Relation distillation

```text
segments:
  4 or 8 segments per video

teacher relation:
  V-JEPA / VideoMAE / flow-track segment similarity

Wan relation:
  adapter(Wan segment feature) similarity

loss:
  relation alignment
```

---

## Step 4. Residual score-level ensemble

```text
score_final =
  text_score
  + λ_pixel * pixel_residual
  + λ_flow  * flow_residual
  + λ_wan   * wan_aligned_residual
```

비교:

```text
Text only
Text + Pixel + Flow
Text + Raw Wan
Text + Pixel + Flow + Raw Wan
Text + Pixel + Flow + Aligned Wan
```

핵심은 마지막 줄이 안정적으로 올라가는지다.

---

## Step 5. Hard temporal subset에서만 VideoLLM adapter

Wan-aligned residual이 다음 조건을 만족하면 VideoLLM adapter로 간다.

```text
1. hard temporal subset에서 +3~5% 이상
2. Repetition Count / Action Order에서 gain
3. Text+Pixel+Flow 대비 추가 gain
4. normal-shuffle/reverse gap 증가
```

그 전에는 VideoLLM adapter full training은 보류한다.

---

## 12. 구체적인 구현 스케치

### 12.1 Wan-MoREPA adapter

```python
# pseudo-code

h_wan = extract_wan_feature(
    video,
    pool="1x1",
    frames=8,
    sampling="high_motion_camera_comp",
)  # [B, T, C]

z_wan = wan_adapter(h_wan)  # [B, D]

with torch.no_grad():
    z_teacher = motion_teacher(video)  # [B, D]
    # e.g. V-JEPA / VideoMAE feature or flow-summary encoder

loss_align = 1 - cosine_similarity(z_wan, z_teacher).mean()

scores = []
for cand in candidates:
    e = text_encoder(question + " " + cand)
    score = residual_scorer(z_wan, e)
    scores.append(score)

loss_rank = cross_entropy(torch.stack(scores, dim=1), answer_idx)

loss = loss_rank + lambda_align * loss_align
```

---

### 12.2 Relation distillation

```python
# h_wan: [B, S, C], S = number of segments
# y_teacher: [B, S, D]

z = normalize(wan_adapter(h_wan), dim=-1)
y = normalize(y_teacher, dim=-1)

R_wan = z @ z.transpose(-1, -2)
R_teacher = y @ y.transpose(-1, -2)

loss_rel = ((R_wan - R_teacher) ** 2).mean()
```

---

### 12.3 Content decorrelation

```python
h_normal = wan(video)
h_static = wan(repeat_first_frame(video))
h_avg    = wan(time_average_video(video))
h_shuf   = wan(shuffle_frames(video))

content_bank = torch.cat([h_static, h_avg, h_shuf], dim=0)

# PCA basis from content_bank
U = pca(content_bank, k=K)

h_motion = h_normal - project(h_normal, U)
```

---

### 12.4 Residual scorer

```python
# frozen text scorer
text_scores = frozen_text_scorer(question, candidates)  # [B, 4]

# video residual scorer
wan_residual_scores = []
for cand in candidates:
    e = text_encoder(question + " " + cand)
    r = residual_scorer(h_motion_wan, e)
    wan_residual_scores.append(r)
wan_residual_scores = torch.stack(wan_residual_scores, dim=1)

final_scores = text_scores + lambda_wan * wan_residual_scores
loss = cross_entropy(final_scores, answer_idx)
```

---

## 13. 추천 평가 프로토콜

모든 실험은 raw accuracy만 보지 말고 다음 지표를 같이 보고한다.

```text
1. text-only gain
   Acc(video + text) - Acc(text-only)

2. Wan residual gain
   Acc(text + pixel + flow + Wan) - Acc(text + pixel + flow)

3. temporal gap
   Acc(normal) - Acc(shuffle/reverse/first-frame/time-average)

4. hard subset gain
   text-only wrong or low-margin samples에서의 성능

5. per-type gain
   Action Order / Repetition Count / Motion Recognition / Motion-related Objects

6. complementarity
   Text wrong, Pixel wrong, Flow wrong, Wan right 샘플 수

7. oracle union
   여러 scorer 중 하나라도 맞히는 upper bound
```

추천 리포트 표:

| Model | Overall | Hard subset | Action Order | Repetition Count | Motion Recognition | Motion Objects | Shuffle gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Text only |  |  |  |  |  |  |  |
| Text + Pixel + Flow |  |  |  |  |  |  |  |
| Text + Raw Wan |  |  |  |  |  |  |  |
| Text + Pixel + Flow + Raw Wan |  |  |  |  |  |  |  |
| Text + Pixel + Flow + Wan-MOFT |  |  |  |  |  |  |  |
| Text + Pixel + Flow + Wan-MoREPA |  |  |  |  |  |  |  |

---

## 14. 바로 만들면 좋은 스크립트

### 14.1 `motionbench_wan_moft.py`

```text
Input:
  motionbench_features.h5
  normal / first-frame / time-average / shuffle Wan features

Process:
  content PCA basis 생성
  normal feature에서 content projection 제거
  motion channel filtering

Output:
  wan_moft_features.h5
```

### 14.2 `motionbench_motion_teacher.py`

```text
Input:
  video files

Targets:
  optical flow summary
  camera-compensated flow
  dominant direction
  periodicity / repetition proxy

Output:
  motion_teacher_features.h5
```

### 14.3 `motionbench_wan_morepa_train.py`

```text
Input:
  Wan feature
  motion teacher feature
  question/candidate text

Train:
  adapter + residual scorer

Loss:
  candidate ranking + motion alignment + temporal contrastive
```

### 14.4 `motionbench_residual_ensemble.py`

```text
Input:
  out-of-fold scores from text / pixel / flow / raw Wan / aligned Wan

Train:
  calibrated score-level ensemble

Output:
  overall / per-type / hard-subset gain
```

### 14.5 `motionbench_relation_distill.py`

```text
Input:
  segment-level Wan features
  segment-level teacher features

Train:
  relation distillation adapter

Output:
  segment-aware Wan features
```

---

## 15. 연구 claim 후보

현재까지 결과와 REPA류 아이디어를 합치면 가장 안전한 claim은 다음이다.

```text
Raw generative video latents are not directly superior for real VideoQA.
However, when aligned to motion-centric video understanding targets,
compact high-motion Wan representations can provide complementary temporal residuals
for hard motion-sensitive questions.
```

논문 제목 후보:

```text
From Generative Latents to Motion Residuals:
Representation Alignment for Using Wan2.1 Features in Video Understanding
```

또는:

```text
Wan-MoREPA:
Motion-Centric Representation Alignment of Video Diffusion Features for VideoQA
```

---

## 16. 최종 추천

하나만 고르라면 다음을 추천한다.

```text
1. Wan 1x1 8f high-motion+camera-comp 추출
2. first-frame / time-average / shuffle 기반 content PCA 제거
3. flow / trajectory / V-JEPA / VideoMAE teacher와 motion alignment
4. candidate reranker는 text_score + Wan_residual 구조로 학습
5. hard temporal subset, Repetition Count, Action Order에서 평가
6. Text+Pixel+Flow 대비 Aligned Wan 추가 gain을 확인
```

핵심은 raw Wan feature를 더 세게 probe하는 것이 아니라, **Wan feature를 motion-centric residual representation으로 재정렬하는 것**이다.

---

## 17. 참고 문헌 및 관련 링크

- REPA: Representation Alignment for diffusion transformers  
  - GitHub: https://github.com/sihyun-yu/REPA

- VideoREPA: Learning Physics for Video Generation through Relational Alignment with Foundation Models  
  - arXiv: https://arxiv.org/abs/2505.23656  
  - Project: https://videorepa.github.io/

- MoAlign: Motion-Centric Representation Alignment for Video Diffusion Models  
  - OpenReview: https://openreview.net/forum?id=OR0ySm4l9h  
  - arXiv: https://arxiv.org/abs/2510.19022

- MOFT: Video Diffusion Models are Training-free Motion Interpreter and Controller  
  - arXiv: https://arxiv.org/abs/2405.14864  
  - Project: https://xizaoqu.github.io/moft/

- GenRec: Unifying Video Generation and Recognition with Diffusion Models  
  - arXiv: https://arxiv.org/abs/2408.15241

- V-JEPA2  
  - arXiv: https://arxiv.org/abs/2506.09985  
  - GitHub: https://github.com/facebookresearch/vjepa2

- VideoMAE  
  - arXiv: https://arxiv.org/abs/2203.12602

- REG: Representation Entanglement for Generation  
  - arXiv: https://arxiv.org/abs/2507.01467
