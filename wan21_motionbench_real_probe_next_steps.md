# Wan2.1 × VideoLLM: MotionBench Real-Video Probe 이후 다음 실험 제안

작성일: 2026-05-09

## 0. 현재 상태 요약

MotionBench real-video probe까지 완료했다.

결과 위치:

```text
VideoGen-LLM/results/motionbench_real_20260509_062454/motionbench_question_type_probe.md
VideoGen-LLM/results/motionbench_real_20260509_062454/motionbench_answer_letter_probe.md
VideoGen-LLM/results/motionbench_real_20260509_062454/motionbench_adapter_probe.md
VideoGen-LLM/results/motionbench_real_20260509_062454/motionbench_subset.jsonl
VideoGen-LLM/results/motionbench_real_20260509_062454/motionbench_features.h5
```

구현 추가:

```text
VideoGen-LLM/experiments/make_motionbench_subset.py
VideoGen-LLM/experiments/run_motionbench_real_probe.sh
```

실험 구성:

- MotionBench 공개 mp4 중 실제로 다운로드 가능한 DEV 샘플 사용
- 질문 타입 4개:
  - Action Order
  - Motion Recognition
  - Motion-related Objects
  - Repetition Count
- 각 타입 24개, 총 96개
- feature extraction modes:
  - `none`
  - `uniform5`
  - `nonuniform5`
- feature types:
  - Wan-VAE grid/global
  - pixel grid
  - optical flow grid

---

## 1. 현재 결과

### 1.1 Question-type 4-way probe

Chance는 25%이다.

| Mode | Feature | Accuracy |
|---|---|---:|
| none | `wan_vae_grid_sequence` | 39.3% |
| none | `wan_vae_global_sequence` | 32.1% |
| none | `pixel_grid_sequence` | 28.6% |
| none | `flow_grid_sequence` | 28.6% |
| uniform5 | `wan_vae_grid_sequence` | 35.7% |
| uniform5 | `flow_grid_sequence` | 35.7% |
| uniform5 | `pixel_grid_sequence` | 32.1% |
| nonuniform5 | `wan_vae_global_sequence` | 39.3% |
| nonuniform5 | `wan_vae_grid_sequence` | 35.7% |
| nonuniform5 | `flow_grid_sequence` | 28.6% |

해석:

- `wan_vae_grid_sequence`는 full-fps real video에서 pixel/flow보다 높게 나왔다.
- low-fps setting에서도 chance 이상은 유지했다.
- 다만 표본이 작기 때문에 강한 결론보다는 **positive signal**로 해석하는 것이 안전하다.
- 현재 test split 기준으로 39.3%, 35.7%, 32.1%, 28.6%는 몇 개 샘플 차이일 수 있다.

---

### 1.2 Adapter probe

| Adapter | Mode | Accuracy |
|---|---|---:|
| `wan_vae_grid` adapter | none | 35.7% |
| `wan_vae_grid` adapter | uniform5 | 28.6% |
| `wan_vae_grid` adapter | nonuniform5 | 39.3% |

해석:

- adapter probe는 linear probe보다 뚜렷하게 좋지 않았다.
- 이는 feature가 쓸모없다는 의미라기보다, 현재 adapter objective가 MotionBench QA에 맞지 않을 가능성이 크다.
- 현재 adapter가 question/answer text를 조건으로 보지 않는다면, QA 성능으로 이어지기 어렵다.

---

### 1.3 Answer-letter probe

`answer_letter_probe`는 강하게 해석하면 안 된다.

이유:

- A/B/C/D는 semantic class가 아니라 선택지 위치 label이다.
- feature-only classifier가 풀어야 할 의미 있는 문제가 아니다.
- task별 test가 약 7개 수준이면 통계적으로 매우 불안정하다.

따라서 answer-letter probe는 QA 성능 근거가 아니라 sanity check 정도로만 사용한다.

---

## 2. 현재 결론

현재 결과는 다음과 같이 정리하는 것이 가장 적절하다.

> Wan-VAE grid feature는 toy/stress setting뿐 아니라 MotionBench real-video subset에서도 weak but positive transfer signal을 보였다. 특히 full-fps question-type classification에서 pixel/flow grid보다 높은 성능을 보였고, low-fps setting에서도 chance 이상을 유지했다. 그러나 작은 sample size와 feature-only proxy task의 한계 때문에, Wan feature만으로 MotionBench QA를 강하게 해결한다고 보기는 어렵다. 다음 단계는 answer-letter classification이 아니라 question-answer conditioned scoring, VideoLLM confidence와의 ensemble, 그리고 question-aware adapter로 가야 한다.

핵심 변화:

```text
이전 질문:
  Wan feature가 motion signal을 담고 있는가?

현재 답:
  담고 있다. toy/stress에서는 강하고, real MotionBench에서도 약한 positive signal이 있다.

다음 질문:
  이 signal을 question/answer-conditioned VideoQA 성능으로 어떻게 변환할 것인가?
```

---

## 3. 가장 중요한 다음 방향

## 3.1 1순위: Candidate-conditioned Wan scorer

`answer_letter_probe` 대신 다음 문제를 풀어야 한다.

```text
video feature + question text + candidate answer text -> candidate score
```

즉 각 선택지마다 점수를 매기고, 가장 높은 점수의 candidate를 고른다.

### 구조

```text
video
  -> wan_vae_grid_sequence
  -> resampler / pooling
  -> v_wan

question + candidate_i
  -> text encoder
  -> e_i

score_i = MLP([v_wan, e_i, v_wan * e_i, |v_wan - e_i|])
answer = argmax_i score_i
```

### Text encoder 후보

| Text encoder | 장점 | 비고 |
|---|---|---|
| Sentence-transformer / BGE류 | 구현 쉬움, semantic matching 강함 | 빠른 baseline |
| CLIP text encoder | vision-language baseline과 비교 쉬움 | candidate text가 짧으면 유용 |
| Wan T5 / UMT5 text encoder | Wan feature와 연구 스토리 일관성 | 가장 흥미로운 방향 |

특히 Wan에서 사용하는 T5/UMT5 계열 text embedding을 쓰면, Wan-VAE visual latent와 Wan text condition을 하나의 generative prior story로 묶을 수 있다.

### 왜 중요한가

기존 answer-letter probe:

```text
video -> A/B/C/D 위치 맞히기
```

이건 의미가 없다.

candidate-conditioned scorer:

```text
video + question + candidate answer -> compatibility score
```

이건 실제 VideoQA와 직접 연결된다.

---

## 3.2 2순위: VideoLLM baseline score와 ensemble

Wan feature-only가 단독으로 강하지 않더라도, VideoLLM이 놓치는 motion case를 보완할 수 있으면 연구 가치가 크다.

### Ensemble scoring

```text
For each candidate answer a_i:

S_final(i) =
  α * S_videollm(i)
  + β * S_wan_candidate(i)
  + γ * S_wan_motion_type(i)
  + δ * S_lowfps_consistency(i)
```

여기서:

- `S_videollm(i)`: VideoLLM의 candidate logprob 또는 confidence
- `S_wan_candidate(i)`: Wan video feature + question-answer text scorer 점수
- `S_wan_motion_type(i)`: question type / motion category 보조 점수
- `S_lowfps_consistency(i)`: full-fps와 low-fps feature가 같은 답을 지지하는지 보는 보조 점수

### 반드시 봐야 할 분석

단순 accuracy보다 complementarity가 더 중요하다.

| 분석 | 의미 |
|---|---|
| VideoLLM만 맞힌 샘플 | Wan이 방해할 가능성 |
| Wan만 맞힌 샘플 | Wan을 붙일 가치 |
| 둘 다 맞힌 샘플 | 쉬운 샘플 |
| 둘 다 틀린 샘플 | 현 접근의 한계 |
| Oracle union accuracy | ensemble의 잠재 상한 |
| Disagreement accuracy | 서로 의견이 다를 때 누가 더 믿을 만한가 |
| Per-question-type gain | Action Order / Recognition / Objects / Count별 효과 |

좋은 패턴:

```text
VideoLLM이 틀린 motion/order/count 샘플 중 일부를 Wan scorer가 맞힘
→ Wan feature가 VideoLLM의 motion failure를 보완한다는 주장 가능
```

---

## 3.3 3순위: DiT를 question-conditioned hidden feature로 재시도

이전 실험에서 zero-text DiT denoising loss는 reversal에서 불안정했다. 따라서 zero-text loss를 primary reranker로 쓰는 것은 비추천이다.

하지만 DiT를 완전히 버릴 필요는 없다. 이제는 question-answer text로 condition된 hidden state를 봐야 한다.

### 구조

```text
prompt_i = "Question: {q} Candidate answer: {a_i}"

video
  -> Wan-VAE latent z
  -> add noise at t=900
  -> z_t

DiT(z_t, text=prompt_i)
  -> layer14 hidden
  -> candidate-conditioned feature h_i
  -> score_i
```

### 비교할 feature

| Variant | 설명 |
|---|---|
| A | VAE grid only |
| B | VAE grid + QA text scorer |
| C | DiT layer14/t900 token_mean + QA text scorer |
| D | DiT layer14/t900 spatial tokens + QA text scorer |
| E | VAE grid + DiT QA-conditioned hidden |

중요한 점:

```text
bad:
  DiT hidden -> token_mean -> 1 token

better:
  DiT hidden -> [T, H, W, C] -> factorized resampler -> 8/16 tokens
```

현재 `token_mean`은 spatial direction 정보를 날릴 가능성이 크다. DiT를 다시 평가하려면 spatial token을 유지하는 버전을 반드시 포함한다.

---

## 4. Adapter probe가 약했던 이유

Adapter probe가 linear probe보다 뚜렷하게 좋지 않았던 것은 이상하지 않다.

가능한 이유:

1. **데이터가 너무 작다**
   - 96개 subset에서 adapter를 학습하면 resampler/MLP가 쉽게 불안정해진다.

2. **question-unconditioned objective일 가능성이 크다**
   - MotionBench QA는 질문과 선택지를 봐야 풀 수 있다.
   - 비디오만 보고 question type을 맞히는 것은 adapter의 최종 목적이 아니다.

3. **question_type classification은 adapter가 잘해야 할 목표가 아니다**
   - adapter는 VideoLLM의 QA 성능을 올리는 것이 목표다.
   - 비디오만 보고 질문 타입을 맞히는 proxy와 다르다.

4. **grid feature는 차원이 커서 작은 데이터에서 overfitting 위험이 크다**
   - Ridge/linear probe가 오히려 더 안정적일 수 있다.

따라서 다음 방향은 adapter를 포기하는 것이 아니라, objective를 바꾸는 것이다.

```text
하지 말 것:
  video -> adapter -> question_type label

해야 할 것:
  video + question + candidate answer -> adapter/scorer -> correct candidate
```

---

## 5. MotionBench probe를 더 신뢰 가능하게 만드는 방법

현재 결과는 pilot으로 좋다. 하지만 논문 근거로 쓰려면 통계적으로 더 단단하게 만들어야 한다.

### 5.1 Repeated stratified split / cross-validation

추천:

```text
- stratified 5-fold CV
- repeated 5-fold CV, seeds 5~10개
- out-of-fold prediction aggregate
- mean ± std
- 95% bootstrap confidence interval
```

리포트 예시:

```text
wan_vae_grid_sequence:
  38.5 ± 4.2%

pixel_grid_sequence:
  29.7 ± 3.8%

paired bootstrap difference:
  +8.8% [CI: ...]
```

가능하면 feature 비교는 paired comparison으로 한다.

추천 통계:

```text
- paired bootstrap confidence interval
- McNemar test
- per-split paired accuracy difference
```

### 5.2 Count table 저장

percentage만 저장하지 말고 count를 반드시 저장한다.

```text
accuracy: 39.3%
correct: 11
total: 28
```

작은 데이터에서는 percentage보다 count가 더 중요하다.

### 5.3 Confusion matrix 저장

question_type probe에서는 confusion matrix가 중요하다.

```text
rows: true question type
cols: predicted question type
```

확인할 것:

- Action Order를 Repetition Count와 헷갈리는가?
- Motion Recognition과 Motion-related Objects를 헷갈리는가?
- 특정 type만 잘 맞히는가?
- low-fps에서 어떤 type이 무너지는가?

---

## 6. Motion signal인지 shortcut인지 확인하는 control

현재 question_type probe가 chance 이상이라는 사실만으로는 motion signal이라고 단정하기 어렵다. MotionBench subset 자체에 appearance/domain bias가 있을 수 있다.

따라서 다음 control을 추가한다.

| Control | 목적 |
|---|---|
| metadata-only classifier | duration, fps, resolution, source, filename pattern bias 확인 |
| first-frame-only classifier | 정지 이미지 appearance cue만으로 풀리는지 확인 |
| time-average frame classifier | motion 없이 평균 이미지로 풀리는지 확인 |
| shuffled video feature | frame order를 섞어도 성능이 유지되는지 확인 |
| reversed video feature | order-sensitive feature인지 확인 |
| background/crop control | 객체/배경 shortcut 확인 |

좋은 결과 패턴:

```text
wan_vae_grid_sequence normal:    39%
wan_vae_grid_sequence shuffled:  27%
first-frame-only:                25~28%
metadata-only:                   25%
```

이런 결과가 나오면 “Wan grid가 temporal signal을 이용했다”는 주장이 강해진다.

나쁜 결과 패턴:

```text
first-frame-only:                38%
metadata-only:                   35%
wan_vae_grid_sequence normal:    39%
```

이 경우에는 Wan이 motion을 본 것이 아니라 dataset shortcut을 잡았을 가능성이 있다.

---

## 7. Low-fps 결과 해석과 다음 평가

현재 low-fps 결과는 강하게 해석하면 안 된다.

```text
uniform5:
  wan grid 35.7
  flow grid 35.7
  pixel grid 32.1

nonuniform5:
  wan global 39.3
  wan grid 35.7
  flow grid 28.6
```

해석:

- low-fps에서도 Wan이 완전히 무너지지는 않는다.
- 하지만 `global > grid` 같은 결론을 내리기에는 표본이 작다.
- accuracy 외에 prediction consistency를 봐야 한다.

### 추가할 metric

```text
consistency = P(pred_normal == pred_lowfps)
```

보고할 것:

```text
- normal accuracy
- uniform5 accuracy
- nonuniform5 accuracy
- normal vs uniform5 prediction consistency
- normal vs nonuniform5 prediction consistency
- lowfps에서만 맞는 샘플
- lowfps에서만 틀리는 샘플
```

Wan grid가 robust한 feature라면 accuracy가 조금 낮더라도 prediction consistency가 높아야 한다.

---

## 8. 바로 만들면 좋은 다음 스크립트

### 8.1 `motionbench_candidate_rerank_probe.py`

목적:

```text
answer-letter가 아니라 candidate semantics로 QA를 풀기
```

입력:

```text
motionbench_subset.jsonl
motionbench_features.h5
```

처리:

```python
for each sample:
    q = question
    choices = [choice_A, choice_B, choice_C, choice_D]
    correct = answer

    for each choice in choices:
        text = f"Question: {q} Answer: {choice}"
        text_emb = text_encoder(text)
        video_emb = wan_feature
        score = scorer(video_emb, text_emb)

    loss = cross_entropy(scores, correct)
```

출력:

```text
- overall QA accuracy
- per question type accuracy
- candidate score margin
- seed mean/std
- 95% bootstrap CI
- confusion among candidates
```

---

### 8.2 `motionbench_videollm_ensemble.py`

목적:

```text
VideoLLM baseline과 Wan scorer의 보완성 확인
```

입력:

```text
videollm_candidate_scores.jsonl
wan_candidate_scores.jsonl
```

Scoring:

```text
final_score = alpha * videollm_score + beta * wan_score
```

Sweep:

```text
alpha, beta
normalization method:
  - raw score
  - z-score per sample
  - rank-normalized score
  - temperature-scaled score
```

출력:

```text
- VideoLLM only accuracy
- Wan only accuracy
- ensemble accuracy
- oracle union accuracy
- disagreement analysis
- per question type gain
```

---

### 8.3 `motionbench_temporal_controls.py`

목적:

```text
motion signal인지 shortcut인지 확인
```

Features / conditions:

```text
normal
first_frame_only
time_average
shuffled
reversed
uniform5
nonuniform5
```

출력:

```text
question_type accuracy
candidate QA accuracy
normal - shuffled gap
normal - first_frame gap
normal - time_average gap
```

---

## 9. VideoLLM adapter는 ranker 다음에 진행

지금 바로 큰 VideoLLM adapter training으로 가는 것은 비효율적이다.

먼저 candidate-conditioned Wan scorer가 QA에서 signal이 있는지 확인해야 한다.

### 의사결정 기준

```text
Case A:
  Wan candidate scorer가 chance 25% 대비 35~40% 이상이고,
  VideoLLM과 disagreement에서 자주 맞음
  -> adapter training 진행

Case B:
  Wan candidate scorer는 약하지만 frame selection에 도움
  -> Wan-based frame selector로 전환

Case C:
  Wan candidate scorer도 약하고 complementarity도 낮음
  -> MotionBench feature-only 방향 축소,
     synthetic / MVBench / 다른 real task에서 재검증
```

### Adapter v2 구조

```text
Wan-VAE grid_sequence
  -> factorized resampler
  -> 16 or 32 motion tokens

question tokens
  -> cross-attention over Wan motion tokens

VideoLLM visual tokens + Wan motion tokens
  -> LLM
```

핵심은 question-aware adapter이다.

단순히 Wan token을 concat하는 것보다, 질문이 어떤 motion segment를 봐야 하는지 adapter 안에서 반영해야 한다.

---

## 10. Frame selector도 강한 대안

feature-only QA가 약하다면, Wan feature를 직접 답변 생성에 쓰기보다 frame selector로 쓰는 방향이 더 잘 먹힐 수 있다.

### 구조

```text
video
  -> Wan-VAE grid_sequence
  -> motion saliency per frame/segment
  -> top-K frames select
  -> VideoLLM에 selected frames 입력
```

### 비교군

```text
uniform sampling
random sampling
pixel motion magnitude sampling
optical-flow magnitude sampling
Wan-VAE grid saliency sampling
Wan question-conditioned saliency sampling
```

### 평가

```text
accuracy vs frame budget:
  8 frames
  16 frames
  32 frames
  64 frames

breakdown:
  Action Order
  Motion Recognition
  Motion-related Objects
  Repetition Count
```

좋은 결과 패턴:

```text
motion-heavy question:
  Wan selector > uniform selector

static/object-heavy question:
  Wan selector ≈ uniform selector
```

---

## 11. SSV2가 막혔을 때 대안

SSV2 selected pair는 gated data 문제로 바로 실행하지 못했다. 대신 다음 순서를 추천한다.

### 11.1 MotionBench subset 확대

이미 MotionBench pipeline이 돌아가므로 가장 낮은 비용의 다음 단계다.

```text
현재:
  4 types x 24 = 96

다음:
  가능한 DEV mp4 전부
  또는 type당 50~100개 목표
```

리포트에 반드시 남길 것:

```text
attempted videos
successfully downloaded videos
failed videos
used videos per type
download success rate per source/type
```

### 11.2 MVBench temporal tasks

다음 task들이 현재 Wan motion feature 실험과 잘 맞는다.

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

### 11.3 Video-MME short/medium subset

Video-MME는 motion-only benchmark는 아니므로, MotionBench/MVBench에서 signal을 확인한 뒤 generalization 확인용으로 쓰는 것이 좋다.

특히 frame selector 실험에 적합하다.

```text
8 frames
16 frames
32 frames
64 frames

uniform vs Wan-selected frames
```

---

## 12. 추천 실행 순서

### Step 1. MotionBench probe 안정화

```text
- repeated stratified 5-fold
- count, CI, confusion matrix 저장
- McNemar / paired bootstrap
- first-frame, time-average, shuffle, reverse control
```

목표:

```text
Wan grid > pixel/flow가 우연인지 확인
```

---

### Step 2. Candidate-conditioned Wan scorer

```text
video feature + question + candidate answer -> score
```

비교 feature:

```text
wan_vae_grid_sequence
wan_vae_global_sequence
pixel_grid_sequence
flow_grid_sequence
dit_l14_t900_token_mean
dit_l14_t900_spatial_tokens
```

목표:

```text
answer-letter가 아니라 실제 QA chance 25%를 넘는지 확인
```

---

### Step 3. VideoLLM score와 ensemble

```text
VideoLLM only
Wan scorer only
VideoLLM + Wan scorer
oracle union
disagreement analysis
```

목표:

```text
Wan이 VideoLLM의 motion failure를 보완하는지 확인
```

---

### Step 4. Question-aware frame selector

```text
uniform sampling vs Wan saliency sampling
```

목표:

```text
VideoLLM 입력 frame budget을 줄이면서 motion QA 성능 개선
```

---

### Step 5. Adapter v2

```text
Wan grid
  -> factorized resampler
  -> question-aware motion tokens
  -> VideoLLM
```

목표:

```text
ranker/selector에서 확인된 Wan signal을 end-to-end adapter로 흡수
```

---

## 13. 지금 하지 않는 게 좋은 것

### 13.1 Answer-letter probe를 더 파지 않기

A/B/C/D 위치 label은 의미가 없고, task별 test가 너무 작아 noise가 크다.

### 13.2 큰 VideoLLM adapter를 바로 학습하지 않기

현재 adapter가 약한 이유가 아직 분리되지 않았다.

가능한 원인:

```text
- feature 자체가 약함
- question conditioning이 없음
- data가 작음
- objective가 잘못됨
- resampler가 부적절함
```

candidate-conditioned scorer와 ensemble을 먼저 확인한다.

### 13.3 MotionBench 96개 subset만으로 강한 claim을 하지 않기

현재는 다음 정도가 적절하다.

```text
real-video pilot에서 positive signal 확인
```

강한 claim은 더 큰 subset, repeated split, control, QA-conditioned 실험 이후에 한다.

---

## 14. 최종 추천

지금 가장 좋은 다음 실험은 다음 하나다.

> MotionBench candidate-conditioned QA reranker를 만들고, VideoLLM baseline score와 ensemble했을 때 motion-heavy question에서 개선되는지 확인한다.

구체적으로:

```text
1. Wan-VAE grid_sequence를 video embedding으로 사용
2. question + candidate answer를 text embedding으로 사용
3. 4-choice ranking loss로 small scorer 학습
4. repeated CV로 Wan-only QA accuracy 측정
5. VideoLLM candidate score와 linear ensemble
6. per-type / disagreement / oracle union 분석
```

이 결과가 좋으면 논문 스토리는 자연스럽게 이어진다.

```text
toy/stress:
  Wan grid는 motion/order/count 정보를 강하게 보존

real MotionBench probe:
  feature-only에서도 weak positive transfer

candidate-conditioned QA:
  Wan feature가 question-answer semantics와 결합될 때 실제 QA signal 생성

VideoLLM ensemble/adapter:
  Wan generative motion prior가 VideoLLM의 fine-grained motion reasoning을 보완
```

현재 결과는 실패가 아니다.

오히려 **feature-only probe의 한계를 확인했고, 다음 단계가 question-conditioned Wan scorer라는 것을 명확히 보여준 결과**로 보는 것이 맞다.

---

## 15. Minimum viable next experiment

바로 실행 가능한 최소 실험은 아래와 같다.

```text
Script:
  experiments/motionbench_candidate_rerank_probe.py

Input:
  results/motionbench_real_20260509_062454/motionbench_subset.jsonl
  results/motionbench_real_20260509_062454/motionbench_features.h5

Feature:
  wan_vae_grid_sequence
  pixel_grid_sequence
  flow_grid_sequence

Text:
  question + candidate answer

Model:
  frozen text encoder + small MLP scorer

Evaluation:
  repeated stratified 5-fold CV
  per-question-type accuracy
  paired bootstrap CI

Success criterion:
  Wan candidate scorer > pixel/flow candidate scorer
  and Wan scorer has non-trivial oracle union gain with VideoLLM baseline
```

---

## 16. Short version for paper notes

```text
The MotionBench real-video pilot shows that Wan-VAE grid features retain weak but positive motion-type transfer signal beyond synthetic tasks. On a balanced 96-sample MotionBench DEV subset, Wan grid features outperform pixel/flow grid baselines in full-fps question-type probing and remain above chance under low-fps variants. However, feature-only and answer-letter probes are insufficient for real VideoQA. The next step is to convert Wan motion features into question-answer conditioned compatibility scores and test whether they complement VideoLLM candidate scores, especially on motion-heavy questions such as action order, motion recognition, and repetition count.
```
