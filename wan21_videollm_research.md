# Wan2.1 같은 Video Generation Model의 Feature를 활용한 VideoLLM 성능 향상 연구 방향

작성일: 2026-05-08

---

## 0. 핵심 결론

**Wan2.1은 VideoLLM 성능 향상을 위한 `feature teacher`, `motion prior`, `generative reranker`로 활용할 만한 가치가 크다.**

특히 기존 VideoLLM이 약한 영역인 다음 문제에서 연구 가능성이 높다.

- fine-grained motion understanding
- temporal order reasoning
- object interaction reasoning
- low-frame-rate video understanding
- long-video temporal localization
- counterfactual video reasoning
- multiple-choice VideoQA reranking

핵심 가설은 다음과 같다.

> VideoLLM은 언어 추론은 강하지만 motion token이 약하다. 반면 Wan2.1 같은 video generation model은 자연스러운 frame transition, object dynamics, motion prior를 학습한다. 따라서 Wan2.1에서 motion-aware feature 또는 generative compatibility score를 추출해 VideoLLM에 결합하면 temporal reasoning 성능을 개선할 수 있다.

---

## 1. 왜 Wan2.1 Feature가 VideoLLM에 도움이 될 수 있는가

많은 VideoLLM은 비디오를 프레임 단위 image encoder feature로 변환한 뒤, 이를 압축해서 LLM에 입력한다. 이 방식은 object recognition이나 scene understanding에는 강하지만, 다음과 같은 문제에는 약한 경우가 많다.

- 어떤 동작이 먼저 일어났는가?
- 물체가 왼쪽에서 오른쪽으로 움직였는가, 오른쪽에서 왼쪽으로 움직였는가?
- 사람이 컵을 잡은 뒤 마셨는가, 내려놓았는가?
- 특정 행동이 몇 번 반복되었는가?
- 두 객체가 접촉했는가, 단순히 가까이 있었는가?
- 중간 프레임이 생략되어도 사건의 흐름을 추론할 수 있는가?

비디오 생성 모델은 이런 문제와 본질적으로 연결되어 있다. 자연스러운 비디오를 생성하려면 모델은 다음을 알아야 하기 때문이다.

- 물체가 시간에 따라 어떻게 이동하는지
- 어떤 동작 전후 관계가 자연스러운지
- 텍스트 조건과 영상 dynamics가 어떻게 대응되는지
- scene layout과 motion trajectory가 어떻게 결합되는지
- 불가능하거나 부자연스러운 frame transition이 무엇인지

따라서 Wan2.1 같은 video diffusion / flow matching model은 단순 생성기를 넘어, **비디오 동역학 prior를 가진 teacher model**로 볼 수 있다.

---

## 2. Wan2.1에서 추출해볼 수 있는 Feature

Wan2.1에 공식적인 “feature extractor API”가 따로 있는 것은 아니지만, 공개 PyTorch 또는 Diffusers 구현을 사용하면 hook을 통해 여러 feature를 추출할 수 있다.

### 2.1 Feature 후보 요약

| Feature | 추출 난이도 | 의미 | VideoLLM 활용 방식 |
|---|---:|---|---|
| Wan-VAE latent | 쉬움 | 비디오를 압축한 spatio-temporal latent | 비디오 token 또는 motion token으로 사용 |
| VAE encoder intermediate feature | 쉬움~중간 | multi-scale texture, structure, motion feature | 기존 visual encoder 보조 feature |
| DiT block hidden states | 중간 | denoising 과정의 semantic + motion representation | Wan motion token으로 LLM에 추가 |
| timestep별 DiT feature | 중간 | noise level에 따른 structure/motion/appearance feature | motion/appearance 분리 분석 |
| predicted velocity / noise residual | 중간 | text-video compatibility score | VideoQA answer reranking |
| denoising loss | 중간 | 관측 비디오가 특정 prompt와 얼마나 잘 맞는지 | generative score로 사용 |
| cross-attention / text-conditioned saliency | 중간~어려움 | 질문 token이 어떤 video token을 참조하는지 | question-aware frame/region selector |
| VACE / I2V / FLF2V conditioning feature | 중간~어려움 | 조건부 생성에서의 dynamics prior | counterfactual reasoning, missing-frame reasoning |

---

## 3. 가장 먼저 검증해야 할 질문

연구를 시작할 때 가장 중요한 질문은 다음이다.

> Wan2.1 feature가 실제로 VideoLLM에 부족한 motion / temporal 정보를 담고 있는가?

이 질문에 답하기 전에는 바로 VideoLLM에 붙이는 것보다, feature probing 실험을 먼저 하는 것이 좋다.

---

## 4. 추천 테스트 1: Wan Feature Probing

### 4.1 목표

Wan2.1에서 추출한 feature가 다음 정보를 담고 있는지 확인한다.

- action class
- motion direction
- temporal order
- frame transition plausibility
- object interaction
- repetition count

### 4.2 비교할 feature

| 비교군 | 설명 |
|---|---|
| CLIP frame average | 프레임별 image feature 평균 |
| VideoMAE / InternVideo / SigLIP video feature | 일반적인 video representation baseline |
| Wan-VAE latent | Wan2.1 VAE encoder output |
| Wan-DiT early layer feature | local motion / geometry 가능성 |
| Wan-DiT middle layer feature | action / interaction 가능성 |
| Wan-DiT late layer feature | text-conditioned semantic alignment 가능성 |
| Wan-DiT high-noise feature | global motion / trajectory 가능성 |
| Wan-DiT low-noise feature | appearance / detail 가능성 |

### 4.3 추천 probing task

| Task | 설명 | 기대 효과 |
|---|---|---|
| Action recognition | Kinetics, SSV2 등에서 frozen feature linear probe | feature의 기본 action 정보 확인 |
| Temporal order classification | 정상 영상 vs reversed video | 시간 방향성 이해 확인 |
| Frame shuffle detection | 정상 영상 vs shuffled frames | temporal coherence 이해 확인 |
| Motion direction classification | left/right/up/down, approach/away | fine-grained motion 확인 |
| Object interaction classification | touch, pick, put, push, pull 등 | relation reasoning 확인 |
| Repetition counting | jump/tap/nod 횟수 예측 | count-sensitive motion 확인 |
| Before/after classification | event A before B 여부 | temporal relation 확인 |

### 4.4 가장 쉬운 sanity check

가장 먼저 해볼 만한 테스트는 다음이다.

```text
정상 비디오와 reversed/shuffled 비디오를 각각 Wan2.1에 넣고,
정상 비디오의 denoising loss가 더 낮은지 확인한다.
```

가설:

```text
Wan2.1이 자연스러운 temporal dynamics를 학습했다면,
정상 시간 순서의 비디오가 reversed 또는 shuffled 비디오보다
더 높은 generative likelihood, 즉 더 낮은 denoising loss를 가져야 한다.
```

이 실험은 별도 label 없이도 가능하므로 초기 검증에 적합하다.

---

## 5. 추천 테스트 2: Multiple-Choice VideoQA Diffusion Reranking

### 5.1 핵심 아이디어

VideoQA에서 후보 답변이 여러 개 있을 때, 각 후보 답변을 text prompt로 바꾸고, Wan2.1의 denoising loss를 계산한다.

정답 후보가 비디오와 더 잘 맞는다면, 그 prompt 조건에서 observed video latent를 복원하는 denoising loss가 더 낮을 가능성이 있다.

### 5.2 예시

질문:

```text
What does the person do after picking up the cup?
```

후보 답변:

```text
A. The person drinks from it.
B. The person throws it away.
C. The person puts it into a bag.
D. The person hands it to another person.
```

각 후보를 다음과 같은 prompt로 변환한다.

```text
Candidate A prompt:
In this video, after picking up the cup, the person drinks from it.

Candidate B prompt:
In this video, after picking up the cup, the person throws it away.
```

그다음 각 prompt에 대해 diffusion denoising loss를 계산한다.

```text
score(answer) = - E_t,eps [ || target_flow(z0, eps, t) - Wan_DiT(z_t, t, text=q+a) ||^2 ]
```

최종 답변 점수는 VideoLLM의 confidence와 Wan score를 결합할 수 있다.

```text
final_score(a) =
  alpha * VideoLLM_logprob(a | video, question)
  - beta * Wan_denoising_loss(video | question, a)
```

### 5.3 장점

이 방식의 장점은 크다.

- VideoLLM을 재학습하지 않고도 테스트 가능하다.
- multiple-choice VideoQA에 바로 적용할 수 있다.
- VideoLLM hallucination을 줄이는 reranker로 쓸 수 있다.
- prompt와 비디오의 compatibility를 생성 모델 관점에서 평가할 수 있다.
- motion-sensitive question에서 특히 유효할 가능성이 높다.

### 5.4 추천 benchmark

- MotionBench
- MVBench
- Video-MME
- NExT-QA
- EgoSchema
- SSV2 기반 temporal QA
- Diving48 기반 fine-grained action QA

---

## 6. 추천 테스트 3: Layer / Timestep Sweep

Wan2.1 feature는 어느 layer와 timestep에서 뽑느냐에 따라 성격이 크게 달라질 수 있다.

### 6.1 Sweep할 요소

| 요소 | 후보 |
|---|---|
| timestep | high noise, mid noise, low noise |
| layer | early, middle, late |
| prompt condition | empty prompt, video caption, question, question+answer |
| feature type | hidden state, predicted velocity, residual, denoising loss |
| pooling | spatial average, temporal average, top-K token, Perceiver resampler |
| normalization | raw feature, PCA, whitening, temporal difference |

### 6.2 예상되는 feature 성격

| 구간 | 예상 역할 |
|---|---|
| high-noise DiT feature | global motion, layout, trajectory, coarse event structure |
| mid-noise DiT feature | object interaction, action semantics, temporal relation |
| low-noise DiT feature | appearance, texture, identity, visual details |
| early layer | local motion, geometry, short-range transition |
| middle layer | action-level representation, object relation |
| late layer | text-conditioned semantic alignment |

### 6.3 중요한 분석

단순히 최고 성능만 보는 것이 아니라, 다음 질문을 분석해야 한다.

```text
1. Wan feature는 CLIP frame feature보다 temporal order를 더 잘 구분하는가?
2. high-noise feature가 low-noise feature보다 motion task에 강한가?
3. question-conditioned feature가 empty-prompt feature보다 VideoQA에 유리한가?
4. denoising loss는 정답 후보와 오답 후보를 분리하는가?
5. layer별로 action, object, motion 정보가 다르게 분포하는가?
```

---

## 7. VideoLLM에 결합하는 Architecture 아이디어

## 7.1 아이디어 A: Wan Motion Token Adapter

가장 정석적인 방법이다.

```text
video
  ├─ 기존 VideoLLM visual encoder
  │      └─ appearance / object tokens
  │
  └─ Wan-VAE + Wan-DiT hooks
         └─ motion / dynamics tokens
                  ↓
        small projector / Q-Former / Perceiver resampler
                  ↓
LLM input = [visual tokens; Wan motion tokens; question tokens]
```

### 핵심 설계

Wan-DiT hidden state는 token 수가 많기 때문에 그대로 LLM에 넣기 어렵다. 따라서 압축이 필요하다.

추천 방식:

```text
Wan hidden h: [B, N_time * N_h * N_w, C]

1. reshape to [B, T', H', W', C]
2. spatial average or top-K saliency pooling → [B, T', C]
3. temporal difference 계산: Δh_t = h_t - h_{t-1}
4. concatenate [h_t, Δh_t]
5. small projector로 LLM hidden dimension에 맞춤
6. 8~64개 motion tokens로 압축
```

### 학습 전략

초기에는 다음을 추천한다.

```text
freeze Wan2.1
freeze or partially freeze VideoLLM
train only motion adapter / projector
```

이렇게 해야 Wan feature의 순수 효과를 안정적으로 확인할 수 있다.

---

## 7.2 아이디어 B: Generative Score Reranker

VideoLLM이 후보 답변을 생성하거나 multiple-choice 후보를 평가한 뒤, Wan2.1이 generative compatibility score로 reranking한다.

```text
VideoLLM score:
P(answer | video, question)

Wan score:
- denoising_loss(video | question, answer)

Final score:
alpha * VideoLLM score + beta * Wan score
```

### 활용 예시

VideoLLM이 다음 답변을 냈다고 하자.

```text
The person throws the ball to the left.
```

그러나 Wan score가 다음 prompt에서 더 높다면:

```text
The person throws the ball to the right.
```

모델은 답변을 교정하거나 uncertainty를 높일 수 있다.

### 장점

- 학습 없이 zero-shot 실험 가능
- candidate answer가 있는 benchmark에서 바로 적용 가능
- VideoLLM hallucination detection에 활용 가능
- temporal / motion question에서 특히 유망

---

## 7.3 아이디어 C: Question-Aware Frame / Region Selector

긴 비디오에서 VideoLLM의 중요한 문제는 어떤 프레임을 볼지 선택하는 것이다.

Wan2.1을 question 또는 question+answer prompt로 condition한 뒤, 다음 신호를 saliency로 사용할 수 있다.

```text
saliency(frame, patch) =
  gradient of denoising loss wrt latent token
  or cross-attention strength between text token and video token
  or candidate loss difference under local masking
```

이 saliency를 통해 top-K frame 또는 patch를 선택하고, 선택된 부분만 VideoLLM에 입력한다.

### 비교 실험

```text
baseline 1: uniform frame sampling
baseline 2: CLIP text-video similarity sampling
baseline 3: motion magnitude / optical flow sampling
ours: Wan score / Wan gradient / Wan attention based sampling
```

### 기대 효과

- 긴 비디오에서 관련 구간을 더 잘 선택
- LLM context length 문제 완화
- fine-grained temporal QA 성능 개선
- low-frame-rate sampling의 정보 손실 완화

---

## 7.4 아이디어 D: Counterfactual Video Augmentation

Wan2.1을 feature extractor로만 쓰지 않고, counterfactual video pair를 생성하는 데 사용할 수 있다.

### 예시 pair

```text
A: The red ball moves from left to right.
B: The red ball moves from right to left.

A: The person picks up the cup before opening the door.
B: The person opens the door before picking up the cup.

A: The dog follows the cat.
B: The cat follows the dog.

A: The hand taps the table three times.
B: The hand taps the table four times.
```

이런 pair는 appearance는 거의 같고, motion/order/count만 달라진다. 따라서 VideoLLM이 temporal reasoning을 학습하기에 좋다.

### 활용 방식

```text
1. 같은 first frame 또는 reference image 사용
2. prompt만 motion/order/count 중심으로 변경
3. Wan2.1 I2V / FLF2V / VACE로 counterfactual pair 생성
4. 자동 filtering 수행
5. contrastive 또는 instruction tuning 데이터로 사용
```

### Filtering 전략

생성 데이터는 artifact와 bias가 있을 수 있으므로 filtering이 중요하다.

```text
1. optical flow로 motion direction 검증
2. tracker로 object trajectory 검증
3. action classifier로 action consistency 검증
4. Wan denoising score로 prompt-video consistency 검증
5. OCR/counting module로 반복 횟수 검증
6. 소량 human validation
7. generated-only evaluation 금지
8. real benchmark transfer 성능 확인
```

---

## 7.5 아이디어 E: Wan Feature Distillation

Wan2.1을 inference 때마다 사용하는 것은 비용이 크다. 따라서 Wan2.1을 teacher로 사용하고, 작은 student motion encoder를 학습하는 방향이 실용적이다.

### 구조

```text
Teacher:
Wan2.1 VAE + DiT hidden feature

Student:
small video encoder / temporal adapter / motion tokenizer

Goal:
Student가 Wan motion representation 또는 Wan score를 모방하도록 학습
```

### 학습 objective 예시

```text
L_total =
  L_QA
  + λ1 * || student_motion_token - Wan_motion_feature ||^2
  + λ2 * temporal_order_contrastive_loss
  + λ3 * diffusion_score_distillation_loss
```

### 장점

- inference 비용 감소
- Wan2.1 없이도 motion-aware VideoLLM 구성 가능
- 실제 배포 가능성 증가
- teacher-student 논문 구조로 명확함

---

## 8. 구현 Sketch

아래는 개념적 pseudo-code이다. 실제 module name은 Wan2.1 repository 또는 Diffusers 구현에 따라 달라질 수 있다.

```python
# Pseudo-code only

video = preprocess(video)  # [B, C, T, H, W]

# 1. Encode observed video into Wan latent
z0 = wan_vae.encode(video).latent_dist.mean

# 2. Add noise at selected timestep
t = sample_timestep(strategy="high_or_mid_noise")
eps = torch.randn_like(z0)
zt = scheduler.add_noise(z0, eps, t)

# 3. Encode text condition
prompt = format_prompt(question, candidate_answer)
prompt_embeds = text_encoder(prompt)

# 4. Register hooks for selected DiT layers
features = {}

def save_hook(layer_id):
    def hook(module, inputs, output):
        features[layer_id] = output.detach()
    return hook

for layer_id in selected_layers:
    wan_dit.blocks[layer_id].register_forward_hook(save_hook(layer_id))

# 5. Forward Wan-DiT
pred = wan_dit(
    hidden_states=zt,
    timestep=t,
    encoder_hidden_states=prompt_embeds,
)

# 6. Compute denoising / flow matching loss
loss = compute_diffusion_loss(pred, target)

# 7. Use hidden states as motion tokens
wan_motion_feature = pool_and_project(features[selected_layer])
```

---

## 9. 실험 설계 제안

## 9.1 Phase 1: Wan Feature가 쓸모 있는지 확인

### 목표

Wan2.1 feature가 motion과 temporal order 정보를 담고 있는지 검증한다.

### 실험

```text
1. CLIP frame feature vs Wan-VAE latent vs Wan-DiT feature 비교
2. linear probe / MLP probe
3. reversed video detection
4. shuffled frame detection
5. SSV2 action classification
6. MotionBench subset QA
```

### 성공 기준

```text
Wan-DiT feature가 CLIP frame average보다
motion-sensitive task에서 명확히 우수해야 한다.
```

---

## 9.2 Phase 2: Diffusion-Loss Reranking

### 목표

Wan2.1의 generative score가 정답 후보와 오답 후보를 분리하는지 확인한다.

### 실험

```text
For each video-question-answer candidate:
  1. prompt = question + candidate answer
  2. observed video latent z0에 noise 추가
  3. Wan-DiT denoising loss 계산
  4. 낮은 loss 후보를 더 높은 score로 간주
  5. VideoLLM score와 ensemble
```

### 분석

```text
1. 정답 후보의 평균 loss가 오답보다 낮은가?
2. motion question에서 loss separation이 더 큰가?
3. appearance question에서는 효과가 약한가?
4. prompt template에 얼마나 민감한가?
5. timestep에 따라 separation이 달라지는가?
```

---

## 9.3 Phase 3: Wan Motion Token Adapter

### 목표

Wan feature를 VideoLLM 입력 token으로 추가해 성능을 개선한다.

### 모델 비교

```text
1. VideoLLM baseline
2. VideoLLM + CLIP temporal tokens
3. VideoLLM + VideoMAE tokens
4. VideoLLM + Wan-VAE tokens
5. VideoLLM + Wan-DiT motion tokens
6. VideoLLM + Wan-DiT motion tokens + diffusion reranker
```

### 학습 방식

```text
1. Wan frozen
2. VideoLLM frozen 또는 LoRA tuning
3. adapter/projector만 학습
4. motion-heavy dataset 중심 fine-tuning
```

### 평가

```text
1. overall QA accuracy
2. motion question accuracy
3. temporal order question accuracy
4. counting question accuracy
5. object interaction question accuracy
6. long-video retrieval accuracy
```

---

## 9.4 Phase 4: Counterfactual Training

### 목표

Wan2.1으로 생성한 counterfactual pair가 real benchmark 성능을 올리는지 확인한다.

### 데이터 생성

```text
1. 같은 scene / object / first frame 고정
2. motion direction 변경
3. temporal order 변경
4. count 변경
5. object relation 변경
```

### 학습 objective

```text
1. contrastive learning
2. multiple-choice instruction tuning
3. pairwise preference learning
4. temporal order classification auxiliary loss
```

### 주의점

```text
Generated benchmark에서만 평가하면 안 된다.
반드시 real video benchmark로 transfer를 확인해야 한다.
```

---

## 10. 가장 추천하는 연구 플랜

우선순위를 매기면 다음과 같다.

### 1순위: Feature Probing + Diffusion Reranking

가장 빠르고 리스크가 낮다.

```text
1. Wan-VAE latent, Wan-DiT hidden state 추출
2. SSV2 / MotionBench subset에서 linear probe
3. multiple-choice QA에서 candidate별 denoising loss reranking
4. layer / timestep / prompt ablation
```

이 단계에서 다음 질문에 답할 수 있다.

```text
Wan2.1 feature가 VideoLLM에 실제로 도움이 될 만한가?
```

---

### 2순위: Wan Motion Token Adapter

1순위 실험에서 유의미한 layer/timestep을 찾은 뒤 진행한다.

```text
1. 선택된 Wan feature를 8~64개 motion token으로 압축
2. VideoLLM visual token 뒤에 concat
3. adapter만 학습
4. motion-heavy benchmark에서 평가
```

---

### 3순위: Counterfactual Video Augmentation

논문 novelty는 크지만 quality control이 어렵다. 따라서 1, 2순위 이후 진행하는 것이 좋다.

```text
1. Wan2.1으로 counterfactual pair 생성
2. 자동 filtering
3. contrastive/instruction tuning
4. real benchmark transfer 평가
```

---

## 11. 논문 아이디어 형태

### 제목 후보

```text
Generative Motion Tokens: Distilling Wan2.1 Video Diffusion Priors for Fine-Grained VideoLLM Reasoning
```

또는:

```text
Can Video Generation Models Teach VideoLLMs to Understand Motion?
```

또는:

```text
Wan2.1 as a Motion Prior: Generative Feature Extraction and Reranking for VideoLLMs
```

### 핵심 contribution

```text
C1. Wan2.1 같은 open video generation model에서 motion-aware feature를 체계적으로 추출한다.

C2. layer, timestep, prompt condition에 따라 어떤 representation이 temporal reasoning에 유효한지 분석한다.

C3. Wan motion tokens를 VideoLLM에 추가해 fine-grained motion QA 성능을 개선한다.

C4. Wan denoising loss를 generative compatibility score로 사용해 multiple-choice VideoQA reranking을 수행한다.

C5. 필요하다면 counterfactual video generation을 통해 temporal reasoning supervision을 강화한다.
```

---

## 12. 실패 가능성과 주의점

### 12.1 VAE latent만으로는 부족할 수 있음

Wan-VAE latent는 reconstruction 중심 representation이다. 따라서 motion이나 texture는 담고 있어도 action semantics나 causal relation이 linear하게 드러나지 않을 수 있다.

따라서 VAE latent만 보지 말고, DiT hidden state와 denoising residual도 함께 봐야 한다.

---

### 12.2 Prompt sensitivity

Diffusion-loss reranking은 prompt wording에 민감할 수 있다.

따라서 여러 prompt template을 ensemble하는 것이 좋다.

```text
Template 1:
In this video, {answer}.

Template 2:
The video shows that {answer}.

Template 3:
Question: {question}
Answer: {answer}

Template 4:
A video of {answer}.
```

---

### 12.3 Compute cost

Wan2.1 14B를 feature extractor로 쓰면 매우 무겁다.

초기 실험은 다음 설정을 추천한다.

```text
1. Wan2.1 1.3B 사용
2. 낮은 해상도 사용
3. selected timestep만 사용
4. selected layer만 hook
5. 1~4개 timestep 평균
6. feature extraction offline caching
```

---

### 12.4 Attention map 추출 문제

Flash attention을 쓰는 구현에서는 attention matrix가 반환되지 않을 수 있다.

대안은 다음과 같다.

```text
1. Q/K/V hook으로 attention 재구성
2. attention backend 변경
3. gradient saliency 사용
4. local masking 후 loss difference 사용
```

---

### 12.5 생성 데이터 bias

Wan2.1으로 만든 counterfactual video만 학습하면, 모델이 생성 artifact를 학습할 위험이 있다.

따라서 반드시 다음 평가가 필요하다.

```text
generated train → real test
real train + generated augmentation → real test
generated-only test 금지 또는 보조 분석으로만 사용
```

---

## 13. 가장 중요한 Ablation

연구를 설득력 있게 만들려면 다음 ablation이 필요하다.

| Ablation | 목적 |
|---|---|
| VAE vs DiT feature | 생성 모델의 어느 부분이 유용한지 확인 |
| high/mid/low timestep | motion 정보가 어느 noise level에 있는지 확인 |
| early/mid/late layer | representation hierarchy 분석 |
| empty prompt vs question prompt | text conditioning 효과 확인 |
| hidden feature vs denoising loss | feature 방식과 score 방식 비교 |
| Wan feature vs CLIP/VideoMAE | 기존 representation 대비 장점 확인 |
| motion question vs appearance question | Wan feature의 강점이 motion에 국한되는지 확인 |
| generated augmentation with/without filtering | synthetic data quality 영향 확인 |
| frozen Wan vs fine-tuned adapter | 학습 안정성과 성능 비교 |
| inference with Wan vs distilled student | 실용성 비교 |

---

## 14. 가장 먼저 실행할 Minimum Viable Experiment

### 목표

작은 비용으로 Wan2.1 feature의 가능성을 확인한다.

### 설정

```text
Model:
- Wan2.1 1.3B
- 기존 VideoLLM 1개

Dataset:
- MotionBench subset
- SSV2 subset
- MVBench motion-related subset

Features:
- CLIP frame average
- Wan-VAE latent
- Wan-DiT high-noise middle-layer feature
- Wan-DiT mid-noise middle-layer feature

Tests:
- reversed video detection
- shuffled frame detection
- motion direction classification
- multiple-choice QA reranking
```

### 판단 기준

다음 중 2개 이상이 성립하면 후속 연구 가치가 높다.

```text
1. Wan-DiT feature가 CLIP baseline보다 temporal order task에서 우수하다.
2. 정답 후보의 Wan denoising loss가 오답보다 평균적으로 낮다.
3. motion question에서 reranking accuracy가 오른다.
4. high/mid-noise feature가 low-noise feature보다 motion task에 강하다.
5. Wan motion token을 붙였을 때 VideoLLM의 motion QA 성능이 오른다.
```

---

## 15. 최종 제안

가장 논문화하기 좋은 방향은 다음이다.

> Wan2.1을 단순한 synthetic video generator가 아니라, VideoLLM을 위한 motion prior teacher로 사용한다. 구체적으로 Wan-DiT hidden state를 motion token으로 추출하고, Wan denoising loss를 answer compatibility score로 사용하며, 필요하면 Wan-generated counterfactual pair로 temporal reasoning supervision을 강화한다.

최종 시스템은 다음과 같이 구성할 수 있다.

```text
Observed video
   ↓
Wan-VAE encoding
   ↓
Noised latent z_t
   ↓
Wan-DiT forward with question/answer prompt
   ↓
[1] hidden state → motion tokens → VideoLLM
[2] denoising loss → answer reranking
[3] saliency → frame/region selection
   ↓
Improved VideoLLM temporal reasoning
```

가장 좋은 첫 실험은 다음이다.

```text
1. Wan2.1 feature를 뽑는다.
2. reversed/shuffled video detection으로 temporal prior를 확인한다.
3. MotionBench 또는 MVBench multiple-choice QA에서 diffusion-loss reranking을 해본다.
4. 잘 되는 layer/timestep을 찾아 motion token adapter로 확장한다.
```

---

## 16. 참고 링크

- Wan2.1 GitHub: https://github.com/Wan-Video/Wan2.1
- Diffusers Wan pipeline: https://huggingface.co/docs/diffusers/api/pipelines/wan
- Diffusers AutoencoderKLWan: https://huggingface.co/docs/diffusers/api/models/autoencoder_kl_wan
- Diffusers WanTransformer3DModel: https://huggingface.co/docs/diffusers/api/models/wan_transformer_3d
- MotionBench: https://arxiv.org/html/2501.02955v1
- GenRec: https://arxiv.org/html/2408.15241v1
- Motion-aware video diffusion feature example: https://arxiv.org/html/2405.14864v3
- Video diffusion features for recognition / representation: https://openreview.net/forum?id=SIZhZrU41O
- Diffusion feature for tracking / motion-related analysis: https://openreview.net/forum?id=BggfTUtZOM
