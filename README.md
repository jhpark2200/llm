# LLM Post-Training Study Plan

<pre> ```text
llm-posttraining-study/
├── README.md
├── docs/
│   ├── interview_stories.md
│   └── notes_rlhf_dpo_grpo.md
├── src/
│   ├── losses/              # dpo/ppo/grpo toy losses
│   ├── data/                # dataset + collate + bucketing
│   ├── train/               # seqcls template / preference training
│   ├── eval/                # lm-eval harness wrapper (optional)
│   └── serve/               # vllm/fastapi skeleton (optional)
├── notebooks/
│   └── eda_tsne_hist.ipynb
├── scripts/
│   ├── run_seqcls.sh
│   ├── run_dpo_toy.sh
│   └── profile_gpu.sh
└── tests/
    ├── test_losses.py
    └── test_collate.py
</pre>



---

# 운영 원칙

- **PR 1개 = 산출물 1개**
- 면접에서 “결과물 링크”를 바로 제시할 수 있도록 관리
- 모든 산출물은 실행 가능하고 재현 가능해야 함

---

# 1주 플랜  
## 목표

- RLHF / DPO / PPO / GRPO 차이를 **말 + 코드**로 설명 가능
- 실무형 HF train template 1개 확보
- Attention / `einsum` 직접 구현 1개

---

## Day 1 ? 개념 압축 + 면접 Q&A 생성

### 주제
- RLHF vs DPO vs PPO vs GRPO
- 실패 모드 및 디버깅 포인트

### 산출물
- docs/notes_rlhf_dpo_grpo.md

(2~4페이지 정리)

### 면접 스토리
- 왜 DPO를 선택하는 팀이 늘었는가
- PPO의 운영 리스크는 무엇인가

---

## Day 2 ? Toy Loss 구현 (핵심 포트폴리오)

### 주제
- DPO loss
- PPO clip loss
- GRPO group-relative loss

### 산출물
- src/losses/toy_losses.py
- tests/test_losses.py


### 면접 스토리
- 수식 → 코드로 내려오는 과정
- shape 실수 방지 전략
- 테스트 설계 방식

---

## Day 3 ? HF 실무형 템플릿 (빠르게 완성)

### 주제
- `transformers` + `datasets`
- sequence classification 학습 루프

### 산출물
- src/train/train_seqcls.py
- scripts/run_seqcls.sh


### 면접 스토리
- 재현성 설계
- 로깅 전략
- collate 방어 코드
- save/load end-to-end 구성

---

## Day 4 ? Attention + einsum 구현

### 주제
- Scaled Dot-Product Attention
- mask 처리
- Multi-Head reshape
- `einsum` vs `matmul` 동치 검증

### 산출물
src/train/attention_from_scratch.py
tests/test_attention.py


### 면접 스토리
- 차원 설계 체크리스트
- reshape/transpose 실수 방지법

---

## Day 5 ? EDA (간결하게 1개)

### 주제
- histogram
- t-SNE
- 샘플링 전략
- summary.json 저장

### 산출물


notebooks/eda_tsne_hist.ipynb
또는
src/data/eda.py


### 면접 스토리
- 데이터 편향이 alignment에 미치는 영향
- 이상치가 preference 학습에 주는 영향

---

## 1주 완료 시 한 줄 요약

> RLHF/DPO/GRPO 목적함수를 toy로 구현하고 테스트로 검증했으며, HF 기반 학습 템플릿과 attention/einsum 구현을 공개 레포로 정리했습니다.

---

# 2주 플랜  
## 목표

- 실제 post-training과 유사한 최소 파이프라인 구축
- GPU 사용 경험 및 최적화 사례 확보

---

## Week 1

- notes
- toy losses
- seqcls template
- attention + einsum

---

## Week 2 ? 실전성 강화

---

### Day 6?7 ? Mini Preference Training (DPO)

#### 주제
- pairwise preference 데이터 (synthetic 가능)
- DPO 학습
- LoRA 옵션 추가

#### 산출물


src/train/train_dpo_mini.py
README 실행 섹션


#### 면접 스토리
- Reward Model 없이 preference 최적화가 가능한 이유
- KL/reference 모델 안정화 전략

---

### Day 8 ? PEFT (LoRA) + 메모리 전략

#### 주제
- LoRA 적용
- batch / sequence / gradient accumulation trade-off

#### 산출물


docs/peft_memory_notes.md

(실행 로그 포함)

#### 면접 스토리
- 제한 GPU 환경에서의 레버 조정 전략

---

### Day 9 ? GPU 프로파일링

#### 주제
- `torch.profiler`
- CUDA memory stats
- 병목 1개 이상 개선

#### 산출물


scripts/profile_gpu.sh
docs/gpu_profiling_report.md


#### 면접 스토리
- 문제 → 진단 → 개선 → 결과 (throughput / peak mem 수치)

---

### Day 10 ? 서빙 시스템 디자인 (1페이지)

#### 주제
- prefill vs decode
- batching
- KV cache
- metric 설계

#### 산출물


docs/serving_system_design.md


#### 면접 스토리
- 학습과 서빙 병목을 모두 이해하는 관점

---

## 2주 완료 시 한 줄 요약

> DPO 미니 파이프라인을 LoRA로 학습하고, torch.profiler로 병목을 분석했으며, 서빙 설계 문서까지 포함한 end-to-end 관점을 확보했습니다.

---

# 4주 플랜  
## 목표

연구 → 실험 → 평가 → 배포까지 축소판 end-to-end 재현

---

## Week 1 ? Foundations

- notes
- toy losses
- seqcls template
- attention/einsum

---

## Week 2 ? Minimal Post-Training

- DPO mini
- PEFT
- GPU profiling
- Serving design

---

## Week 3 ? Evaluation & Experiment Discipline

---

### Day 11?12 ? 평가 파이프라인 구축

#### 주제
- 실행 스크립트
- 결과 JSON
- 요약 리포트 자동 생성

#### 산출물


src/eval/run_eval.py
docs/eval_report.md


#### 면접 스토리
- 어떤 메트릭으로 "개선"을 증명했는가

---

### Day 13 ? 실험 관리 체계

#### 주제
- seed 고정
- config 관리
- 폴더 구조 설계
- 재실행 가능성 확보

#### 산출물


src/train/config.py 정비
docs/experiment_playbook.md


#### 면접 스토리
- 실험 복잡도를 통제하는 규율 설계

---

### Day 14?15 ? 알고리즘 선택 매트릭스

#### 주제
- PPO vs DPO vs GRPO
- 팀 상황별 선택 기준

#### 산출물


docs/algorithm_selection_matrix.md


#### 면접 스토리
- 데이터/리소스/안전성 기준 의사결정

---

## Week 4 ? Serving & Scaling

---

### Day 16?17 ? 서빙 스켈레톤

#### 주제
- streaming
- batching
- API 구조

#### 산출물


src/serve/app.py
docs/serving_runbook.md


#### 면접 스토리
- 모델을 제품 형태로 노출하는 구조 이해

---

### Day 18 ? 실패 모드 실험 1개

#### 주제 예시
- length bias
- over-optimization
- 데이터 노이즈

#### 산출물


docs/failure_mode_case_study.md


#### 면접 스토리
- 문제 재현 → 원인 격리 → 완화 전략 적용

---

### Day 19?20 ? 면접 스토리 패키징

#### 산출물


docs/interview_stories.md


#### 포함 스토리

1. DPO 파이프라인 구축
2. GPU OOM / 프로파일링 해결
3. Attention / einsum 디버깅
4. 평가 및 실험 관리 체계

---

## 4주 완료 시 한 줄 요약

> DPO 중심 post-training 미니 파이프라인을 구축하고, 실험·평가·프로파일링·서빙까지 축소판 end-to-end 사이클을 구현했으며, 실패 모드 재현 및 완화 사례까지 정리했습니다.

---

# GitHub 산출물 체크리스트 (면접 친화형)

각 PR / 산출물 README에 반드시 포함:



What:
무엇을 만들었는가 (한 줄)

Why:
어떤 문제를 해결하거나 검증했는가

How:
실행 커맨드 (복붙 가능)

Result:
수치 / 로그 / 표 / 스크린샷

Interview:
면접에서 말할 핵심 포인트 3개


--- 

이 문서는 포트폴리오를 "연구자"가 아니라 "즉시 투입 가능한 엔지니어"처럼 보이게 만드는 것을 목표로 설계되었습니다.