# klue-sts
Solving Semantic Textual Similarity task for KLUE Benchmark dataset within 12 days

# NLU 문장 유사도 계산 (STS)

> 3,4주차 기업과제

## Process
1. 학습 데이터 전처리 
    
    - 맞춤법, 영숫특문제거
    - khaiii 형태소 분석기 사용
2. Pretrained Model 선정, 불러오기
    - klue-RoBERTa-base 모델
3. Train 코드 작성
    - transformers.Trainer 클래스를 이용한 훈련
4. 최적 하이퍼 파라미터 서치
    - Optuna를 이용한 하이퍼 파라미터 서치
5. Pearsonr, f1 score로 모델 훈련 결과 평가
6. FastAPI를 통한 모델 서빙

---

### 과제 목표

- 한국어 문장의 유사도 분석 모델 훈련 및 서비스화
- semantic textual similarity (의미적 텍스트 유사도)
- input) 2개의 한국어 문장
- output) 의미적 유사도 점수 출력

### 학습데이터셋

- KLUE-STS
    - AIRBNB - 리뷰
    - policy - 뉴스
    - parakQC - 스마트홈 쿼리

### 주의사항

- Train set만 사용하여 train/val로 나누어서 훈련시키기
- 공개된 pretrained 모델 사용 가능 (출처 명시)

### 제출사항

- 학습된 모델 (모델 자유 선택)
    Pretrained KLUE RoBERTa Large
- 학습 방식 보고서
    - 어떤 모델을 선택했나
    - 어떤 파라미터를 튜닝했나
    - 어떤 훈련 과정을 거쳤는가
- dev set score(dev set의 모든 문장을 pair에 대한 유사도 추론 결과와 F1 점수)
- REST API를 통해 모델을 이용하여 두 문장의 유사도를 분석하는 Server Code

