# klue-sts
Solving Semantic Textual Similarity task for KLUE Benchmark dataset within 12 days

# NLU 문장 유사도 계산 (STS)

> 3,4주차 기업과제

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
    Pretrained KLUE RoBERTa Base
- 학습 방식 보고서
    - 어떤 모델을 선택했나
    - 어떤 파라미터를 튜닝했나
    - 어떤 훈련 과정을 거쳤는가
- dev set score(dev set의 모든 문장을 pair에 대한 유사도 추론 결과와 F1 점수)
- REST API를 통해 모델을 이용하여 두 문장의 유사도를 분석하는 Server Code
---

# Intro

“한국어 문장의 유사도 분석 모델 훈련 및 서비스화” 프로젝트를 진행하였습니다. 

의미적 텍스트 유사도(Semantic Textual Similarity, 이하 STS)는 두 문장 사이의 의미적 동등성의 정도를 측정하는 Task입니다. 

주어진 학습 데이터 셋을 사용하여 STS 모델을 훈련하고, 그 후 두 개의 한국어 문장을 입력받아 두 문장의 STS 정도를 출력하는 모델을 생성하였습니다. 

좋은 데이터일수록 좋은 결과가 나오므로, 한국어 데이터의 의미적 정확도를 높이기 위해 다양한 전처리를 진행했습니다. 맞춤법, 띄어쓰기 교정이나 형태소 분석 라이브러리를 통해 문장에서 형태소만 분리하는 등의 방법 또한 시행하였습니다. 과적합을 방지하기 위해 단어의 위치 교환, 동의어 등을 이용한 방법(EDA)으로 데이터를 증강시켰습니다. 

프로젝트에 사용한 데이터셋은 [「KLUE: Korean Language Understanding Evaluation」(Park et al., 2021)](https://arxiv.org/pdf/2105.09680.pdf)을  통해 공개된 KLUE-STS 데이터입니다. 모델은 KLUE 데이터셋을 사전학습한 RoBERTa를 fine-tuning하여 BaseModel로 사용하였습니다. 

# Process

1. Data EDA (exploratory data analysis)
2. Data Preprocessing
    - 맞춤법, 영특문 제거
    - khaiii 형태소 분석기 사용
3. Data Augumentation
    - EDA (easy data augmentation)
4. Pretrained Model 선정, 불러오기
    - klue-RoBERTa-base
5. Fine-Tuning
    - transformers.Trainer 클래스를 이용한 훈련
6. Hyperparameter Search
    - Optuna
7. Evaluation Metric
    - Pearson’s r score, f1 score
8. Serving
    - FastAPI

# Data EDA & Preprocess
## EDA - 탐색

  [KLUE STS 데이터](https://klue-benchmark.com/tasks/67/data/description)는 STS task를 해결하기 위해 만들어진 한국어 데이터셋이며, AIRBNB(구어체 리뷰), Policy(격식체 뉴스), ParaKQC(구어체 스마트홈 쿼리)의 세 가지 도메인으로 구성되어 있습니다. 전체 데이터 개수는 총 13,224개로, Train 데이터 11,668개, Dev 데이터 519개, Test 데이터 1,037개인 약 20:1:2의 비율로 구성되어 있습니다.  본 분석에서는 Train 데이터를 분리하여 Train과 Dev로 Dev 데이터를 Test로 간주하여 진행하였습니다. Train 데이터 내의 Dev의 비율이 9:1이 되도록 11,668개를 10494, 1167개로 분리하였습니다. 그래서 Train:Dev:Test의 데이터 개수를 10494:1167:519 개로 재구성하였습니다. 데이터셋은 guid, source, sentence1, sentence2, labels, annotations 총 6개의 칼럼으로 구성되어 있으며 label은 real-label, label, binary-label 3가지 값을 갖고 있습니다. real-label은 0에서 5까지 범위에서 두 문장의 유사도를 실수형으로 표현한 라벨이며, label은 real-label을 소숫점 둘째 자리에서 반올림 한 값, 그리고 binary-label은 threshold 3을 기준으로 이하면 0, 이상이면 1로 표현한 값입니다. 

![binary-label은 Well-balanced 되어 있다](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/f5c3a29c-c231-439f-a44d-a5b65ef826a0/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220417%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220417T105850Z&X-Amz-Expires=86400&X-Amz-Signature=d1855297d40a4d26a17dc70f7dc86eb975287770dfdbb60826aadb51f17bbcc4&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)binary-label은 Well-balanced 되어 있다

![정수형 label의 경우 0에 가까운 값이 가장 많다](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/d8f47ef5-b098-42ec-8880-e8adc47d6a55/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220417%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220417T105923Z&X-Amz-Expires=86400&X-Amz-Signature=6002ac86ce6c2499ebfec44c2b61de1ea22f5fc7dedc10da4e935059d1fcf2e7&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)정수형 label의 경우 0에 가까운 값이 가장 많다

   KLUE STS Task에서 f1-score metric의 타겟 값으로 사용한 Binary label은 0.55 : 0.45의 비율로 다소 balanced 되어 있는 데이터셋이었습니다. Train셋에서 중복 행을 발견하여 하나씩만 남기고 제거해 주었습니다. ‘Sentence1’과 ‘Sentence2’ 내용이 같으면서 labels 까지 모든 값이  같은 중복 데이터가 5개가 존재했습니다. 결측치는 존재하지 않았습니다. 

## Preprocess

 KLUE STS 데이터셋에는 일반인이 직접 쓴 AirBnB(리뷰)가 포함되어 있기 때문에, 맞춤법이나 띄어쓰기 교정이 유의미할 것이다 판단하여 이에 대한 전처리를 수행하였습니다. 한국어의 의미적 유사도를 측정하는 것이기 때문에 한국어를 제외한 영어, 한자, 일본어, 특수문자 등 의미를 해칠 수 있는 문자들을 제거하였습니다. 다양한 라이브러리를 활용하여 전처리를 수행한 내용은 다음과 같습니다.

1. `pyKoSpace`를 이용해 띄어쓰기에 대한 전처리를 수행하였습니다.
2. 각종 특수 문자 및 영어, 일본어, 한자를 제거하였습니다.
3. `py-hanspell`을 이용해 맞춤법을 수정하였습니다.
4. 카카오의 형태소 분석기인 [‘Khaiii’](https://brunch.co.kr/@kakao-it/308)를 이용하여 형태소를 분리하였습니다.
    1. 문장의 의미 유사도를 비교하기 위해선 중요한 의미를 담고 있는 명사, 동사 위주의 형태소 분리가 필요하다고 판단하였습니다. 
    2. 불용어로 접속사, 조사 등의 문장의 의미에 큰 영향을 끼치지 않는다 판단되는 부분을 제거하였습니다. 품사리스트를 참고하여 각종 조사와 어미를 제거했습니다. 
    
    ![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/5f498444-a24e-4699-b9f0-63f80d5640c2/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220417%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220417T110025Z&X-Amz-Expires=86400&X-Amz-Signature=5ab573239bc0826c6f563fb0819bcd0b573ad15542e8be8861ef9bde193246da&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

KLUE STS 데이터 셋에 대해 총 4단계의 전처리를 적용한 데이터셋을 가지고 훈련을 진행하였습니다. 각 단계별로 모델을 구분하여 Wandb로 성능을 모니터링 하였습니다. 아래 결과에서 확인할 수 있듯이 Batch_size 128개 모델로 khaiii 형태소 분석기를 거친 데이터셋이 가장 높은 validation score를 기록한 것을 확인할 수 있습니다. 하지만 khaiii 라이브러리를 설치할 때 5분 정도 시간이 필요했는데, 소요 시간 대비 성능에 큰 차이가 없었습니다. 별도로 전처리를 하지 않은 데이터셋 또한 0.9558로 다소 괜찮은 validation score를 기록한 것을 확인할 수 있었습니다. 따라서 최종 분석에서는 영어 등의 기타 문자만 제거하는 cleanse 전처리만 거치고 추론하도록 하였습니다.

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/95c9cb62-8d40-495d-80a8-7c5cc69abcd4/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220417%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220417T105823Z&X-Amz-Expires=86400&X-Amz-Signature=d63eddeef49b131550d3e25e909ac4c94941bd86c09a94a4a5e727d84187326e&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

- 전처리 후 데이터의 갯수 : train, val (10494, 1167)
