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


# Process

1. Data EDA (exploratory data analysis)
2. Data Preprocessing
    - Cleansing
    - khaiii 형태소 분석기 사용
3. Data Augmentation
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

  [KLUE STS 데이터](https://klue-benchmark.com/tasks/67/data/description)는 STS task를 해결하기 위해 만들어진 한국어 데이터셋이며, AIRBNB(구어체 리뷰), Policy(격식체 뉴스), ParaKQC(구어체 스마트홈 쿼리)의 세 가지 도메인으로 구성되어 있습니다. 전체 데이터 개수는 총 13,224개로, Train 데이터 11,668개, Dev 데이터 519개, Test 데이터 1,037개인 약 20:1:2의 비율로 구성되어 있습니다.  본 분석에서는 Train 데이터를 분리하여 Train과 Dev로 Dev 데이터를 Test로 간주하여 진행하였습니다. Train 데이터 내의 Dev의 비율이 9:1이 되도록 11,668개를 10494, 1167개로 분리하였습니다.[(비율 참고 레퍼런스)](https://stackoverflow.com/questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio) 그래서 Train:Dev:Test의 데이터 개수를 10494:1167:519 개로 재구성하였습니다. 데이터셋은 guid, source, sentence1, sentence2, labels, annotations 총 6개의 칼럼으로 구성되어 있으며 label은 real-label, label, binary-label 3가지 값을 갖고 있습니다. real-label은 0에서 5까지 범위에서 두 문장의 유사도를 실수형으로 표현한 라벨이며, label은 real-label을 소숫점 둘째 자리에서 반올림 한 값, 그리고 binary-label은 threshold 3을 기준으로 이하면 0, 이상이면 1로 표현한 값입니다. 

![binary-label은 Well-balanced 되어 있다](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/f5c3a29c-c231-439f-a44d-a5b65ef826a0/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220417%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220417T105850Z&X-Amz-Expires=86400&X-Amz-Signature=d1855297d40a4d26a17dc70f7dc86eb975287770dfdbb60826aadb51f17bbcc4&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)binary-label은 Well-balanced 되어 있다

![정수형 label의 경우 0에 가까운 값이 가장 많다](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/d8f47ef5-b098-42ec-8880-e8adc47d6a55/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220417%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220417T105923Z&X-Amz-Expires=86400&X-Amz-Signature=6002ac86ce6c2499ebfec44c2b61de1ea22f5fc7dedc10da4e935059d1fcf2e7&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)정수형 label의 경우 0에 가까운 값이 가장 많다

   KLUE STS Task에서 f1-score metric의 타겟 값으로 사용한 Binary label은 0.55 : 0.45의 비율로 다소 balanced 되어 있는 데이터셋이었습니다. Train셋에서 중복 행을 발견하여 하나씩만 남기고 제거해 주었습니다. ‘Sentence1’과 ‘Sentence2’ 내용이 같으면서 labels 까지 모든 값이  같은 중복 데이터가 5개가 존재했습니다. 결측치는 존재하지 않았습니다. 

## Preprocess

 KLUE STS 데이터셋에는 일반인이 직접 쓴 AirBnB(리뷰)가 포함되어 있기 때문에, 맞춤법이나 띄어쓰기 교정이 유의미할 것이다 판단하여 이에 대한 전처리를 수행하였습니다. 하지만 오히려 의미를 해치거나, 평균 0.56개의 맞춤법 에러를 고침으로써 시간 대비 성능이 나오지 않아 최종 데이터셋엔 반영하지 않았습니다. 한국어의 의미적 유사도를 측정하는 것이기 때문에 한국어를 제외한 영어, 한자, 일본어, 특수문자 등 의미를 해칠 수 있는 문자들을 제거하였습니다. 또 이번 프로젝트에서 **형태소 분석**에 가장 많은 시간을 투자하였습니다. 이는 문장의 의미적 유사도를 비교하기 위해선 중요한 의미를 담고 있는 명사, 동사 위주의 형태소 분리가 필요하다고 판단하였습니다. 좋은 형태소 분석기를 활용한다면 많은 성능을 끌어올릴 수 있다고 생각하였습니다. 전처리를 수행한 내용은 다음과 같습니다.

1. `pyKoSpace`를 이용해 띄어쓰기에 대한 전처리를 수행하였지만 오히려 의미를 해치는 경우가 발생하여 적용하지 않았습니다. 
2. `py-hanspell`을 이용해 맞춤법을 수정하였습니다.
    1. 전체 데이터셋에 적용한 후 고친 부분이 있을시 error로 표시하였는데, 평균 0.56개의 맞춤법 에러가 발생하였고, 평균적으로 에러가 1개도 있지 않다는 뜻이기 때문에 적용하지 않았습니다. 
3. 각종 특수 문자 및 영어, 일본어, 한자를 제거하였습니다.
4. 카카오의 형태소 분석기인 [‘Khaiii’](https://brunch.co.kr/@kakao-it/308)를 이용하여 형태소를 분리하였습니다.
    1. 불용어로 접속사, 조사 등의 문장의 의미에 큰 영향을 끼치지 않는다 판단되는 부분을 제거하였습니다. 품사리스트를 참고하여 각종 조사와 어미를 제거했습니다.
        
    ![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/5f498444-a24e-4699-b9f0-63f80d5640c2/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220417%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220417T110025Z&X-Amz-Expires=86400&X-Amz-Signature=5ab573239bc0826c6f563fb0819bcd0b573ad15542e8be8861ef9bde193246da&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)
        
    2. 파이썬 한국어 패키지 konlpy의 코모란(komoran), 꼬꼬마(Kkma), Okt(open korean text)나 한나눔, mecab 등의 다양한 형태소 분석기 외에 딥러닝 기반 khaiii를 선택한 이유는 실험을 통한 성능 비교 때문입니다. [해당 포스트](https://iostream.tistory.com/144?utm_source=gaerae.com&utm_campaign=%EA%B0%9C%EB%B0%9C%EC%9E%90%EC%8A%A4%EB%9F%BD%EB%8B%A4&utm_medium=social)를 참고하였습니다. 
        1. 분석 시간
        <img src = "https://s3.us-west-2.amazonaws.com/secure.notion-static.com/509d9ef2-970c-4a46-b06d-7335cbdfe8fa/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220423%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220423T080729Z&X-Amz-Expires=86400&X-Amz-Signature=ba52a4259804cc5cdcd5c7ae209b05a19a60e55171da0d8618c8fd43345d55d9&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="40%" height="40%">
            
        문장의 갯수가 늘어날수록, 꼬꼬마 분석기가 분석 시간이 오래 걸리고 꼬꼬마를 제외했을때 mecab 분석기가 가장 빠르게 형태소 분석을 하는 것을 확인할 수 있습니다. 그 다음으로 딥러닝 기반 분석기인 khaiii가 빠른 것을 확인하였습니다. 한 문장을 분석하는 속도의 경우, Komoran을 제외한 다른 분석기들 모두 0.0016초에서 0.0001초 사이로 빠른 것을 확인 할 수 있습니다. 
            
        2. 성능
            
            KLUE STS 데이터셋의 문장들은 대부분 맞춤법과 띄어쓰기가 제대로 되어 있습니다. 데이터셋의 문장 일부를 사용하여 형태소 분석기의 성능을 테스트하였습니다. 
            
            | 분석기 | 문장1 '그냥 모든게 다 완벽했던 에어비엔비 였어요’ | 문장2 '너가 생각하긴 거실을 가장 효과적으로 청소하려면 어떻게 해야될 것 같아?’ | 제거 문장 |
            | --- | --- | --- | --- |
            | Okt | [('그냥', 'Noun'), ('모든', 'Noun'), ('게', 'Josa'), ('다', 'Adverb'), ('완벽했던', 'Adjective'), ('에어', 'Noun'), ('비엔비', 'Noun'), ('였어요', 'Verb')] | [('너', 'Noun'), ('가', 'Josa'), ('생각', 'Noun'), ('하긴', 'Verb'), ('거실', 'Noun'), ('을', 'Josa'), ('가장', 'Noun'), ('효과', 'Noun'), ('적', 'Suffix'), ('으로', 'Josa'), ('청소', 'Noun'), ('하려면', 'Verb'), ('어떻게', 'Adjective'), ('해야', 'Verb'), ('될', 'Verb'), ('것', 'Noun'), ('같아', 'Adjective'), ('?', 'Punctuation')] | 1. 그냥 모든 다 완벽했던 에어 비엔비 였어요 2. 너 생각 하긴 거실 가장 효과 적 청소 하려면 어떻게 해야 될 것 같아 ? |
            | kkma | [('그냥', 'MAG'), ('모든', 'MDT'), ('것', 'NNB'), ('이', 'JKS'), ('다', 'MAG'), ('완벽', 'NNG'), ('하', 'XSV'), ('었', 'EPT'), ('더', 'EPT'), ('ㄴ', 'ETD'), ('에어', 'NNG'), ('비', 'NNG'), ('에', 'JKM'), ('는', 'JX'), ('비', 'NNG'), ('이', 'VCP'), ('었', 'EPT'), ('어요', 'EFN')] | [('너', 'NP'), ('가', 'JKS'), ('생각', 'NNG'), ('하', 'XSV'), ('기', 'ETN'), ('는', 'JKS'), ('거실', 'NNG'), ('을', 'JKO'), ('가장', 'MAG'), ('효과적', 'NNG'), ('으로', 'JKM'), ('청소', 'NNG'), ('하', 'XSV'), ('려면', 'ECE'), ('어떻', 'VA'), ('게', 'ECD'), ('하', 'VV'), ('어야', 'ECD'), ('되', 'VV'), ('ㄹ', 'ETD'), ('것', 'NNB'), ('같', 'VA'), ('아', 'ECD'), ('?', 'SF')] | 1. 그냥 모든 다 완벽 었 더 ㄴ 에어 비 에 비 이 었 어요 2. 너 생각 거실 가장 효과적 으로 청소 려면 어떻 게 하 어야 되 ㄹ 같 아 ? |
            | khaiii | [('그냥', 'MAG'), ('모', 'VA'), ('든', 'MM'), ('게', 'JKB'), ('다', 'MAG'), ('완벽', 'NNG'), ('하', 'XSA'), ('였', 'EP'), ('던', 'ETM'), ('에어비엔비', 'NNG'), ('이', 'VCP'), ('었', 'EP'), ('어요', 'EC')] | [('너', 'NP'), ('가', 'JKS'), ('생각', 'NNG'), ('하', 'XSV'), ('기', 'ETN'), ('ㄴ', 'JX'), ('거실', 'NNG'), ('을', 'JKO'), ('가장', 'MAG'), ('효과', 'NNG'), ('적', 'XSN'), ('으로', 'JKB'), ('청소', 'NNG'), ('하', 'XSV'), ('려면', 'EC'), ('어떻', 'VA'), ('게', 'EC'), ('하', 'VV'), ('여야', 'EC'), ('되', 'XSV'), ('ㄹ', 'ETM'), ('것', 'NNB'), ('같', 'VA'), ('아', 'EF'), ('?', 'SF')] | 1. 그냥 모 든 다 완벽 에어비엔비 이 2. 너 생각 거실 가장 효과 청소 어떻 하 같 ? |
            | mecab | [('그냥', 'MAG'), ('모든', 'MM'), ('게', 'NNB+JKS'), ('다', 'MAG'), ('완벽', 'NNG'), ('했', 'XSA+EP'), ('던', 'ETM'), ('에어', 'NNG'), ('비', 'XPN'), ('엔비', 'NNG'), ('였', 'VCP+EP'), ('어요', 'EF')] | [('너', 'NP'), ('가', 'JKS'), ('생각', 'NNG'), ('하', 'XSV'), ('긴', 'ETN+JX'), ('거실', 'NNG'), ('을', 'JKO'), ('가장', 'MAG'), ('효과', 'NNG'), ('적', 'XSN'), ('으로', 'JKB'), ('청소', 'NNG'), ('하', 'XSV'), ('려면', 'EC'), ('어떻게', 'MAG'), ('해야', 'VV+EC'), ('될', 'VV+ETM'), ('것', 'NNB'), ('같', 'VA'), ('아', 'EF'), ('?', 'SF')] | 1. 그냥 모든 게 다 완벽 했 에어 비 엔비 였 2.너 생각 긴 거실 가장 효과 청소 어떻게 해야 될 같 ? |
            
            도메인이 다른 두 문장에 대해 형태소 분석한 결과 좀 더 성능이 좋다고 느끼는 분석기는 명사형을 제대로 인식하고, 더 축약되는 khaiii였습니다. 이번엔 사용하기 좀 더 용이한 khaiii를 사용했지만, 형태소 분석기로 좀 더 활발하게 사용되고 있는 mecab을 다음 프로젝트때는 사용해보고자 했습니다. 이번엔 khaiii 분석기를 선택하여 전체 데이터셋에 대한 형태소 분리를 진행하였습니다. 
            

KLUE STS 데이터 셋에 대해 총 4단계의 전처리를 적용한 데이터셋을 가지고 훈련을 진행하였습니다. 각 단계별로 모델을 구분하여 Wandb로 성능을 모니터링 하였습니다. 아래 결과에서 확인할 수 있듯이 Batch_size 128개 모델로 khaiii 형태소 분석기를 거친 데이터셋이 가장 높은 validation score를 기록한 것을 확인할 수 있습니다. 

<img src = "https://s3.us-west-2.amazonaws.com/secure.notion-static.com/95c9cb62-8d40-495d-80a8-7c5cc69abcd4/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220423%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220423T081406Z&X-Amz-Expires=86400&X-Amz-Signature=cf217ea558986d6a8afdacb9bdb46d99725eed3e62696c00c5d040045e2bab6d&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="70%" height="70%">

- 전처리 후 데이터의 갯수 : train, val (10494, 1167)

# Data Augmentation

KLUE-STS 데이터셋은 1만여개의 한국어 문장 쌍으로 구성되어 있습니다. 하지만 다른 STS Task 프로젝트에서 훈련하는 데이터셋의 크기가 30k, 40k 정도로 확연히 크기 때문에, 우리 프로젝트의 성능과 과적합을 방지하고 모델의 일반화를 돕기 위해 EDA(Easy Data Augmentation) 기법을 선택하여 데이터를 증강하였습니다([[EDA: Easy Data Augmentation Techniques for Boosting Performance on
Text Classification Tasks(2019)]](https://arxiv.org/pdf/1901.11196.pdf)). CV 프로젝트에서는 자주 사용하는 일정 노이즈나 의미를 해치지 않는 변환을 부여하여 데이터를 늘리는 방식입니다. NLP에서는 다음 4가지 방법 중 하나를 각각의 문장에 임의로 선택하여 데이터셋을 강제로 증강합니다. 

1. SR(Synonym Replacement) : 불용어가 아닌 n개의 단어들을 선택해 임의로 선택한 동의어로 바꾼다.
2. RI(Random Insertion) : 불용어가 아닌 임의의 단어를 선택해 해당 단어의 임의의 유의어를 임의의 포지션에 삽입한다. 이를 n번 반복한다.  
3. RS(Random Swap) : 문장 내 임의의 두 단어의 위치를 바꾼다. 이를 n번 반복한다. 
4. RD(Random Deletion) : 문장 내 임의의 단어를 p의 확률로 삭제한다. 

n개의 단어를 선택하는 방식은 문장의 길이에 따라 문장 의미의 변질 정도가 달라지므로, SR, RI, RS는 문장의 길이 l을 사용하여 다음의 공식 n=αl에 따라 n을 결정하게 됩니다. α는 RD에서 p와 같은 값을 갖는, 문장 내 변하는 단어들의 비율을 지칭하는 매개변수입니다. 참고 논문에서는 이 α값을 train 데이터셋 크기에 따라 추천하며, 해당 프로젝트에서 원본 데이터가 1만여개이므로 α를 0.1로, 증강할 문장을 원본 문장당 4개로 진행하였습니다. 

![전체 데이터셋의 크기가 작을수록(500) 성능 향상이 더 높은 것을 확인할 수 있다. ](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/bdf5f96e-1b58-452a-bc02-b1313c0a345d/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220423%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220423T081616Z&X-Amz-Expires=86400&X-Amz-Signature=d2370bb5e847d944662b2fe02311dafb9311ba9d61d2555f8c1ea9aa127c6842&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

전체 데이터셋의 크기가 작을수록(500) 성능 향상이 더 높은 것을 확인할 수 있다. 

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a040fe19-6e52-4554-b4c6-2b2b2917cf14/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220423%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220423T081634Z&X-Amz-Expires=86400&X-Amz-Signature=65df966d8e6054673fdf5e5d7dd1f2352afc3dd83a30a25e679f16bb532c6772&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

증강 기법 중 RI, SR의 경우 동의어를 찾을 wordnet이 필요합니다. 상기 논문을 한국어로 변환한 [KorEDA 프로젝트](https://github.com/catSirup/KorEDA) 에서는 해당 wordnet을 KAIST에서 배포한 [Korean WordNet(KWN)](http://wordnet.kaist.ac.kr/)을 사용했고, 그대로 적용해 보았으나 동의어로 제대로 변환되지 않았고 중복 문장이 다수 발생하는 등의 어려움이 있었습니다. 따라서 새로운 유의어 사전이 필요하였고 국립국어원에서 제공하는 [모두의 말뭉치](https://corpus.korean.go.kr/main.do) - 어휘 관계 자료: NIKLex를 유의어 사전으로 활용하여 RI, SR에 이용하였습니다.NKLex 자료는 비슷한말, 반대말, 상위어, 하위어 등 어휘 관계를 총 5만명의 언어 사용자가 평가한 자료로서 어휘 관계 기초 자료 20만 쌍 중 비슷한 말 60,000쌍으로 제공하는 단어 수가 9714개인 KWN보다 더 풍부한 개체수를 갖고 있어 기존 어려움을 해결하였습니다. [KorEDA 프로젝트](https://github.com/catSirup/KorEDA)의 코드를 참고하여 Sentence1의 문장의 의미를 변질하지 않으면서 변형하여 증강하였습니다. 기존 Sentence1 문장의 짝인 Sentence2를 증강된 문장에 쌍으로 추가하였습니다. 전처리 된 Train 데이터 셋의 크기가 10,494개의 문장쌍을 갖고 있었는데, 61,389개의 데이터셋으로 증강시킬 수 있었습니다. 증강 후 중복되는 문장은 삭제하였습니다. 

비록 프로젝트 마감 내에 증강된 데이터를 코랩 환경의 한계로 끝까지 돌려볼 수 없었지만, 프로젝트 끝나고 보강하는 과정에서 시험해볼 수 있었습니다. 

# Select Model

  어떤 모델을 사용할 것인가에 대한 논의를 할 때, STS task 관련한 다양한 연구 및 논문을 서치하여 가장 성능이 높은 모델을 공부하였습니다.  [KLUE’s benchmark scores](https://github.com/KLUE-benchmark/KLUE#baseline-scores), [Tunib-Electra’s benchmark scores](https://github.com/tunib-ai/tunib-electra), [KoElectra’s benchmark scores](https://github.com/monologg/KoELECTRA) 등등의 벤치마크 스코어를 바탕으로도 고민 해 보았으며, 다른 언어의 STS Task를 잘 수행한다고 평가받은 [Sentence Transformers](https://arxiv.org/abs/1908.10084)를 이용한 모델도 시도했습니다. 결국 pretrained model로 [KLUE-RoBERTa-Base](https://arxiv.org/abs/2105.09680) 모델을 사용하였습니다. 해당 모델을 선정하는 데 있어서 한국어 적합성, 모델 크기 및 개발 환경을 기준으로 고려하였습니다. 

1. 한국어 적합성
    
    해당 모델은 KLUE 코퍼스를 사전학습한 RoBERTa base 임베딩 모델로서 KLUE 데이터셋의 한글 문장 유사도 측정을 할 때 최적화된 모델이며 높은 성능을 보일 것이라 판단하였습니다. 또 KLUE STS task의 벤치마크 리더보드에서 가장 높은 성능을 보인 모델이기 때문에, 파인 튜닝에 몰입하여 성능을 개선할 수 있다고 생각하였습니다. 
    다중어 모델 기준 STS를 포함한 다양한 task에서 높은 성능을 기록한 multi-use Sentence-BERT 기반 모델 [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), all-MiniLM-L6-v2 등의 pretrained model로 시도해 보았습니다. 해당 모델은 **`[microsoft/mpnet-base](https://huggingface.co/microsoft/mpnet-base)`**모델을 사전학습하여 1B 문장 쌍으로 파인 튜닝한 모델입니다. Contrastive learning을 사용한 모델로 이는 문장 페어 중 하나가 주어지면 모델은 랜덤하게 샘플링된 문장들 중 유사하지 않은 문장을 걸러내고 제일 유사한 페어를 예측하는 문장 벡터화 방법론입니다. 주로 unbalanced label을 갖고 있는 데이터셋에 효과적으로 활용하는 방법인데 한국어 context를 학습한 모델에 비해 적합성이 떨어지는 것을 확인하였습니다. 
    
2. 모델 크기 및 개발 환경

    KLUE 데이터셋을 사전학습한 모델의 경우 BERT base 모델과 RoBERTa base 모델이 있었고 각각 임베딩 사이즈와 레이어, 헤드 수로 large, small로 분류되었습니다. 이 중 KLUE 벤치마크 baseline 모델 중 STS task에서 가장 높은 점수인 pearsons’ r 93.35를 기록한 모델은 RoBERTa-large 모델이었습니다. 하지만 개인 랩탑(맥북 프로 intel 2019 모델)에서 구글 colab으로 모델을 훈련시키는 환경 상 large 모델의 경우 batch size를 조절해도 계속 메모리 오류가 발생하는 경우가 있었습니다. 현재 사용 가능한 자원 안에서 효율적으로 프로젝트를 진행하기 위해 base 모델 중 BERT 보다 1.65 정도 더 높은 성능을 기록한 92.5 스코어의 RoBERTa-base를 선택하게 되었습니다. 또 API 응답 속도를 고려하여 layer 수가 적절하면서 최대한 가벼운 모델을 불러올 수 있도록 결정하였습니다. 
    
    
    | Model | Embedding Size | Hidden Size | # Layers | # Heads |
    | --- | --- | --- | --- | --- |
    | KLUE-BERT-base | 768 | 768 | 12 | 12 |
    | KLUE-RoBERTa-base | 768 | 768 | 12 | 12 |
    | KLUE-RoBERTa-small | 768 | 768 | 6 | 12 |
    | KLUE-RoBERTa-large | 1024 | 1024 | 24 | 16 |

# Training Model & Hyperparameter tuning

### Training Model

먼저, 실무에서 현재 활발하게 사용하고 있다는 말을 NLP 실무자인 같은 팀원에게 들었고 처음 접해 보았기에 프로젝트에서 활용하고 싶어 선택하게 되었습니다.  [공식](https://huggingface.co/docs/transformers/main_classes/trainer) 도큐먼트의 모든 파라미터를 건드려보았다고 해도 좋을 정도로 심도 있게 활용해 볼 수 있었습니다. 
두번째로, TrainingArgument라는 batch size, optimizer, evaluator 등의 다양한 파라미터들을 입력하여 쉽게 학습할 수 있는 점이 가장 큰 장점이었습니다. layer를 freeze 또 f1_score, pearsonr 을 계산할 때 변수 하나만으로 쉽게 모델을 따로 학습시킬 수 있어서 편리했습니다. 
다음으로 Trainer의 경우 docs나 튜토리얼의 설명이 매우 잘 되어 있어서 오류가 발생하거나 하고 싶은 작업이 있을때마다 쉽게 이해하며 파인 튜닝을 진행할 수 있었습니다. 저는 NLP를 공부하는 학생의 입장에서 모델 학습 과정 뿐만 아니라, 그 과정에서 매개변수의 역할 또한 직관적으로 받아들이며 공부할 수 있어서 좋았습니다. 

### TrainingArgument
<img src = "https://s3.us-west-2.amazonaws.com/secure.notion-static.com/f7b89426-9c47-4367-ad9c-e7581bc0b3e2/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220423%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220423T081907Z&X-Amz-Expires=86400&X-Amz-Signature=6a6b02dcdeedb4f4d1f7a7e9c808235f6a99e133e5b52549e959e2e639b96634&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="40%" height="40%">

TrainingArgument에는 크게 Dataset, Optimizer, Regularization, Evaluation 관련 매개인자 값을 지정할 수 있습니다. 저는 학습 epoch를 10으로 잡고, eval epoch는 8로 데이터 크기를 고려하여 그보다 더 적게 잡았습니다. objective function으로는 Cosine Similarity 기반 MSE Loss를 사용하였습니다. Optimizer는 AdamW를 지정하였습니다. 학습률 Learning rate 6e-5, weight decay는 0.01로 trainer class의 default 값으로 지정하였습니다. 학습이 모두 끝나면, push_to_hub 파라미터로 HuggingFace Hub에 업로드 되도록 했고 나중에 훈련이 끝난 모델을 바로 허브에서 간편하게 끌어올 수 있도록 하였습니다. 

`fp16` 파라미터를 처음 접했는데, 모델이 학습할 때 32-bit Floating Point가 아닌, 16-bit Floating Point를 사용하는 방식이라고 합니다. True 값을 줘서 쉽게 사용할 수 있으며 모델 학습시 성능은 비슷하지만 약 60% 가량 향상된 속도로 학습을 진행할 수 있다는 장점이 있었습니다. 

### Hyperparameter tuning

Trainer에 존재하는 hyperparameter_search라는 메소드를 이용해 Hyperparameter tuning을 위한 최적값을 찾았습니다. Trainer에는 raytune, optuna, sig0pt과 같은 하이퍼파라미터 최적화 프레임워크를 사용할 수 있으며 저는 [optuna](https://github.com/optuna/optuna)를 선택하였습니다. 그 이유는 t설치부터 사용이 간편하고 프레임워크의 크기 및 구조가 가볍습니다. 또 조건 및 루프가 친숙한 python 구문을 사용하여 서치 범위를 지정할 수 있으며, 코드를 거의 변경하지 않고 팀원 모두가 이해할 수 있도록 간결하게 사용할 수 있습니다. 다음과 같이 optuna가 찾을 hyperparameter들을 함수로 정의했습니다.

하이퍼 파라미터 서치에 따른 여러 모델의 성능 평가는 Weights&Bias(Wandb)를 사용하여 모니터링 하였습니다. 하이퍼 파라미터 값에 따른 성능 추이를 시각화하여 제공하기 때문에 학습이 잘 이뤄지고 있는지, 최적의 파라미터 조합은 무엇인지 알 수 있습니다. Sweep 기능 등 각 하이퍼 파라미터 별로 범위를 지정하거나 영향력을 직관적으로 알 수 있는 기능에 대해 뒤늦게 알게 되어 아쉬웠고, 다음 프로젝트때 꼭 사용해보고자 합니다. 

![스크린샷 2022-03-24 오전 1.57.58.png](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/cf5517fa-9bba-4e8d-8dcf-ea0026923da3/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-03-24_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_1.57.58.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220423%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220423T082036Z&X-Amz-Expires=86400&X-Amz-Signature=9f5a019653a03b47a0a98feb37b2d26db12423b31b8cc3dd48fa11f4b558ae24&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA%25202022-03-24%2520%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB%25201.57.58.png%22&x-id=GetObject)

optuna는 hyperparameter search 방식인 grid search, random search, Bayesian method 중 베이지안 방식을 사용하면서, 속도가 굉장히 빠른 편입니다. 

optuna를 사용해서 조정한 parameter는 learning_rate, train_epochs, batch_size, weight_decay, warmup_steps입니다. 위 다섯 개 파라미터를 고른 이유는 Wandb 웹사이트의 ‘[HuggingFace Transformers를 위한 Hyperparameter 최적화](https://wandb.ai/amogkam/transformers/reports/Hyperparameter-Optimization-for-Hugging-Face-Transformers--VmlldzoyMTc2ODI)’ 칼럼에서 참고하였습니다. 

<img src = "https://s3.us-west-2.amazonaws.com/secure.notion-static.com/3b3d65c2-7be7-4ad4-a2e4-22441552328f/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220423%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220423T082053Z&X-Amz-Expires=86400&X-Amz-Signature=29e0d0d568eae7e92f7fc0b74bb1bc32934b8b8afed2a8c526429d1cd1030841&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="40%" height="40%">

<img src = "https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b0a2b487-c118-4e31-8bba-db669c3ff776/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220423%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220423T082124Z&X-Amz-Expires=86400&X-Amz-Signature=0f5d29dc53892f7c0f328edab8ee58f9d8c5f11ae6f0b592d38366714af337d0&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject" width="40%" height="40%">

13개의 hyperparameter 서치를 60회 트라이얼 동안 진행하면서 warmup_steps의 중요도가 가장 높게 나온 것을 확인할 수 있습니다. 음의 상관관계가 제일 높으며, warmup_steps가 낮을수록 accuracy는 높게 나옵니다. 해당 실험을 통해 가장 중요한 다섯가지 파라미터를 구할 수 있었으며, 가장 영향을 크게 주는 범위로 지정하여 hyperparameter search를 진행하였습니다. 

learning_rate는 너무 작으면 수렴하기까지 많은 iteration이 필요해 비효율적이지만, 너무 많은 iteration은 자칫하면 학습 데이터셋에 과적합되어 robust한 모델을 만들 수 없다는 단점이 존재합니다. 너무 크면 global minimum에 수렴하지 못하고 발산하는 문제가 발생할 수 있습니다. 그리고 데이터셋이 작아 과적합을 방지하기 위해 batch size를 크게 설정하였기 때문에, 이에 맞는 learning rate를 찾는 것이 중요하였습니다. num_train_epochs는 너무 많은 학습은 학습데이터에 과적합될 수 있기 때문에, 너무 큰 값으로 설정하면 안됩니다. per_device_batch_size도 너무 크면 메모리에 의해 제한될 수 있고, 너무 작으면 가중치를 더 많이 업데이트 하게 하기 때문에, 적절한 사이즈여야한다고 [Reference](https://medium.com/@aiii/how-to-tune-hyper-parameters-in-deep-learning-a0fa4bc1d782) 에서 언급하고 있습니다. warmup steps는 [Reference](https://moviecultists.com/what-is-warmup-learning-rate) 를 살펴보면, 학습의 시작 전후에 learning rate가 낮을 때, warmup steps의 단계를 지나 regular한 learning rate를 갖게하는 수치입니다. 이는 저희가 쓰고 있는 Adam같은 optimizer가 정확하게 gradient의 통계량을 계산할 수 있게 해줍니다. [Reference](https://medium.com/analytics-vidhya/deep-learning-basics-weight-decay-3c68eb4344e9#:~:text=Weight%20decay%20is%20a%20regularization,weights%20and%20not%20the%20bias.)를 보면 weight decay를 Gradient exploding을 피하고, Overfitting을 방지하기 위해 사용한다고 언급하고 있습니다. 보통 SGD optimizer나 Adam optimizer를 쓸때 사용하게 되는 Parameter인데 저희의 모델의 optimizer도 Adam이기 때문에, 지정해줘야한다고 생각했습니다. 그래서 이 Hyperparameter들에 대한 최적 탐색을 수행했고, 그 결과를 바탕으로 hyperparameter tuning을 진행하였습니다.

# Result

### Evaluation

> 💡 Pearsons’r score of 0.932
>   
> F1 score of 0.728

훈련에 사용되지 않은 제공된 Validation set을 Evaluation에 사용하여, 최종 스코어를 위와 같이 기록하였습니다. F1 score의 경우, Trainer class의 metric을 ‘f1’으로 지정하여 모델을 크게 수정하지 않고 모델을 훈련하였는데 그 과정에서 제대로 훈련되지 못했기 때문에 정확도가 매우 낮은 것으로 생각되었습니다. Pearsonr metric의 경우 KLUE-STS Leaderboard에서 2등 성적을 기록하였습니다. threshold가 3점인 것을 활용하여, 유사도 점수 3을 넘길 경우 1, 이하일 경우 0으로 binary-label을 예측한다면 f1 score가 상승할 것으로 생각됩니다. 

### Serving

FastAPI 프레임워크를 사용하여 Fine-Tuning한 Klue-RoBERTa-Base 모델을 서빙했습니다. 입력값으로 한국어 문장 2개를 넣으면 모델이 유사도를 측정하여 inference 결과를 출력합니다. STS task를 위한 Rest API 서버 코드는 https://github.com/honeybeat1/klue-sts-serving 해당 깃헙에서 확인하실 수 있습니다. Fine-Tuning된 model, tokenizer를 huggingface에서 받아오기 때문에 첫 접속시 로딩이 있습니다. 서버 코드를 실행하고 로컬 IP address([http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs))로 접속하면 post method를 통해 비교하고자 하는 Sentence1, Sentence2를 json 형식으로 전송합니다. Inference 결과는 비교할 두 문장, Cosine Similarity로 측정한 실수형 유사도 값, threshold 3점을 기준으로 하는 ‘두 문장이 비슷한가?”에 대한 답인 이진형 라벨 값이 반환됩니다. 코사인 유사도 값이 3 이상일시 ‘Yes’를, 이하일시 ‘No’를 반환합니다. 

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/24e32e3e-9d11-4ff6-ab3e-2911deb81b94/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220423%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220423T082302Z&X-Amz-Expires=86400&X-Amz-Signature=62127d8e4bf93a7b2c36991706820b1200633ff8e1504c19f75bda11cb8d4c67&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

## Works to try Next time

- 12일이라는 시간의 압박과 자원의 한계로 인해 Trainer 모듈을 사용하여 Fine-tuning을 진행했는데, 다음에는 별도 LSTM과 같은 bidirectional layer를 추가하여 모델의 성능 향상을 시도해보고 싶습니다. 또 NLP 프로젝트는 처음이라 현재 실무자로 일하고 있는 선배에게 실용적이고 SOTA 모델에 대해 많은 질문을 했었는데, pretrained 모델을 사용하지않고 토크나이저부터 새로 만들어 쓴다면 sentencepiece 모델을 활용해 보는 것도 다음에 시도하고자 합니다.
- 모델 epoch가 지날수록 과적합이 발생하여 Validation loss는 늘어나고 train loss는 줄어드는 상황이 발생했을때, 모델 자체에 early stopping 파라미터를 추가하지 않아 시간이 낭비되는 상황이 계속 발생했습니다. 이 때 모델을 돌려놓고 그 시간동안 하이퍼 파라미터나 데이터 전처리 검색을 계속 했던 상황이라 일손이 부족하여 이런 일이 발생했습니다. 다음에는 프로젝트 생산성에 초점을 맞춰보면서 진행할 예정입니다.
- Wandb를 이번에 처음으로 사용하면서 wandb가 제공하는 다양한 시각화 기능들을 심도 있게 사용하지 못해 아쉬웠습니다. 파라미터의 영향력과 accuracy에 미치는 영향 등을 다음에는 성능과 함께 모니터링 하고자 합니다.
- Data Augmentation 부분에서 EDA뿐만 아니라 Round-trip translation / Back Translation도 자주 쓰이는 방식인 것을 이번에 알게 되어, 다음에는 꼭 적용해보고 싶습니다.

# Acknowledgement

Role : 논문 리서치 / 모델링 / 데이터 전처리 / 데이터 증강 / 하이퍼 파라미터 튜닝 / 기타 코드 작업 / 보고서 작성 

저는 이번 NLP 프로젝트에서 데이터를 전처리하는 가장 첫 코드부터 마지막 API 서버 코드까지 완성해본 귀중한 경험을 했습니다. 이제껏 팀 프로젝트에서 스크래치부터 서버 코드까지 만져본 적은 없었는데, 처음으로 모든 부분을 진행했습니다. 자연어처리 프로젝트란 이런 프로세스를 거치는 것이구나 직접 알게 되어 프로젝트 말미인 지금 어떤 산봉우리에 도착한 느낌입니다. 공부하면 할수록 재밌고 더 적용하고 싶은 부분이 생겨서 너무나 재밌게 프로젝트를 진행했습니다. 특히 새로운 Tool을 docs를 열심히 읽어가며 사용해본 경험이 좋았습니다. 처음엔 Wandb의 존재도 몰라서 Notion에 별도로 try를 정리하는 수준이었습니다. 하지만 이제는 Wandb로 다양한 try들의 성능 모니터링이 가능합니다. 

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/c39b79d4-05f6-48c6-8c57-8452a1dd94e4/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220423%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220423T082322Z&X-Amz-Expires=86400&X-Amz-Signature=4db75456854c7975b8e3789096b31a4a60d7f1489473cc43fcea07c25e4a021a&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject)

이번 심화 과정에 참여하면서, 좋았던 점은 NLP 프로젝트가 처음이었지만 5주간 차근 차근 튜토리얼처럼 기초부터 알려주신 덕분에 따라할 수 있었던 점입니다. 제가 논문과 구글링을 통해 해보고 싶었던 전처리나, 모델 등을 시간 내에 다 적용해 볼 수 없었지만 프로젝트 보고서를 정리하면서 3주간 일일 과제를 통해 배웠던 기초를 바탕으로 집중적으로, 특히 sts task에 대하여 끝까지 파본 기분이라 뿌듯했습니다. 첫 import 코드부터 api 코드까지 플젝 모든 과정에 참여해본 적은 처음이라서 몸이 고되도 정말 많이 배웠다는 생각에 보람찼던 기간이었습니다. 감사합니다.

### Reference

1. A disciplined approach to neural network hyper-parameters
[https://arxiv.org/abs/1803.09820](https://arxiv.org/abs/1803.09820)
2. [How to Tune Hyper-Parameters in Deep Learning | by Neil Zhang | Medium](https://medium.com/@aiii/how-to-tune-hyper-parameters-in-deep-learning-a0fa4bc1d782)
3. Sentence Transformers
[https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)
4. Why do high learning rate diverges the weight updates?
[https://medium.com/@prash24goel/why-do-high-learning-rate-diverges-the-weight-updates-c39d9b3b326d](https://medium.com/@prash24goel/why-do-high-learning-rate-diverges-the-weight-updates-c39d9b3b326d)
5. Learning rate의 Max, Min
[https://arxiv.org/pdf/1506.01186.pdf](https://arxiv.org/pdf/1506.01186.pdf)
6. 카카오 형태소 분석기 Khaiii
[https://brunch.co.kr/@kakao-it/308](https://brunch.co.kr/@kakao-it/308)
7. KLUE benchmark
[https://arxiv.org/pdf/2105.09680.pdf](https://arxiv.org/pdf/2105.09680.pdf)
8. Weight decay
[https://medium.com/analytics-vidhya/deep-learning-basics-weight-decay-3c68eb4344e9#:~:text=Weight decay is a regularization,weights and not the bias](https://medium.com/analytics-vidhya/deep-learning-basics-weight-decay-3c68eb4344e9#:~:text=Weight%20decay%20is%20a%20regularization,weights%20and%20not%20the%20bias).
9. KoELECTRA
[https://github.com/monologg/KoELECTRA](https://github.com/monologg/KoELECTRA)
10. tunib-electra
[https://github.com/tunib-ai/tunib-electra](https://github.com/tunib-ai/tunib-electra)
11. EDA : Easy Data Augmentation for Boosting Performance on Text Classification Tasks
[https://github.com/jasonwei20/eda_nlp](https://github.com/jasonwei20/eda_nlp)
12. [konlpy] 형태소 분석기별 명사(noun) 분석 속도 비교
[https://needjarvis.tistory.com/691](https://needjarvis.tistory.com/691)
