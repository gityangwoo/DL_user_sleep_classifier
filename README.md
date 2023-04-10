# DL_user_sleep_classifier
---
## 📚**1. Project 소개**
- 운전자 부주의로 인한 교통사고 통계
  - [<흡연>담배를 피는 운전자 10명 중 3.6명이 운전 중 흡연으로 인해 교통사고](http://www.gjtnews.com/news/articleView.html?idxno=31792)
  - [<졸음> 2019년부터 2021년까지 봄철(3월부터 5월까지) 하루 평균 7건 가량의 졸음 운전 사고 발생](https://www.iwjnews.com/news/articleView.html?idxno=52644)
  - [<통화> 운전 중 전화통화로 인한 사고는 정상 운행 상태에 비해 4배 이상의 사고 확률을 기록](https://www.korea.kr/news/reporterView.do?newsId=115084354#reporter)

- 운전자 이미지를 정상, 하품, 졸음, 흡연, 통화로 분류하는 모델을 구현하여 운전자 부주의 행태 모니터링 시스템 구축에 기여

## 📌**2. 결과**
### ① 운전자 이미지를 입력합니다.
(운전자 이미지 넣기)

### ② 운전자 행태별로 이미지가 해당 행태를 나타낼 확률 값을 출력합니다.
(모델에 운전자 이미지 input값으로 넣었을 때 출력되는 값을 주피터 코드 캡처해서 넣기)

### ③ 가장 높은 확률 값을 가지는 운전자 행태를 예측 값으로 실제 값과 비교합니다.
- Validation Accuracy: 0.8874
<img width="778" alt="스크린샷 2023-04-09 230903" src="https://user-images.githubusercontent.com/122243187/230874763-0e2b07f7-fa5c-40c2-9a9e-6e6af8f5b6e5.png">

- Data set에 대하여 모델의 입출력값 비교
<img width="477" alt="스크린샷 2023-04-09 230947" src="https://user-images.githubusercontent.com/122243187/230874447-fcfbe33f-40b3-4b98-88d4-8ba4c26f992b.png">


## **3. 과정**
### ① 데이터
- AI-HUB 데이터 중 '졸음운전 예방을 위한 운전자 상태 정보 영상'의 데이터 활용  
```
※ 출처: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=173
```
- 실제 도로 주행 데이터, 준 통제 환경 데이터, 통제 환경 데이터 중 **통제 환경 데이터 활용**
  - 데이터 수집을 위한 신뢰성 있는 졸음 및 부주의 상황을 판단하기 위한 다양성, 정밀성, 정량적 예측가능성 확보
  - 실제 승용차량을 완벽히 통제된 실험실 환경에서 일반운전 상황과 부주의 운전 상황을 시나리오에 따라 사람의 연기를 통해 연출한 영상을 수집한 데이터
  - 수집한 데이터를 얼굴윤곽, 눈, 코, 입, 소지품등을 바운딩박스로 가공한 데이터

### ② Frame Work
- 본 프로젝트에서는 Tensorflow-Keras 프레임워크를 중심으로 진행하였습니다.
```!pip install tensorflow```

### ③ EDA & 전처리

|**변경 전**|**변경 후**|
|:---:|:---:|
|이미지별로 Annotation 파일이 Json파일 형식으로 분산 저장|EDA 용이성을 위해 하나의 DataFrame에 저장|
|촬영 객체별로 폴더가 분리되어 저장되어 있음 |flow from directory 모듈 사용을 위해 label별로 폴더 생성 후 shutil.move()|
|이미지 1280 * 800의 흑백이미지(channel = 1)| 직접 build한 모델 적용시 컴퓨팅 파워 고려, 80 * 50 , 160 * 100, 320 * 200 등으로 변환|
|이미지 1280 * 800의 흑백 이미지(channel = 1)|전이학습시 사전학습한 imagenet input shape 고려, 224 * 224의 컬러 이미지(channel = 3)으로 변환|
|이미지 1장씩 label별로 저장| flow from directory 모듈을 활용하여 4장씩 batch 형성, shuffle = True|

- 위의 전처리를 통해서 우리가 사용할 최종 데이터를 가공했습니다.
- 이를 바탕으로 데이터를 분석해본 결과는 다음과 같습니다.

```
ⓐ Train data set은 100303장, Validation data set은 12563장으로 구성되어 있음.
ⓑ 운전자 행태는 정상, 졸음, 통화, 하품, 흡연이 있으며 각 행태별 이미지는 50215장, 24954장, 12618장, 12547장, 12532장으로 약 4:2:1:1:1 비율임.
ⓒ 모든 이미지에 대하여 Face Bounding Box가 있음.
ⓓ Cigar Bounding Box는 100331장에 대하여 부재하고, 12535장에 대하여 형성되어 있음.
ⓔ Phone Bounding Box는 100248장에 대하여 부재하고, 12618장에 대하여 형성되어 있음.
```

### ④ Modeling 
- 컴퓨팅 파워를 고려하여 축약된 이미지를 기준으로 경량화된 모델부터 적용
- 점차 이미지 shape을 늘리거나, 정교한 모델로 갱신하여 모델링
- BN은 Batch Normalization의 약자
- TL은 Transfer Learning의 약자(pretrained data = imagenet)
- Resnet50 TL은 top layer만 train
- EfficientNetB0 TL은 총 238개 layer 중 200th layer부터 train
※ 원천 데이터 중 일부 데이터만 테스트 한 경우가 있고 flow from directory 파라미터 shuffle을을 True로 설정한 관계로 iteration마다 성능 차이 있을 수 있음.

||**loss**|**accuracy**|**val_loss**|**val_accuracy**|
|:---:|:---:|:---:|:---:|:---:|
|VGGnet|1.2667|0.4270|1.3418|0.3333|
|Resnet w/o BN|0.0533|0.9886|12.9373|0.3125
|Resnet w BN|0.4187|0.8184|0.5127|0.7788|
|Resnet50 TL|0.6958|0.7158|0.7224|0.7044|
|EfficientNetB0 TL|0.0720|0.9790|0.6489|0.8732|

### ⑤ Final Model
(sj님 모델링 끝나면 작성)

## **4. 한계 및 개선사항**
(sj님 모델링 끝나면 작성)

## **5. 참고문헌**
(sj님 모델링 끝나면 작성)
