# DL_user_sleep_classifier
---
## 📚**1. Project 소개**
- 운전자 부주의로 인한 교통사고 통계
  - [<흡연> 담배를 피는 운전자 10명 중 3.6명이 운전 중 흡연으로 인해 교통사고](http://www.gjtnews.com/news/articleView.html?idxno=31792)
  - [<졸음> 2019년부터 2021년까지 봄철(3월부터 5월까지) 하루 평균 7건 가량의 졸음 운전 사고 발생](https://www.iwjnews.com/news/articleView.html?idxno=52644)
  - [<통화> 운전 중 전화통화로 인한 사고는 정상 운행 상태에 비해 4배 이상의 사고 확률을 기록](https://www.korea.kr/news/reporterView.do?newsId=115084354#reporter)

- 운전자 이미지를 정상, 하품, 졸음, 흡연, 통화로 분류하는 모델을 구현하여 운전자 부주의 행태 모니터링 시스템 구축에 기여

## 📌**2. 결과**
### ① 운전자 이미지를 입력합니다.
![224_G3_34_정면광원_계기판_졸음재현_20201117_215450_02046](https://user-images.githubusercontent.com/122243187/230890185-00e3c960-6def-406d-90f3-a037d5455e45.jpg)

### ② 운전자 행태별로 이미지가 해당 행태를 나타낼 확률 값을 출력합니다.
<img width="425" alt="스크린샷 2023-04-10 194352" src="https://user-images.githubusercontent.com/122243187/230887099-1ef78f88-a107-4e25-b117-1bb847eb7ffb.png">

### ③ 가장 높은 확률 값을 가지는 운전자 행태를 최종 예측 값으로 간주하여 실제 값과 비교합니다.
- Validation Accuracy: 0.9312
<img width="580" alt="image" src="https://user-images.githubusercontent.com/122243187/233257337-76530c34-6144-4044-bb29-2d39e8beef7c.png">

- Confusion Matrix : Data set에 대하여 모델의 실제 label과 예측치 비교 <br>
<img width="270" alt="image" src="https://user-images.githubusercontent.com/122243187/233257469-80cbbcb7-fd4d-459b-b373-db0cf6bf478e.png">


## 📋**3. 과정**
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
- 본 프로젝트에서는 Tensorflow-Keras 프레임워크를 중심으로 진행하였습니다.<br>
```!pip install tensorflow```

### ③ EDA & 전처리

|**변경 전**|**변경 후**|
|:---:|:---:|
|이미지별로 Annotation 파일이<br> Json파일 형식으로 분산 저장|EDA 용이성을 위해 하나의 DataFrame에 저장|
|촬영 객체별로 폴더가 분리되어 저장되어 있음 |flow from directory 모듈 사용을 위해 label별로 폴더 생성 후 shutil.move()|
|이미지 1280 * 800의 흑백이미지<br>(channel = 1)| 직접 build한 모델 적용시 컴퓨팅 파워 고려,<br> 80 * 50 , 160 * 100, 320 * 200 등으로 변환|
|이미지 1280 * 800의 흑백 이미지<br>(channel = 1)|전이학습시 사전학습한 imagenet input shape 고려,<br> 224 * 224의 컬러 이미지(channel = 3)으로 변환|
|이미지 1장씩 label별로 저장| flow from directory 모듈을 활용하여 4장씩 batch 형성, shuffle = True|

- 위의 전처리를 통해서 우리가 사용할 최종 데이터를 가공했습니다.
- 이를 바탕으로 데이터를 분석해본 결과는 다음과 같습니다.

```
ⓐ Train data set은 100303장, Validation data set은 12563장으로 구성되어 있음.
ⓑ 운전자 행태는 정상, 졸음, 통화, 하품, 흡연으로 구성되어 있음.
ⓒ 상기 행태별 이미지는 50215장, 24954장, 12618장, 12547장, 12532장으로 약 4:2:1:1:1 비율임.
ⓓ 모든 이미지에 대하여 Face Bounding Box가 있음.
ⓔ Cigar Bounding Box는 100331장에 대하여 부재하고, 12535장에 대하여 형성되어 있음.
ⓕ Phone Bounding Box는 100248장에 대하여 부재하고, 12618장에 대하여 형성되어 있음.
```

### ④ Modeling 
- 컴퓨팅 파워를 고려하여 축약된 이미지를 기준으로 경량화된 모델부터 적용
- 점차 이미지 shape을 늘리거나, 정교한 모델로 갱신하여 모델링
- BN은 Batch Normalization의 약자
- TL은 Transfer Learning의 약자(pretrained data = imagenet)
- Resnet50 TL은 top layer만 train
- EfficientNetB0 TL은 총 238개 layer 중 200th layer부터 train<br>
- EfficientNetB3 TL은 총 385개 layer 중 350th layer부터 train<br>
※ 원천 데이터 중 일부 데이터만 테스트 한 경우 有<br>
※ flow from directory 파라미터 shuffle을 True로 설정한 관계로 iteration마다 성능 차이 있을 수 있음.

||**loss**|**accuracy**|**val_loss**|**val_accuracy**|
|:---:|:---:|:---:|:---:|:---:|
|VGGnet|1.2927|0.4396|1.3010|0.4021|
|Resnet w/o BN|6.1449|0.4396|3.9817|0.3918|
|Resnet w BN|0.4187|0.8184|0.5127|0.7788|
|Resnet50 TL|0.6958|0.7158|0.7224|0.7044|
|EfficientNetB0 TL|0.0731|0.9787|0.4358|0.8898|
|EfficientNetB3 TL|0.1174|0.9662|0.2571|0.9312|

### ⑤ Final Model
- accuracy와 val_accuracy가 가장 높은 EfficientNetB3의 전이학습 모델(350번째 layer부터 train) 채택


## 🔎 **4. 한계 및 개선사항**
### 1) 한계
- 프로젝트 내에서 활용한 데이터는 통제환경에서의 데이터로 학습데이터와 테스트데이터가 모두 같은 형식의 이미지 데이터임<br>
  따라서 다른 형식의 이미지 데이터(새로운 데이터)에 대한 추가적인 테스트가 필요<br>
  ex) 실제 운전 환경에서 촬영한 이미지 등
- 각자의 개인 컴퓨터를 이용했기 때문에 큰 해상도의 이미지로 학습을 시키지 못함
### 2) 개선사항
- EfficientNetB3를 전이학습 시킬 때 이번 프로젝트에서는 큰 의미를 두고 350번째 Layer를 trainable시킨 것이 아니라 추후에 이를 개선할 여지가 있음

## 💡**5. 참고문헌**
- [CNN 성능향상](https://velog.io/@ruinak_4127/CNN-%EC%84%B1%EB%8A%A5%ED%96%A5%EC%83%81)
- [y_pred와 y_true 값 추출하기](https://stackoverflow.com/questions/66636157/how-can-i-plot-a-confusion-matrix-for-image-dataset-from-directory)
- [confusion matrix 작성하기](https://benn.tistory.com/18)
- [batch normalization 성능 향상](https://eehoeskrap.tistory.com/430)
- [ResNet50 전이학습](https://velog.io/@dlskawns/Deep-Learning-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%B2%98%EB%A6%AC-%EC%A0%84%EC%9D%B4%ED%95%99%EC%8A%B5Transfer-Learning-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8-%EA%B5%AC%ED%98%84-%EC%8B%A4%EC%8A%B5#2-%EC%A0%84%EC%9D%B4%ED%95%99%EC%8A%B5%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%B4-%EC%B5%9C%EC%A2%85-%EB%B6%84%EB%A5%98%EA%B8%B0%EB%A5%BC-%EB%AA%A9%EC%A0%81%EC%97%90-%EB%A7%9E%EA%B2%8C-%EB%B0%94%EA%BE%B8%EA%B8%B0)
- [EfficientNetB3 전이학습](https://deep-learning-study.tistory.com/563)
- [EfficientNetB3 미세조정](https://luvbb.tistory.com/39)


## 🌈**6. 조원**
|**이름**|**링크**|
|:---:|:---:|
|강민주||
|강사라||
|김양우||	
|원수진||	
