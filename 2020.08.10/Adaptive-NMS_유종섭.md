---
layout: post
title:  "Adaptive NMS"
date:   2020-08-10 23:16:13
categories: Deep_Learning
---



# Adaptive NMS

> Adaptive NMS : Refining Pedestrian Detection in a Crowd | [CVPR' 19]
>
> Songtao Liu - [Megvii Technology](https://www.google.com/search?source=hp&ei=5jExX-WdJJe7wAPsjb3ABQ&q=Megvii+Technology&oq=Megvii+Technology&gs_lcp=CgZwc3ktYWIQAzICCAAyAggAMgQIABAeMgQIABAeMgQIABAeMgQIABAeMgQIABAeMgQIABAeMgQIABAeMgQIABAeUPwBWPwBYNsDaABwAHgAgAGPAYgBjwGSAQMwLjGYAQCgAQKgAQGqAQdnd3Mtd2l6&sclient=psy-ab&ved=0ahUKEwjlqLO9xpDrAhWXHXAKHexGD1gQ4dUDCAc&uact=5)
>
> Di Huang - Associate Professor, School of Computer Science and Engineering, Beihang Univ.
>
> Yunhong Wang - Professor, School of Computer Science and Engineering, Beihang Univ.

- 논문의 저자는 Pedestrian Detection에서 prediction의 결과를 이미지화 하기 위해 여러개의 불필요한 박스들을 제거하는 과정이 얼마나 까다로운 작업인지 강조한다.

  논문 저자에 따르면, 한 이미지에 사람이 적게 나올때는 일반적인 greedy NMS나 soft-NMS를 사용해도 괜찮지만, 사람들이 많이 몰린 경우, 즉 검출의 대상이 실제로 많이 겹쳐있어서 ground truth 자체가 겹쳐진 경우에 문제가 생긴다고 한다.

  보통 NMS는 iou나 score를 기준으로 threshold값을 정하는데, 이 threshold값을 조절할때 사람들이 마주한 딜레마는 다음과 같다.

  1.  threshold값을 낮추어 더 적게 겹치는 박스까지 제거하는 경우 : 

     사람이 많이 몰려있는 이미지에서 miss rate를 상승시키는 요인이 된다. 이는 TP수를 줄여 detector의 성능이 낮아진다.

  2. threshold값을 높여 더 많은 박스를 남기는 경우 : 

     TP는 높게 유지되겠지만, 사람이 모여있지 않고 띄엄띄엄 있는 경우 한 사람에 대해 여러 박스가 나올 수 있으므로 FP수가 증가

  > 참고 : 
  >
  > Confusion matrix
  >
  > TP(True Positive) : 맞는 물체를 맞았다고 한 경우 / 실제 True, 모델 True (ex) class label 일치 & high iou
  >
  > TN(True Negative) : 맞는 물체를 틀렸다고 한 경우 / 실제 True, 모델 False (ex) class label 일치 & low iou
  >
  > FP(False Positive) : 틀린 물체를 맞았다고 한 경우 / 실제 False, 모델 True (ex) class label 불일치 & high iou
  >
  > FN(False Negative) : 틀린 물체를 틀렸다고 한 경우 / 실제 False, 모델 False(ex) class label 불일치 & low iou
  >
  > Precision(정밀도) = TP / (TP + FP) : 모델이 True로 분류한 것 중 실제 True인 것
  >
  > Recall(재현율) = TP / (TP + FN) : 실제 True중 모델이 True로 예측한 것 



## Non-Maximum Suppression(NMS)

- CNN layer를 통해 얻어낸 box에 대한 prediction 결과는 필요 이상으로 많이 나온다.

- 아래 그림처럼 필요 이상으로 얻어진 prediction들을 특정 threshold값을 적용하여 같은 object에 대한 결과로 인식되는 박스들을 하나만 남긴 채 없애준다.

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/adaptiveNMS/nms_img.png)



### NMS 알고리즘

- 가장 기본적인 NMS 알고리즘은 Greedy NMS로, 가장 score가 높은 박스를 기준삼아 그 박스와 나머지 박스의 iou값이 특정 threshold값보다 높은, 즉 highly-overlap된 박스들을 모두 제거해 주는 알고리즘이다.

- Greedy NMS 알고리즘은 iou값이 threshold값 이상이면 그 박스의 score를 0으로 만들어 물체가 없다고 판단하게 한다. 하지만 이 방법은 TP수를 줄게하는 요인이 되기 때문에 soft-NMS 알고리즘이 대두되었다.

  soft-NMS는 위와 같은 상황에서 score를 극단적으로 0으로 만드는 것이 아니라 특정 공식에 의한 패널티를 주어 score를 낮추는 방식이다. 이때 사용되는 패널티는 Gaussian penalty 공식으로, 다음과 같다.
  
  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/adaptiveNMS/gaussian_penalty.png)
  
  > D : final detection
  >
> s_i : score
  >
  > b_i : i번째 box prediction
  >
  > 참고 논문 : [Improving Object Detection With One Line of Code](https://arxiv.org/pdf/1704.04503.pdf)
  
- 다음은 Greedy NMS의 pseudo-code 이다.

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/adaptiveNMS/NMS_pseudo.png)



### Adaptive NMS

- 위에서 언급한것 처럼 threshold값을 높이면 물체들을 더 많이 남겨서 이미지에 물체가 빽빽히 분포된 경우에 검출 성능이 좋아지고, threshold값을 낮추면 물체들을 더 많이 제거해서 이미지에 물체가 듬성듬성하게 분포된 경우에 검출 성능이 좋아진다. 

- Pedestrian detection의 문제에서 NMS는 다음과 같은 목표를 가지고 진행된다.

  1. M은 objectness score가 가장 높은 물체이고, 이 박스와 멀리 위치한 박스일수록 그 박스가 FP일 likelihood는 낮기 때문에 이들은 제거하지 않는다.

  2. highly-overlapped인 이웃 박스들에 대해서는 물체 M과 얼마나 겹치느냐의 정도 뿐만 아니라 M이 빽빽한 물체들 사이에 위치해 있느냐의 문제에 대해서 생각해야 한다.

     만약 빽빽한 공간에 M이 위치했다면, 그와 highly-overlapped인 이웃 박스들은 TP일 확률이 높기 때문에 이들은 제거하지 않아야 하지만, 듬성한 공간에 M이 위치했다면 이때의 이웃박스들은 FP일 확률이 높기 때문에 큰 패널티를 주어야 한다.

- 이 밸런스를 더 잘 조절하기 위해 Adaptive NMS는 object density를 다음과 같이 정의하고 사용한다.
  
  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/adaptiveNMS/density_function.png)
  
  여기서 물체 i에 대한 density는 전체 ground truth 박스들과 물체 i와의 iou 최대값이다. 
  
- 위에서 얻어낸 object density를 이용하여 박스를 제거하는 과정은 다음과 같다.
  
  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/adaptiveNMS/pruning_step.png)
  
  > N_M : 물체 M의 threshold
  >
  > d_M : 물체 M의 density
  
- 위의 과정의 세가지 특징을 살펴보면,

  1. 물체 M과의 이웃 박스가 M에서 멀리 떨어져 있다면(iou < threshold), 그 이웃 박스들은 기존 NMS처럼 그대로 유지된다.
  2. 만약 물체 M이 물체들이 빽빽한 곳에 위치한다면(density > threshold), 그때의 density가 NMS의 threshold로 사용된다. density는 M과 ground truth 사이의 iou값중 가장 높은 값인데, 이를 threshold로 사용하게 되면 이웃 박스들은 M과의 iou가 density보다 낮은 경우가 많기 때문에 이웃 박스들이 제거되지 않는다.
  3. 만약 물체가 듬성듬성 분포된 곳에 위치한다면(density <= threshold), 그때는 기존의 threshold를 그대로 사용한다. 따라서 이 경우에는 기존의 NMS처럼 작동을 하게되며 아주 가까이 위치한(많이 겹치는) 박스들은 FP로 분류되어 제거된다.

- Adaptive NMS의 pseudo-code는 다음과 같다.

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/adaptiveNMS/Adaptive_NMS_pseudo.png)

- Adaptive NMS 알고리즘의 계산 복잡도는 greedy NMS, soft NMS와 비교할때 차이가 없다. 다만 Adaptive NMS는 물체의 density를 모아놓은 리스트 하나를 더 갖고있는데, 이는 요즘 컴퓨터 성능에서는 거의 차이가 없다고 봐도 무방하다. 



### Density Prediction

- density를 사용하는 법은 위에 설명했지만, 이를 얻어내는 것은 언급하지 않았다.

- 논문의 저자는 density prediction 문제를 regression 문제로 생각하고, density의 계산은 위에서 설명한 정의대로 시행햇다.

- training loss는 Smooth-L1 loss를 사용했다.

- 아래 그림과 같이 세개의 convolution layer가 Density subnet으로서 추가되었다. 이 subnet의 특징은 one-stage detector와 two-stage detector 모두 적용 가능하다는 점이다.

  1. two-stage detector인 경우, Density subnet은 Region Proposal Network(RPN)뒤에 붙는다.

     먼저 1x1 conv layer를 사용하여 feature map의 차원을 감소시키고, 이를 RPN의 결과와 concatenate하여 입력값으로 사용한다.

     마지막에는 5x5 사이즈의 큰 커널을 사용하여 convolution 연산을 함으로써 결과를 얻는다.

  2. one-stage detector인 경우, Density subnet은 마지막 detection network 뒤에 추가되어 two-stage detector의 방식과 마찬가지로 진행된다.

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/adaptiveNMS/density_subnet.png)

