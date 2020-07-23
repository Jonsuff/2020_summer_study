---
layout: post
title:  "CenterNet"
date:   2020-07-22 11:58:13
categories: Deep_Learning
---



# CenterNet: Keypoint Triplets for Object Detection

> ***Kaiwen Duan, Song Bai, Lingxi Xie, Honggang Qi, Qingming Huang, Qi Tian\***; Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 6569-6578
>
> github -> https://github.com/Duankaiwen/CenterNet



## 1. 모델 특징

### One-stage Detector

- CenterNet은 One-stage Detector 이다.

  Object Detector를 구현하기 위해서는 물체를 구별하는 Classification과 물체의 위치를 인식하는 Localization을 수행해야 한다. 만약 이 두 기능을 동시에 진행한다면 One-stage Detector라고 부르고(YOLO, CornerNet, CenterNet, ...), 순차적으로 Classification과 Locallization을 나누어 진행한다면 Two-stage Detector라고 부른다(R-CNN, fast R-CNN ...)



# 2. 네트워크 구조

### Hourglass Network

![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/centernet/hourglass_net.png)

- CornerNet에 사용된 네트워크 형식으로, 이미지의 여러 scale에 대한 정보를 downsampling 과정에서 추출하고, 이를 upsampling 과정에 반영하여 다양한 scale에서의 feature를 얻어낼 수 있다.



### CenterNet

![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/centernet/model_structure.png)

- 기본적으로 CornerNet의 네트워크를 유지하고, 코너를 찾아내는 작업 이외에 center region에 대한 정보를 찾아내는 작업을 추가하여 기존의 pair 데이터를 이용하는것과 달리 (TL, BR, C)의 triplet 데이터를 이용한다.

  

### 물체 인식 방법

- CenterNet의 기초가 되는 CornerNet.

  기본적인 모델 구조는 CornerNet: Detecting Objects as Paired Keypoints 논문에서 소개한 CornerNet과 거의 유사하다.

  CornerNet은 물체를 인식할 때 기존의 One-stage Detector들이 사용한 다수의 anchor box를 이용하는 방식이 아닌 물체의 좌측 상단과 우측 하단 좌표를 한 쌍의 keypoint로 이용하는 방식(corner pooling)을 채택했다.

  

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/centernet/cornerpool.png)

  물체의 코너는 이미지의 가로, 세로 끝부분과 물체와 배경의 경계(픽셀값 차이가 많은곳)의 거리값이 가장 큰 부분을 찾아낸다.

  

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/centernet/cornerdetect.png)

  

  위의 이미지처럼 물체의 좌측상단(TL(x,y)), 우측하단(BR(x,y))의 정보를 가지고 물체를 인식하는데, 이 알고리즘은 물체의 외곽선 인식에 민감하다.

  만약 한 쌍의 코너 벡터를 얻었을 경우에는  두 점 사이의 거리값을 기준으로 같은 물체의 코너인지, 다른 물체의 코너인지 구별한다(거리값 threshold값보다 크다면 그들은 서로 다른 물체의 코너이다).

  CenterNet에서는 중앙좌표(C(x,y))를 사용하여 물체를 인식하는데, 물체의 중앙부분이 히트맵이 되어 중앙좌표에서 세로로 가장 먼 점과 가로로 가장 먼 점을 통해 박스를 만들어 물체를 인식한다. 이를 통해 One-stage형식을 유지하면서 RoI pooling을 사용하는 효과를 얻어낼 수 있다.

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/centernet/centerdetect.png)

  > RoI(Region of Interest) pooling : 이미지에서 관심영역 일부를 추출하여 max_pooling을 진행하는 것. CenterNet의 중앙 region이 RoI pooling의 관심영역 추출과 비슷한 동작을 한다.

  하지만 물체의 중간부분에 대한 feature는 물체를 구별하는데 크게 도움을 주지는 못한다. 그 이유는 물체의 중간일수록 주변 픽셀과의 차이가 명확하지 않고 비슷한 데이터를 갖기 때문이다(예를 들어 우리는 머리, 얼굴, 손, 발, 다리 등 말단부분의 특징으로 사람을 구별하는 것이 배, 등과 같은 몸의 중심부의 특징으로 사람을 구별하는 것보다 쉬운것처럼). 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/centernet/cacadecorner.png)

  corner만으로 물체를 인식하는것은 물체의 바깥부분에 대한 정보만을 갖기때문에 이를 보안하기 위해 cascade corner pooling 기법을 사용한다. 이는 물체의 바깥 테두리를 코너로 찾고난 후 물체의 내부, 즉 코너를 찾을때 사용된 픽셀에서 물체 안쪽으로 비슷한 픽셀끼리의 최대값을 구하여 코너로 만들어진 테두리 내부의 feature까지도 이용할 수 있다.

- 수식 정보

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/centernet/tl_br_cv.png)
  $$
  ctl_x = {(n+1)tl_x + (n-1)br_x \over 2n}\\
  ctl_y = {(n+1)tl_y + (n-1)br_y \over 2n}\\
  cbr_x = {(n-1)tl_x + (n+1)br_x \over 2n}\\
  cbr_y = {(n-1)tl_y + (n+1)br_y \over 2n}\\
  $$
  논문의 저자는 n = 3, 5로 사용하였다.



### 학습 방법

- 입력 이미지의 크기는 511 x 511이고, 히트맵 크기는 128 x 128이다.

- optimizer는 adam을 사용한다.

- loss함수는 다음과 같다.

  
  $$
  \\
  L = L^{co}_{det} + L^{ce}_{det} + \alpha L^{co}_{pull} + \beta L^{co}_{push} + \gamma(L^{co}_{off} + L^{ce}_{off})
  \\
  $$
  $$L^{co}_{det}$$와 $$L^{ce}_{det}$$는 코너와 중앙 keypoint를 학습시키는데 사용되는 focal loss이다

  (잘 예측한 class에 대해서 가중치를 적게두어 변화를 적게하고, 잘못 예측하면 가중치를 높게두어 변화를 크게한다).

  

  $$L^{co}_{push}$$는 코너의 push loss로, 각기 다른 물체의 코너끼리의 거리를 최대가 되게하여 학습시킨다.

  

  $$L^{co}_{off}$$와 $$L^{ce}_{off}$$는 fast r-cnn에서 사용한 $$l_1 loss$$를 사용한다.

  

  $$\alpha, \beta, \gamma$$는 각각 $$L_{det}, L_{pull}, L_{push}$$에 대한 가중치이며 [0.1, 0.1, 1]로 설정한다.





## 3. 결과 및 성능

### One-stage Detector중 우수한 성적



![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/centernet/result_cocoAP.png)

- coco AP 성능지표를 보면 YOLOv3보다 속도와 정확도면에서 앞선다



![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/centernet/different_network.png)

- feature를 추출할때 Hourglass-104에서 높은 성능을 보였고, ResNet-18을 사용할때 FPS가 142로 높게 나왔다



![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/centernet/different_model.png)

- 다른 논문에 사용된 네트워크와의 비교표