---
layout: post
title:  "SfmLearner"
date:   2020-08-25 02:55:13
categories: VODE
---



# SfmLearner

| 제목 |   Unsupervised Learning of Depth and Ego-Motion from Video   |
| :--: | :----------------------------------------------------------: |
| 저자 | Tinghui Zhou(UC Berkeley), Matthew Brown(Google), Noah Snavely(Google), David G. Lowe(Google) |
| 출판 |                          CVPR, 2017                          |

- SfM(Structure-from-Motion) :  움직이는 연속적인 이미지(ex 비디오)를 이용하여 3D형태의 구조물(혹은 이미지)를 복원해 내는 것.
- Visual Odometry : 로봇의 현재 위치와 로봇이 바라보는 방향을 카메라와 같은 시각적 이미지를 통해 측정하는 것.
- Depth Estimation : 2D형태의 이미지 안에서 깊이(Depth)를 추정하는 것.
- Visual Odometry + Depth Estimation = VODE ( = DL-VODE : Deep Learning based -)



### 연구 목적 및 연구 내용 요약

- 인간은 인생을 살면서 자연적으로 시각정보를 3차원으로 이해하며 그것을 토대로 반응하거나 행동한다. 이는 인간이 오랜 기간 연속적인 시각정보를 통해 세상의 일반성을 학습하는 과정을 겪으며 성장하기 때문에 가능한 것으로 추정된다.

  한 가지 예로 인간은 두 눈으로 서로 다른 시점을 통해 사물을 바라보면서 입체감을 느낄 수 있다. 하지만 한 쪽 눈을 감더라도 완벽하진 않지만 대략적인 입체감을 예상하여 느낄 수 있다(실제로 정교한 작업을 하기에는 힘들다고 한다). 이는 인간이 인생을 살면서 두 눈으로 보며 학습한 입체적 정보를 통해 세상을 일반화 시켰기 때문에 한 눈으로도 예측이 가능한 것이다(세상의 일반화라 함은, 인간이 본능적으로 물체를 인식할 수 있는 것을 말한다. 예를 들어 평평한 곳은 땅, 푸른것은 하늘 등등..).

- 인간이 느끼는 입체감에 정답이 없이 오직 경험으로 쌓인 데이터에 의존하여 학습한 것 처럼, 본 논문의 method는 현재 pose에 대한 gt값 없이 target image와 그와 연속된 sequential scene을 학습 데이터로 사용하여 카메라의 ego-Motion을 예측하는 방법이다.



### SfmLearner의 특징

1. 본 논문의 method는 unsupervised한 방법이다.

   > supervised learning : 맞춰야 할 값이 있는 것. 즉 정답이 존재하고, 예측과 정답의 차이를 통해 맞음과 틀림이 존재. ex) 분류 문제, 회귀 문제
   >
   > unsupervised learning : 맞춰야 하는 target value(혹은 label)이 없는 것. 즉 정답이 존재하지 않고, 시스템이 정한 기준을 통해 물체를 분류하거나 스스로 기준을 만들어 내는 것. ex) 클러스터링
   >
   > 참고자료 : https://process-mining.tistory.com/98

2. Unlabeled video sequence의 캡쳐 이미지를 입력으로 사용하고, Single-view depth CNN과 camera pose estimation CNN을 이용하여 학습한다.

   ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/sfmlearner/1.png)

   위 그림과 같이 target이 되는 I_t와 1프레임 전, 후의 I_t-1, I_t+1을 통해 학습이 진행된다.

   - depth : 2D로 표현된 이미지에서 깊이를 나타냄
   - pose : unsupervised learning 이므로 gt값이 존재하지 않는다. 따라서 target 이미지와 sequential한 1프레임 전, 후 이미지를 이용하여 target 예측

3. depth model과 pose estimation은 test 단계에서 서로 독립적으로 사용된다.

4. 2번에서 언급한 Unlabeled video sequence란, 움직이는 카메라(혹은 카메라를 고정시킨 물체가 움직임)를 통해 얻어낸 짧은 image sequence 이다.



### SfmLearner의 연산

- 앞서 언급했듯이 sfm은 움직이는 연속적인 이미지를 이용하여 3D 데이터를 복원하는 것이며, 이는 depth network와 pose network를 통해 진행이 된다.

  - depth network는 한 장의 이미지로부터 depth를 추정.

  - pose network는 두 장의 이미지로부터 둘 사이의 상대적 pose를 추정.

    이때 target으로 지정한 이미지 전, 후 프레임을 source 이미지로 놓고 그것을 warping하여 예측 target 이미지를 만든다. 

    최종적으로 예측 target과 실제 target 사이의 photometric error를 loss로 사용하여 학습을 진행한다.

    ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/sfmlearner/2.png)

  

- 위의 식에서 I_t는 target 이미지들의 집합이고, I^_s는 source 이미지들을 입력값으로 사용하여 target 좌표계로 warp시킨 값들이다(예측된 target이다).

  그렇다면 target 이미지 픽셀(p_t)에 해당하는 source 이미지 픽셀(p_s)는 어떻게 계산할까? 다음 계산식을 보자.

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/sfmlearner/3.png)

  논문에서 K는 camera intrinsics라고 언급하고 있다. 이를 이해하기 위해서는 [카메라 캘리브레이션](https://darkpgmr.tistory.com/32)에 대한 내용을 알아야 한다. 쉽고 빠르게 정리해 보면, 카메라를 통해 보여지는 이미지는 실제로는 x,y,z를 가지는 3차원 세상이지만 이미지로 표현된 것은 2차원이다. 즉 카메라는 3차원 공간상의 점들을 2차원 평면에 투사함으로써 이미지를 얻어내는데, 이때의 projection을 K로 사용했다.

  다시 정리해보면 K는 image projection, D는 depth, T는 target to source 변환 이다. 이 용어들을 수식에 적용하여 수식을 이해해보면 다음과 같은 과정임을 알 수 있다.

  1. 먼저 p_t에 projection의 inverse와 depth를 곱하여 target 좌표계에서의 3차원 좌표를 얻는다.
     $$
     \\
     \hat{D}_t(p_t)K^{-1}p_t
     \\
     $$

  2. 그 다음 target 좌표계의 좌표를 source 좌표계로 옮기고
     $$
     \\
     \hat{T}_{t \rightarrow s}
     \\
     $$

  3. 다시 image projection을 곱해주면 이미지 픽셀 좌표인 p_s를 구할 수 있다.
     $$
     \\
     p_s \sim K\hat{T}_{t \rightarrow s}\hat{D}_t(p_t)K^{-1}p_t
     \\
     $$

  이 연산은 이미지의 모든 픽셀에 대하여 진행하고, 이렇게 한 픽셀씩 옮기다보면 연산 결과는 sourge 이미지 좌표를 target 좌표계에서 직접 찍은것 처럼 synthesize할 수 있다(이것이 예측 target이 된다). 



- 위에서 p_s는 얻어냈지만 우리가 loss에 사용할 것은 p_s가 아닌 I^_s(p_t) 이다. 이는 예측된 target 이미지가 모인 집합이므로 다른 표현을 사용한다면
  $$
  \\
  \hat{I}_s(p_t) = I_s(p_s)
  \\
  $$
  위와 같은 관계임을 알 수 있다. source 이미지는 target 이미지의 전후 프레임 이미지를 사용하기 때문에 target 이미지의 픽셀 좌표에 해당하는 source 이미지 픽셀 좌표는 두 가지가 될 것이고, 이를 통해 하나의 target 픽셀을 추론해야 한다.

  이때 선형보간법(linearly interpolation : 끝 점들이 주어졌을 때 사이의 값을 직선 거리에 따라 선형적으로 계산하는 방법)을 사용하여 다음 그림의 가운데 점과 같이 네 좌표를 얻어낸다.

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/sfmlearner/5.png)

  이제 가운데 그림에서 구해진 네 가지 점을 이용하여 p_t를 구해야 한다. 이때는 다음과 같은 공식을 사용한다.
  $$
  \\
  \hat{I}_s(p_t) = I_s(p_s) = \Sigma_{i \in{t,b}, j\in{l,r}} w^{ij} I_s({p_s}^{ij})
  \\
  $$
  여기서 w는 p_s점과 각각의 네 점 사이의 직선 거리의 상대적인 비율이고, 한 픽셀당 모든 w의 값의 합은 1이다.



### 모델 구조

- 기본적인 depth와 pose network의 구조는 다음과 같다.

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/sfmlearner/4.png)

  - depth network : 

    기본적으로 depth network는 DispNet 구조를 사용했다. 이는 encoder-decoder 구조이다. encoder 부분에서는 resolution(해상도)를 줄이면서 채널 수를 늘리고, decoder 부분에서는 다시 해상도를 늘리면서 채널 수를 줄인다.

    위의 그림에서 화살표를 잘 보면 encoder와 decoder 사이에 skip connection이 있는 것을 볼 수 있다. 이는 convolution 연산이 많이 진행될수록 고수준 feature를 얻게 되어 feature의 고유 특성이 강조되고, 연산이 적게 진행될수록 저수준 feature를 얻어 공간적 정보가 강조되는 특성을 이용한 것이다. 즉 encoder부분의 low-level feature와 decoder의 high-level feature를 합쳐 더욱 다양한 정보를 이용하게 되고, 이를 통해 depth를 예측한다.

    모든 conv layer들 뒤에는 ReLU activation이 붙는고(prediction 단계 제외), 예측값이 납득할만한 범위 안의 양수로 출력될 수 있도록 다음과 같은 파라미터를 사용한다.
    $$
    \\
    {1 \over (\alpha * sigmoid(x) + \beta)}
    \\
    \alpha = 10, \beta = 0.1
    \\
    $$

  - pose network : 

    pose estimation network에 입력값은 모든 source 이미지와 target 이미지가 concatenate된 데이터 이고, 출력값은 target 이미지와 그에 해당하는 source 이미지 사이의 상대적 pose이다. 

    네트워크 앞단에서 7개의 stride-2 conv를 거친 후 1x1 conv를 6*(N - 1) 채널로 연산을 한다(여기서 3개는 오일러 각도에 해당하고, 나머지는 3차원 변환에 해당한다).

    > 질문 : N에 대한 정보를 못찾았습니다...

    마지막으로 global average pooling을 사용하여 pose를 출력한다.

  - Explainability mask : 

    이 network는 pose network와 처음 다섯개의 feature layer를 공유한다. 이 feature layer 뒤에는 5개의 deconvolution layer가 따른다(이 결과로 이미지와 크기가 같아진다). 모든 conv/deconv layer 뒤에는 ReLU가 붙고 prediction layer에서는 붙지 않는다.

    각 prediction의 출력 채널수는 2 * (N - 1)개인데, 각 두개 채널마다 softmax function을 사용하여 normalize 한 후 이를 source-target pair에 각각 해당하는 explainability prediction으로 사용한다.

    이 값은 photometric error를 구할 때 weight 역할을 한다고 한다. 한 장의 이미지에서 움직임으로 인해 초점이 맞지 않는 물체나 엄폐물에 의해 가려진 부분이 많은 경우와 같이 depth를 추정하기 어려운 영역들에 낮은 weight를 줄 수 있는 방법이다.



### Loss

- 위에서 지속적으로 언급됐던 photometric loss는 다음과 같은 식으로 표현된다. 
  $$
  \\
  \mathcal{L}_{vs} = \sum_{<I_1, \dots , I_N> \in S} \sum_p \hat{E}_s(p)|I_t(p) - \hat{I}_s(p)|
  \\
  $$
  이 식의 의미를 살펴보면, 특정 target 이미지(I_t(p))에서 나온 depth 주변으로 여러가지 이미지를 source to target으로 변환하여 모든 픽셀에 대한 오차를 더하는데, 앞에 weight로서 E_s가 곱해지는 것을 알 수 있다.

- 논문의 저자는 우리가 E_s에 대한 정확한 정보는 갖지 않고, 그저 예측 한 것에 불과하기 때문에 학습이 진행되어 갈수록 E_s는 0에 가까워져 픽셀간의 오차와 관계없이 loss가 0에 수렴하는 문제가 생긴다고 설명한다. 

  이에 저자는 E_s에 regularization을 적용시켰다. 이는 0값이 아닌 예측 픽셀들과 상수값 1사이의 cross-entropy loss를 최소화함으로써 0이 아닌 예측값들에 힘을 실어준다고 한다. 다시 말해 네트워크는 서로 다른 시점의 이미지를 합성하는것은 최소화 하되(변화를 작게한다), 모델이 고려하지 못한 특정한 변화(변화가 큰 값)는 허용함으로써 학습이 진행된다.

- 위와 같은 단순한 photometric loss는 이미지 내에서 변화가 별로 없는 영역에서 학습 효과가 떨어진다. 따라서 더 큰 관점에서 depth를 추정할 수 있도록 depth network를 가운데가 좁은 encoder-decoder 구조로 설계하고 loss를 multi-scale로 연산한다.

  따라서 전체적인 loss함수는 multi-scale에 대하여 photometric loss + smoothness loss + regularized explinability term으로 계산하게 된다.
  $$
  \\
  \mathcal{L}_{final} = \sum_l \mathcal{L}^l_{vs} + \lambda_s \mathcal{L}^l_{smooth} + \lambda_e \sum_s \mathcal{L}_{reg}(\hat{E}^l_s)
  \\
  $$
  여기서 l은 각각 다른 이미지 스케일에 대한 인덱스이고, 람다들은 각각 smoothness loss에 대한 weight, explainability regularization에 대한 weight 이다.



### 연구 결과

- Cityscapes dataset으로만 학습한 결과 : 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/sfmlearner/6.png)

  

  

- Cityscapes dataset으로 훈련한 모델을 바로 Kitti dataset에 적용한 결과와 Kitti dataset으로도 학습하여 적용한 결과 : 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/sfmlearner/7.png)





- 다른 논문의 연구 결과와 비교한 결과 : 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/sfmlearner/8.png)





- Kitti + Cityscapes로 학습한 결과를 바로 Make3D dataset에 테스트한 결과 : 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/sfmlearner/9.png)