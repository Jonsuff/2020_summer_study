---
layout: post
title:  "UnDeepVO"
date:   2020-08-31 22:16:13
categories: VODE
---



# UnDeepVO

| 제목 | UnDeepVO: Monocular Visual Odometry through Unsupervised Deep Learning |
| ---- | :----------------------------------------------------------: |
| 저자 |      Ruihao Li, Sen Wang, Zhiqiang Long and Dongbing Gu      |
| 출판 |                          ICRA, 2018                          |



### 연구 목적 및 연구 내용 요약

- Sfmlearner와 비슷한 방식으로 camera pose와 depth를 예측한다. 다른점이 있다면 UnDeepVO는 입력값으로 stereo image 쌍을 이용하여 실제 scale까지 학습할 수 있다는 점이다.

  > stereo image pair : (left image, right image)

  학습 과정의 입력으로 한 쌍의 이미지들을 사용하지만, 테스트 과정에서는 한 시점으로 본 연속적인 이미지만 사용하기 때문에 monocular system이라고 할 수 있다.

- system overview : 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/undeepvo/1.png)

- 예측하는 값 : 

  1. pose estimator를 통한 camera pose
  2. depth estimator를 통한 dense depth



### 모델 특징

- 학습 단계 시각화 : 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/undeepvo/2.png)

  학습에 사용되는 loss의 종류는 위의 그림에서 보는것과 같이 Spatial Image Loss와 Temporal Image Loss가 있다.

  기존 Sfmlearner처럼 monocular image의 연속된 이미지 프레임을 사용하여 depth를 학습하는 방법은 UnDeepVO의 Temporal Image Loss부분과 유사하다. 이렇게 한 시점에서 이미지 시퀀스를 통해 depth를 학습하면 물체들간의 상대적인 depth를 알 수 있지만 실제 scale은 알 수 없다.

  UnDeepVO에서 추가된 가장 중요한 특징은 Spatial Image Loss 부분이다. 이 부분에서는 왼쪽 시점의 카메라로 찍은 영상을 오른쪽 시점으로 찍은것처럼 변환하고, 그 변환된 이미지와 실제 오른쪽 이미지의 차이를 줄이는 방향으로 pose와 depth를 학습한다. monocular image와 다른점은, 왼쪽 카메라와 오른쪽 카메라의 실제 거리를 알고 있기 때문에 절대적 depth를 알 수 있고, 따라서 실제 scale을 학습할 수 있기 때문에 ground truth가 없는 unsupervised learning이지만 실제 scale까지 학습 가능하다는 장점이 있다.

  

  네트워크 모델 구조는 다음과 같다.

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/undeepvo/3.png)

  depth map을 예측하는 encoder-decoder 구조의 네트워크 하나, rotation과 translation을 각각 따로 학습시키는 pose estimation을 위한 네트워크 하나가 있다.

  

### Loss

- Spatial Image Loss : 

  spatial image loss는 다음과 같은 loss들로 구성되어 있다.

  1. *Photometric Consistency Loss* : 

     왼쪽 이미지와 오른쪽 이미지의 projective photometric error로 계산되는 loss이다. 쉽게 말하면 오른쪽 카메라로 얻은 이미지를 왼쪽 카메라의 시점에서 본 이미지로 변환하고, 이것과 원래 왼쪽 카메라로 얻어낸 이미지를 비교하여 얻어지는 loss이다. 

     이 과정을 단계별로 살펴보면 다음과 같다.

     우선 왼쪽 이미지와 오른쪽 이미지에 같은 사람이 나와있다고 생각해 보자. 만약 이 사람의 머리를 구성하는 이미지 픽셀들은 왼쪽과 오른쪽 모두에 존재하지만, 서로 시점이 다르기 때문에 두 픽셀의 위치는 동일하지 않을것이다. 이때 생기는 두 픽셀의 차이를 $$D_p$$(disparity)라고 한다. Disparity는 왼쪽 시야와 오른쪽 시야로부터 보이는 특징의 position 사이의 2D vector라고 설명할 수 있으며, 이는 depth와 반비례하다.

     > 왼쪽 픽셀 : $$p_l(u_l, v_l)$$, 오른쪽 픽셀 : $$p_r(u_r, v_r)$$이라면
     >
     > $$u_l = u_r$$ 일때 $$v_l = v_r + D_p$$

     그렇다면 이때의 $$D_p$$는 어떻게 구할까? 

     > B : stereo 카메라의 baseline(두 카메라 렌즈 사이의 거리)
     >
     > f : focal length(카메라의 초점 거리)
     >
     > $$D_{dep}$$ : 해당 픽셀의 depth값(depth estimator에서 예측)

     위의 값들을 사용하여 아래의 연산을 진행하여 $$D_p$$값을 계산할 수 있다.
     $$
     \\
     D_p = Bf/D_{dep}
     \\
     $$
     이렇게 두 픽셀사이의 차이를 얻어내고 나면, 우리는 한 시점에서 바라본 다른 시점의 이미지를 예측할 수 있고, 예측한 이미지와 실제 정답 이미지의 loss를 다음 연산을 통해 구할 수 있다.

     ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/undeepvo/5.png)

     > $$I_l,I_r$$  : 원본 왼쪽, 오른쪽 이미지
     >
     > $$I'_l$$ : 원본 오른쪽 이미지를 왼쪽 이미지로 변환한 것
     >
     > $$I'_r$$ : 원본 왼쪽 이미지를 오른쪽 이미지로 변환한 것
     >
     > $$L^{SSIM}$$ : Structural Similarity Index, 구조적 유사 지수 - [참고자료](https://m.blog.naver.com/PostView.nhn?blogId=y4769&logNo=220505513170&proxyReferer=https:%2F%2Fwww.google.com%2F)
     >
     > $$L^{l_1}$$ : L1 distance(x축값 + y축값)

  2. *Disparity Consistency Loss* : 

     왼쪽과 오른쪽 이미지의 disparity map은 다음과 같이 구할 수 있다.
     $$
     \\
     D_{dis} = D_p \times I_w
     \\
     $$

     > $$I_w$$ : 원본 이미지의 width

     원본 영상과 예측 영상의 disparity map은 같아야 하므로 다음과 같은 L1 distance를 loss로 사용한다.

     ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/undeepvo/6.png)

  3. *Pose Consistency Loss* : 

     논문의 저자는 같은 시각에 대해서 왼쪽과 오른쪽 이미지 각각 이미지 sequence 사이의 pose는 서로 같아야 한다고 말한다. 따라서 이 둘의 차이를 loss로 사용하여 학습한다.

     ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/undeepvo/7.png)

     > $$[\mathbf{x'}_l, \boldsymbol{\phi'}_l]$$과 $$[\mathbf{x'}_r, \boldsymbol{\phi'}_r]$$ : 왼쪽, 오른족 이미지에서 각각 예측된 pose

- Temporal Image Loss : 

  temporal image loss는 stereo image가 아닌 monocular image를 이용하는 loss이며, 다음과 같은 종류가 있다.

  1. *Photometric Consistency Loss* : 

     spatial image loss에서 왼쪽 이미지와 오른쪽 이미지를 서로의 시점으로 변환하여 예측한 이미지와 실제 이미지를 비교하여 loss를 구했다면, photometric image loss에서는 단안 카메라로 찍은 이미지 시퀀스 중에, k+1시간의 이미지를 한 프레임 이전인 k 시간의 시점(pose)에서 찍은것처럼 변환한 예측 이미지 $$I'_k$$와 실제 이미지 $$I_k$$ 사이의 차이를 SSIM과 L1 disatnce를 통해 loss를 구한다.

     ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/undeepvo/8.png)

  2. *3D Geometric Registration Loss* : 

     시간 k+1의 포인트클라우드를 시간 k의 포인트클라우드로 변환한 $$P'_k$$와 실제 k시간의 포인트클라우드 $$P_k$$사이의 차이를 L1 distance로 계산하여 loss를 구한다.

     ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/undeepvo/9.png)



### 연구 결과

- KITTI dataset을 사용한 depth 예측 결과 : 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/undeepvo/10.png)