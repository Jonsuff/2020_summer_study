---
layout: post
title:  "YOLOv4"
date:   2020-09-10 13:56:13
categories: Deep_Learning
---



# YOLOv4

| 제목 | YOLOv4: Optimal Speed and Accuracy of Object Detection  |
| ---- | :-----------------------------------------------------: |
| 저자 | Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao |
| 출판 |                       arXiv 2020                        |

- 깃헙 주소 : https://github.com/AlexeyAB/darknet

- yolo의 알고리즘은 Joseph Redmon이라는 사람에 의해 탄생했고 발전해왔지만, 2020년 2월 그는 다음과 같은 이유로 더이상 Computer Vision 분야를 연구하지 않겠다고 말했다.

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_creator.png)



### 연구 목적 및 연구 내용 요약

- 실시간 객체 검출 시스템은 더 많은 수의 GPU의 성능을 요구해왔다(배치 사이즈가 커지면 커질수록 많은 GPU가 필요하므로). 하지만 이 논문의 저자는 다양한 스킬을 사용하여 하나의 GPU에서도 모델의 성능을 좋게 만들었다.

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_perform.png)

- 이 논문의 주된 목표는 다음과 같다.

  1. 하나의 GPU로도 좋은 성능을 내는 것

     > 저자의 말에 따르면 1080Ti나 2080Ti 하나에서 아주 빠르고 정확한 객체검출을 할 수 있다고 한다.

  2. 최신 BOF(Bag of freebies), BOS(Bag of specials) 기법이 성능에 미치는 영향을 증명

  3. CBN, PAN, SAM을 포함한 기법을 활용하여 single GPU training에 최적화

     

- YOLOv4에서 사용된 최신 딥러닝 기법 : 

  1. WRC(Weighted-Residual-Connections)
  2. CSP(Cross-Stage-Partial-Connections)
  3. CBN(Cross mini-Batch Normalizations)
  4. SAT(Self-Adversarial-Training)
  5. Mish Activation
  6. Mosaic Data Augmentation
  7. Drop Block Regularization
  8. CIOU Loss



### 관련 연구(Related Work)

- 2.1. Object Detection Models : 

  이 절에서는 object detector들의 모델 구조에 대하여 설명하고 있다. 최신 detector는 feature map을 만드는 백본(backbone)과 class label, bounding box를 예측하는 헤드(Head)로 구성되어 있다.

  본 논문에서 사용한 백본과 헤드는 다음 그림에서 볼 수 있다.

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_model_structure.png)

  모델 구조를 살펴보면 백본과 헤드(prediction 구간) 사이에 넥(Neck)이 존재하는걸 볼 수 있다. 넥에서는 백본에서 추출된 feature map을 내가 사용할 데이터 형태에 맞게 재구성한다.

  대표적인 넥 구조로는 FPN(Feature Pyramid Network), PANet(Path Aggregation Network), Bi-FPN등이 있다.

  

- 2.2. Bag of Freebies(BOF) : 

  inference 중에 추가 계산 비용을 발생시키지 않으면서 object detection 네트워크의 성능을 향상시키기 위해 조합하여 사용할 수 있는 여러 가지 기법들을 통칭한 용어이다. 대표적인 BOF 기법은 다음과 같다.

  >  Inference refers to the process of using a trained machine learning algorithm to make a prediction

  1. **데이터 증강** : 

     (ex) CutMix, Mosaic 

  2. **BBox Regression의 loss함수** : 

     (ex) IOU loss, CIOU loss



- 2.3 Bag of Specials(BOS) : 

  이는 BOF와는 다르게 inference는 소폭 상승하지만 성능 향상이 되는 딥러닝 기법이다. 대표적인 BOS 기법은 다음과 같다.

  1. **receptive field의 증가** : 

     (ex) SPP([Spatial Pyramid Pooling](https://yeomko.tistory.com/14)), ASPP([Atrous Spatial Pyramid Pooling](https://gaussian37.github.io/vision-segmentation-aspp/)), RFB([Receptive Field Block](https://seongkyun.github.io/papers/2019/04/17/rfb_net/))

  2. **feature 통합** : 

     (ex) skip-connection, hyper-column, Bi-FPN

  3. **activation functions** 최적의 선택 : 

     (ex) ReLU, Mish



- 3.1. Selection of architecture : 

  저자는 본 절에서 입력 네트워크의 resolution,  conv layer의 수, parameter 수, 그리고 output의 수들 사이의 최적의 밸런스를 찾아내는것을 목표로 연구가 진행되었다고 말한다. 

  하지만 classification에 최적화된 기법들이 detector에서는 썩 좋은 성능을 내지 못하는 경우가 생긴다고 하는데, 

  1. 입력 네트워크가 큰 경우(higher resolution) : 여러개의 작은 물체를 검출하는데 문제가 생김
  2. 더 깊은 layer : layer가 깊어지면 feature size가 작아지므로, 보다 큰 receptive field를 위해서는 입력 이미지의 크기가 커야 함
  3. 더 많은 파라미터 : 한 이미지에서 다양한 크기의 여러 객체를 검출하기 위해서는 다수의 파라미터가 필요함

  단순히 이론적으로 생각해본다면, 더 큰 receptive field와 더 많은 파라미터를 갖는 네트워크가 백본으로 선택되어야 한다. 

  다음 표는 CSPResNeXt50, CSPDarknet53, EfficientNet-B3을 백본으로 사용할 경우의 파라미터 스펙이다.

  ![](./yolo/param.png)

  표에 의하면 receptive field size, parameter 수, output layer 크기들의 밸런스를 고려하면 CSPDarknet53이 가장 효과적이라고 한다.

  또한 백본뒤에 추가적으로 SPP 모듈을 더하고, 넥 구간에는 YOLOv3의 FPN과는 달리 PANet(Path-aggregation neck)을 사용하였으며, 헤드는 YOLOv3와 동일하게 구성하였다.

  

- 3.2. Selection of BOF and BOS : 

  YOLOv4에서는 다음과 같은 BOF를 사용하였다.

  1. **CutMix** : 데이터 증강법의 일종으로, 한 이미지에 2개의 class를 넣은것이 특징이다. 

     ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_cutmix.png)

     

  2. **Mosaic** Data Augmentation : CutMix와 마찬가지로 데이터 증강법의 일종으로, 한 이미지에 4개의 class를 넣은것이 특징이다.

     ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_mosaic.png)

     

     > 번외로 MixUp이라는 기법도 존재하는데, 이는 여러 이미지를 겹쳐서 입력으로 사용하는 방식이다.
     >
     > ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_mixup.png)

     

  3. **DropBlock** Regularization : DropOut과 같은 regularization 기법의 일종으로, 랜덤한 비율로 drop시키는 DropOut과 달리 feature의 일정 범위를 drop 한다.

     ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_dropblock_concept.png)

     알고리즘 구현 방법은 다음과 같으며, 이 [링크](https://arxiv.org/pdf/1810.12890.pdf)를 통해 자세한 논문을 읽어볼 수 있다.

     > DropBlock 논문 정리 참고자료 : https://norman3.github.io/papers/docs/dropblock.html

     ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_dropblock.png)

     

  4. Class **label Smoothing** : 

     라벨링의 새로운 방법으로, 기존의 라벨링은 정답이면 1, 아니면 0과 같은 딱 떨어지는 정수로 표현했지만 label smoothing 기법을 사용하면 label을 0.2, 0.8과 같은 확률로 표시할 수 있다. 

     이 기법을 통해 사람이 labeling을 한 데이터셋에서 발생할 수 있는 misslabeling에 대응할 수 있다. 결과적으로 calibration 및 regularization 효과를 얻게되어 overfitting문제를 방지할 수 있다고 한다.

     > tensorflow에서는 label smoothing에 대한 implementation을 제공한다. 아래와 같이 softmax_cross_entropy의 파라미터로 label_smoothing 값을 전달할 수 있다.
     >
     > ```python
     > tf.losses.softmax_cross_entropy(
     > 	onehot_labels,
     > 	logits,
     > 	weights=1.0,
     > 	label_smoothing=0,
     > 	scope=None,
     > 	loss_collection=tf.GraphKeys.LOSSES,
     >     reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
     > )
     > ```
     >
     > 만약 label_smoothing이 0이 아니면(클 수록 smoothing 증가), 다음과 같은 공식으로 smoothing이 진행된다.
     >
     > new_onehot_labels = onehot_labels * (1 - label_smoothing) + label_smoothing / num_classes
     >
     > 단 0 < label_smoothing < 1

  

  YOLOv4에서는 다음과 같은 BOS를 사용하였다.

  1. Mish Activation : 

     활성화 함수의 최신 기법중 하나로, 수식과 그래프는 다음과 같다.
     $$
     \\ f(x) = x \times tanh(ln(1+e^x))\\ 
     $$
     ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_mish.png)

     이 함수의 특징으로는

     - 범위가 [-0.31, $$\infty$$) 이다

     - 약간의 음수를 허용하기 때문에 ReLU보다 gradient의 흐름이 smooth하다

     - 그래프가 무한대로 뻗어나가기 때문에 0에 가까운 gradient로 인한 포화(Saturation)문제가 방지된다

       > Saturation : 0에 가까운 기울기로 인해 훈련 속도가 급격히 느려지는 현상

     - 함수의 형태가 아래로 볼록한 형태이기 때문에 strong regulariation이 발생하고, 이에 따라 overfitting 문제가 방지된다.

     - 참고자료 : https://eehoeskrap.tistory.com/440

     

  2. CSPNet(Cross-Stage Partial connections) : 

     기존 CNN 연산의 효율을 증가시키기 위해 개발된 기법으로, 학습할때 중복으로 사용되는 gradient정보를 버림으로써 연산량이 줄어들도록 하여 성능을 높였다고 한다.

     이 기법을 제시한 논문의 저자는 DenseNet을 사용하며 예시를 들었다. DenseNet의 매커니즘은 다음 그림과 같이 이전 layer의 정보를 그 하위 layer에 계속 연결지어주는 방식이다.

     ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_densnet.png)

     논문의 저자는 이 네트워크에서 역전파가 일어나며 가중치가 업데이트 되는 과정을 언급하며 다음과 같이 수많은 중복되는 gradient가 재사용 된다고 주장했다. 이들은 결국 중복된 gradient를 반복적으로 학습할 뿐이라고 말하며 이들을 제거해줄 필요성을 강조했다.

     ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_grad.png)

     따라서 이러한 문제를 다음과 같은 수식으로 중복되는 기울기를 제거했다고 한다.

     ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_CSP.png)

     그림과 함께 수식을 해석해 보면, Base layer의 출력값을 두개로 나누어(split) 두개의 grad-path를 만든다($$X_0 = [x_{0'}, x_{0''}]$$, $$x_{0'}$$ in part1, $$x_{0''}$$ in part2). 그 다음 part1은 바로 이 stage의 끝단과 연결되고, part2는 Dense Block을 지나게 된다. 

     part2에서 Dense Block을 지난 출력인 $$[x_{0''}, x_1,...,x_k]$$는 Transition layer를 통과한 출력값인 $$x_T$$는 윗단의 $$x_{0'}$$과 concatenate되어 또다른 Transition layer를 통과하고, 그 결과로 $$x_U$$를 얻게 된다.

     결과적으로 $$W_T'$$와 $$W_T''$$를 구성하는 gradient를 보면 서로 중복되지 않은것을이 사용된 것을 볼 수 있다.

     이때 Transition layer는 DenseNet에서 사용된것과 동일한데, 이 layer의 역할은 feature map의 사이즈와 개수를 줄이는 역할을 담당한다.

     DenseNet과 CSPnet의 차이점은 다음 그림을 통해 눈으로 확인할 수 있다.

     ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_dense_vs_csp.png)

     > 관련 논문 : https://arxiv.org/pdf/1911.11929.pdf
     
     

  yolov4에서 사용한 BOF(for backbone)는 다음과 같다.

  1. CIoU-Loss(Complete iou loss) : 

     CIoU를 설명하기 전에 GIoU(Generalized Intersection over Union)에 대해 먼저 알아보자.

     GIoU는 다음과 같은 공식으로 표현할 수 있다.
     $$
     \\
     GIoU = {|A \cap B| \over |A \cup B|} - {C - |A \cup B| \over C}
     \\
     $$
     여기서 C는 A박스와 B박스를 동시에 감싸는 가장 작은 박스이다.

     ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_C.png)

     일반적으로 IoU값은 두개의 박스가 겹치는 구간이 없는 경우에 결과값이 0이 되므로 학습에 영향을 미치지 못하지만, GIoU는 겹치는 구간이 없더라도 C를 통해 구해낸 결과를 통해 학습에 영향을 미칠 수 있다.

     distance관점에서 볼때, IoU와 마찬가지로 다음과 같은 값을 사용한다.
     $$
     \\
     \mathcal{L}_{GIoU} = 1-GIoU
     \\
     $$
     

     다음은 CIoU에 대해 알아보자. CIoU를 처음 제시한 논문([Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/pdf/1911.08287.pdf))의 저자는 두 개의 박스를 사용하여 겹치는 구간의 비율을 알아내는 IoU의 개념에 두 박스들의 중앙점의 거리정보를 추가한 DIoU를 먼저 제시했다.

     일반적으로 IoU Loss는 다음과 같은 공식으로 얻어낼 수 있는데,
     $$
     \\
     \mathcal{L}_{IoU} = 1-IoU + \mathcal{R}(A, B)
     \\
     $$
     여기서 뒤에 쓰여진 R은 예측된 박스 A와 타겟박스 B에 대한 penalty term이라고 말하며 적절한 R을 설계한 결과가 DIoU와 CIoU라고 한다. CIoU를 설명하는 과정에 DIoU의 개념이 함께 들어가 있으니 CIoU에 대해 알아보도록 하자.

     CIoU는 다음과 같은 공식으로 얻어낸다.
     $$
     \\
     CIoU = S(A, B)+D(A,B)+V(A,B)
     \\
     $$
     여기서 각각 S, D, V는 다음과 같은 의미를 갖는다.

     - S : Overlap Area
       $$
       \\
       S = 1-IoU
       \\
       $$
       즉 일반적인 IoU Loss이다.

     

     - D : Normalized central point distance

       각 박스의 중앙점 사이의 거리에 대해 normalization을 적용한 결과 이다. 이를 통해 어떠한 scale에서도 동일한 값을 얻어낼 수 있다.

       ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_norm_dist.png)

       

       D의 normalized값은 다음과 같은 공식으로 구할 수 있다.

       ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_norm_dist2.png)
       $$
       \\
       D = {(Euclidean\ distance(A, B))^2 \over c^2}
       \\
       $$
       여기서 구해진 D값을 위에서 언급한 penalty term으로 두면 그 결과가 DIoU가 된다.
       $$
       \\
       \mathcal{L}_{DIoU} = 1-IoU+D
       \\
       $$
       
  - V : Aspect Ratio
    
    Aspect Ratio는 쉽게 말해 가로 세로 비율이다. D에서 박스들의 중앙점의 거리를 구한 후 각각 박스의 가로 세로 비율을 구해 그 차이를 이용하여 박스의 모양에 대한 정보를 맞추는것 같다.
    
    ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_norm_dist3.png)
    
    단 여기서 계수 알파는 trade-off 파라미터로, IoU값에 대한 조건에 따라 위와같이 정의된다.
    
  
  논문의 저자는 위의 방법을 통해 구해낸 CIoU와 GIoU의 결과를 비교하면 아래의 그림과 같다고 주장한다.
  
  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_ciou.png)

  
  
2. CmBN(Cross-mini Batch Normalization)
  
   CmBN을 이해하기 위해 먼저 BN(Batch Normalization)의 정확한 의미와 사용 목적에 대해 알아본다.
  
   최근 딥러닝에는 대부분 GPU연산을 사용하고 있으며, GPU를 효율적으로 사용할 수 있도록 병렬 연산을 하기 위해 32~256 크기를 갖는 mini batch를 사용한다. 이는 train 데이터를 mini batch 단위로 쪼개서 연산을 한다는 의미이다. 
  
   위와 같은 방식으로 학습을 하는 경우 현재 layer의 입력은 모든 이전 layer의 파라미터 변화에 영향을 받게 되고, 이것이 신경망이 깊어질수록 반복되면 신경망이 깊어짐에 따라 이전 layer에서의 작은 파라미터 변화가 깊은 망을 거치며 증폭되어 뒷단에 큰 영향을 끼치는 현상이 발생한다. 
  
   이처럼 학습하는 도중에 이전 layer의 파라미터 변화로 인해 현재 layer의 입력 분포가 바뀌는 현상을 **Co-Variate Shift**라고 한다.
  
   Co-Variate Shift를 줄이는 대표적인 방법중 하나가 바로 BN([Batch Normalization: Accelerating Deep Network Training b y Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf))이며 학습시 BN 방법은 다음과 같다.
  
   ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_bn.png)
  
     마지막 단계에서 scale과 shift연산을 위해 $\mathcal{\gamma}$와 $\mathcal{\beta}$가 추가되었다. 이들은 학습을 통해 얻어지는 파라미터이며, 다음과 같이 설정하면 이들을 통해 정규화된 부분을 원래대로 되돌리는 identity mapping도 가능하다.
     $$
     \\
     \mathcal{\gamma}^{(k)} = \sqrt{Var[x^{(k)}]}
     \\
     \mathcal{\beta}^{(k)} = E[x^{(k)}]
     \\
     $$
   
  
   BN은 다음 그림과 같이 non-linear 활성화 함수 앞쪽에 배치되고, 신경망에 포함되기 때문에 backpropagation을 통해 학습이 가능하며, 아래 그림과 같은 chain rule이 적용된다.
  
   ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_bn2.png)
  
   본론으로 돌아와서 CmBN에 대해 알아보면, 기본적으로 BN과 마찬가지로 평균과 분산을 사용하는것은 같으나 타겟이 되는 layer 이전의 특정 개수의 layer에서의 평균들과 분산들을 이용한다는 점이 다르다.
  
   새로운 평균과 분산값은 다음과 같이 계산이 된다.
  
   ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_cbn.png)
  
   이렇게 구해진 새로운 평균과 분산을 이용하여 Normalization을 진행하는 것이 정통적인 **CBN**(Cross Batch Normalization) 방법이다. 
  
   ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_cbn2.png)
  
   
  
   YOLOv4의 저자는 CBN을 수정한 자신들만의 CmBN을 사용하였다고 말하며 그 구조는 다음과 같다. 하지만 왜 CBN을 수정할 필요가 있었는지에 대한 언급은 나와있지 않다.
  
   ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_cmbn.png)
  
detector를 위한 BOS로는 다음을 사용하였다.
  
1. SPP-block([Spatial Pyramid Pooling](https://arxiv.org/pdf/1406.4729.pdf))
  
   기존의 CNN은 input 이미지로 고정된 크기의 이미지가 필요했다(CNN의 convolution layer에서는 convolution연산을 진행하며 원하는 크기의 feature map을 생성할 수 있지만 FC layer는 고정된 크기의 입력이 필요하다. 따라서 고정된 이미지의 크기가 필요하다). 따라서 해상도가 다른 이미지를 같은 크기의 이미지로 만들때 crop, 혹은 warp같은 방식을 사용했다.
  
   ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_crop-ex.png)
  
   하지만 이들을 사용하면 다음과 같은 이미지 원본 데이터가 손상되는 문제점이 발생한다.
  
     1. crop을 사용한 경우 crop된 region은 해당 물체의 전체를 포함하지 않는다(요다의 머리가 잘린것처럼).
   2. warp를 사용한 경우 물체의 기하학적 왜곡이 발생한다(요다가 찌부러진것 처럼).
  
   하지만 SPP layer를 사용하면 고정된 이미지만을 사용할 필요가 없어져 이미지 크기에 대해 자유로워질 수 있다. 이 논문의 저자는 다음과 같은 단계로 네트워크를 구성하여 일반적인 CNN과의 차이를 보여준다.
  
   ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_spp.png)
  
   SPP layer는 다음 그림과 같은 구조로 이루어져 있다. SPP layer는 multi-level spatial bins를 사용하여 fixed-size 이미지 이슈에서 벗어날 수 있었다고 한다.
  
   ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolov4/yolov4_spp2.png)
  
   > spatial bins란 위 그림에서 보이는 파란, 초록, 회색 네모칸 하나 이다.
  
   이때의 bins의 개수는 고정되어있기 때문에 spatial bins 하나의 크기는 이미지 크기에 비례한다. 각각의 spatial bins에서 max-pooling을 통해 값을 구해내고, SPP의 최종 output은 **kM** 차원을 갖는다.
  
     - k = 마지막 conv layer의 필터 수
   - M = bins 수
  
   이로써 SPP의 최종 결과는 FC layer의 입력값이 된다.
  
     > 입력 이미지가 어떤 크기이더라도, conv layer를 통과하며 feature map은 고정이 되고, 그 결과에 고정된 필터수(bins 수)만큼 max-pooling을 진행하여 최종적으로 모두 같은 차원의 벡터값을 얻어낼 수 있다.
     
2. SAM-block(Spatial Attention Module ; [CBAM : Convolutional Block Attention Module](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)) : 
  
   Attention 기법(중요한 feature에 더욱 집중하고, 덜 중요한 feature는 억누르는 것)을 사용하여 channel과 spatial 두개의 차원에 대한 모듈을 구현한다. 결과적으로 중요한 특징은 증폭되고, 중요하지 않은 특징은 억제되는 효과를 얻어낸다.
  
   ![](C:\Users\RILAB_JONSUFF\Desktop\Jonsuff\jonnote\images\yolov4\yolov4_cbam1.png)
  
   위의 그림은 CBAM을 그림으로 표현한 것이다. 그림에서 볼 수 있듯이, channel attention module에서는 1D형태의 맵을, spatial attention module에서는 2D형태의 맵을 구성한다. 
  
     입력 feature map F는
     $$
     \\
     \bold{F} \in \mathbb{R}^{C \times H \times W}
     \\
     $$
     와 같고, 이를 다음과 같이 두 가지 차원(channel, spatial)의 형태로 만드는 것이다
     $$
     \\
     \bold{M_c} \in \mathbb{R}^{C \times 1 \times 1}, 
     \\
     \bold{M_s} \in \mathbb{R}^{1 \times H \times W}
     \\
     $$
     전체적인 attention 연산과정은 다음과 같이 표시한다.
     $$
     \\
     \bold{F'} = \bold{M_c(F) \otimes F},
     \\
     \bold{F''} = \bold{M_s(F') \otimes F'}
     \\
     $$
   여기서 $$\otimes$$ 연산자는 element-wise 곱연산이다. 곱연산 과정에서 channel 차원에 존재하지 않는 H, W에 대한 값들은 C에 대한 값들이 자동으로 broadcasting되고, 반대로 spatial 차원에 존재하지 않는 C에 대한 값들은 H, W에 대한 값들이 broadcasting된다.
  
   - channel attention module : 
  
     channel module에서는 feature map의 각각 채널이 feature detector로 작용한다. 즉 입력 이미지에서 의미있는 부분에 집중하는 것이다.
  
     ![](C:\Users\RILAB_JONSUFF\Desktop\Jonsuff\jonnote\images\yolov4\yolov4_cbam3.png)
  
     위의 그림처럼, 입력 feature의 채널부분을 각각 max pooling, average pooling하고, 그 출력을 한 개의 hidden layer를 갖는 MLP(Multi-layer perceptron) network에 통과시킨 후 이 둘을 element-wise 합연산을 통해 결과를 얻는다.
  
       이는 다음과 같은 수식으로 정리할 수 있다.
       $$
       \\
       \bold{M_c(F)} = \sigma(MLP(AvgPool(\bold{F})) + MLP(MaxPool(\bold{F})))\\
       = \sigma(\bold{W_1}(\bold{W_0}(\bold{F^c_{avg}})) + \bold{W_1}(\bold{W_0}(\bold{F^c_{max}}))),
       \\
       \bold{W_0} \in \mathbb{R}^{C/r \times C}, \bold{W_1} \in \mathbb{R}^{C \times C/r}
       \\
     $$
  
       > r = reduction ratio : 파라미터를 줄이기 위한 축소 비율
       >
       > $$W_0$$ 뒤에는 ReLU가 붙는다
       >
     > 본 논문에서는 사용하지 않았다.
  
   - spatial attention module : 
  
     channel module에서 '어떤' feature가 중요한가에 대해 집중했다면, spatial module에서는 '어디에' 중요한 feature가 있는가에 집중한다.
  
     ![](C:\Users\RILAB_JONSUFF\Desktop\Jonsuff\jonnote\images\yolov4\yolov4_cbam4.png)
  
       위의 그림과 같이 channel에 대해 max pooling, average pooling을 진행하고 그 결과값을 concatenate한다. 그 다음은 아래의 수식처럼 convolution 연산을 통해 2D spatial attention map을 만든다.
       $$
       \\
       \bold{M_s(F)} = \sigma(f^{7 \times 7}([AvgPool(\bold{F});MaxPool(\bold{F})]))
       \\
       =\sigma(f^{7 \times 7}([\bold{F^s_{avg}};\bold{F^s_{max}}]))
       \\
     $$
  
     > $$f^{7 \times 7}$$ : convolution operation ;filter size(7)
  
     2D spatial attention map의 결과는 다음과 같이 highlighting informative regions로 나타낼 수 있다.
  
     ![](C:\Users\RILAB_JONSUFF\Desktop\Jonsuff\jonnote\images\yolov4\yolov4_cbam5.png)
  
     
  
   CBAM 논문에서는 2D 데이터가 element-wise로 연산이 진행되는데, yolov4논문에서는 이것을 다음과 같이 point-wise 연산으로 바꾸어 진행하였다고 한다.
  
   ![](C:\Users\RILAB_JONSUFF\Desktop\Jonsuff\jonnote\images\yolov4\yolov4_cbam2.png)
  
   > max pooling, average pooling으로 feature를 축소하지 않고 convolution 연산을 진행한다.
  
3. PAN(Path Aggregation Network) block ([Path Aggregation Network for Instance Segmentation](https://arxiv.org/pdf/1803.01534.pdf)) : 
  
   PANet은 하단 layer의 feature와 상단 layer 사이의 information path를 단축시켜 원활한 information flow를 만들겠다는 의도로 만들어졌다. PANet은 다음과 같은 단계로 구성되어 있다.
  
   ![](C:\Users\RILAB_JONSUFF\Desktop\Jonsuff\jonnote\images\yolov4\yolov4_cbam6.png)
  
   1. (a) FPN :
  
      FPN은 conv layer가 깊어질수록 최하단 layer에서 최상단 layer까지 도달하는데 지나는 conv layer가 많아지기 때문에 back propagation이 원활하게 진행되지 않는다는 단점이 있다.
  
   2. (b) Bottom-up path augmentation : 
  
      따라서 PANet 논문의 저자는 간단한 conv layer로 이루어진 새로운 $$N_n$$피라미드를 구성하여 최하단 layer가 최상단까지 도달하는 shortcut을 만들어주는 Bottom-up path augmentation을 제시했다.
  
      새로운 $$N_n$$레이어들은 resolution은 각각 다르지만 동일한 필터개수(256개)를 가지며, $$N_{n-1}$$을 stride 2의 3x3 convolution 연산으로 다운샘플링한 결과를 FPN의 $$P_n$$과 concatenate하여 $$N_n$$을 구현한다.
  
   3. (c) Adaptive feature pooling : 
  
      각각의 $$N$$마다 관심 영역을 추출하여 모든 $$N_n$$레이어들의 ROI를 elemental-wise로 pooling한다. 이때 논문에서 제시한 것은 max pooling과 sum pooling이고, 각각의 $$N$$마다 관심 영역을 추출하는 것은 Mask R-CNN에서 사용한 ROIAlign 기법을 사용하여 진행한다.
  
      > ROIAlign : 단순히ROI를 설정했을때 feature의 ROI와 output의 ROI pixel이 정확하게 일치하지 않는것을 보정해주는 기법(feature map은 최종 output과 resolution 차이가 발생하기 때문에 이러한 경우가 발생)
  
   4. (d) Box Branch : 
  
      단순히 box로 구분할때 pooling의 결과를 fully connected layer로 연결해 예측한다.
  
   5. (e) fully connected fusion : 
  
      segmentation을 위한 단계이다. pooling의 결과를 conv layer와 fully connected layer로 분기시킨 뒤 다시 융합한다. 저자는 conv filter가 spatial 정보를 담고있고, FC layer는 feature에 대한 정보량이 풍부하기 때문에 이들을 융합하면 더 정확한 예측이 가능하다고 주장했다.
  
      >  PANet은 segmentation에서도 좋은 성능을 냈다.
  
   PANet 논문에서는 (c)단계에서 모든 $$N_n$$레이어들의 ROI를 elemental-wise로 sum pooling(혹은 max pooling)했지만, yolov4에서는 이들을 concatenate하여 사용했다고 한다.
  
     ![](C:\Users\RILAB_JONSUFF\Desktop\Jonsuff\jonnote\images\yolov4\yolov4_cbam7.png)



### YOLOv4

YOLOv4는 다음과 같은 모델구조를 갖는다.

- backbone : CSPDarknet53
- Neck : SPP, PAN
- Head : YOLOv3

또한 위에서 설명한 여러 기법들을 보기 쉽게 나열하면

1. Bag of Freebies *for backbone* : 
   - CutMix and Mosaic augmentation
   - DropBlock regularization
   - Class label smoothing
2. Bag of Specials *for backbone* : 
   - Mish activagtion
   - Cross-stage partial connections
3. Bag of Freebies *for detector* : 
   - CIoU loss
   - CmBN
   - DropBlock regularization
   - Mosaic augmentation
4. Bag of Specials *for detector* : 
   - Mish activation
   - SPP-block
   - SAM-block
   - PAN path-aggregation block
   - DIoU-NMS



#### Result

다양한 기법들이 추가된 만큼 실험도 다양한 단계로 진행되었다.

- Influence of different features on Classifier training : 

  CSPResNeXt-50에 BoF와 Mish를 추가하여 사용한 결과를 보면 이들이 결과에 영향을 얼마나 주는지 알 수 있다. (수치는 분류 정확도를 뜻한다.)

  ![](./yolo/cspresnext_aug.png)

  

  다음은 CSPDarknet-53에 BoF와 Mish를 추가한 결과이다.

  ![](./yolo/data_aug_table.png)

  

- Influence of different features on Detector training :

  다음 표는 디텍터에 다양한 BoF를 사용한 결과를 보여준다.

  ![](./yolo/detector_train.png)

  - S : Elimination of grid sensitivity

    YOLOv3에서 박스의 좌표를 구할때 아래와 같은 식을 사용하는데, 이때 뒤에 붙는 c는 각 그리드마다의 위치이기 때문에 언제나 상수가 들어간다. 
    $$
    \\
    b_x=\sigma(t_x)+c_x\\
    b_y=\sigma(t_y)+c_y
    \\
    $$
    이 경우 t에 크기가 상당히 큰 수가 들어와야 b가 c값이나 c+1값에 가까워질 수 있고, 만약 큰수가 들어오지 않는다면 대부분 c+0.5수준으로 b가 결정될 것이다. 이렇게 되면 해당 grid의 중앙에만 물체가 있다고 판단이 될 것이고, 그 결과 정확히 객체를 감싸는 박스가 아닌 살짝 어긋난 박스가 예측될 수 있다.

    YOLOv4에서는 $$\sigma(t_x)$$부분에 1.0 이상의 값을 곱하여 값을 키워줌으로써 보다 fit한 박스를 예측할 수 있게 한다.

  - M : Mosaic data augmentation

    4장의 이미지를 Mosaic기법으로 사용한다.

  - IT : IoU threshold

    다양한 앵커들 중에서 IoU > IoU threshold인 gt만을사용한다.

  - GA : Generic algorithm

    최적의 하이퍼파라미터값들을 전체 학습과정의 전반부 10%에서 결정되게 학습하여 사용한다.

  - LS : Class label smoothing

  - CBN : CmBN

  - CA : Cosine annealing scheduler

     학습과정에서 learning rate을 가변적으로 학습한다.

  - DM : Dynamic mini-batch size

    작은 resolution을 학습할때 자동적으로 mini-batch 사이즈를 증가시킨다.

  - OA : Optimized Anchors

    512 x 512 사이즈에 최적화된 앵커 크기를 사용한다.

  - GIoU, CIoU, DIoU, MSE :

    bounding box regression에 다양한 loss algorithm을 사용한다.

  

- Influence of different backbones and pre-trained weightings on Detector training : 

  다음 표는 다양한 네트워크를 backbone으로 사용한 결과이다.

  ![](./yolo/backbone.png)

- Influence of different mini-batch size on Detector training : 

  다음 표는 mini-batch 사이즈를 달리했을 때의 결과이다.

  ![](./yolo/batchsize.png)

- result : 

  다음은 다른 모델과 비교한 결과이다.

  ![](./yolo/resulttable.png)



### Conclusions

- YOLOv4는 다양한 최신 기법이 사용된 real time object detector 이다. 
- 논문의 저자는 다양한 BoF, BoS기법을 backbone과 detector에 적용하여 효과가 있었음을 증명했다.
- 일반적인 성능의 GPU로도 좋은 성능을 낼 수 있다.





