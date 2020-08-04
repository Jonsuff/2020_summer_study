## [논문 리뷰]Single-Shot Refinement Neural Network for Object Detection

### 주저자: Shifeng Zhang

- ##### [Institute of Automation, Chinese Academic of Sciences](http://english.ia.cas.cn/) 박사과정 중
- ##### 최근 CVPR 2020에서 Best paper award 수상



## Abstract

- **SSD** 계열의 one stage model
- two stage model의 장점과 one stage model의 장점을 가지기 위해 만들어진 모델
- 두개의 **inter-connected module**로 이루어진 모델
- **Multi-Scale Testing** 방식을 사용하여 정확도를 향상
- 2018년 당시 **SOTA** 모델의 정확도를 뛰어넘어 **SOTA** 모델이됨



## Model Architecture

![image-20200731133953888](C:\Users\MIlab\AppData\Roaming\Typora\typora-user-images\image-20200731133953888.png)

  Refinedet 모델은 backbone network로 VGG-16과 ResNet-101(ILSVRC CLS-LOC datset으로 pretrained됨)을 사용했고 크게 3가지의 module로 구성된다.

#### inter-connected modules

---

- ##### Anchor refinement module (ARM)

  - negative anchor를 제거해 search space를 줄이게 해주는 모듈
  - anchor의 영역등을 재 조정해서 anchor의 위치를 보정
  - 가장 나은 초기치를 전달하기위한 모듈



- **Object detection module (ODM)**
  - ARM에서 전달된 anchor를 가지고 object detection을 하는 모듈
  - regression과 multi class label을 예측



- **Transfer connection block(TCB)**
  - ARM에서 보내는 특징인 anchor의 위치, 사이즈, 클래스 label 등을 전달해주는 모듈
  - ARM의 특징을 ODM에 맞게 변환해줌
  - 또 다른 기능으로는 정확도 향상을 위해 특징을 고차원으로 만들어주고 데이터들 간에 차원을 맞춰주기 위해 Deconvolution 등을 사용
  - 본 논문에서는 효율적으로 ARM에서의 특징을 전달하도록 만들어졌다고함

![image-20200731161219599](C:\Users\MIlab\AppData\Roaming\Typora\typora-user-images\image-20200731161219599.png)

> To ensure the effectiveness, we design the TCB to transfer the features in the ARM to handle more challenging tasks, i.e., predict accurate object locations, sizes and class labels, in the ODM.

- ##### Cascade Regression

  - 단계적으로 앵커의 위치와 크기를 조정하면서 최적의 regression값을 도출하는 방법



- **Negative Anchor Filtering**
  - ARM 단계에서 기준에 부합하지 않은 Negative anchor를 제거하는 방식
  - ARM 단계에서 Refined 된 Anchor인 경우 ODM에서 regression으로 나온 confidence 값을 이용해 기준 T 보다 크면 다시 한번 제거됨



## Training에 사용된 방법

- **Anchors Design and Matching**

  - 서로 다른 크기의 물체를 detect 하기위해 4개의 서로 크기가 다른 feature layer를 사용(stride 는 8, 16, 32, 64로 서로 다르게 사용)
  - Ground truth와 Anchor를 비교하기 위해 IOU가 아닌 Jaccard overlap(다른 말로는 IOU값 ) 방식을 사용해서 겹침 정도를 계산
  - 겹침 정도 계산결과를 가지고 0.5 이상인 값만 남김

- **Hard Negative Mining**

  - 보통 matching을 하면 positive에 비해 훨씬 많은 negative sample 생기게 되는데 이러면 sample간에 불균형이 커서 제대로 학습이 되지 않음
  - 이러한 문제를 해결하기 위해 negative sample에서 가장 높은 점수를 sample들을 negatvie로 학습시킴으로써 class Imbalance 문제를 완화
  - 논문에서는 positive와 negative의 비율을 1:3 으로 하는게 가장 좋다고 함

  

- ##### LOSS Function

![image-20200731173025219](C:\Users\MIlab\AppData\Roaming\Typora\typora-user-images\image-20200731173025219.png)


$$
l^{*}_{i} = ground\ truth\ class\ label\ of anchor\\
g^{*}_{i} = ground\ truth\ location\ and\ size\ of\ anchor\\
p_{i} = predict\ confidence\\
x_{i} = refined\ cordinates\ of\ anchor\\
c_{i} = predicted\ object\ cla
t_{i} = predicted\ object\ coordinates\ of\ bounding\ box\\
N_{arm}/N_{ODM} =  number\ of\ positive\ anchor\ in\ (ARM/ODM)\\
L_{b} = cross-entropy/log\ loss\ over\ two\ classes\ (object vs. not object)\\
L_{m} = softmax\ loss\ over\ multiple\ classes\ confidences
$$


- ##### Train option& Environment

  - ##### Xavier 초기 가중치 method 사용

  - ##### ILSVRC CLS-LOC로 Pre-trained된 모델 사용

  - ##### SGD optimizer 사용

  - ##### NVIDIA Titan X, Cuda 8.0, cuDNN v6

  

- ##### multi-scale testing

![](https://hoya012.github.io/assets/img/object_detection_sixth/5.PNG)

> 그림 출처: https://hoya012.github.io/assets/img/object_detection_sixth/5.PNG

 Multi scale testing을 이용하면 검출 성능을 높일 수 있다.

 위의 그림을 보면 원본에서는 벌레 한마리를 detect하지 못했지만 Multi-Scale Testing을 이용하여 검출하는 것을 보여주고 있으며 실제로도 전반적인 Recall은 증가한다.

 

## Train result

- #### 	PASCAL VOC Dataset

  ![image-20200731174829699](C:\Users\MIlab\AppData\Roaming\Typora\typora-user-images\image-20200731174829699.png)

RefinDet 뒤에 +가 붙은것은 multi scale기법을 적용한 모델이다. mAP 수치를 본다면 multi scale이 적용된 모델이 가장 mAP값이 높은 것을 볼 수 있다. 또한 **YOLO**보다 FPS값이 부족하지만 mAP의 값이 더 높은 것을 알 수 있다.



- #### MS COCO Data set

![image-20200731174951591](C:\Users\MIlab\AppData\Roaming\Typora\typora-user-images\image-20200731174951591.png)

MS COCO data에서 RefineDet이 대부분 가장 높은 mAP값을 가지는 것을 볼 수 있다.



- #### 주요 기능을 제외한 모델의 mAP

![image-20200731182541668](C:\Users\MIlab\AppData\Roaming\Typora\typora-user-images\image-20200731182541668.png)

위의 표를 보면 RefineDet에서 3가지 기능들이 mAP를 높이는데 많은 도움을 줬다는 것을 알 수 있다.



## Conclusion

- ##### SSD 계열의 모델 구조를 사용하여 모델을 설계

- ##### ARM, ODM module을 사용한 two inter-connected 구조를 사용

- ##### ARM에서 anchor를 수정 및 refine해준 후 ODM에서 regression 및 multi class classification을 수행

- ##### 결과적으로 위의 방식을 사용해 빠른 FPS와 높은 정확도를 가지게됨



감사합니다.



