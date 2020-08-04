---
layout: post
title:  "YOLOv3"
date:   2020-08-03 00:58:13
categories: Deep_Learning
---



# YOLOv3

> YOLOv3: An Incremental Improvement | [arXiv' 18] - Joseph Redmon, Ali Farhadi
>
> Joseph Redmon - University of Washington - Type Theory, Computer Vision, Machine Learning
>
> **YOLO**,  **Xnor-net**,  etc...
>
> more info $$\rightarrow$$ [click here](https://pjreddie.com/)



## YOLO

- One-stage detector

- Grid cell에서 물체를 검출

  ![](https://raw.githubusercontent.com/Jonsuff/MLstudy/master/images/YOLO_figure2.png)





## 모델 구조

### Backbone

- YOLO는 darknet이라는 backbone 네트워크를 통해 feature map을 추출한다.

  darknet은 YOLOv1부터 발전하여 YOLOv3에서는 darknet-53을 사용한다.

- darknet-53의 구조 : 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolo/darknet-53.png)



### 피라미드 구조를 이용한 yolo network

- YOLOv3는 세 가지 scale에 따른 feature map을 추출하고, 검출할 물체의 크기에 맞는 feature를 사용하여 학습을 진행한다.

- 세 가지 scale에 대한 feature map은 피라미드 구조를 통해 만들어 낸다. 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/images/yolo/feature-pyramid.png)

  > Feature Pyramid Networks for Object Detection - Tsung-Yi Lin, Piotr Dollár, ...

  backbone의 layer27, layer44와 마지막 output을 각각 route1, route2, route3으로 지정한다.

  물체의 크기별로 large_box, medium_box, small_box로 feature를 추출하는데, feature map의 사이즈가 작을수록 실제 이미지에서 feature의 grid cell 하나가 차지하는 픽셀이 크기 때문에 피라미드의 최상단, 즉 route3 자체가 large scale에 대한 feature이고, 이를 통해 large_box에 대한 prediction을 구할 수 있다.

  다음은 route3을 upsampling 하여 크기를 route2와 맞게 한 뒤 이 둘을 concatenate하여 medium scale에 대한 feature를 만들고, 이를 통해 medium_box에 대한 prediction을 구할 수 있다.

  마지막으로 medium scale에 대한 feature를 upsampling하여 크기를 route1와 맞게 한 뒤 위와 마찬가지로 둘을 concatenate하여 small scale에 대한 feature를 만들고, 이를 통해 small_box에 대한 prediction을 구한다.

  YOLOv3에서 input image의 크기는 (416 x 416 x 3)을 사용하며, 세 가지 scale의 비율은 다음과 같다.

  ```
  scale : large
  ratio : 1/32
  grid  : 13 x 13
  
  scale : medium
  ratio : 1/16
  grid  : 26 x 26
  
  scale : small
  ratio : 1/8
  grid  : 52 x 52
  ```



### Anchor box

- YOLOv3는 anchor box를사용하여 scale별로 물체의 크기를 나눈다.

- 데이터를 준비하는 과정에서  scale별로 데이터를 나눠야 하며, 일반적으로 사용되는 anchor 값은 다음과 같다.

  ```
  anchor = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
  ```

  이를 [9, 2] 형태로 reshape하고, 정답 box와 iou를 계산한다.

  만약 정답 박스의 height와 width값이 작다면, 작은 anchor와 iou가 가장 높을것이다. 다시 말해  물체가 작을수록 anchor의 앞순서부터 세 가지씩 사용한다.

  ```
  ex)
  small anchor = [[10, 13], [16,30], [33, 23]]
  ```



### Decoding

- conv layer를 통과한 값을 box데이터로 decoding 하는 작업을 진행한다.

- x, y좌표에 대해서는 sigmoid를 사용하여 각각의 grid cell값을 더해주고, width, height에 대해서는 해당 값을 지수함수의 계수로 사용하여 scale에 맞는 anchor에 곱해준다. 
  $$
  \\
  b_{(x, y)} = \sigma (t_{(x, y)}) + c_{(x. y)} \\
  b_{(w, h)} = p_{(w, h)} e^{t_{(w, h)}} \\
  $$

  ```python
  box_yx = feature(..., 0:2)
  box_hw = feature(..., 2:4)
  box_yx = tf.sigmoid(box_yx)
  box_hw = tf.exp(box_hw) * anchors
  ```
  
  
  
- confidence score(objectness)는 sigmoid를 사용한다.

  ```python
  objectness = feature(..., 4)
  objectness = tf.sigmoid(objectness)
  ```

  

- class probability 부분은 softmax를 사용하지 않는다고 한다. 논문 저자의 말을 인용하면

  "We do not use a softmax as we have found it is unnecessary for good performance, instead we simply use independent logistic classifiers. During training we use binary cross-entropy loss for the class predictions."

  라고 하는데, 해석해보면 보다 좋은 성능을 위해 softmax를 사용하지 않고, 단순히 각각 class label마다 logistic classifier를 적용하기 위해 sigmoid activation과 binary cross-entropy를 사용한다고 한다.

  softmax를 사용하게 되면, 보다 높은 복잡성을 지닌 dataset을 사용할때 겹치는 label에 대한 문제를 해결하기 어렵다고 하는데...(Women과 Person처럼) 

  > 하지만 coco dataset처럼 복잡성이 높지 않은 dataset에서는 상관없지 않을까..?

  ```python
  class_probs = feature(..., 5:)
  # class_probs.get_shape() = (batch, height, width, anchor, 80)
  class_probs = tf.nn.softmax(class_probs, axis=-1)
  ```

  

### 최종 output

- backbone과 yolo network를 통과한 최종 output의 shape은 다음과 같다.

  ```
  feature : (batch, height, width, anchor, (y,x,h,w) + confidence + num_class_label)
  
  for feature_l : height, width = 13
      feature_m : height, width = 26
      feature_s : height, width = 52
  
  number of class label in coco dataset : 80
  
  then)
  feature_l = (batch, 13, 13, 3, 85)
  feature_m = (batch, 26, 26, 3, 85)
  feature_s = (batch, 52, 52, 3, 85)
  ```

  여기서 anchor 차원이 3인 이유는 각각의 grid마다 anchor box가 3개씩 나오게 하기 위함이다.



## Training

### Training step

- 전체적인 학습 과정은 다음과 같다.

  1. 이미지 데이터를 model에 통과시켜 prediction을 얻어낸다.
  2. 얻어낸 prediction에 대해 loss를 계산한다.
  3. 계산된 loss를 gradient tape를 통해 자동미분 연산을 하여 trainable weights를 구한다.
  4. 구해진 trainable weights를 적용시켜 학습시킨다.

  ```python
  model = YoloNet((cfg.BATCH_SIZE, cfg.SIZE_H, cfg.SIZE_W, 3)).get_model()
  with tf.GradientTape() as tape:
  	preds = model(features["image"])
  	losses = self.compute_loss(features, preds)
  grads = tape.gradient(losses, model.trainable_weights)
  self.optimizer.apply_gradients(zip(grads, model.trainable_weights))
  ```

  

### Loss

- YOLOv3는 다음과 같은 세 가지 종류의 loss를 사용한다.

  1. bounding box loss
  2. objectness(confidence) loss
  3. category(class label) loss

- **bounding box loss** :

  bounding box loss는 해당 grid의 박스 좌표에 대한 loss이다.

  정답 좌표와 prediction 좌표 사이의 제곱 오차 합 결과를 사용한다. 

  > height와 width에 대한 loss는 제곱 오차에 루트를 씌우면 더 좋다는 의견이 있다.

  ```python
  # predictions.get_shape() = (batch, height, width, anchor, num_class + 5)
  pred_yx = predictions[..., 0:2]
  pred_hw = predictions[..., 2:4]
  grtr_yx = features[..., 0:2]
  grtr_hw = features[..., 2:4]
  yx_loss = tf.square(grtr_yx - pred_yx)
  hw_loss = tf.square(grtr_hw - pred_hw)
  bbox_loss = tf.concat([yx_loss, hw_loss], axis=-1)
  ```

  

- **objectness loss** : 

  objectness loss는 해당 grid에 물체가 있는지, 없는지를 판단하는 score에 대한 loss이다.

  objectness 부분에는 decoding에서 sigmoid를 통과한 결과값이 들어있기 때문에 binary cross entropy를 사용한다.

  ```python
  pred_obj = predictions[..., 4]
  grtr_obj = features[..., 4]
  pred_obj = pred_obj[..., tf.newaxis]
  grtr_obj = grtr_obj[..., tf.newaxis]
  obj_loss = tf.losses.binary_crossentropy(grtr_obj, pred_obj)
  ```

  

- **category loss** :

  category loss는 해당 박스가 어떤 class label에 속해있는지에 대한 loss 이다. 

  위에 설명했듯이 논문에서는 sigmoid와 binary cross entropy를 사용했지만, coco dataset을 사용할때는 softmax와 categorical cross entropy를 사용해도 될것 같다...

  ```python
  pred_category = predictions[..., 5:]
  grtr_category = features[..., 5]
  category_loss = tf.losses.sparse_categorical_crossentropy(grtr_category,pred_category)
  ```

  원래 categorical cross entropy는 정답의 class label이 원핫 인코딩이 되어있어야 작동한다. 하지만 tensorflow 2.0버전에서 지원하는 tf.losses.sparse_categorical_crossentropy()는 원핫 인코딩이 되어있지 않고 단순히 정수로 label id 하나를 입력값으로 사용해도 똑같은 동작을 수행한다.



## Validation

### NMS(Non-Maximum Suppression)

- YOLOv3는 grid cell 하나에서 3가지의 박스를 예측한다. 따라서 한 물체에 대해서 3개의 겹치는 박스가 결과로 출력될 수 있는데, 이때 같은 물체를 예측한 겹치는 박스를 없애주는 작업이 필요하다.

  Non-Maximum Suppression은 예측된 박스들 중 동일한 class로 분류된 박스들을 confidence score 순으로 정렬하고, 가장 score가 높은 상자와 나머지 상자들을 비교하여 일정 iou이상인 박스들이 동일한 물체를 예측했다고 판단하여 제거한다.

- tensorflow 2.0버전부터 tf.image.combined_non_max_suppression()을 사용하여 nms 결과를 쉽게 얻어낼 수 있다.

  ```python
  tf.image.combined_non_max_suppression(
      boxes, scores, max_output_size_per_class, max_total_size, iou_threshold=0.5,
      score_threshold=float('-inf'), pad_per_class=False, clip_boxes=True, name=None
  )
  
  
  # boxes : 4-D tensor of shape [batch, num_boxes, q, 4]
  # 만약 q가 1이면 박스가 모든 class에 대해 사용되고, q가 class수와 같다면 특정 class에 대해서만 사용된다.
  
  # scores : 3-D tensor of shape [batch, num_boxes, num_classes]
  # 각 박스가 갖는 하나의 score. score = confidence * class_probs로 입력한다.
  
  # max_output_size_per_class : A scalar integer tensor
  # class 하나당 최대로 가질 수 있는 박스의 개수
  
  # max_total_size : A scalar integer tensor
  # 전체 박스들의 최대 개수
  
  # iou_threshold : A float
  # 특정 값 이상의 iou를 갖는 박스들을 없애주는 threshold 값
  
  # score_threshold : A float
  # 특정 값 이하의 score를 갖는 박스들을 없애주는 threshold 값
  
  # pad_per_class : if false => nms가 적용된 박스들이 max_total_size의 수에 의해 제한됨
  #					 true  => nms가 적용된 박스의 수가 max_size_per_class * num_classes
  
  # clip_boxes : if true  => 박스 좌표가 [0, 1]사이로 제한됨
  #				  false => 박스 좌표가 제한되지 않음
  ```

  