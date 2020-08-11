# EfficientNet: Rethinking Model Scaling for CNN.

#### 저자 : Mingxing Tan , Quoc V. Le 

#### Mingxing Tan

#### 소속

- Google software engineer
- Google Brain 연구팀 소속

#### 주요 논문

- Mnasnet: Platform-aware neural architecture search for mobile
- Searching for mobilenetv3
- Efficientdet: Scalable and efficient object detection



#### Quoc V. Le

#### 소속

- Google software engineer
- Google Brain 연구팀 소속

#### 주요논문

- https://scholar.google.com/citations?user=vfT6-XIAAAAJ&hl=en



## EfficientNet

#### Abstract

- ##### Compound scaling method를 사용해 Depth, height, resolution을 조절하는 새로운 방법을 제안

- ##### 위의 방법을 사용하여 연산량을 획기적으로 줄이고 SOTA를 달성

- ##### 전 Best Conv model과 비교해 8.4배 작고  6.1배의 속도 향상을 가짐

<img src="https://github.com/Jonsuff/2020_summer_study/blob/Dongun/2020.08.10/EfficientNet_%ED%95%9C%EB%8F%99%EC%9A%B4/img/image-20200807204029546.png?raw=true" />



### Model scaling method

![image-20200807204055246](https://github.com/Jonsuff/2020_summer_study/blob/Dongun/2020.08.10/EfficientNet_%ED%95%9C%EB%8F%99%EC%9A%B4/img/image-20200807204055246.png?raw=true)



#### 각  scaling의 특징

- ##### Width scaling

  - filter(channel)의 개수를 조절하는 방법
  - 기존 연구에 따르면 Width를 넓게할 수록 fine-gained feature를 많이 찾을 수 있음

- ##### Depth scaling

  - layer의 깊이를 늘리는 방법
  - 신경망을 깊게 쌓는데 한계가 있음(ResNet-1000과 ResNet-101은 성능이 비슷함)

- ##### Resolution

  - input 이미지의 해상도를 높이는 방법
  - 최신 연구 Gpipe에서는 480X480 해상도의 이미지 사용
  - Object detection 영역에서 600X600 해상도 이미지를 사용하면 좋은 성능을 낸다고함



 위의 모델은 각 Model Scaling 방식을 포현을 보여준다. 기존 Model Scaling을 통해 모델의 정확도를 높이기 위한 연구는 많이 시도되었는데 Figure. 2에서 (b), (c), (d) 와 같이 한 특징을 집중적으로 scaling하는 방식을 사용했다.

 반면 본 논문에서는 기존에 잘 고려되지 않던 Depth, Height, Resolution을 동시에 고려하는 Scaling 방법을 사용한다.

![image-20200807210522460](https://github.com/Jonsuff/2020_summer_study/blob/Dongun/2020.08.10/EfficientNet_%ED%95%9C%EB%8F%99%EC%9A%B4/img/image-20200807210522460.png?raw=true)

 Figure. 3은 본 논문에서 실시한 실험결과를 보여준다. 제일 왼쪽 실험 결과를 보면 Depth를 Scaling함에 따라 정확도의 변화를 볼 수 있고 Width를 늘리면 정확도가 높아지다가 어느 순간부터는 정확도가 포화되는 것을 알 수 있다.

중간(Depth)와 오른쪽(Resolution)도 같은 방식을 사용해 실험하였고 왼쪽 실험과 비슷하게 어느 순간부터는 정확도가 포화되는 것을 알 수 있다.

이와 비슷하게 Width를 고정하고 Depth와 Resolution을 변화시키는 실험을 진행하였다.

<img src="https://github.com/Jonsuff/2020_summer_study/blob/Dongun/2020.08.10/EfficientNet_%ED%95%9C%EB%8F%99%EC%9A%B4/img/image-20200807211654755.png?raw=true" style="zoom:80%;" />



Figure. 4의 실험결과를 보면 조합에 따라 동일한 FLOPS에서 정확도가 크게 차이나는 것을 알 수있다. 위 실험에서는 Resolution을 Scaling하는것이 가장 좋은 효과를 가지는 것을 알 수있다. 또한  Depth, Width, Resolution을 1, 2가지만을 활용한 Scaling보다 3가지를 동시에 Scaling하는것이 가장 좋은 성능을 가질 수 있다는 점을 보여준다.

 

### Model Architecture

<img src="https://github.com/Jonsuff/2020_summer_study/blob/Dongun/2020.08.10/EfficientNet_%ED%95%9C%EB%8F%99%EC%9A%B4/img/image-20200807222215980.png?raw=true" style="zoom:80%;" />

> <img src="https://github.com/Jonsuff/2020_summer_study/blob/Dongun/2020.08.10/EfficientNet_%ED%95%9C%EB%8F%99%EC%9A%B4/img/image-20200811120126759.png?raw=true" style="zoom:60%;" />
>
> 
>
>  MBConv: Moblie Inverted convolution(MobileNet V2 구조 사용)
>
> 여기서 나오는 **ReLU6**는 기존 ReLU에서 0이상의 상한 값을 6으로 제한한 함수
>
> 참조:
>
>  https://seongkyun.github.io/papers/2019/01/09/mobilenetv2/
>
> https://n1094.tistory.com/29
>
> [ReLU6](https://gaussian37.github.io/dl-concept-relu6/)

 실험 진행시에 Baseline Network 자체가 성능이 떨어지면 scaling 조절을 해도 좋은 결과를 내기에 한계가 존재하기 때문에 본 논문에서는 MnasNet(Architect search를 이용해 만들어진 Model)과 거의 동일하게 AutoML을 활용해 모델을 탐색해했고 이 과정을 통해 찾은 모델을 EfficientNet-B0라고 부르며 활용한다.

> Architect search: 사람이 모델을 설계하는것이 아닌 학습을 통해 최적의 모델 구조를 설계하는 방법
>
> 참조:
>
> https://research.sualab.com/review/2018/09/28/nasnet-review.html





### Compound Scaling



​                                                 ![image-20200807214103348](C:\Users\milab\Documents\Study\EfficientNet_Rethinking Model Scaling for CNN\image-20200807214103348.png)
$$
\phi:유저가 \ 가지고\ 있는 resource에\ 따른\ 조정\ 하이퍼파라미터\\
\alpha*\beta*\gamma = 2\ 고정한\ 이유\ 2배\ 4배등으로\ scaling-up하기\ 위함
$$

> Beta와 Gamma가 제곱인 이유는 depth와 같은 경우는 2배가 되면 FLOPS도 2배가 증가하지만 Width나 Resolution같은 경우는 가로 세로가 같이 증가하기 때문에 연산이 4배가 되기 떄문에 제곱이 붙여진다.

 위의 실험을 토대로 Depth/Width/Resolution 3가지를 동시에 Scaling하기 위해 본 논문에서는

 **Compound scaling**을 사용한다.  

 위의 모델에서 파이 값을 1로 고정한 후 Small Grid Search를 이용한 결과 Alpha = 1.2, Beta = 1.1 , Gamma = 1.15 의 비율들을 도출할 수 있었다.



### Model Performance



<img src="C:\Users\milab\Documents\Study\EfficientNet_Rethinking Model Scaling for CNN\image-20200807224902011.png" alt="image-20200807224902011" style="zoom:80%;" />





![image-20200807223720858](C:\Users\milab\Documents\Study\EfficientNet_Rethinking Model Scaling for CNN\image-20200807223720858.png)

Table. 2에서 가장 중요한 정보는 Parameter이다. 기존 모델과 Compound Scaling이 적용된 EfficientNet과 비교해보면 정확도는 더 높지만 Parameter 수가 획기적으로 줄어든 것을 볼 수 있다.

가장 하단부에 보면 가장 최신 모델인 GPipe보다 정확도가 높고 Parameter가 8.4배 정도 차이가 나는 것을 볼 수 있다.

<img src="C:\Users\milab\Documents\Study\EfficientNet_Rethinking Model Scaling for CNN\image-20200807224127387.png" alt="image-20200807224127387" style="zoom:80%;" />

Table. 3은 기존 모델에 Compound Scaling을 적용한 결과를 나타낸 표이다. 표를 보면 Compound Scaling을 적용했을시에 정확도가 더 높아지는 것을 알 수 있다.



![image-20200807224457252](C:\Users\milab\Documents\Study\EfficientNet_Rethinking Model Scaling for CNN\image-20200807224457252.png)

Parameter가 정말 획기적으로 줄어들었다는것을 확인할 수 있다.



### Class Activation Map (CAM)

![image-20200807224607736](C:\Users\milab\Documents\Study\EfficientNet_Rethinking Model Scaling for CNN\image-20200807224607736.png)

 위 Figure 7은 Compound scaling 적용시에 얼마나 feature를 잘 찾아내는가를 **CAM**으로 보여준다.  다른것들과 달리 Compound Scaling을 적용한 모델이 물체의 특징을 더 세밀하게 잘 찾아내는 것을 알 수 있다.

> 빨간색일수록 중요한 특징



<img src="C:\Users\milab\Documents\Study\EfficientNet_Rethinking Model Scaling for CNN\image-20200807224902011.png" alt="image-20200807224902011" style="zoom:80%;" />





### Conclusion

- 기존에는 Model Scaling에서 Depth/Width/Resolution을 동시에 scaling하지 않았는데 본 논문에서는 Compound Scaling method를 통해 Depth/Width/Resolution 3가지를 동시에 Scaling
- Compound Scaling을 적용해 Parameter를 획기적으로 낮추고 높은 정확도를 확보
- Efficient 모델에 Compound Scaling을 적용해  2019년 **SOTA**를 달성 
