

![](C:\Users\RILAB_JONSUFF\Desktop\Jonsuff\2020_summer_study\2020.12.21\image\3d object detection list.png)



# PointNet

| 제목 | PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation |
| ---- | :----------------------------------------------------------: |
| 저자 |              Charles R. Qi, Hao Su, Kaichun Mo               |
| 출판 |                          CVPR 2017                           |

> https://github.com/charlesq34/pointnet : Pointnet github



## Introduction

### 2차원 데이터와 3차원 데이터

일반적으로 객체에서 3차원 data를 사용할때는 이미지와는 다르게 geometric 정보가 추가된다. 이는 x, y, z축의 정보로 이루어진 point들을 data로 사용하는 것이다.

![](.\image\1.png)



2차원 이미지는 행렬로 data가 얻어지고, (x, y) pixel grid에 대하여 RGB 3차원의 정보를 얻어온다. 이 때 pixel grid는 행렬 (x, y)에서의 위치에 대응한다. 

3차원 데이터는 이미지와는 다르게 depth를 고려해야 하므로 3차원 데이터는 point들로 정보를 얻어온다. 이때 point 데이터는 행렬로 이루어지지 않고 각 point별로 (x, y, z)축에 대한 정보를 가지고 있다. 여기에 RGB값이 더해지면 3차원 데이터의 전체적인 구성은 (x, y, z, r, g, b)와 같이 6차원이 된다.



### 3차원 데이터 랜더링

2차원 이미지는 행렬로 얻어지기 때문에 이를 regular format으로 이루어져 있다고 말한다. 하지만 point들은 행렬이 아니므로 irregular format으로 이루어져 있다. 즉 point들이 행렬과 같이 특정한 규칙에 의해 정보들이 모여있는 것이 아니기 때문에 데이터 내부 원소들의 순서가 아무런 의미가 없고, 이때문에 convolution 연산이 이루어질 수 없다.

위와같은 문제점 때문에 3차원 데이터는 받아온 point정보를 통해 3차원 공간에 랜더링하는 작업을 진행한 후 사용한다. 3차원 데이터의 랜더링 방식은 다음과 같다.

![](.\image\2.png)

하지만 high resolution으로 랜더링된 3차원 데이터는 GPU 메모리와 RAM용량을 매우 많이 잡아먹기 때문에 본 논문에서는 랜더링 작업을 하지 않고 point 데이터를 직접 다루는 방법을 고안했다.



### Point 데이터를 직접 사용하기 위한 성질

본 논문의 저자는 unordered, unstructured 데이터인 point 데이터를 다루기 위해서는 다음 두 가지 성질을 만족해야 한다고 말한다.

1. Permutation invariant : 

   point 들이 unordered 상태로 주어지기 때문에 네트워크는 어떠한 순서로 데이터가 입력되더라도 output이 달라지지 않도록 해야 한다.

   만일 n개의 point가 있다면, 이들이 얻어지는 순서의 경우는 n! 수만큼 존재한다. 따라서 모든 n!의 경우의 수에서 output 결과가 동일하게 네트워크가 동작해야 한다.

   논문의 저자는 이 성질을 만족하기 위해 symmetric function(변수들의 순서가 바뀌어도 결과가 같은 함수)을 사용했으며 그 종류는 max pooling이라고 말한다. 

2. Rigid motion invariant : 

   transform이 진행되어도 rigid body가 유지되는 것을 말한다.



### 본 논문의 특징

3차원 데이터는 irregular form으로 되어있기 때문에 데이터의 구조와 순서가 무분별하다. 따라서 기존 연구들은 이를 질서있는 다른 데이터 형식으로 랜더링을 완료한 후(mesh, voxel) 3d convolution을 진행했지만, 이 경우 여러 하드웨어적 문제가 발생했다(lack of GPU memory).

본 논문은 point 데이터가 permutation invariance를 충족하여 입력으로 사용될 수 있게 만드는 방식을 제시하고 object classification, part segmentation에서 복합적으로 사용할 수 있는 네트워크 구조를 소개한다.





## 네트워크 구조 및 원리

classification network와 segmentation network의 구조와 흐름은 다음과 같다.

![](.\image\3.png)



기본 구조를 가볍게 살펴보면 다음과 같다.

1. classification network : 

   단순히 생각하면 입력으로 들어오는 point들 각각에 대해 MLP를 사용하여 feature를 추출하고, 모든 point의 차원에 대해 max pooling을 진행하여 한 개의 차원으로 줄인다. 이 과정의 결과를 global feature로 정의하고 마지막단에 MLP를 한번 더 진행하여 output score를 뽑아내고, 이를 classification score로 사용한다.

2. segmentation network : 

   classification은 input 전체에 대해서 하나의 output을 계산하는 문제이지만,  segmentation은 각 픽셀별로 어떤 class에 속하는지를 계산해야 하므로 local, global 정보가 모두 필요하다.

   segmentation network구조를 살펴보면 64개의 feature가 생성된 중간 단계의 layer를 local feature로 취급하고, 그 뒤에  global feature를 모두 concatenate 시킨다. 이후 MLP를 통해 class별 score를 추출한다.



### Global feature extraction을 위한 symmentric function

앞서 언급했듯이 point 데이터는 irregular form 데이터이다. 이를 해결하기 위한 방법은 다음과같이 존재한다.

1. 정돈되지 않은 입력 데이터를 규칙에 의해 정렬한다.

2. 입력 데이터를 RNN 학습용 sequence라고 생각하고, 모든 조합에 대해 고려한다.

3. 간단한 symmetric function을 사용하여 각각의 point에 대한 정보들을 집계(통합)한다. 

   symmentry function이란 n개의 입력벡터가 들어올때 n개의 입력 내부에서 순서가 바뀌더라도 출력은 변화가 없는 연산을 말하며, +와 *가 그 예이다.

   > 쉽게 말해서 입력을 벡터가 아닌 숫자로 생각한다면 , 입력 (1,2,3)이나 (3,2,1)에 대해서 +연산을 진행하면 출력은 모두 (6)이므로 이는 symmentric function인 것이다.



첫 번째 방법은 무분별한 차원의 입력을 기준이 되는 차원으로 줄이면서 spatial 관계까지 유지하기가 일반적인 상황에 모두 적용되기는 불가능하고, 두 번째 방법은 주로 문자열 학습에 사용되지만 point데이터는 그 차원의 수가 너무 많기 때문에 모든 조합의 경우의수를 다 고려하기는 불가능하다. 따라서 본 논문은 세 번째 방법의 사용에 초점을 맞추었다.

본 논문은 다음 식에 의해 transform된 data를 입력으로 사용한다.
$$
f(\{x_1, ..., x_n\}) \approx g(h(x_1), ..., h(x_n))
$$

$$
f : 2^{R^N} \to \mathbb{R}, \qquad h : \mathbb{R}^N \to \mathbb{R}^K, \qquad g: \underbrace{\mathbb{R}^K \times \cdots \times \mathbb{R}^K}_{n} \to \mathbb{R}
$$

이때 g가 symmetric function이다.

자세한 설명을 덧붙이면, h는 MLP연산에 의해 얻어내고, g는 단일 변수 함수와 max pooling을 사용한다(이는 실험에서 잘 작동되었다). 

실제로 논문에 사용된 각각의 point별 global feature 추출은 다음 그림과 같이 진행된다.

![출처 : https://ganghee-lee.tistory.com/50](.\image\4.png)
$$
f(x_1, ..., x_n) = \gamma \cdot g(h(x_1), ..., h(x_n))
$$


### Local feature extraction을 위한 T-net

앞에서 언급했듯이 segmentation task를 위해서는 global feature뿐만 아니라 local feature를 얻어내야 한다. 이는 rigid motion invariant조건을 충족하기 위함이고, 본 논문에서는  spatial transformer network(STN)의 아이디어를 이용한 T-net을 소개한다.

> STN paper : https://arxiv.org/pdf/1506.02025.pdf

STN 논문에서는 이미지에 대해 rigid motion invariant를 만족시키기 위해서 transformation matrix를 orthogonal하게 만든다.

![](.\image\5.png)

이 과정을 수행하기 위해 STN에서는 다음과 같이 동작한다.

1. 입력 이미지를 orthogonal하게 만들기 위해 어떠한 transformation이 적용되어야 하는지를 계산한다.
2. 계산된 transformation을 기존 입력 이미지에 곱하여 변형이 일어나지 않은 상태의 output 이미지를 만든다.

본 논문의 T-net은 위의 아이디어를 이용하여 다음과 같은 구조를 만들었다.

![](.\image\6.png)

T-net에서 point 데이터들을 orthogonal한 상태의 공간(canonical space)로 보내기 위해 적용되어야 하는 transformation matrix를 MLP와 max pooling을 통해 얻어내고, 이를 입력 데이터에 행렬연산하여 transform된 이미지를 얻어낸다. 이는 전체적인 네트워크 구조에서 input transform에 해당한다.

전체적인 네트워크 구조에서는 input transform 이외에 feature transform도 존재하는것이 보이는데, 이때 feature transform의 shape은 (64 x 64)가 되어야 하는것을 알 수 있고, 이 커다란 차원의 transformation matrix를 예측하고 optimize하기 쉽지 않다. 따라서 본 논문의 저자는 여기에 아래와 같은 정규화 식을 추가한다.
$$
L_{reg} = ||I - AA^T ||^2_F, \qquad A=transformation
$$
위 식의 의미를 생각해 보면 $AA^T$가 $I$에 가까워질수록 A는 orthogonal matrix가 되어야 하고, transformation matrix가 orthogonal matrix가 되면 rigid motion이 되기 때문에 입력 이미지에 대해 rigid body가 유지되는 것을 유추할 수 있다. 다시 말해 transformation matrix가 rigid motion이 되도록 정규화 식을 추가한 것이다.





## 실험 결과

본 논문은 ModelNet데이터셋을 사용하여 성능테스트를 진행했다.

![](.\image\7.png)



![](.\image\8.png)



## Conclusion

본 논문의 Pointnet은 입력받은 3차원 데이터가 irregular form임에도 불구하고 그 데이터를 그대로 네트워크의 input 데이터로 사용할 방법을 제시했다. 이는 이전 3D segmentation이나 classification 문제에서 사용됐던 voxel-base input data보다 파라미터수가 크게 줄어들었고, GPU 메모리도 절약하면서 성능은 높아진 결과를 도출했다.



> 그림, 자료 출처 : https://ganghee-lee.tistory.com/50