

# BRIEF, BRISK, ORB 

컴퓨터 비전에서의 feature descriptor는 많은 영역에서 핵심적인 역할을 한다. 많은 데이터를 제한적인 자원에서 빠르고 정확하게 처리하기 위해서는 이러한 descriptor의 계산과 매칭을 빠르게 진행해야 하고, 메모리또한 효율적으로 사용해야 한다.

- Feature descriptor : 매칭이나 인식과 같은 단계에서 같은 그림인가의 여부를 판단하기 위해 사용함.

  descriptor는 아래와 같은 필수 요구 조건을 만족해야 한다.

  - 분별력(discriminating power)이 높을수록 좋다.

  - 변환에 대한 불변성(invariance)을 가져야 한다.

    (ex) 회전, 축소등 변환에 대하여 특징이 변하지 않는다.

  - descriptor의 차원이 적을수록 좋다.

    매칭은 descriptor간의 거리(차이)를 계산하여 판단하므로 차원이 높으면 계산 시간이 그에 비례하여 늘어난다. 보통 한 영상에서 발생하는 특징의 개수가 수천에 이르기 때문에 거리 계산에 소모되는 시간을 단축하는 일은 매우 중요하다.

descriptor는 계산 방식에 따라 다양한 종류가 있는데, 그 중 **binary descriptor**는 특징점 주변에 있는 두개의 화소를 비교 쌍으로 삼아 그들의 명암값을 비교하여 0 또는 1의 이진값을 생성하여 사용한다. 이는 XOR 비트연산자로 서로를 비교할 수 있는 방식이기 때문에 낮은 CPU성능과 메모리 환경에서 빠르고 정확한 연산을 가능하게 한다.

![](.\descriptor\binary_descriptor.png)

이번에 소개할 **BRIEF, BRISK, ORB**는 binary descriptor의 대표적인 조사 패턴이다.



## BRIEF

|      |    BRIEF : Binary Robust Independent Elementary Features     |
| ---- | :----------------------------------------------------------: |
| 저자 | Michael Calonder, Vincent Lepetit, Christoph Strecha, and Pascal Fua |
| 출판 |                          eccv 2010                           |



BRIEF는 위에서 언급했듯이 binary descriptor의 조사 패턴중 하나로, 이미지 전체에서 부분 이미지 패치를 설정하고 그 내부에서 짝지어질 수 있는 픽셀 쌍중 일부만을 sampling하여 비교하는 방식으로 진행된다. 

> "수많은 픽셀 쌍중 일부만을 선택해서 비교해도 해당 이미지 패치 분류가 효과적으로 진행될 것이다"라는 가정은 *Randomized classification trees*나 *Naive Bayesian classifier*등을 이용한 기존 연구에서 검증되었다.



### Basic Method

BRIEF에서 패치 $p$ 내부의 두 픽셀을 비교하는 테스트 함수는 다음과 같다.
$$
\tau(p; x, y) := \begin{cases}
1 &\ \text{if }p(x) \lt p(y) \\
0 &\ \text{otherwise}
\end{cases}
$$

> $\tau$ : test 
>
> p : patch of size S x S

여기서 $p(x)$는 패치 내 $x = (u, v)^T$ 위치의 픽셀 값을 의미하고, 패치는 원본 이미지에 smoothing을 적용한 pixel intensity 이다. 

패치 내의 두 위치 $(x, y)$ 쌍을 겹치지 않도록 $n_d$개를 골라 다음과 같이 구성하여 $n_d$의 비트 길이를 갖는 BRIEF descriptor를 계산할 수 있다.
$$
f_{n_d}(p) := \sum_{1 \leq i \leq n_d} 2^{i-1} \tau(p;x_i,y_i)
$$
논문에서는 $n_d$의 크기를 128, 256, 512 세가지를 사용하여 좋은 결과가 도출되었다고 한다. 결과표에는 사용된 비트 수에 따라 $k = n_d / 8$로 계산하여 각 버전을 BRIEF-k로 표시했다.



### Smoothing & Spatial Arrangement

#### Smoothing

앞서 언급한 테스트 함수는 픽셀단위로 비교가 진행되기 때문에 노이즈에 상당히 민감하다. 따라서 사전에 이미지 패치에 대해 스무딩을 진행한다. 

논문의 저자는 가우시안 커널의 표준편차를 0~3까지 변화시켜 실험했지만 성능 차이가 크지 않았고, 결과적으로 2를 사용하였다고 한다. 또한 커널의 크기는 9 x 9를 사용했다.



#### Spatial Arrangement

S x S 크기의 패치 내에서 $(x, y)$쌍을 어떻게 선택하느냐에 따라 비트 벡터가 달라진다. 본 논문에서는 

다음과 같은 5가지 방법을 사용했다.

1. $(X, Y) \sim \text{i.i.d. Uniform}(-\frac{S}{2}, \frac{S}{2}) $

2. $(X, Y) \sim \text{i.i.d. Gaussian}(0, \frac{S^2}{25})$

   ${s \over 2} = {5 \over 2}\sigma \Longleftrightarrow \sigma^2 = {1 \over 25}S^2$ 

3. $X \sim \text{i.i.d. Gaussian}(0, \frac{S^2}{25}), Y \sim \text{i.i.d. Gaussian}(x, \frac{S^2}{100}) $

4. $(x_i, y_i)$ 극 좌표계(polar grid)에서 랜덤 샘플링되어 x-y 좌표계로 양자화된 좌표

5. $x_i= (0, 0)^T$이고 $y_i$는 극좌표계에서 가능한 모든 좌표

![](.\descriptor\spatial_arrangement.png)

실험 결과 5번이 가장 성능이 좋지 않았고, 나머지들 중 2번이 근소하게 좋은 결과를 보여서 2번을 사용했다고 한다. 



### Result

다음과 같은 6가지 이미지 특성에 대해 실험을 진행하였다.

![](.\descriptor\test1.png)

![](.\descriptor\test2.png)

![](.\descriptor\result.png)

- 한계 : BRIEF는 같은 크기의 가우시안에서 비교 쌍을 생성하기 때문에 스케일 변환과 회전 변환에 대처하지 못한다.



## BRISK

|      |      BRISK: Binary Robust Invariant Scalable Keypoints       |
| ---- | :----------------------------------------------------------: |
| 저자 | Stefan Leutenegger, Margarita Chli and Roland Y. Siegwart Autonomous Systems Lab, ETH Zurich |
| 출판 | IEEE 2011, git : http://www.asl.ethz.ch/people/lestefan/personal/BRISK |

BRISK는 스케일이 변하거나 회전변환이 생기는 경우에도 keypoint를 찾아낼 수 있는 알고리즘이다. 이는 크게 detection과 description step으로 나뉘며, 코너를 찾아내는데 AGAST 알고리즘을 사용하여 계산시간을 줄였다.

> AGAST : Adaptive and Generic Corner Detection Based on the Accelerated Segment Test



### Basic Method

#### Detection

corner detection은 다음과 같은 순서로 진행된다.

![](.\descriptor\brisk_scale.png)

1. 원본 이미지 $c_0$을 절반 크기로 downsampling한 결과를 $c_1$으로 삼고, 그것을 $c_n$까지 반복한다.

   $c_0$과 $c_1$간의 크기차이가 너무 큰것을 염려해 intra-octave인 $d_0$을 그 사이에 배치한다. 이는 $c_0$를 3/4 크기로 downsampling한 결과이다.

   위를 반복하여 점차 scale을 키운다. 일반적으로 이 octave층은 4층으로 구성하여 4가지 scale을 사용하는데, intra-octave까지 포함하면 총 8개의 층으로 구성되는 것이다.

   scale factor t는 다음과 같이 계산할 수 있다.
   $$
   \\octave:t(c_i)=2^i \\ intra-octave:t(d_i)=2^i *1.5\\
   $$
   
2. AGAST 알고리즘 중 하나인 FAST 9-16 알고리즘을 모든 octave / intra-octave 층에 적용하여 다양한 scale에서 corner를 찾아낸다.

   FAST 9-16 알고리즘의 동작 원리는 아래의 그림을 통해 설명할 수 있다.

   ![](.\descriptor\agast.png)

   1) 중심점 p를 기준으로 반지름이 3인 원을 그린다.

   2) 원에 걸쳐진(빨간 박스로 표현된) 픽셀중 중심점보다 어둡거나 밝은 픽셀이 연속으로 9개 이상 발생하면 corner로 판별한다.

   3) non-maximal suppression을 적용하여 corner로 판별되는 값들 중 확률이 높은 것들만 골라낸다.

   - non-maximal suppression 적용 방법 : 

     corner라고 인식된 픽셀들을 아래 식과 같이 maximum condition을 진행하여 score *s*값을 얻어낸다.
     $$
     \\FAST\mathbf{s} = max(\sum_{x \in brighter} |x-p|-t, \sum_{x \in darker}|p-x|-t)\\
     $$

3. ![](.\descriptor\brisk.png)

   위와 같이 FAST score를 i+1, i, i-1의 연속된 세개 층에서 얻어냈을 때 가운데 i값이 위 아래층보다 크다면 이를 corner로 인식한다.

   위의 조건이 만족한 경우 세로축이 $log_2(t)$이고 가로축이 FAST score인 그래프에 점들을 찍어 2차원 포물선 그래프로 연결지어준 후 이때의 최대값을 scale층에 매칭하면 해당 keypoint가 가진 true scale층을 얻을 수 있다.



#### Keypoint Description

corner detection에서 scale에 대해 알고리즘을 진행했다면, description에서는 Sampling Pattern and Rotation Estimation을 통해 keypoint들이 회전 변환에 불변성을 가지도록 한다.

- Sampling Pattern and Rotation Estimation : 

  BRISK descriptor는 keypoint 주번의 픽셀을 샘플링하기 위해 pattern을 사용했다.

  아래 그림과 같이 N=60인 pattern에서 파란 원이 샘플링 위치이고, 빨간 원은 pattern에서 한 point의 intensity를 샘플링할 때 각각의 원 안에 있는 점들 사이의 거리에 비례하여 가우시안 스무딩의 표준편차값을 반지름으로 한 원이다.

  ![](.\descriptor\sampling.png)

  이미지의 특정 keypoint k에 대해 pattern을 얻어내기 위해 $N *(N-1)/2$개의 샘플링 쌍중 하나 $(p_i, p_j)$가 있다고 하자. 각각의 샘플링된 점에서의 intensity가 $I(p_i, \sigma_i)$, $I(p_j, \sigma_j)$일때 이들은 local gradient $g(p_i, p_j)$를 구하기 위해 사용된다.
  $$
  \\g(\mathbf{p}_i, \mathbf{p}_j) = (\mathbf{p}_j - \mathbf{p}_i) \cdot {I(\mathbf{p}_j,\sigma_j) - I(\mathbf{p}_i, \sigma_i) \over \lVert \mathbf{p}_j - \mathbf{p}_i \rVert^2}\\
  $$
  모든 샘플링 쌍을 $\mathcal{A}$로 고려하면 아래와 같고,
  $$
  \\
  \mathcal{A} = {(\mathbf{p}_i, \mathbf{p}_j) \in \mathbb{R}^2 \times \mathbb{R}^2 | i<N \and j<i \and i, j \in N}\\
  $$
  short-distance를 $\mathcal{S}$, long-distance를 $\mathcal{L}$이라고 구분하면 
  $$
  \\
  \mathcal{S} = {(\mathbf{p}_i, \mathbf{p}_j) \in \mathcal{A} | \lVert \mathbf{p}_j - \mathbf{p}_i \rVert < \delta_{max}} \subseteq \mathcal{A} \\
  \mathcal{L} = {(\mathbf{p}_i, \mathbf{p}_j) \in \mathcal{A} | \lVert \mathbf{p}_j - \mathbf{p}_i \rVert > \delta_{min}} \subseteq \mathcal{A}\\
  $$

  > $\delta_{max}$ : 9.75$t$
  >
  > $\delta_{min}$ : 13.67$t$

  위와 같이 정리할 수 있다.

  $\mathcal{L}$을 이용하여 keypoint k의 전체적인 pattern orientation을 예측할 수 있다.
  $$
  \\
  \mathbf{g} = {g_x \choose g_y} = {1 \over L} \cdot \sum_{(\mathbf{p}_i, \mathbf{p}_j) \in \mathcal{L}} \mathbf{g}(\mathbf{p}_i, \mathbf{p}_j)\\
  $$



#### Building the Descriptor

BRISK는 $\alpha = arctan2(g_y, g_x)$만큼 회전된 샘플링 pattern에 적용할 수 있다.

비트 벡터 descriptor $d_k$는 $\mathcal{S}$에 속하는 모든 샘플 $\mathbf{p}_i ^\alpha, \mathbf{p}_j ^ \alpha \in \mathcal{S}$의 short-distance intensity comparison을 통해 계산할 수 있다.
$$
\\b = \begin{cases}
1, & I(\mathbf{p}_j ^\alpha, \sigma_j) > I(\mathbf{p}_i ^\alpha, \sigma_i) \\
0, & \mbox{otherwise}
\end{cases}\\
\forall (\mathbf{p}_i ^\alpha, \mathbf{p}_j ^\alpha) \in \mathcal{S}\\
$$
위의 예시와 같이 N=60 points인 샘플링 pattern과 threshold를 가지면 길이 512의 bit-string을 갖게 되고, 두 개의 BRISK descriptor를 매칭하기 위해서는 Hamming distance 연산(XOR)을 사용하면 된다.





## ORB

|      |       ORB: an efficient alternative to SIFT or SURF       |
| ---- | :-------------------------------------------------------: |
| 저자 | Ethan Rublee, Vincent Rabaud, Kurt Konolige, Gary Bradski |
| 출판 |                         IEEE 2011                         |

BRIEF와 BRISK에 사용된 FAST keypoint detector가 합쳐진 형태로, 속도가 굉장히 빠른 descriptor이다.



#### oFAST : FAST Detector with Orientation

FAST는 BRISK에서도 사용된 실시간으로 keypoint를 찾아내는 알고리즘이다. 이 알고리즘은 각각의 scale에 대해 단계적으로 적용되면 scale에 따른 feature들을 찾아낼 수 있는 반면 방향에 대한 detection은 지원하지 않는다. 하지만 ORB논문의 저자는 Intensity Centroid 방법을 사용해 방향성을 얻어내고자 했다. 이때 FAST-9(원의 반지름이 9)를 사용했더니 성능이 좋게 나왔다고 한다.

- Intensity Centroid : 

  이미지 패치의 moment m을 다음과 같이 정의하고,
  $$
  \\m_{pq} = \sum_{x, y} x^p y^q I(x,y)\\
  $$
  이 moment들을 통해 centroid를 구하면 다음과 같다.
  $$
  \\C = 	\left( {m_{10} \over m_{00}}, {m_{01} \over m_{00}} \right)\\
  $$
  중심점 O에서 centroid C로 향하는 벡터 $\overline{OC}$를 만들 수 있고, 이 벡터의 방향에 근거하여 orientation을 다음과 같이 정리할 수 있다.
  $$
  \\
  \theta = atan2(m_{01}, m_{10})
  \\
  $$

  > atan2는 조건에 따라 4사분면에 대한 arctan값이다.

  회전 변환의 invariance를 향상시키기 위해 x와 y의 영역을 원형의 형태로 유지하도록 하였고, 이때의 반지름 r은 $|C| \sim 0$이 되도록 하는 $[-r,r]$의 범위에서 결정되었다.



#### rBRIEF : BRIEF descriptor with Rotation

기존의 BRIEF 알고리즘은 시점, 조명, 블러에 강한 모습을 보였지만, 이미지가 조금이라도 회전되면 성능이 확연하게 저하되는 단점이 있었다. ORB논문의 저자는 회전 변환에도 영향을 크게 받지 않는 BRIEF 알고리즘을 rBRIEF라는 이름으로 언급하고 적용하였다.

- r-BRIEF : 

  n개의 binary test의 위치를 $(\mathbf{x}_i, \mathbf{y}_i)$라고 하면, 다음과 같이 $\mathbf{S}$행렬을 정의한다.
  $$
  \\
  \mathbf{S} = {\mathbf{x}_1, ..., \mathbf{x}_n \choose \mathbf{y}_1, ..., \mathbf{y}_n}\\
  $$
  앞에서 구한 patch orientation $\theta$에 해당되는 rotation matrix $\mathbf{R}_\theta$를 사용하여 $\mathbf{S}$의 "steered" 버전인 $\mathbf{S}_\theta$는 
  $$
  \mathbf{S}_\theta = \mathbf{R}_\theta \mathbf{S}
  $$
  위와 같이 정의한다. 

  이때 $\theta = 12^\circ$로 고정하여 항상 12도만큼 회전해 있다고 가정한 후 BRIEF 알고리즘을 적용한다.



## Conclusion

BRIEF, BRISK, ORB에 대해 알아보았다. 이들은 모두 binary descriptor이지만 각각 해당되는 조건이 달랐다. 이를 표로 정리해 보면 다음과 같다.

|       | scale invariance | rotation invariance | feature vector bits |
| :---: | :--------------: | :-----------------: | :-----------------: |
| BRIEF |      **X**       |        **X**        |      256 bits       |
|  ORB  |      **X**       |        **O**        |      512 bits       |
| BRISK |      **O**       |        **O**        |      512 bits       |

binary descriptor는 계산 속도가 SIFT보다 수십배 빠르기 때문에 SLAM과 같이 제한적인 환경에 실시간성이 중요한 과제에 효율적이다.