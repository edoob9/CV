### 'EfficientNet, Rethinking Model Scaling for Convolutional Neural Networks' review

#### EfficientNet : The most common way is to scale up ConvNets by their depth, width, or image resolution
논문에서 제공한 모델에 대한 (number of parameters - imagenet top-1 accuracy) 관계를 그래프에서 볼 수 있다.

EfficientNet-B0(Efficient 모델 중 가장 가벼운) 모델이 연산량(파라미터 총수)가 적음에도 ResNet-50, Inception-v2보다 성능이 좋은 것을 볼 수 있다. 
그리고 EfficientNet-B7이 제일 복잡한 모델이다. → 성능이 압도적이다.

scaling up Convolution is widely used to achieve better accuracy.
→ ResNet은 depth를 늘려가면서 점점 scaling up을 진행했고, 층이 쌓이면서 점점 accuracy가 증가했다.(리소스에 대한 언급X)
(ResNet-18, ResNet50, ResNet 101…)

#### CNN 모델을 깊게 설계하면(파라미터가 증가) 성능을 올라가지만, 그만큼 리소스(memory, FLOPS :  연산)가 소요(trade off 관계)

- <strong>최적의 깊이(depth), 너비(width), 해상도(resolution) 조합 찾아낸다면, 최소환의 리소스로 뛰어난 성능을 보여줌.</strong>
⇒ 깊이(depth), 너비(width), 해상도(resolution)의 balance를 찾는 기법들 : `compound scaling methods`

<Model Scaling>
논문에서 제시한 그림(f.2)을 통해서
(b) width scaling : kernel의 수를 증가시킨다. 
(c) depth scaling : model become deeper
(d) resolution scaling : input size를 크게 하는 것

입력의 이미지가 H,W,C로 이루어져있다. L : layer 

Network-N(d,w,r)의 accuracy가 최적의 성능을 가질 수 있게! H,W,R을 찾는것이다.

단, `Memory(N) ≤ target_memory` ’네트워크의 메모리가 target memory보다 낮아야한다.’ 같은 방식으로 연산량도 낮아야한다.
- 위의 H,W,r의 세 요소를 각각 나머지 두 요소는 고정시키고, 한 개의 요소만 변화시키면서 accuracy를 확인한다. 세 변수 모두 특정 구간에서부터 `saturation되는 현상`이 발생된다.

<strong>compound scaling method</strong>

알파, 베타, 감마는 small grid search로 결정되는 상수이다. FLOPS는 w, r에 따라 제곱배로 상승하며, 논문에서는 w,r의 제곱을 2로 제한하고, 제한된 범위에서 최적의 h,w,r을 찾았다.
- model : EfficientNet-B0(base model) 사용
그리고 논문에서는 ResNet과 MobileNet에 compound scaling을 적용한 결과를 보면서 성능이 더 좋게 향상된 것을 확인할 수 있다.
