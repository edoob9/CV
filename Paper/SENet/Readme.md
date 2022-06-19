### SENet이란?

#### SENet은 GoogLeNet, ResNet에 이어 2017 ILSVRC에서 1등한 CNN 구조이다.

#### 논문을 쓰게된 목적은 다음과 같다.
CNN interleaves a series of convolution layers, down-sampling, non-linear activation to produce image representation.→ spatial correlations.
Q. What about channel-wise correlation? → channel dependencies are implicitly embedded !
즉, Spatial info에 집중하는 다른 연구와는 다르게 channel-wise feature recalibration을 착안! 쉽게 말하면, ‘채널별로도 가중치가 존재하지않을까?‘ 이런 idea에 기반된 모델이다.

#### 논문에서 제시한 Figure 1.을 보면, 과정은 다음과 같다.
(SE block 과정)
 ⇒ input: weight, height, channel(C) 각각 GAP를 하면, 1x1xC라는 결과값이 나온다.(Produce a channel descriptor by aggregating features map across their spatial dimension [HxW])
feature map size:[HxWxC] → (GAP) → [1x1xC]
⇒ excitation 과정에서 각각 가중치가 학습(반영)되어있다.(Produce a collection of per-channel modulation weights)
feature map size: [1x1xC] → (two)fully connected layers around the non-linearity → [1x1xC] 
“중요한 채널에 대해서는 가중치를 높게 설정하고, 비교적 덜 중요한 채널에는 가중치를 낮춘다.”

##### 실험결과에 대해 정리된 내용을 보자. 
논문에서 SE block을 ResNet-50에 적용한 SE-ResNet-50 네트워크를 이용하여 계산 복잡도를 측정했다. 기존 ResNet-50이 ~3.86 GFLOPs가 필요했다면, SE-ResNet-50은 ~3.87 GFLOP이 필요하다. 하지만, SE-ResNet-50는 ResNet-50(7,48%)와 비교해서 설명하자면, validation error를 6.62% 기록했다. 이 결과는 ResNet-101(6.52%) 결과 만큼 근사한 수치를 가진다고 설명했다.

FLOPS는 딥러닝에서의 FLOPS는 단위 시간이 아닌 절대적인 연산량(곱하기, 더하기 등)의 횟수를 지칭한다.
- 내적(dot product)
    y = w[0]*x[0] + w[1]*x[1] + … + w[n-1]*x[n-1]
    곱허가 n번, 더하기 n-1번 하니 총 2n-1 FLOPS이다.
