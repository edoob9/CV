## GoogLeNet(Going Deeper with Convolutions) - review!
ILSVRC-2014에서 VGG를 이기고 1등을 차지한 모델이다.
# <논문이 발표된 이유- 논문에서 제공>
Neural Network가 깊어지게 설계하고자 한다. 이 말의 의미는  depth, width가 증가한다는 의미는 parameter 수가 증가한다는 의미이다. ⇒ overfitting(’parameter들이 각각 data를 학습한다’), computation load가 될 가능성이 높아진다. 이러한 것들을 sparse한 연결 구조, but dense한 연산을 해보자!라는 취지에서 시작한 논문이자 Model로 해석했다.

# <Model 간략한 설명!>

- GoogLeNet contains 1x1 convolution at the middle of the network.
- Using GAP(global average pooling) instead of using fully connected layers.
- inception module is to have different size/types of convolutions for the same input and stacking(concat) all the output.

논문에서 주장하는 목표를 정리하자면,
- 최대한 파라미터를 줄이면서 네트워크를 깊게 디자인!
- layer가 깊더라도 연결이 sparse한다면, 파라미터 수가 줄어든다. 그리고 오버피팅을 방지하는 효과가 있다. 하지만, 연산은 dense하게 처리하는게 목표
- 딥러닝 계산은 matrix의 곱으로 처리되는데, 이때 matrix의 값이 sparse한다면, 낭비되는 연산이 많아진다. → 이걸 방지하는게 목표이다.

논문에서 제공한 그림을 통해서 보다가 parameter, 총 연산량에 대한 개념을 알고싶어서 Andrew Ng 교수님이 제시하신 내용을 보고 정리하자면, Naive Inception module에서 마지막에 연산 결과들이 concat이 된 값을 보면, 28x28(크기)x672(차원)에서 논문에서 제시한 모델을 사용하면, 28x28x480으로 줄어드는 것을 확인할 수 있다.
fully connected layer을 사용하면, weight의 갯수가 7*7*1024*1024 =51.3M을 학습해야하지만, GAP를 사용하면, 7*7 이미지 x 1024개 channel이라면, 그냥 weight없이 한 개 이미지마다 채널 수(1024)개의 벡터를 결과로 만드는 것이기 때문에 학습량의 차이가 많이 난다.
