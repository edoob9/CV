# AlexNet은 무엇일까?

650,000개의 뉴런을 갖는 신경망은 5개의 컨볼루션 레이어로 구성되며, 그 중 일부는 max-pooling 레이어와 3개의 완전히 연결된 레이어와 최종 softmax로 구성되어있습니다.
To make training faster, we used non-saturating neurons and a very efficient GPU implementation of the convolution operation!

데이터 세트의 구성은 ImageNet은 약 22,000개 범주에 속하는 1,500만개 이상의 고해상도 이미지의 데이터 세트이며, 사람들이 직접 라벨링한 데이터셋으로 알려져있다.
test data에서 37.5% 및 17.0%의 top-1 및 top-5 오류율을 달성했으며 이는 이전의 최신 기술보다 훨씬 우수하다고 설명하고있습니다.(논문이 완료된 시점에서)

## ReLU Nonlinearity
ReLU Nonlinearity를 사용하여 overfitting(vanishing gradient problem)을 방지하고 (gradient가 크게 크게 전달되기 때문에) 학습속도가 훨씬 빨라졌다. 모든 layer에 적용된다. 
→ 대략 6배 빨라짐.

## Local Response Normalization
VGG 기법(논문)에서 반박되어서 사라졌다고 하는 부분이지만, 기본 개념을 알 필요가 있기 때문에 학습을 진행하는 부분이다. → 현재에는 Batch normalization 기법을 사용한다.
activate, deactivate된 값들을 response normalization을 통해서 channel-wise하게 값들을 더해서 위에 제시한 식을 통해서 하나의 feature map을 만든다!

## Overlapping pooling
논문에서는 당시 컴퓨팅 기술의 제한으로 인하여 병렬 연산으로 진행했다.

그리고 overfitting을 줄이기 위해서 data augmentation를 이용해서 랜덤하게 추출하게 이미지 개수를 증대했으며, fully connected layer 2개 층에서 dropout을 이용했다.
