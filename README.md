# Computer Vision
컴퓨터 비전 관련된 논문 리뷰 및 코드 구현


-----
## 이미지 분석
ImageNet(large-scale), MNIST, CIFAR dataset
MNIST나 CIFAR는 idea에 대한 검증 목적으로 사용
- MNIST: 0부터 9까지의 '28 x 28 손글씨 사진'을 모은 데이터셋 (학습용: 60,000개 / 테스트용: 10,000)
- CIFAR-10: 10개의 클래스로 구분된 '32 x 32 사물 사진'을 모은 데이터셋 (학습용: 50,000개 / 테스트용: 10,000개)
- ImageNet : (Amazon Mechanical Turk) 서비스를 이용하여 일일이 사람이 분류한 데이터셋


## 파일 설명
### > Solution_overfitting.ipynb
overfitting을 극복하는 방법으로는 data augmentation, batch normalization, drop out이 있다. 이를 통해서 얼마나 고쳐지는지 확인한다.
