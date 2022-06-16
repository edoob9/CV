### Computer Vision

컴퓨터 비전 관련된 논문 리뷰 및 코드 구현
* 최신 논문 위주로, 많은 인기를 끌고 있는 다양한 딥러닝 논문을 소개합니다.

#### Paper review folder

논문에 대한 정리를 노션에서 주로 하지만, 나중에 소스 코드 관리를 위해서 간략한 설명을 추가했다.
- LeNet
- AlexNet
- GoogLeNet
- VGG
- ResNet
- SENet
- EfficientNet
- GAN
- CycleGAN
- MobileNet-v1.v2.v3
- YOLO

#### Image Recognition (이미지 인식)

#### CV 분야 논문
* Histograms of Oriented Gradients for Human Detection(HOG)
    * [Original Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1467360&tag=1) / Code Practice
* GradientBased Learning Applied to Document Recognition(LeNet)
    * [Original Paper Link]([https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1467360&tag=1](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)) / [Code Practice](https://github.com/edenLee94/CV/blob/main/Paper/LeNet/LeNet.ipynb)
* ImageNet Classification with Deep Convolutional Neural Networks(AlexNet)
    * [Original Paper Link](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) / [Code Practice](https://github.com/edenLee94/CV/blob/main/Paper/AlexNet/AlexNet_pr.ipynb)
* Very Deep Convolution Networks for Large-scale image recognition(VGG)
    * [Original Paper Link](https://arxiv.org/pdf/1409.1556.pdf%20http://arxiv.org/abs/1409.1556.pdf) / [Code Practice](https://github.com/edenLee94/CV/blob/main/Paper/VGG/vgg.py)
* Going deeper with convolutions(GoogLeNet)
    * [Original Paper Link](https://arxiv.org/pdf/1409.4842.pdf) / [Code Practice](https://github.com/edenLee94/CV/blob/main/Paper/GoogLeNet/_GoogLeNet_pr.ipynb)
* Deep Residual Learning for Image Recognition(ResNet)
    * [Original Paper Link](https://arxiv.org/pdf/1512.03385.pdf) / [Code Practice](https://github.com/edenLee94/CV/blob/main/Paper/ResNet/ResNet_50.ipynb)
* EfficientNet, Squeeze-and-Excitation Networks(SENet)
    * [Original Paper Link](https://arxiv.org/pdf/1709.01507v4.pdf) / Code Practice
* Rethinking Model Scaling for Convolutional Neural Networks(EfficientNet)
    * [Original Paper Link](https://arxiv.org/pdf/1905.11946.pdf) / Code Practice
-----
ImageNet(large-scale), MNIST, CIFAR dataset
MNIST나 CIFAR는 idea에 대한 검증 목적으로 사용
- MNIST: 0부터 9까지의 '28 x 28 손글씨 사진'을 모은 데이터셋 (학습용: 60,000개 / 테스트용: 10,000)
- CIFAR-10: 10개의 클래스로 구분된 '32 x 32 사물 사진'을 모은 데이터셋 (학습용: 50,000개 / 테스트용: 10,000개)
- ImageNet : (Amazon Mechanical Turk) 서비스를 이용하여 일일이 사람이 분류한 데이터셋


## 파일 설명
### > Solution_overfitting.ipynb
overfitting을 극복하는 방법으로는 data augmentation, batch normalization, drop out이 있다. 이를 통해서 얼마나 고쳐지는지 확인한다.

### > CNN_cifar10.ipynb
https://tutorials.pytorch.kr/beginner/basics/optimization_tutorial.html

pytorch에서 제공한 tutorials에서 제공하는 설정을 이용해서 신경망 설계 및 최적화 코드를 반복하여 수행하는 train와 테스트 데이터로 모델의 성능을 측정하는 evaluate를 정의!
GPU 설정을 '펭귄브로의 3분 딥러닝, 파이토치맛' 책의 내용을 학습해서 진행하였다.
