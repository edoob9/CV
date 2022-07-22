Semantic Segmentation의 경우, 각 픽셀을 객체 1과 배경 중에 어떤 클래스에 속하는지 분류하고 있습니다. Instance Segmentation의 경우 객체 1과 배경을 구분해줄 뿐만 아니라 객체 1에 속해있는 각각의 물체들(ex.농부들)끼리도 구분해주고 있는 것을 확인할 수 있습니다.

FCN (2014) : https://arxiv.org/abs/1411.4038

### U-Net (2015) 
[paper](https://arxiv.org/abs/1505.04597) / [code]

U-Net이라 불리는 인코더(다운샘플링)와 디코더(업샘플링)를 포함한 구조는 정교한 픽셀 단위의 segmentation이며, Encoder-decoder 구조 또한 semantic segmentation을 위한 CNN 구조로 자주 활용했다.
- Encoder 부분 : 점진적으로 spatial dimension을 줄여가면서 고차원의 semantic 정보를 convolution filter가 추출
- Decoder 부분 : Encoder에서 spatial dimension 축소로 인해 손실된(spatial) 정보를 점진적으로 복원하여 보다 정교한 boundary segmentation

SegNet (2015) : https://arxiv.org/abs/1511.00561

PSPNet (2016) : https://arxiv.org/abs/1612.01105

DeepLab V3+ (2018) : https://arxiv.org/abs/1802.02611
