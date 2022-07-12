## 2 stage detection

Many stage라고 적혀있는 방법에서는 object가 있을 법한 위치의 후보(proposals) 들을 뽑아내는 단계, 이후 실제로 object가 있는지를 Classification과 정확한 바운딩 박스를 구하는 Regression을 수행하는 단계가 분리되어 있습니다. 대표적으로는 Faster-RCNN이다.
[2 stage detection lecture - Andrew Ng]

### <R-CNN>
R-CNN에서 image가 먼저 들어와서 region proposal을 진행한다. 

AlexNet을 backbone으로 feature extractor 역할을 시킨다. 그러면 AlexNet의 특징으로 227*227 정사각형 size(fixed image size)로 ****Image Warping 진행한다.(압축된 상태로 들어가면, 이미지 소실을 발생시킬 수 있다.) 그리고 R-CNN에서는 classification을 위해서 svm(특히, `binary svm`)을 사용한다.
  
1000개의 class에 대해서 구별할 수 있게 학습된 AlexNet을 가져왔다. RCNN에서 사용되는 VOC data는 클래스가 20개이다. 그래서 AlexNet에서 수정해서 21개(background 1 + classification 20)개가 추출할 수 있게 transfor learning을 한다.(fine tune for extection) → 그리고 5번째 pooling layer feature을 추출해서 disk에 저장한다. 해당 이유는 논문에서 실험결과로 설명하기로는 mAp가 5번째까지만 크게 증가하지않았고, 6,7,8,…layer가 넘어가면서 특정 data의 specific하게 학습이 진행된 것을 알 수 있다. 
  
For each class, train linear regression model - GT boxes to make up for ‘slightly wrong’ proposals

dx, dy, dw, dh에 대한 학습을 어떻게 진행할 것인가?
  
- predicted box 와 GT box간의 관계식
    - Gx, Gy는 중심점을 평행이동시킨 것이다.
- P와 G의 차이(scale 고려)
    - tx, ty, tw, th관계식을 구하고, 그 gap을 줄이는게 목표이다.(loss function을 최소화할 수 있는 학습해서 파라미터 찾기) 그리고 논문에서 보여지는 L1,L2 norm에 대해서는 소수의 큰값에 의해서 좌우되어서 편향되지않기 위해서(overfitting을 막기 위해서) 진행한다.
    

#### <R-CNN problem>

- Training is slow(84h), a lot of disk space!(slow at test-time)
    - 이미지마다 Selective search 수행하여 2000개 region proposal 2000개에 대해 CNN feature map 생성(need to run full forward pass of CNN for each region proposal)
- detection is slow
- Complex multistage training pipeline
- Ad hoc training objective
    - SVMs and regressors are post-hoc
        - CNN features not updated in response to SVMs and regressors
        

### <Fast R-CNN>

R-CNN의 경우 region proposal을 selective search로 수행한 뒤 약 2,000개에 달하는 후보 이미지 각각에 대해서 convolution 연산을 수행 → 하나의 이미지에서 feature을 반복해서 추출하기 때문에 비효율적이고 느리다는 단점이다. 그래서 Fast R-CNN에서는 후보 영역의 classification과 Bounding Box regression을 위한 feature을 한 번에 추출하여 사용한다. 그리고 R-CNN과의 차이점은  이미지를 Sliding Window 방식으로 잘라내는 것이 아니라 해당 부분을 CNN을 거친 Feature Map에 투영해, Feature Map을 잘라낸다는 것이다.
  
input image → convNet을 통해서 feature map을 그리고, 거기에 ROI projection을 진행한다.(convNet에 의해서 만들어진 feature map에 동일한 비율의 영역을 projection을 한다.) → ROI pooling layer → FC

Region of interest pooling(ROI)

input(Hi-res input image)(3x800x600) → conv & pooling → feature map(CxHxW) ‘selective search에서 영역으로 나온 좌표를 projection을 해서 동일한 위치에 동일한 scaling을 통해서 feature amp에도 그려넣는것이다.’ → projected region into Hxw grid

What is ROI pooling?

RCNN에서 warping이 꼭 필요했지만, Fast R-CNN은 고정된 vector값으로 표현하기 위해서 RoI pooling과정을 통해서 feature map에 적용하면 된다. → **후보 영역에 해당하는 특성을 원하는 크기가 되도록 pooling하여 사용하는 것이다.**

3x16 = 48크기의 vector를 만드는 것이다.

- (8x8x3) → (2x2 grid)
- (12x8x3) → (3x2 grid) // grid의 크기만 달라질 뿐이다.
  
  
### <Faster R-CNN>

selective search를 CPU에서만 작동하니까 느린 거 맞지않나요? → region proposal를 위한 clustring 단계에서 느리다.(Fast R-CNN은 반복되는 CNN 연산을 크게 줄여냈지만 region proposal 알고리즘이 병목) ⇒ GPU 환경에서 이용하려면 어떤 행동을 해야할까?

Image → CNN(feature map) → RPN(Region proposal network) : slide a small window on the feature map

이때 부터 Anchor box라는 개념이 들어가 있다. → 각 object는 이전과 같이 중심점이 있는 cell에 배정되고, **실제 ground truth와 object 사이 가장 높은 IOU를 갖는 grid cell과 Anchor box에 배정( grid cell에서 Anchor box에 대한 object 할당은 IoU로 할 수 있습니다. 인식 범위 내에 object가 있고 두 개의 Anchor box가 있는 경우 IoU가 더 높은 Anchor box에 object를 할당하게 됩니다.)

지금껏 본 Object Detection의 문제점 중 하나는 각각의 grid cell이 오직 하나의 object만 감지할 수 있느냐였습니다. grid cell이 여러 개의 object를 감지하고 싶다면 어떻게 해야 할까요?→ use Anchor Boxes

#### Faster R-CNN은 분할하는 알고리즘 대신 CNN을 사용하여 Fast R-CNN보다 조금 더 빠릅니다. 하지만 R-CNN 알고리즘은 YOLO보다 느리다는 한계점이 있습니다.
