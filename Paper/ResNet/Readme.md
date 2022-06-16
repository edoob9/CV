# ResNet(Deep Residual Learning for Image Recognition) 논문 - review

https://arxiv.org/pdf/1512.03385.pdf

2014년의 GoogLeNet이 22개 층으로 구성된 것에 비해, 152개의 layer(VGG 대비 8배 많은 depth)[**약 7배나 깊어졌다!**] → 훨씬 더 깊은 layer(`Residual block`을 추가해서 가능했다!) 
마이크로소프트에서 제안한 모델이며, ILSVRC 2015 image classfication에서 top-5 test error 3.57%! 우승(인간이 이미지를 인식할 때 발생하는 error rate를 5~10%라고 하는데, 이걸 능가했다.)

## ## <Introduce>

We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping.
  
There exists a solution by construction to the deeper model: the added layers are identity mapping, and the other layers are copied from the learned shallower model.

In our case, the shortcut connections simply perform identity mapping, and their outputs are added to the outputs of the stacked layers (Fig. 2).

Identity shortcut connections add neither extra parameter nor computational complexity.

## <핵심 개념>
  
논문에서 제공한 figure2. 해당 그림은 논문에서 제시한 그림이고, C = F(x) + B의 개념을 설명했다. 그림을 해석하면, 별도의 루트를 하나 만들어서 학습을 하는데, 이 말은 즉, 원본 데이터와 추상화된 데이터 더해서 이걸 새롭게 mapping해주는 것!
Residual Learning building block을 이용해서 정보가 소실되지않게 데이터를 mapping한 것이 주요 개념이다. **ResNet은 F(x) + x를 최소화하는 것을 목적으로 한다!
