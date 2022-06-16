# ResNet(Deep Residual Learning for Image Recognition) 논문 - review

https://arxiv.org/pdf/1512.03385.pdf

2014년의 GoogLeNet이 22개 층으로 구성된 것에 비해, 152개의 layer(VGG 대비 8배 많은 depth)[**약 7배나 깊어졌다!**] → 훨씬 더 깊은 layer(`Residual block`을 추가해서 가능했다!) 
마이크로소프트에서 제안한 모델이며, ILSVRC 2015 image classfication에서 top-5 test error 3.57%! 우승(인간이 이미지를 인식할 때 발생하는 error rate를 5~10%라고 하는데, 이걸 능가했다.)

![FCB05A52-CD78-40F8-B7E0-A2FDB058A3AD_4_5005_c](https://user-images.githubusercontent.com/47210353/174119326-9d3e4370-900f-49ca-9d9e-282cd2e403b9.jpeg)
