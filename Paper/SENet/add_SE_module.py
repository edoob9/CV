# ResNet 기본 모델에서 이 코드를 추가하면 된다.
# 단, Residual_Block에 하나의 코드를 추가해야한다.
# x = SEmodule(x, f2, 16) 추가해줘야한다.


def SEmodule(pre_layer, ch, r):

    x = GlobalAveragePooling2D()(pre_layer)
    x = Dense(int(ch/r), activation='relu')(x)
    x = Dense(ch, activation='sigmoid')(x)
    x = Reshape((1, 1, ch))(x)
    x = multiply([pre_layer,x])
    
    return x
 
