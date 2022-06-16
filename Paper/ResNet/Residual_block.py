def Residual_Block(x, filter):

    x_skip = x 
    f= filter

    x = Conv2D(f, kernel_size=(3,3), strides=1, padding='same')(x) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(f, kernel_size=(3,3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # projection 진행
    x_skip = Conv2D(f, kernel_size=(1,1),strides=(1,1),padding='same')(x_skip) #1x1 conv을 f만큼 진행해서 projection을 한다.
    x_skip = BatchNormalization()(x_skip)    

    x = add([x, x_skip]) # concat을 처음에 진행했는데 조금 다른 값이 나온거 같다. why?
    x = Activation('relu')(x)    
    
    return x
# 
