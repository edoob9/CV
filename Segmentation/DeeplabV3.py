def DeeplabV3(image_size, num_classes):
  model_input = keras.Input(shape=(image_size, image_size,3))
  resnet50 = keras.applications.ResNet50(weights='imagenet', include_top=False, input_tensor=model_input)

  x = resnet50.get_layer('conv4_block6_2_relu').output
  x = DilatedSpatialPyramidPooling(x)
  input_a = layers.UpSampling2D(size=(image_size//4//x.shape[1],
                                      image_size//4//x.shpae[2]),
                                interpolation='bilinear')(x)

  input_b = resnet50.get_layer('conv2_block3_2_relu').output
  input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

  x = layers.Concatenate(axis=-1)([input_a, input_b])
  x = convolution_block(x)
  x = convolution_block(x)
  x = layers.UpSampling2D(size=(image_size // x.shape[1],
                                image_size // x.shape[2]),
                          interpolation = 'bilinear')(x)
  model_output = layers.Conv2D(num_classes, kernel_size=(1,1), padding='same')
  return keras.Model(inputs= model_input, outputs=model_output)

model = DeeplabV3(image_size = IMAGE_SIZE, num_classes=NUM_CLASS)
model.summary()
