def vgg16_encoder_decoder(input_size = (200,200,1)):
    #################################
    # Encoder
    #################################
    inputs = Input(input_size, name = 'input')

    conv1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name ='conv1_1')(inputs)
    conv1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name ='conv1_2')(conv1)
    pool1 = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool_1')(conv1)

    conv2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name ='conv2_1')(pool1)
    conv2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name ='conv2_2')(conv2)
    pool2 = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool_2')(conv2)

    conv3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name ='conv3_1')(pool2)
    conv3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name ='conv3_2')(conv3)
    conv3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name ='conv3_3')(conv3)
    pool3 = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool_3')(conv3)

    conv4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name ='conv4_1')(pool3)
    conv4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name ='conv4_2')(conv4)
    conv4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name ='conv4_3')(conv4)
    pool4 = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool_4')(conv4)

    conv5 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name ='conv5_1')(pool4)
    conv5 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name ='conv5_2')(conv5)
    conv5 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name ='conv5_3')(conv5)
    pool5 = MaxPooling2D(pool_size = (2,2), strides = (2,2), name = 'pool_5')(conv5)

    #################################
    # Decoder
    #################################
    #conv1 = Conv2DTranspose(512, (2, 2), strides = 2, name = 'conv1')(pool5)

    upsp1 = UpSampling2D(size = (2,2), name = 'upsp1')(pool5)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'conv6_1')(upsp1)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'conv6_2')(conv6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'conv6_3')(conv6)

    upsp2 = UpSampling2D(size = (2,2), name = 'upsp2')(conv6)
    conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'conv7_1')(upsp2)
    conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'conv7_2')(conv7)
    conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'conv7_3')(conv7)

    upsp3 = UpSampling2D(size = (2,2), name = 'upsp3')(conv7)
    conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', name = 'conv8_1')(upsp3)
    conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', name = 'conv8_2')(conv8)
    conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', name = 'conv8_3')(conv8)

    upsp4 = UpSampling2D(size = (2,2), name = 'upsp4')(conv8)
    conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', name = 'conv9_1')(upsp4)
    conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', name = 'conv9_2')(conv9)

    upsp5 = UpSampling2D(size = (2,2), name = 'upsp5')(conv9)
    conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', name = 'conv10_1')(upsp5)
    conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', name = 'conv10_2')(conv10)

    conv11 = Conv2D(3, 3, activation = 'relu', padding = 'same', name = 'conv11')(conv10)

    model = Model(inputs = inputs, outputs = conv11, name = 'vgg-16_encoder_decoder')

    return model