from multiprocessing import pool
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, EfficientNetV2B3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, GlobalAveragePooling2D, MaxPooling2D, Dropout

from config import efficientNet_config


def frank_model(cls):
    input_shape  = efficientNet_config['input_shape_B0']
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)) # input_shape=(32, 32, 3)
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.2))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(Flatten())
    # model.add(Dense(16, activation='relu'))
    model.add(Dense(cls, activation='softmax'))
    model.summary() 
    return model


def mobilenetv2_model(cls):
    input_shape  = efficientNet_config['input_shape_B0']
    mobilenetv2 = MobileNetV2(include_top=False,
                              weights='imagenet', 
                              input_shape=input_shape,
                              pooling='avg'
                             )
    
    model = Sequential()
    model.add(mobilenetv2)
    model.add(Dense(cls, activation='softmax'))   # cls = class
    # 輸出網絡模型參數
    model.summary() 
    # dot_img_file = 'EfficientNetV2B0.png'
    # tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

    # 卷基層參與訓練
    mobilenetv2.trainable = True    # 解凍與否
    
    return model


def efficientNetV2B0_model(cls):
    input_shape  = efficientNet_config['input_shape_B0']
    efnv2b0 = EfficientNetV2B0(include_top=False,
                               weights='imagenet', 
                               input_shape=input_shape,
                               pooling='avg'
                            )
    
    model = Sequential()
    model.add(efnv2b0)
    #model.add(Dropout(0.5))
    model.add(Dense(cls, activation='softmax'))   # cls = class
    # 輸出網絡模型參數
    model.summary() 
    # dot_img_file = 'EfficientNetV2B0.png'
    # tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

    # 卷基層參與訓練
    efnv2b0.trainable = True 
    
    return model


def resnet50_model(cls):
    input_shape  = efficientNet_config['input_shape_B0']
    resnet50 = ResNet50(include_top=False,
                  weights='imagenet', 
                  input_shape=input_shape,
                  pooling='avg'
                            )
    
    model = Sequential()
    model.add(resnet50)
    model.add(Dense(cls, activation='softmax'))   # cls = class
    # 輸出網絡模型參數
    model.summary() 
    # dot_img_file = 'EfficientNetV2B0.png'
    # tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

    # 卷基層參與訓練
    resnet50.trainable = True    # 解凍與否
    
    return model


def vgg16_model(cls):
    input_shape  = efficientNet_config['input_shape_B0']
    vgg16 = VGG16(include_top=False,
                  weights='imagenet', 
                  input_shape=input_shape,
                  pooling='avg'
                            )
    
    model = Sequential()
    model.add(vgg16)
    model.add(Dense(cls, activation='softmax'))   # cls = class
    # 輸出網絡模型參數
    model.summary() 
    # dot_img_file = 'EfficientNetV2B0.png'
    # tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

    # 卷基層參與訓練
    vgg16.trainable = True    # 解凍與否
    
    return model


def efficientNetV2B3_model():
    input_shape  = efficientNet_config['input_shape_B3']
    efnv2b3 = EfficientNetV2B3(include_top=False,
                               weights='imagenet', 
                               input_shape=input_shape,
                               pooling='avg'
                            )

    model = Sequential()
    model.add(efnv2b3)
    # model.add(GlobalAveragePooling2D())
    # model.add(Dropout(dropout_rate, name="dropout_out"))
    model.add(Dense(1, activation='sigmoid')) 
    # 輸出網絡模型參數
    model.summary() 

    # 卷基層參與訓練
    efnv2b3.trainable = True # True:全部解凍(超參數會很多)
    
    return model