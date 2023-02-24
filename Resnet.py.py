
import numpy as np
import pandas as pd
import os
import cv2
import sklearn
import random as python_random

from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation
from tensorflow.keras.layers import add, Add

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import Sequence


from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense , Conv2D , Dropout , Flatten , Activation, MaxPooling2D , GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam , RMSprop 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau , EarlyStopping , ModelCheckpoint , LearningRateScheduler

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ZeroPadding2D, MaxPooling2D


IMAGE_SIZE = 128
BATCH_SIZE = 64

# identity block은 shortcut 단에 conv layer가 없는 block 영역
def identity_block(input_tensor, middle_kernel_size, filters, stage, block):
    '''
    함수 입력 인자 설명
    input_tensor는 입력 tensor
    middle_kernel_size 중간에 위치하는 kernel 크기. identity block내에 있는 두개의 conv layer중 1x1 kernel이 아니고, 3x3 kernel임. 
    3x3 커널이 이외에도 5x5 kernel도 지정할 수 있게 구성. 
    filters: 3개 conv layer들의 filter개수를 list 형태로 입력 받음. 첫번째 원소는 첫번째 1x1 filter 개수, 두번째는 3x3 filter 개수, 세번째는 마지막 1x1 filter 개수
    stage: identity block들이 여러개가 결합되므로 이를 구분하기 위해서 설정. 동일한 filter수를 가지는 identity block들을  동일한 stage로 설정.  
    block: 동일 stage내에서 identity block을 구별하기 위한 구분자
    ''' 
    
    # filters로 list 형태로 입력된 filter 개수를 각각 filter1, filter2, filter3로 할당. 
    # filter은 첫번째 1x1 filter 개수, filter2는 3x3 filter개수, filter3는 마지막 1x1 filter개수
    filter1, filter2, filter3 = filters
    # conv layer와 Batch normalization layer각각에 고유한 이름을 부여하기 위해 설정. 입력받은 stage와 block에 기반하여 이름 부여
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # 이전 layer에 입력 받은 input_tensor(함수인자로 입력받음)를 기반으로 첫번째 1x1 Conv->Batch Norm->Relu 수행. 
    # 첫번째 1x1 Conv에서 Channel Dimension Reduction 수행. filter1은 입력 input_tensor(입력 Feature Map) Channel 차원 개수의 1/4임. 
    x = Conv2D(filters=filter1, kernel_size=(1, 1), kernel_initializer='he_normal', name=conv_name_base+'2a')(input_tensor)
    # Batch Norm적용. 입력 데이터는 batch 사이즈까지 포함하여 4차원임(batch_size, height, width, channel depth)임
    # Batch Norm의 axis는 channel depth에 해당하는 axis index인 3을 입력.(무조건 channel이 마지막 차원의 값으로 입력된다고 가정. )
    x = BatchNormalization(axis=3, name=bn_name_base+'2a')(x)
    # ReLU Activation 적용. 
    x = Activation('relu')(x)
    
    # 두번째 3x3 Conv->Batch Norm->ReLU 수행
    # 3x3이 아닌 다른 kernel size도 구성 가능할 수 있도록 identity_block() 인자로 입력받은 middle_kernel_size를 이용. 
    # Conv 수행 출력 사이즈가 변하지 않도록 padding='same'으로 설정. filter 개수는 이전의 1x1 filter개수와 동일.  
    x = Conv2D(filters=filter2, kernel_size=middle_kernel_size, padding='same', kernel_initializer='he_normal', name=conv_name_base+'2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)
    
    # 마지막 1x1 Conv->Batch Norm 수행. ReLU를 수행하지 않음에 유의.
    # filter 크기는 input_tensor channel 차원 개수로 원복
    x = Conv2D(filters=filter3, kernel_size=(1, 1), kernel_initializer='he_normal', name=conv_name_base+'2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base+'2c')(x)
    # Residual Block 수행 결과와 input_tensor를 합한다. 
    x = Add()([input_tensor, x])
    # 또는 x = add([x, input_tensor]) 와 같이 구현할 수도 있음. 

    # 마지막으로 identity block 내에서 최종 ReLU를 적용
    x = Activation('relu')(x)
    
    return x

def conv_block(input_tensor, middle_kernel_size, filters, stage, block, strides=(2, 2)):
    '''
    함수 입력 인자 설명
    input_tensor: 입력 tensor
    middle_kernel_size: 중간에 위치하는 kernel 크기. identity block내에 있는 두개의 conv layer중 1x1 kernel이 아니고, 3x3 kernel임. 
                        3x3 커널 이외에도 5x5 kernel도 지정할 수 있게 구성. 
    filters: 3개 conv layer들의 filter개수를 list 형태로 입력 받음. 첫번째 원소는 첫번째 1x1 filter 개수, 두번째는 3x3 filter 개수, 
             세번째는 마지막 1x1 filter 개수
    stage: identity block들이 여러개가 결합되므로 이를 구분하기 위해서 설정. 동일한 filter수를 가지는 identity block들을  동일한 stage로 설정.  
    block: 동일 stage내에서 identity block을 구별하기 위한 구분자
    strides: 입력 feature map의 크기를 절반으로 줄이기 위해서 사용. Default는 2이지만, 
             첫번째 Stage의 첫번째 block에서는 이미 입력 feature map이 max pool로 절반이 줄어있는 상태이므로 다시 줄이지 않기 위해 1을 호출해야함 
    ''' 
    
    # filters로 list 형태로 입력된 filter 개수를 각각 filter1, filter2, filter3로 할당. 
    # filter은 첫번째 1x1 filter 개수, filter2는 3x3 filter개수, filter3는 마지막 1x1 filter개수
    filter1, filter2, filter3 = filters
    # conv layer와 Batch normalization layer각각에 고유한 이름을 부여하기 위해 설정. 입력받은 stage와 block에 기반하여 이름 부여
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # 이전 layer에 입력 받은 input_tensor(함수인자로 입력받음)를 기반으로 첫번째 1x1 Conv->Batch Norm->Relu 수행. 
    # 입력 feature map 사이즈를 1/2로 줄이기 위해 strides인자를 입력  
    x = Conv2D(filters=filter1, kernel_size=(1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base+'2a')(input_tensor)
    # Batch Norm적용. 입력 데이터는 batch 사이즈까지 포함하여 4차원임(batch_size, height, width, channel depth)임
    # Batch Norm의 axis는 channel depth에 해당하는 axis index인 3을 입력.(무조건 channel이 마지막 차원의 값으로 입력된다고 가정. )
    x = BatchNormalization(axis=3, name=bn_name_base+'2a')(x)
    # ReLU Activation 적용. 
    x = Activation('relu')(x)
    
    # 두번째 3x3 Conv->Batch Norm->ReLU 수행
    # 3x3이 아닌 다른 kernel size도 구성 가능할 수 있도록 identity_block() 인자로 입력받은 middle_kernel_size를 이용. 
    # Conv 수행 출력 사이즈가 변하지 않도록 padding='same'으로 설정. filter 개수는 이전의 1x1 filter개수와 동일.  
    x = Conv2D(filters=filter2, kernel_size=middle_kernel_size, padding='same', kernel_initializer='he_normal', name=conv_name_base+'2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)
    
    # 마지막 1x1 Conv->Batch Norm 수행. ReLU를 수행하지 않음에 유의.
    # filter 크기는 input_tensor channel 차원 개수로 원복
    x = Conv2D(filters=filter3, kernel_size=(1, 1), kernel_initializer='he_normal', name=conv_name_base+'2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base+'2c')(x)
    
    # shortcut을 1x1 conv 수행, filter3가 입력 feature map의 filter 개수
    shortcut = Conv2D(filter3, (1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base+'1')(input_tensor)
    shortcut = BatchNormalization(axis=3, name=bn_name_base+'1')(shortcut)
    
    # Residual Block 수행 결과와 1x1 conv가 적용된 shortcut을 합한다. 
    x = add([x, shortcut])
    
    # 마지막으로 identity block 내에서 최종 ReLU를 적용
    x = Activation('relu')(x)
    
    return x
    



# input_tensor로 임의의 Feature Map size를 생성. 
input_tensor = Input(shape=(56, 56, 256), name='test_input')
# input_tensor의 channel수는 256개임. filters는 256의 1/4 filter수로 차원 축소후 다시 마지막 1x1 Conv에서 256으로 복원
filters = [64, 64, 256]
# 중간 Conv 커널 크기는 3x3
kernel_size = (3, 3)
stage = 2
block = 'a'

# identity_block을 호출하고 layer들이 어떻게 구성되어 있는지 확인하기 위해서 model로 구성하고 summary()호출 
output = identity_block(input_tensor, kernel_size, filters, stage, block)
identity_layers = Model(inputs=input_tensor, outputs=output)
identity_layers.summary()

input_tensor = Input(shape=(56, 56, 256), name='test_input')
x = identity_block(input_tensor, middle_kernel_size=3, filters=[64, 64, 256], stage=2, block='a')
x = identity_block(x, middle_kernel_size=3, filters=[64, 64, 256], stage=2, block='b')
output = identity_block(x, middle_kernel_size=3, filters=[64, 64, 256], stage=2, block='c')
identity_layers = Model(inputs=input_tensor, outputs=output)
identity_layers.summary()

input_tensor = Input(shape=(56, 56, 256), name='test_input')
# conv_block() 호출 시 strides를 2로 설정하여 입력 feature map의 크기를 절반으로 줄임. strides=1이면 크기를 그대로 유지
x = conv_block(input_tensor, middle_kernel_size=3, filters=[64, 64, 256], strides=2, stage=2, block='a')
x = identity_block(x, middle_kernel_size=3, filters=[64, 64, 256], stage=2, block='b')
output = identity_block(x, middle_kernel_size=3, filters=[64, 64, 256], stage=2, block='c')
identity_layers = Model(inputs=input_tensor, outputs=output)
identity_layers.summary()




def do_first_conv(input_tensor):
    # 7x7 Conv 연산 수행하여 feature map 생성하되 input_tensor 크기(image 크기)의 절반으로 생성.  filter 개수는 64개 
    # 224x224 를 input을 7x7 conv, strides=2로 112x112 출력하기 위해 Zero padding 적용. 
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_tensor)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    # 다시 feature map 크기를 MaxPooling으로 절반으로 만듬. 56x56으로 출력하기 위해 zero padding 적용. 
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    return x

input_tensor = Input(shape=(224, 224, 3))
output = do_first_conv(input_tensor)
model = Model(inputs=input_tensor, outputs=output)
model.summary()


def create_resnet(in_shape=(224, 224, 3), n_classes=10):
    input_tensor = Input(shape=in_shape)
    
    #첫번째 7x7 Conv와 Max Polling 적용.  
    x = do_first_conv(input_tensor)
    
    # stage 2의 conv_block과 identity block 생성. stage2의 첫번째 conv_block은 strides를 1로 하여 크기를 줄이지 않음. 
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    
    # stage 3의 conv_block과 identity block 생성. stage3의 첫번째 conv_block은 strides를 2(default)로 하여 크기를 줄임 
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # stage 4의 conv_block과 identity block 생성. stage4의 첫번째 conv_block은 strides를 2(default)로 하여 크기를 줄임
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # stage 5의 conv_block과 identity block 생성. stage5의 첫번째 conv_block은 strides를 2(default)로 하여 크기를 줄임
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    
    # classification dense layer와 연결 전 GlobalAveragePooling 수행 
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(200, activation='relu', name='fc_01')(x)
    x = Dropout(rate=0.5)(x)
    output = Dense(n_classes, activation='softmax', name='fc_final')(x)
    
    model = Model(inputs=input_tensor, outputs=output, name='resnet50')
    model.summary()
    
    return model

model =  create_resnet(in_shape=(224,224,3), n_classes=10)



def zero_one_scaler(image):
    return image/255.0

def get_preprocessed_ohe(images, labels, pre_func=None):
    # preprocessing 함수가 입력되면 이를 이용하여 image array를 scaling 적용.
    if pre_func is not None:
        images = pre_func(images)
    # OHE 적용    
    oh_labels = to_categorical(labels)
    return images, oh_labels

# 학습/검증/테스트 데이터 세트에 전처리 및 OHE 적용한 뒤 반환 
def get_train_valid_test_set(train_images, train_labels, test_images, test_labels, valid_size=0.15, random_state=2021):
    # 학습 및 테스트 데이터 세트를  0 ~ 1사이값 float32로 변경 및 OHE 적용. 
    train_images, train_oh_labels = get_preprocessed_ohe(train_images, train_labels)
    test_images, test_oh_labels = get_preprocessed_ohe(test_images, test_labels)
    
    # 학습 데이터를 검증 데이터 세트로 다시 분리
    tr_images, val_images, tr_oh_labels, val_oh_labels = train_test_split(train_images, train_oh_labels, test_size=valid_size, random_state=random_state)
    
    return (tr_images, tr_oh_labels), (val_images, val_oh_labels), (test_images, test_oh_labels )


# 입력 인자 images_array labels는 모두 numpy array로 들어옴. 
# 인자로 입력되는 images_array는 전체 32x32 image array임. 
class CIFAR_Dataset(Sequence):
    def __init__(self, images_array, labels, batch_size=BATCH_SIZE, augmentor=None, shuffle=False, pre_func=None):
        '''
        파라미터 설명
        images_array: 원본 32x32 만큼의 image 배열값. 
        labels: 해당 image의 label들
        batch_size: __getitem__(self, index) 호출 시 마다 가져올 데이터 batch 건수
        augmentor: albumentations 객체
        shuffle: 학습 데이터의 경우 epoch 종료시마다 데이터를 섞을지 여부
        '''
        # 객체 생성 인자로 들어온 값을 객체 내부 변수로 할당. 
        # 인자로 입력되는 images_array는 전체 32x32 image array임.
        self.images_array = images_array
        self.labels = labels
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.pre_func = pre_func
        # train data의 경우 
        self.shuffle = shuffle
        if self.shuffle:
            # 객체 생성시에 한번 데이터를 섞음. 
            #self.on_epoch_end()
            pass
    
    # Sequence를 상속받은 Dataset은 batch_size 단위로 입력된 데이터를 처리함. 
    # __len__()은 전체 데이터 건수가 주어졌을 때 batch_size단위로 몇번 데이터를 반환하는지 나타남
    def __len__(self):
        # batch_size단위로 데이터를 몇번 가져와야하는지 계산하기 위해 전체 데이터 건수를 batch_size로 나누되, 정수로 정확히 나눠지지 않을 경우 1회를 더한다. 
        return int(np.ceil(len(self.labels) / self.batch_size))
    
    # batch_size 단위로 image_array, label_array 데이터를 가져와서 변환한 뒤 다시 반환함
    # 인자로 몇번째 batch 인지를 나타내는 index를 입력하면 해당 순서에 해당하는 batch_size 만큼의 데이타를 가공하여 반환
    # batch_size 갯수만큼 변환된 image_array와 label_array 반환. 
    def __getitem__(self, index):
        # index는 몇번째 batch인지를 나타냄. 
        # batch_size만큼 순차적으로 데이터를 가져오려면 array에서 index*self.batch_size:(index+1)*self.batch_size 만큼의 연속 데이터를 가져오면 됨
        # 32x32 image array를 self.batch_size만큼 가져옴. 
        images_fetch = self.images_array[index*self.batch_size:(index+1)*self.batch_size]
        if self.labels is not None:
            label_batch = self.labels[index*self.batch_size:(index+1)*self.batch_size]
        
        # 만일 객체 생성 인자로 albumentation으로 만든 augmentor가 주어진다면 아래와 같이 augmentor를 이용하여 image 변환
        # albumentations은 개별 image만 변환할 수 있으므로 batch_size만큼 할당된 image_name_batch를 한 건씩 iteration하면서 변환 수행. 
        # 변환된 image 배열값을 담을 image_batch 선언. image_batch 배열은 float32 로 설정. 
        image_batch = np.zeros((images_fetch.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3), dtype='float32')
        
        # batch_size에 담긴 건수만큼 iteration 하면서 opencv image load -> image augmentation 변환(augmentor가 not None일 경우)-> image_batch에 담음. 
        for image_index in range(images_fetch.shape[0]):
            #image = cv2.cvtColor(cv2.imread(image_name_batch[image_index]), cv2.COLOR_BGR2RGB)
            # 원본 image를 IMAGE_SIZE x IMAGE_SIZE 크기로 변환
            image = cv2.resize(images_fetch[image_index], (IMAGE_SIZE, IMAGE_SIZE))
            # 만약 augmentor가 주어졌다면 이를 적용. 
            if self.augmentor is not None:
                image = self.augmentor(image=image)['image']
                
            # 만약 scaling 함수가 입력되었다면 이를 적용하여 scaling 수행. 
            if self.pre_func is not None:
                image = self.pre_func(image)
            
            # image_batch에 순차적으로 변환된 image를 담음.               
            image_batch[image_index] = image
        
        return image_batch, label_batch
    
    # epoch가 한번 수행이 완료 될 때마다 모델의 fit()에서 호출됨. 
    def on_epoch_end(self):
        if(self.shuffle):
            #print('epoch end')
            # 원본 image배열과 label를 쌍을 맞춰서 섞어준다. scikt learn의 utils.shuffle에서 해당 기능 제공
            self.images_array, self.labels = sklearn.utils.shuffle(self.images_array, self.labels)
        else:
            pass


# CIFAR10 데이터 재 로딩 및 OHE 전처리 적용하여 학습/검증/데이터 세트 생성. 
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

(tr_images, tr_oh_labels), (val_images, val_oh_labels), (test_images, test_oh_labels) = \
    get_train_valid_test_set(train_images, train_labels, test_images, test_labels, valid_size=0.2, random_state=2021)
print(tr_images.shape, tr_oh_labels.shape, val_images.shape, val_oh_labels.shape, test_images.shape, test_oh_labels.shape)


tr_ds = CIFAR_Dataset(tr_images, tr_oh_labels, batch_size=BATCH_SIZE, augmentor=None, shuffle=True, pre_func=resnet_preprocess)
val_ds = CIFAR_Dataset(val_images, val_oh_labels, batch_size=BATCH_SIZE, augmentor=None, shuffle=False, pre_func=resnet_preprocess)

print(next(iter(tr_ds))[0].shape, next(iter(val_ds))[0].shape)
print(next(iter(tr_ds))[1].shape, next(iter(val_ds))[1].shape)
# 채널별 값 - [103.939, 116.779, 123.68]
print(next(iter(tr_ds))[0][0])

resnet_model = create_resnet(in_shape=(128, 128, 3), n_classes=10)

resnet_model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 5번 iteration내에 validation loss가 향상되지 않으면 learning rate을 기존 learning rate * 0.2로 줄임.  
rlr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, mode='min', verbose=1)
ely_cb = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

history = resnet_model.fit(tr_ds, epochs=30, 
                    #steps_per_epoch=int(np.ceil(tr_images.shape[0]/BATCH_SIZE)),
                    validation_data=val_ds, 
                    #validation_steps=int(np.ceil(val_images.shape[0]/BATCH_SIZE)), 
                    callbacks=[rlr_cb, ely_cb]
                   )

test_ds = CIFAR_Dataset(test_images, test_oh_labels, batch_size=BATCH_SIZE, augmentor=None, shuffle=False, pre_func=resnet_preprocess)
resnet_model.evaluate(test_ds)




input_tensor = Input(shape=(128, 128, 3))
base_model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)
bm_output = base_model.output

# classification dense layer와 연결 전 GlobalAveragePooling 수행 
x = GlobalAveragePooling2D(name='avg_pool')(bm_output)
x = Dropout(rate=0.5)(x)
x = Dense(200, activation='relu', name='fc_01')(x)
x = Dropout(rate=0.5)(x)
output = Dense(10, activation='softmax', name='fc_final')(x)

pr_model = Model(inputs=input_tensor, outputs=output, name='resnet50')
pr_model.summary()

pr_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = pr_model.fit(tr_ds, epochs=30, 
                    validation_data=val_ds,
                    callbacks=[rlr_cb, ely_cb]
                   )

test_ds = CIFAR_Dataset(test_images, test_oh_labels, batch_size=BATCH_SIZE, augmentor=None, shuffle=False, pre_func=resnet_preprocess)
pr_model.evaluate(test_ds)

// end
