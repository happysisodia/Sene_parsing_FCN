dir_data = "/content/drive/My Drive/Colab Notebooks/dataset1"
dir_seg = dir_data + "/annotations_prepped_train/"
dir_img = dir_data + "/images_prepped_train/"


import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
from sklearn.utils import shuffle
from keras import optimizers
import keras

import pandas as pd
import random


#-------------------methods-------------------------------------
def resize_image_arr(path, width, height):
    img = cv2.imread(path, 1)
    img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    return img


def resize_segmentation_arr(path, nLabels, width, height):
    seg_labels = np.zeros((height, width, nLabels))
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (width, height))
    img = img[:, :, 0]

    for c in range(nLabels):
        seg_labels[:, :, c] = (img == c).astype(int)
    return seg_labels


def give_color_to_seg_img(seg, no_of_labels):
    '''
    seg : (width,height,3)
    '''

    if len(seg.shape) == 3:
        seg = seg[:, :, 0]
    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3)).astype('float')
    colors = sns.color_palette("hls", no_of_labels)

    for c in range(no_of_labels):
        segc = (seg == c)
        seg_img[:, :, 0] += (segc * (colors[c][0]))
        seg_img[:, :, 1] += (segc * (colors[c][1]))
        seg_img[:, :, 2] += (segc * (colors[c][2]))

    return (seg_img)

def FullyCN(nLabels, height=224, width=224):
    assert height % 32 == 0
    assert width % 32 == 0
    image_ordering = "channels_last"

    Inp_image = Input(shape=(height, width, 3))  ## Assume 224,224,3
    # Activation function used in each layer is eLU

    ## Layer 1
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='layer1_conv1', data_format=image_ordering)(Inp_image)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='layer1_conv2', data_format=image_ordering)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='layer1_pool', data_format=image_ordering)(x)
    f1 = x

    # Layer 2
    x = Conv2D(128, (3, 3), activation='elu', padding='same', name='layer2_conv1', data_format=image_ordering)(x)
    x = Conv2D(128, (3, 3), activation='elu', padding='same', name='layer2_conv2', data_format=image_ordering)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='layer2_pool', data_format=image_ordering)(x)
    f2 = x

    # Layer 3
    x = Conv2D(256, (3, 3), activation='elu', padding='same', name='layer3_conv1', data_format=image_ordering)(x)
    x = Conv2D(256, (3, 3), activation='elu', padding='same', name='layer3_conv2', data_format=image_ordering)(x)
    x = Conv2D(256, (3, 3), activation='elu', padding='same', name='layer3_conv3', data_format=image_ordering)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='layer3_pool', data_format=image_ordering)(x)
    pool3 = x

    # Layer 4
    x = Conv2D(512, (3, 3), activation='elu', padding='same', name='layer4_conv1', data_format=image_ordering)(x)
    x = Conv2D(512, (3, 3), activation='elu', padding='same', name='layer4_conv2', data_format=image_ordering)(x)
    x = Conv2D(512, (3, 3), activation='elu', padding='same', name='layer4_conv3', data_format=image_ordering)(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='layer4_pool', data_format=image_ordering)(
        x)

    # Layer 5
    x = Conv2D(512, (3, 3), activation='elu', padding='same', name='layer5_conv1', data_format=image_ordering)(pool4)
    x = Conv2D(512, (3, 3), activation='elu', padding='same', name='layer5_conv2', data_format=image_ordering)(x)
    x = Conv2D(512, (3, 3), activation='elu', padding='same', name='layer5_conv3', data_format=image_ordering)(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='layer5_pool', data_format=image_ordering)(
        x)


    vgg = Model(Inp_image, pool5)
    vgg.load_weights(VGG_Weights_path)
    n = 4096
    o = (Conv2D(n, (7, 7), activation='elu', padding='same', name="conv6", data_format=image_ordering))(pool5)
    conv7 = (Conv2D(n, (1, 1), activation='elu', padding='same', name="conv7", data_format=image_ordering))(o)

    conv7_4 = Conv2DTranspose(nLabels, kernel_size=(4, 4), strides=(4, 4), use_bias=False, data_format=image_ordering)(
        conv7)

    pool411 = (Conv2D(nLabels, (1, 1), activation='elu', padding='same', name="pool4_11", data_format=image_ordering))(
        pool4)
    pool411_2 = (
        Conv2DTranspose(nLabels, kernel_size=(2, 2), strides=(2, 2), use_bias=False, data_format=image_ordering))(
        pool411)

    pool311 = (Conv2D(nLabels, (1, 1), activation='elu', padding='same', name="pool3_11", data_format=image_ordering))(
        pool3)

    o = Add(name="add")([pool411_2, pool311, conv7_4])
    o = Conv2DTranspose(nLabels, kernel_size=(8, 8), strides=(8, 8), use_bias=False, data_format=image_ordering)(o)
    o = (Activation('softmax'))(o)

    model = Model(Inp_image, o)

    return model




class Test(object):
    def __init__(self, x_test, y_test, tmodel):
        self.x_test=x_test
        self.y_test=y_test
        self.trained_model= tmodel
    def print_accuracy(self):
        results = model.evaluate(self.x_test, self.y_test, verbose=1)
        print('Test loss:', results[0])
        print('Test accuracy:', results[1] * 100)


    def Draw_loss(self):
            plt.figure(figsize=(12,10))
            for key in ['loss', 'val_loss']:
                plt.plot(self.trained_model.history[key],label=key)
            plt.legend()
            plt.xlabel('Epochs', fontsize=12)
            plt.title('Loss')
            plt.show()


    def Draw_accuracy(self):
        plt.figure(figsize=(12,10))
        for key in ['accuracy', 'val_accuracy']:
            plt.plot(self.trained_model.history[key],label=key)
        plt.legend(['train','test'])
        plt.ylabel('Accuracy %', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)
        plt.title('Accuracy')



def Intersection_over_Union(Yi, y_predi):
    ## mean Intersection over Union

    Intersection_over_Unions = []
    Nclass = int(np.max(Yi)) + 1
    for c in range(Nclass):
        TP = np.sum((Yi == c) & (y_predi == c))
        FP = np.sum((Yi != c) & (y_predi == c))
        FN = np.sum((Yi == c) & (y_predi != c))
        Intersection_over_Union = TP / float(TP + FP + FN)
        print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, Intersection_over_Union={:4.3f}".format(c, TP, FP, FN, Intersection_over_Union))
        Intersection_over_Unions.append(Intersection_over_Union)
    mIntersection_over_Union = np.mean(Intersection_over_Unions)
    print("_________________")
    print("Mean Intersection_over_Union: {:4.3f}".format(mIntersection_over_Union))






#--------------------------------Checking number of labels and sample printing original--------------

sns.set_style("whitegrid", {'axes.grid' : False})

#sample_image
sample_name = np.array(os.listdir(dir_seg))

file_name = sample_name[5]
print(file_name)

## read in the original image and segmentation labels
seg = cv2.imread(dir_seg + file_name ) # (360, 480, 3)
img_is = cv2.imread(dir_img + file_name )

## Check the number of labels
mi, ma = np.min(seg), np.max(seg)
no_of_labels = ma - mi + 1

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)
ax.imshow(img_is)
ax.set_title("original image")
plt.show()

fig = plt.figure(figsize=(15,10))

sns.set_style("whitegrid", {'axes.grid' : False})


sample_name = np.array(os.listdir(dir_seg))
## show one sample image
file_name = sample_name[5]

## read in the original image and segmentation labels
seg = cv2.imread(dir_seg + file_name ) 
img_is = cv2.imread(dir_img + file_name )

## Check the number of labels
mi, ma = np.min(seg), np.max(seg)
no_of_labels = ma - mi + 1

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)
ax.imshow(img_is)
ax.set_title("original image")
plt.show()

fig = plt.figure(figsize=(15,10))




#------------------------------showing resized sample input images---------------
height, width = 224, 224
output_height, output_width = 224, 224

sample_name = np.array(os.listdir(dir_seg))
for file_name in sample_name[np.random.choice(len(sample_name), 1, replace=False)]:
    file_name = file_name.split(".")[0]
    seg = cv2.imread(dir_seg + file_name + ".png")  # (360, 480, 3)
    img_is = cv2.imread(dir_img + file_name + ".png")
    seg_img = give_color_to_seg_img(seg, no_of_labels)

    fig = plt.figure(figsize=(20, 40))
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(seg_img)

    ax = fig.add_subplot(1, 4, 2)
    ax.imshow(img_is / 255.0)
    ax.set_title("original image {}".format(img_is.shape[:2]))

    ax = fig.add_subplot(1, 4, 3)
    ax.imshow(cv2.resize(seg_img, (height, width)))

    ax = fig.add_subplot(1, 4, 4)
    ax.imshow(cv2.resize(img_is, (output_height, output_width)) / 255.0)
    ax.set_title("resized to {}".format((output_height, output_width)))
    plt.show()

#---------------configuring GPU support-----------------------------------------------
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
import pandas as pd
warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
config = tf.compat.v1.ConfigProto(device_count = {'GPU': 0})
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.visible_device_list = "0"
tf.compat.v1.Session(config=config)


#------resizing both input images and segemtation images and returning in np.array type------

images = os.listdir(dir_img)
images.sort()
segmentations = os.listdir(dir_seg)
segmentations.sort()

X = []
Y = []
for im, seg in zip(images, segmentations):
    X.append(resize_image_arr(dir_img + im, width, height))
    Y.append(resize_segmentation_arr(dir_seg + seg, no_of_labels, output_width, output_height))

X, Y = np.array(X), np.array(Y)
print(X.shape, Y.shape)



#---------------Assigning weights----------------------------------------------

VGG_Weights_path = "/content/drive/My Drive/Colab Notebooks/vgg16_weights.h5"

#--------------------Running Model------------------------------------------

model = FullyCN(nLabels=no_of_labels,
             height=224,
             width=224)
model.summary()


#-------------------------training and testing ---------------------------

train_rate = 0.85
index_train = np.random.choice(X.shape[0], int(X.shape[0] * train_rate), replace=False)
index_test = list(set(range(X.shape[0])) - set(index_train))

X, Y = shuffle(X, Y)
X_train, y_train = X[index_train], Y[index_train]
X_test, y_test = X[index_test], Y[index_test]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)




sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
#rmsprop = keras.optimizers.RMSprop(learning_rate=0.001, decay=5**(-4))
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

hist1 = model.fit(X_train,y_train,
                  validation_data=(X_test,y_test),
                  batch_size=32,epochs=100,verbose=2)


#--------------------Results-------------------------------------------------

test = Test(X_test, y_test, hist1)
test.print_accuracy()
test.accuracy()
test.Draw_loss()





y_pred = model.predict(X_test)
y_predi = np.argmax(y_pred, axis=3)
y_testi = np.argmax(y_test, axis=3)
print(y_testi.shape,y_predi.shape)



Intersection_over_Union(y_testi, y_predi)

shape = (224, 224)
no_of_labels = 10

for i in range(10):
    img_is = (X_test[i] + 1) * (255.0 / 2)
    seg = y_predi[i]
    segtest = y_testi[i]

    fig = plt.figure(figsize=(10, 30))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img_is / 255.0)
    ax.set_title("original")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(give_color_to_seg_img(seg, no_of_labels))
    ax.set_title("predicted class")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(give_color_to_seg_img(segtest, no_of_labels))
    ax.set_title("true class")
    plt.show()









