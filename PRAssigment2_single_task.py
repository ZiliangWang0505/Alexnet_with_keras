#  encoding:utf-8

# single-input and single-output models
from PIL import Image
import numpy as np
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.models import load_model


print('------------load data------------')
y_train = []
y2_train = []
with open('data/train.txt') as fr:
    for i in range(1, 15001):
        line = fr.readline().split()
        y_train.append(int(line[1]))
        y2_train.append([float(line[2]), float(line[3]), float(line[4])])
y_train = np.array(y_train)
y2_train = np.array(y2_train)
print('y_train ok')

x_test = []
train_str = 'data/test/'
for i in range(1, 3001):
    img_string = train_str + str(i) + '.jpg'
    im = Image.open(img_string).convert("L")
    jpg_data = im.getdata()
    jpg_data = np.array(jpg_data)
    x_test.append(jpg_data)
x_test = np.array(x_test)
print('x_test ok')

y_test = []
y2_test = []
with open('data/test.txt') as fr:
    for i in range(1, 3001):
        line = fr.readline().split()
        y_test.append(int(line[1]))
        y2_test.append([float(line[2]), float(line[3]), float(line[4])])
y_test = np.array(y_test)
y2_test = np.array(y2_test)
print('y_test ok')

batch_size = 100
epochs = 50
num_classes = 2

# input image dimensions
img_rows, img_cols = 227, 227
input_shape = (img_rows, img_cols, 1)


# convert class vectors to binary class matrices
# x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
# x_train = x_train.astype("float64")
x_test = x_test.astype("float64")
# x_train /= 255
x_test /= 255
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# build CNN
model = Sequential()

# Conv layer 1 output shape (55, 55, 48)
model.add(Conv2D(
    kernel_size=(11, 11), 
    data_format="channels_last", 
    activation="relu",
    filters=48, 
    strides=(4, 4), 
    input_shape=input_shape
))
model.add(Dropout(0.25))

# Conv layer 2 output shape (27, 27, 128)
model.add(Conv2D(
    strides=(2, 2), 
    kernel_size=(5, 5), 
    activation="relu", 
    filters=128
))
model.add(Dropout(0.25))

# Conv layer 3 output shape (13, 13, 192)
model.add(Conv2D(
    kernel_size=(3, 3),
    activation="relu", 
    filters=192,
    padding="same",
    strides=(2, 2)
))
model.add(Dropout(0.25))

# Conv layer 4 output shape (13, 13, 192)
model.add(Conv2D(
    padding="same", 
    activation="relu",
    kernel_size=(3, 3),
    filters=192
))
model.add(Dropout(0.25))

# Conv layer 5 output shape (128, 13, 13)
model.add(Conv2D(
    padding="same",
    activation="relu", 
    kernel_size=(3, 3),
    filters=128
))
model.add(Dropout(0.25))

# fully connected layer 1
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.25))

# fully connected layer 2
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.25))

# output
model.add(Dense(num_classes, activation='softmax'))

# optimizer=SGD
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# optimizer=Ada
# model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

print(model.summary())

with open('log.text','w') as fw:
    train_str = 'data/train/'
    for l in range(epochs):
        for j in range(0, 150):
            x_train = []
            for i in range(j * 100 + 1, j * 100 + 101):
                img_string = train_str + str(i) + '.jpg'
                im = Image.open(img_string).convert("L")
                jpg_data = im.getdata()
                jpg_data = np.array(jpg_data)
                x_train.append(jpg_data)
            x_train = np.array(x_train)
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_train = x_train.astype("float64")
            x_train /= 255
            model.train_on_batch(x_train, y_train[j * 100:j * 100 + 100])
        score = model.evaluate(x_test, y_test)
        print('Epochs:', l + 1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        fw.write('Epochs:'+ str(l + 1) + '\n')
        fw.write('Test loss:' + str(score[0]) + '\n')
        fw.write('Test accuracy:' + str(score[1]) + '\n')

model.save('model_Alexnet2_SGD_50.h5')
