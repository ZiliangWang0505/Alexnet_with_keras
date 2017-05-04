#  encoding:utf-8

# single-input and multi-output models
from PIL import Image
import numpy as np
import keras
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, merge, Dense, Activation, Conv2D, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.optimizers import SGD

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
num_attitude = 3

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

# input
main_input = Input(shape=input_shape, dtype='float64', name='main_input') 
# Conv layer 1 output shape (55, 55, 48)
conv_1 = Convolution2D(
    nb_filter=48,
    nb_row=11,
    nb_col=11,
    subsample=(4, 4),
    activation='relu',
    name='conv_1',
    init='he_normal',
    dim_ordering='tf',
)(main_input) 
conv_1 = Dropout(0.25)(conv_1)

# Conv layer 2 output shape (27, 27, 128)
conv_2 = Convolution2D(
    nb_filter=128,
    nb_row=5,
    nb_col=5,
    subsample=(2, 2),
    activation='relu',
    name='conv_2',
    init='he_normal'
)(conv_1)
conv_2 = Dropout(0.25)(conv_2)

# Conv layer 3 output shape (13, 13, 192)
conv_3 = Convolution2D(
    nb_filter=192,
    nb_row=3,
    nb_col=3,
    subsample=(2, 2),
    border_mode='same',
    activation='relu',
    name='conv_3',
    init='he_normal'
)(conv_2)
conv_3 = Dropout(0.25)(conv_3)

# Conv layer 4 output shape (13, 13, 192)
conv_4 = Convolution2D(
    nb_filter=192,
    nb_row=3,
    nb_col=3,
    border_mode='same',
    activation='relu',
    name='conv_4',
    init='he_normal'
)(conv_3)
conv_4 = Dropout(0.25)(conv_4)

# Conv layer 5 output shape (13, 128, 128)
conv_5 = Convolution2D(
    nb_filter=128,
    nb_row=3,
    nb_col=3,
    activation='relu',
    border_mode='same',
    name='conv_5',
    init='he_normal'
)(conv_4)
conv_5 = Dropout(0.25)(conv_5)

# fully connected layer 1
flat = Flatten()(conv_5)
dense_1 = Dense(2048, activation='relu', name='dense_1', init='he_normal')(flat)
dense_1 = Dropout(0.25)(dense_1)

# fully connected layer 2
dense_2 = Dense(2048, activation='relu', name='dense_2', init='he_normal')(dense_1)
dense_2 = Dropout(0.25)(dense_2)

# output
label = Dense(num_classes, activation='softmax', name='label', init='he_normal')(dense_2)
attitude = Dense(num_attitude, activation='softmax', name='attitude', init='he_normal')(dense_2)

# build CNN model
model = Model(input=main_input, output=[label, attitude])

# optimizer=SGD
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss={'label':'categorical_crossentropy', 'attitude':'mean_squared_error'}, metrics=['accuracy'], loss_weights={'label':0.5, 'attitude':0.5})
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
            model.train_on_batch(x_train, [y_train[j * 100:j * 100 + 100], y2_train[j * 100:j * 100 + 100]])
        score = model.evaluate(x_test, [y_test, y2_test])
        print('Epochs:', l + 1)
        print('Test loss:', score[0])
        print('Test loss of label:', score[1])
        print('Test loss of attitude:', score[2])
        print('Test accuracy of label:', score[3])
        print('Test accuracy of attitude:', score[4])
        fw.write('Epochs:'+ str(l + 1) + '\n')
        fw.write('Test loss:' + str(score[0]) + '\n')
        fw.write('Test loss of label:' + str(score[1]) + '\n')
        fw.write('Test loss of attitude:' + str(score[2]) + '\n')
        fw.write('Test accuracy of label:' + str(score[3]) + '\n')
        fw.write('Test accuracy of attitude:' + str(score[4]) + '\n')

model.save('model_Alexnet3_SGD_50.h5')
