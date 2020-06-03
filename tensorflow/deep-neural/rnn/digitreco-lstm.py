import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils



#go lstm
nb_classes = 10
img_rows, img_cols = 28, 28

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
nb_classes = 10
img_rows, img_cols = 28, 28
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# one-hot encoding:
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#print()
#print(X_train.shape())
train_size = 60000
test_size = 10000


model = Sequential()
model.add(LSTM(50,input_shape=(28,28))) #lstm cell with 50 units
model.add(Dense(10, init='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 10

history = model.fit(X_train,
                    Y_train,
                    epochs=epochs,
                    batch_size=128,
                    verbose=1)
