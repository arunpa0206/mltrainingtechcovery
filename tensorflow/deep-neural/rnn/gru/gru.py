from keras.datasets import imdb
from keras.layers import GRU, LSTM, CuDNNGRU, CuDNNLSTM, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import numpy as np

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)





num_words = 30000
maxlen = 300

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = num_words)

# pad the sequences with zeros
# padding parameter is set to 'post' => 0's are appended to end of sequences
X_train = pad_sequences(X_train, maxlen = maxlen, padding = 'post')
X_test = pad_sequences(X_test, maxlen = maxlen, padding = 'post')

X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))

def gru_model():
    model = Sequential()
    model.add(GRU(50, input_shape = (300,1), return_sequences = True))
    model.add(GRU(1, return_sequences = False))
    model.add(Activation('sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

model = gru_model()


model.fit(X_train, y_train, batch_size = 100, epochs = 4)

scores = model.evaluate(X_test, y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))
