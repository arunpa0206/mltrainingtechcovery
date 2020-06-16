import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

#height of 200 people between 1 to 6 feet
height = data  = np.linspace(1,6,200)
#randomly generate weight
weight = height*4 + np.random.randn(*height.shape) * 0.3


model = Sequential()
#input = height  and output =weight
model.add(Dense(1, input_dim=1, activation='linear'))

model.compile(optimizer='sgd', loss='mse', metrics=['mse'])

weights = model.layers[0].get_weights()
#weight is represemted as a matrix and is  stored at 0th location
w_init = weights[0][0][0]
#bias is stored a vector in 1st location
b_init = weights[1][0]
print('Linear regression model is initialized with weights w: %.2f, b: %.2f' % (w_init, b_init))


model.fit(height,weight, batch_size=1, epochs=30, shuffle=False)

weights = model.layers[0].get_weights()
#zeroth element is the weight matrix first element is the bias vector

w_final = weights[0][0][0]
b_final = weights[1][0]
print('Linear regression model is trained to have weight w: %.2f, b: %.2f' % (w_final, b_final))

predict = model.predict(data)

plt.plot(data, predict, 'b', data , weight, 'k.')
plt.show()
