import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(2, input_dim=2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

sgd=SGD(lr=0.3)
model.compile(loss='binary_crossentropy',optimizer=sgd)

model.fit(x,y,epochs=1000,batch_size=1,verbose=2)

model.summary()

print(model.predict(x,verbose=1))

