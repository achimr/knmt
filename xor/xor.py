import numpy as np
import keras
import matplotlib.pyplot as plot
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

history = model.fit(x,y,epochs=1000,batch_size=1,verbose=2)

model.summary()

print(model.predict(x,verbose=1))

loss_values = history.history['loss']
epochs = range(1, len(loss_values) + 1)

plot.plot(epochs, loss_values, 'bo', label='Training loss')
plot.title('Training loss')
plot.xlabel('Epochs')
plot.ylabel('Loss')
plot.legend()
plot.show()
