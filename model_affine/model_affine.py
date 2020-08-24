#-----------------------------------------------------------------------------------
# Basic exmple showing affine function f(x) = ax + b 
# 
#-----------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


x = np.array([1, 5, 8, 9 ,10, 15,13, 3,-2],np.float32)
y = np.array([-2,-5, -7, -12 ,-15, -5, -12,-10,-5],np.float32)


BATCH_SIZE  = 1
num_epochs = 200

dataset = tf.data.Dataset.from_tensor_slices(( x , y ))


dataset = tf.data.Dataset.from_tensor_slices(( x , y ))
dataset = dataset.shuffle(9).batch(BATCH_SIZE, drop_remainder=True).repeat(num_epochs)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, activation='linear',input_shape=(1,)))

model.layers[0].set_weights([np.array([[7.3]],np.float32),np.array([5.5],np.float32)])

model.compile(optimizer='sgd', loss='mse')

model.fit(dataset, epochs=num_epochs, steps_per_epoch=tf.data.experimental.cardinality(dataset).numpy() ,  batch_size = 1, verbose=0)

print(model.layers[0].get_weights())



# Save model to SavedModel format
tf.saved_model.save(model, "./model_affine/affine/")

yy= [model((np.array([[xi]],np.float32))).numpy()[0,0] for xi in x]

plt.figure(1)

plt.scatter(x, y, c='b', label='Donnees')

plt.plot(x, yy, c='r', label='model')

plt.show()

#y = -0.3388672 x   -6.046595

x_in = [0,1]
y_out = model.predict(x_in)

print(y_out)


delta =(y_out[0] -  y_out[1]) / ( x_in[0] - x_in[1])
print(delta)
