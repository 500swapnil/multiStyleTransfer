#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np
import sys
import _pickle as cPickle
import os
import gzip
import tensorflow
import matplotlib.pyplot as plt


# In[2]:


f = gzip.open('mnist.pkl.gz', 'rb')
data = cPickle.load(f, encoding='bytes')

(x_train, y_train), (x_test, y_test) = data

x_train = np.reshape(x_train.astype('float32'), (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test.astype('float32'), (len(x_test), 28, 28, 1)) 


# In[6]:


def encoder():
    input_img = Input(shape=(28, 28, 1)) 

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded1 = MaxPooling2D((2, 2), padding='same')(x)

    model1 = Model(input_img, encoded1, name="encoder")
    return model1


# In[7]:


def decoder():
    input_img = Input(shape=(4, 4, 8))

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model2 = Model(input_img, decoded, name="decoder")
    return model2


# In[8]:


model1 = encoder()
model2 = decoder()

inputs = Input(shape=(28, 28, 1) ,name='input')
encoder_out = model1(inputs)
final_out = model2(encoder_out)

model = Model(inputs, final_out, name='final_model')

model.compile(optimizer='adam',
          loss='mean_squared_error')

model.fit(x_train, x_train,
epochs=5, batch_size=32, verbose=1)

model1.save_weights("model1.h5")
model2.save_weights("model2.h5")


# In[38]:


fc1 = model1.predict(x_train[0].reshape(1,28,28,1))

new_im = x_train[8]
fs1 = model1.predict(new_im.reshape(1,28,28,1))

final_ans = model2.predict(fc1)

plt.figure(figsize=(28, 28))
plt.imshow(final_ans.reshape(28, 28))
#plt.savefig('output_on_style.png')


# In[ ]:




