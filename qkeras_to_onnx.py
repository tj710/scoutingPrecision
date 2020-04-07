#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
os.environ['TF_KERAS'] = '1'
import sys
import numpy as np
import keras
from keras.layers import Input, Reshape, Concatenate, Dense
from keras.models import Model, model_from_json
from qkeras.utils import model_save_quantized_weights, quantized_model_from_json, load_qmodel
from qkeras import *
from qkeras import QActivation, QDense, QBatchNormalization
from qkeras import quantized_bits
import onnx
import onnxmltools
import time
import keras.backend as K
import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Concatenate, Dense
from tensorflow.keras.models import Model


# In[2]:


# try also loading quantized weights by adding '_quantized_weights' to model_name
def load_model_quantized(model_name):
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = quantized_model_from_json(loaded_model_json)
    model.load_weights(model_name + '.h5')
    
    return model


# In[3]:


model = load_model_quantized('models/qkeras_models/qkeras_model_2_4_2020_3layers_vol100')
model.summary()


# In[4]:


model_2 = load_qmodel('models/qkeras_models/qkeras_model_2_4_2020_3layers_vol100_h5file.h5')
model_2.summary()


# In[7]:


# didn't use this always (for making sure input and output are compatible with the board)
lays = model_2.layers

n_input_features = lays[0].input_shape[0][1]
inputs = Input(shape=(1,1,n_input_features,))
hidden_layers = Reshape((n_input_features,), name='reshaped_hidden')(inputs)

for l in lays[1:-3]:
    hidden_layers = l(hidden_layers)

phi_weights = lays[-3].get_weights()
eta_weights = lays[-2].get_weights()
pt_weights = lays[-1].get_weights()

combined_W = np.concatenate((phi_weights[0], eta_weights[0], pt_weights[0]), -1)
combined_b = np.concatenate((phi_weights[1], eta_weights[1], pt_weights[1]), -1)
new_output_layer = QDense(3, activation='linear', weights=[combined_W, combined_b], name='output3')(hidden_layers)

inference_model = Model(inputs=inputs, outputs=[new_output_layer])
inference_model.summary()


# In[8]:


# doesn't work --> because of tf version
import keras2onnx as k2o
import tf2onnx
onnx_model = k2o.convert_keras(model, model.name)
k2o.save_model(onnx_model, 'models/qkeras_models/finaly.onnx')


# In[ ]:


print("save tensorflow in format \"saved_model\"")

# creates assets dir, variables dir and saved_model.pb file
# customized = have input_shape=(1,1,num_features) and 1 output
tf.keras.models.save_model(model, 'models/qkeras_models/qkeras_model_7_4_2020_quant')

""" RUN this in terminal: 
        python -m tf2onnx.convert \
        --saved-model ./output/saved_model \
        --output ./output/mnist1.onnx \
        --opset 11 """
""" ./output/saved_model --> repository where is file saved_model.pb"""
""" Opset version ai.onnx --> for onnx version 1.6.0 => opset = 11 """

