#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import keras
from keras.layers import Input, Reshape, Concatenate, Dense
from keras.models import Model, model_from_json
from qkeras.utils import quantized_model_from_json
import onnx
import onnxmltools
import time


# In[2]:


def load_model(model_name):
    name = model_name + '.json'
    json_file = open(name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_name + '.h5')
    
    return model

def load_model_quantized(model_name):
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = quantized_model_from_json(loaded_model_json)
    model.load_weights(model_name  + '.h5')
    
    return model


# In[3]:


model_name = 'models/scouting_models/integer_scouting_7_4_2020_3layers'


# In[4]:


model = load_model(model_name)
model.summary()


# In[ ]:


# Adding layers to create inference model - if model has 1 output
lays = model.layers

n_input_features = lays[0].input_shape[1]
inputs = Input(shape=(1,1,n_input_features,))
hidden_layers = Reshape((n_input_features,))(inputs)

for l in lays[1:]:
    hidden_layers = l(hidden_layers)

inference_model = Model(inputs=inputs, outputs=hidden_layers)
inference_model.summary()


# In[7]:


# Adding layers to create inference model - if model has 3 outputs
lays = model.layers

n_input_features = lays[0].input_shape[1]
inputs = Input(shape=(1,1,n_input_features,))
hidden_layers = Reshape((n_input_features,))(inputs)

for l in lays[1:-3]:
    hidden_layers = l(hidden_layers)

phi_weights = lays[-3].get_weights()
eta_weights = lays[-2].get_weights()
pt_weights = lays[-1].get_weights()

combined_W = np.concatenate((phi_weights[0], eta_weights[0], pt_weights[0]), -1)
combined_b = np.concatenate((phi_weights[1], eta_weights[1], pt_weights[1]), -1)
new_output_layer = Dense(3, activation='linear', weights=[combined_W, combined_b], name='output3')

output = new_output_layer(hidden_layers)

inference_model = Model(inputs=inputs, outputs=output)
inference_model.summary()


# In[8]:


# Converting inference model to onnx
onnx_model = onnxmltools.convert_keras(inference_model)
onnx.save(onnx_model, model_name + '.onnx')


# In[ ]:




