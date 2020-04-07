#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['TF_KERAS'] = '1'
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import math
import keras.optimizers
from tensorflow.keras.optimizers import Adadelta
from keras import regularizers
import keras.backend as K
import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import load_model, model_from_json
import joblib
from scipy.stats import norm
import matplotlib
from scipy.signal import argrelextrema
import tensorflow as tf

from qkeras import *
from qkeras import QActivation, QDense, QBatchNormalization
from qkeras import quantized_bits
from tensorflow.keras.layers import Input, Reshape, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from qkeras import print_qstats
from qkeras import ternary, binary
from qkeras.utils import model_save_quantized_weights, quantized_model_from_json, load_qmodel


# In[2]:


print(K.tensorflow_backend._get_available_gpus())


# In[4]:


#load data (barrel + endcap + overlap dataset)

#barrel
df_1= pd.read_csv('../mu_barrel_tight_zb.csv', index_col=False, delimiter=',')
df_2= pd.read_csv('../mu_barrel_tight_zb2018D.csv', index_col=False, delimiter=',')
#df_base = pd.concat([df_1, df_2], ignore_index=True)

# add the additional feature in the dataset (1 = barrel, 2 = endcap, 3 = overlap)
feature = 1
additional_array = feature*np.ones(df_1.shape[0], dtype=int)
df_1['dataset'] = additional_array

feature = 1
additional_array = feature*np.ones(df_2.shape[0], dtype=int)
df_2['dataset'] = additional_array

#endcap
df_3= pd.read_csv('../mu_endcap_tight_zb.csv', index_col=False, delimiter=',')
df_4= pd.read_csv('../mu_endcap_tight_zb2018D.csv', index_col=False, delimiter=',')
#df_base = pd.concat([df_3, df_4], ignore_index=True)

feature = 2
additional_array = feature*np.ones(df_3.shape[0], dtype=int)
df_3['dataset'] = additional_array

feature = 2
additional_array = feature*np.ones(df_4.shape[0], dtype=int)
df_4['dataset'] = additional_array

#overlap
df_5= pd.read_csv('../mu_overlap_tight_zb.csv', index_col=False, delimiter=',')
df_6= pd.read_csv('../mu_overlap_tight_zb2018D.csv', index_col=False, delimiter=',')
#df_base = pd.concat([df_5, df_6], ignore_index =True)

feature = 3
additional_array = feature*np.ones(df_5.shape[0], dtype=int)
df_5['dataset'] = additional_array

feature = 3
additional_array = feature*np.ones(df_6.shape[0], dtype=int)
df_6['dataset'] = additional_array

df_base = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6], ignore_index = True)

print(df_base.shape)


# In[5]:


# check if quality is always the same
def check_quality(dataset):
    if (dataset['quality'] == 12).all(): return True
    else:
        print(dataset[dataset['quality'] != 12]['quality'].shape)
        return False

print(check_quality(df_base))


# In[6]:


#p_t restriction from 2.5 to 45 and quality = 12
df_base = df_base.drop(df_base[df_base.pTL1 < 2.5].index)
df_base = df_base.drop(df_base[df_base.pTL1 > 45].index)

df_base = df_base.drop(df_base[df_base.quality != 12].index)
df_base = df_base.drop(df_base[df_base.quality != 12].index)

print(df_base.shape)


# In[7]:


def save_model(model_save_name, model):
    with open(model_save_name + '.json', 'w') as file:
        file.write(model.to_json())

    model.save_weights(model_save_name + '.h5')
    model_save_quantized_weights(model, model_save_name + '_quantized_weights' + '.h5')
    model.save(model_save_name + '_h5file.h5')


# In[15]:


def load_model_quantized(model_name):
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = quantized_model_from_json(loaded_model_json)
    model.load_weights(model_name + '_quantized_weights' + '.h5')
    
    return model


# In[9]:


def calculate_delta_phi(phi_approx, phi_true):
    return np.array([x - 2*math.pi if x > math.pi else x + 2*math.pi if x < -math.pi else x                           for x in phi_approx - phi_true])

def calculate_phi(phi_approx, delta_phi):
    return np.array([x - 2*math.pi if x > math.pi else x + 2*math.pi if x < -math.pi else x                           for x in phi_approx - delta_phi])


# In[10]:


#create int values for input
def get_int_values_from_detector(values, range_dict=None, range_len=None):
    if not range_dict:
        range_dict = {float_value: int_index for int_index, float_value in enumerate(sorted(set(values)))}
        assert len(range_dict.keys()) == range_len
        return [range_dict[float_value] for float_value in values]
        
    return [range_dict[round(float_value, 7)] for float_value in values]    

def print_stats(values):
    print(np.min(values), np.max(values))
    
#create range for each parametar    
def create_range_float_int(start, stop, step, before=None, after=None):
    rng = np.arange(start, stop, step)
    if before:
        rng = np.insert(rng, 0, before)
    if after:
        rng = np.append(rng, after)
        
    range_dict = {round(float_value, 7): int_index for int_index, float_value in enumerate(sorted(rng))}
    
    return range_dict

def create_range_int_float(start, stop, step, before=None, after=None):
    rng = np.arange(start, stop, step)
    if before:
        rng = np.insert(rng, 0, before)
    if after:
        rng = np.append(rng, after)
        
    range_dict = {int_index: round(float_value, 7) for int_index, float_value in enumerate(sorted(rng))}
    
    return range_dict


# In[11]:


def add_integer_values_to_df(df_base):
    #new_phi_r = np.array([x if x > 0 else x + 2*math.pi for x in df_base['phi_r']])
    new_phi_e = np.array([x if x > 0 else x + 2*math.pi for x in df_base['phiVtxL1']])
    
    #df_base['int_phi_r'] = (new_phi_r / (2*math.pi / 576)).astype(int)
    df_base['int_phi_e'] = (new_phi_e / (2*math.pi / 576)).astype(int)
    
    #df_base['int_eta_r'] = (df_base['eta_r'] / 0.010875).astype(int)
    df_base['int_eta_e'] = (df_base['etaVtxL1'] / 0.010875).astype(int)

    df_base['int_pt_r'] = (df_base['pTL1'] / 0.5).astype(int)
    return df_base


# In[12]:


df_base = add_integer_values_to_df(df_base)

print_stats(df_base['int_phi_e'])
print_stats(df_base['int_eta_e'])
print_stats(df_base['int_pt_r'])
print_stats(df_base['charge'])


# In[13]:


#divide train/test
percentage = 0.8
train_len = int(percentage * len(df_base))
df_train = df_base[:train_len]
df_test = df_base[train_len:]


# In[16]:


models_dir = 'models/qkeras_models/'
model_name = 'qkeras_model_2_4_2020_3layers_vol100'


# In[14]:


""" When using concatenated dataset ---> add feature 'dataset' after 'charge' and also one more input in NN """
#train data
X_train = np.array(df_train[['int_phi_e', 'int_eta_e', 'int_pt_r', 'charge']])
Y_train = np.array(df_train[['deltaPhi', 'deltaEta', 'deltaPt']])

#test data
X_test = np.array(df_test[['int_phi_e', 'int_eta_e', 'int_pt_r', 'charge']])
Y_test = np.array(df_test[['deltaPhi', 'deltaEta', 'deltaPt']])
#for later use
#X_test_orig = np.array(df_test[['phi_r', 'phi_e', 'eta_r', 'eta_e', 'pt_r', 'qual', 'charge']])


# In[17]:


#scaling/transforming input data => not normal distribution = MinMaxScaler
X_scaler = MinMaxScaler((0,1))
Y_scaler = StandardScaler((0,1))

#X_scaler = joblib.load(models_dir + model_name + '_X_scaler.dat')
#Y_scaler = joblib.load(models_dir + model_name + '_Y_scaler.dat')

X_train = X_scaler.fit_transform(X_train)
Y_train = Y_scaler.fit_transform(Y_train)

X_test = X_scaler.transform(X_test)
Y_test = Y_scaler.transform(Y_test)

#saving scaler
joblib.dump(X_scaler, models_dir + model_name + '_X_scaler.dat')
joblib.dump(Y_scaler, models_dir + model_name + '_Y_scaler.dat')


# In[ ]:


def custom_loss(y_true, y_pred):
    coefs = [7.718, 2.1184316, 1.7462137, 2.7549687, 4.7066404, 7.6163553, 11.723778]
    
    pt_true = y_true[2]
    loss_total = (y_pred - y_true + K.softplus(-2. * (y_pred - y_true)) - K.log(2.))
    loss = K.switch(tf.math.logical_and(tf.greater(pt_true, 3), tf.less(pt_true, 4)),         tf.math.multiply(loss_total, coefs[0]),         K.switch(tf.less(pt_true, 5), tf.math.multiply(loss_total, coefs[1]),         K.switch(tf.less(pt_true, 6), tf.math.multiply(loss_total, coefs[2]),         K.switch(tf.less(pt_true, 7), tf.math.multiply(loss_total, coefs[3]),         K.switch(tf.less(pt_true, 8), tf.math.multiply(loss_total, coefs[4]),         K.switch(tf.less(pt_true, 9), tf.math.multiply(loss_total, coefs[5]),         K.switch(tf.less(pt_true, 10), tf.math.multiply(loss_total, coefs[6]), loss_total)))))))

    loss = K.mean(loss)
    return loss

#keras.losses.custom_loss = custom_loss


# In[18]:


# QKeras parameters
bits = 16
integer = 8


# In[19]:


#network parameters
#k_reg = keras.regularizers.l1(1e-2)
#a_reg = keras.regularizers.l2(1e-1)
k_reg = None
a_reg = None
#constraint = keras.constraints.min_max_norm(0, 1)
num_nodes_h = 32


# In[20]:


# QDense model

""" When using concatenated dataset ---> add one more input = (5,)"""

inputs = Input(shape=(4,), name='inputs_0')
#i = QActivation("quantized_relu(8,8)", name="act_i")(inputs)
#hidden 1
hidden_layer = QDense(num_nodes_h, kernel_quantizer=quantized_bits(bits, integer),                 bias_quantizer=quantized_bits(bits, integer),                 kernel_regularizer=k_reg, activity_regularizer=a_reg,                 name="dense_2")(inputs)
hidden_layer = QBatchNormalization(name='bn_2')(hidden_layer)
hidden_layer = QActivation("quantized_relu(16,8)", name="relu_2")(hidden_layer)
# hidden 2
hidden_layer = QDense(num_nodes_h, kernel_quantizer=quantized_bits(bits, integer),                bias_quantizer=quantized_bits(bits, integer),                 kernel_regularizer=k_reg, activity_regularizer=a_reg,                 name="dense_3")(hidden_layer)
hidden_layer = QBatchNormalization(name='bn_3')(hidden_layer)
hidden_layer = QActivation("quantized_relu(16,8)", name="relu_3")(hidden_layer)
# hidden 3
hidden_layer = QDense(num_nodes_h, kernel_quantizer=quantized_bits(bits, integer),                bias_quantizer=quantized_bits(bits, integer),                 kernel_regularizer=k_reg, activity_regularizer=a_reg,                 name="dense_4")(hidden_layer)
hidden_layer = QBatchNormalization(name='bn_4')(hidden_layer)
hidden_layer = QActivation("quantized_relu(16,8)", name="relu_4")(hidden_layer)
# hidden 4
"""hidden_layer = QDense(num_nodes_h, kernel_quantizer=quantized_bits(bits, integer),                bias_quantizer=quantized_bits(bits, integer),                 kernel_regularizer=k_reg, activity_regularizer=a_reg,                 name="dense_5")(hidden_layer)
hidden_layer = QBatchNormalization(name='bn_5')(hidden_layer)
hidden_layer = QActivation("quantized_relu(16,8)", name="relu_5")(hidden_layer)"""


#outputs - 3 layers with 1 output
output_1 = QDense(1, name='phi', activation='linear')(hidden_layer)

output_2 = QDense(1, name='eta', activation='linear')(hidden_layer)

output_3 = QDense(1, name='p_t',  activation='linear')(hidden_layer)
#create a model
model = Model(inputs = inputs, outputs = [output_1, output_2, output_3])

model.summary()


# In[21]:


OPTIMIZER = Adam()
NUM_EPOCH = 500
BATCH_SIZE = 2**13
VERBOSE = 1
losses = ['logcosh', 'logcosh', 'logcosh']
#losses = ['logcosh']
#loss_weights=[1]
#losses = [custom_loss, custom_loss, custom_loss]
#losses = ['squared_hinge', 'squared_hinge', 'squared_hinge']
VALIDATION_PERC = 0.2

model.compile(loss=losses, optimizer=OPTIMIZER)


# In[22]:


class GetWeights(tf.keras.callbacks.Callback):
    # Keras callback which collects values of weights and biases at each epoch
    def __init__(self):
        super(GetWeights, self).__init__()
        self.weight_dict = {}

    def check_weight_range(self, weight):
        if np.min(weight) >= -1 and np.max(weight) <= 1:
            return True
        return False
    
    def on_epoch_end(self, epoch, logs=None):
        # this function runs at the end of each epoch

        # loop over each layer and get weights and biases
        for layer_i in range(1,len(self.model.layers)):
            if 'Activation' in self.model.layers[layer_i].__class__.__name__:
                continue
            w = self.model.layers[layer_i].get_weights()[0]
            #print("\nRange weights in -1 to 1: ", self.check_weight_range(w.flatten()))
            
            b = self.model.layers[layer_i].get_weights()[1]
            #print("\nRange bias in -1 to 1: ", self.check_weight_range(b.flatten()))
            #print('\nLayer %s has weights of shape %s and biases of shape %s' %(
                #layer_i, np.shape(w), np.shape(b)))

            # save all weights and biases inside a dictionary
            if epoch == 0:
                # create array to hold weights and biases
                self.weight_dict['w_'+str(layer_i+1)] = w
                self.weight_dict['b_'+str(layer_i+1)] = b
            else:
                # append new weights to previously-created weights array
                self.weight_dict['w_'+str(layer_i+1)] = np.dstack(
                    (self.weight_dict['w_'+str(layer_i+1)], w))
                # append new weights to previously-created weights array
                self.weight_dict['b_'+str(layer_i+1)] = np.dstack(
                    (self.weight_dict['b_'+str(layer_i+1)], b))


# In[23]:


#[np.array([Y_train[:,0], df_train['pTReco']]).T , np.array([Y_train[:,0], df_train['pTReco']]).T, np.array([Y_train[:,0], df_train['pTReco']]).T]
# remove this if don't want to store weights
gw = GetWeights()

callbacks = [keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True), gw]
#[Y_train[:,0], Y_train[:,1], Y_train[:,2]]
history = model.fit(X_train, [Y_train[:,0], Y_train[:,1], Y_train[:,2]], 
            callbacks=callbacks, epochs=NUM_EPOCH, batch_size=BATCH_SIZE, 
            validation_split=VALIDATION_PERC, shuffle=True, verbose=VERBOSE)


# In[24]:


save_model(models_dir + model_name, model)


# In[ ]:


for key,value in gw.weight_dict.items():
    print(str(key) + ' shape: %s' %str(np.shape(gw.weight_dict[key])))
    print(value)


# In[ ]:


""" WHEN WE HAVE 3 DENSE LAYERS, 1 OUTPUT EACH """
loss = history.history['phi_loss']
val_loss = history.history['val_phi_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
loss = history.history['eta_loss']
val_loss = history.history['val_eta_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
loss = history.history['p_t_loss']
val_loss = history.history['val_p_t_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


prediction = model.predict(X_test, batch_size=2**11)
#remove single-dimensional entries
prediction = np.squeeze(prediction)
prediction = np.array(prediction).T


# In[ ]:


#inverse transform of prediction
pred = Y_scaler.inverse_transform(prediction)

delta_phi_pred = pred[:,0]
delta_eta_pred = pred[:,1]
delta_pt_pred = pred[:,2]

#calculate phi_pred,eta_pred,pt_pred
phi_pred = calculate_phi(df_test['phiVtxL1'], delta_phi_pred)
eta_pred = np.array(df_test['etaVtxL1'] - delta_eta_pred)
pt_pred = np.array(df_test['pTL1'] - delta_pt_pred)


#calculate difference between prediction and true(reconstructed) values of phi,eta,pt
delta_phi_p = calculate_delta_phi(phi_pred, df_test['phiReco'])
delta_eta_p = np.array(eta_pred - df_test['etaReco'])
delta_pt_p = np.array((pt_pred - df_test['pTReco'])/df_test['pTReco'])


# In[ ]:


def get_range_99_perc(data):
    r = 0.1
    while r < 2:
        cnt = 0
        for d in data:
            if d >= -r and d <= r:
                cnt += 1
        
        if cnt / len(data) >= 0.99:
            print(round(r,2))
            break
            
        r += 0.1
        
    return r


# In[ ]:


# check range for delta_phi and delta_phi_predicted
range_phi = get_range_99_perc(df_test['deltaPhi'])
range_phi_pred = get_range_99_perc(delta_phi_p)
range_phi = max(range_phi, range_phi_pred)

# check range for delta_eta and delta_eta_predicted
range_eta = get_range_99_perc(df_test['deltaEta'])
range_eta_pred = get_range_99_perc(delta_eta_p)
range_eta = max(range_eta, range_eta_pred)

# check range for delta_pt and delta_pt_predicted
range_pt = get_range_99_perc(df_test['deltaPt']/df_test['pTReco'])
range_pt_pred = get_range_99_perc(delta_pt_p)
range_pt = max(range_pt, range_pt_pred)


# In[ ]:


#plotting difference between extrapolated - true values VS. predicted - true values
#values = delta phi, delta eta, delta pt
#better results when p-true has smaller std and higher peek

plt.hist(df_test['deltaPhi'], bins=100, alpha=0.5, histtype='step',          fill=False, range=(-range_phi,range_phi))
plt.hist(delta_phi_p, bins=100, alpha=0.5, histtype='step', fill=False, range=(-range_phi,range_phi))
plt.title('delta phi distribution')
plt.legend(['ext - true', 'pred - true'])
plt.show()

plt.hist(df_test['deltaEta'], bins=100, alpha=0.5, histtype='step',          fill=False, range=(-range_eta,range_eta))
plt.hist(delta_eta_p, bins=100, alpha=0.5, histtype='step', fill=False,          range=(-range_eta,range_eta))
plt.title('delta eta distribution')
plt.legend(['ext - true', 'pred - true'])
plt.show()

plt.hist(df_test['deltaPt']/df_test['pTReco'], bins=100, alpha=0.5, histtype='step',          fill=False, range=(-range_pt,range_pt))
plt.hist(delta_pt_p, bins=100, alpha=0.5, histtype='step', fill=False, range=(-range_pt,range_pt))
plt.title('delta pt distribution')
plt.legend(['reco - true', 'pred - true'])
plt.show()


# In[ ]:


from scipy.stats import norm

print(norm.fit(delta_phi_p))
print(norm.fit(delta_eta_p))
print(norm.fit(delta_pt_p))


# In[ ]:


data = {
    'delta_phi_pred': delta_phi_p,
    'delta_eta_pred': delta_eta_p,
    'delta_pt_pred': delta_pt_p,
}

dataframe = pd.DataFrame (data, columns = ['delta_phi_pred','delta_eta_pred', 'delta_pt_pred'])
dataframe.to_csv(r'predicted_deltas_QKeras_all_2_4_2020_3layers.csv', index=False)


# In[ ]:





# In[ ]:




