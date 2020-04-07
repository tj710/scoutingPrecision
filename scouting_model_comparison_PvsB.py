#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, '/home/epuljak/muon_scouting/SDK/')
import os
os.environ['TF_KERAS'] = '1'
import numpy as np
import SDK
import keras
from keras.layers import Input, Reshape, Concatenate, Dense
from keras.models import Model, model_from_json
import onnx
import onnxmltools
import fwdnxt
import pandas as pd
import joblib
from matplotlib import pyplot as plt
import math
import tensorflow as tf
#from qkeras import utils as ut


# In[2]:


def clear_pairs_dataset(df_pairs):
    df_pairs['remove_row'] = [True if abs(phi_r_even - phi_r_odd) < 0.0001 else False                               for phi_r_even, phi_r_odd in zip(df_pairs['phi_r'], df_pairs['phi_r'].shift())]
    
    indices_to_exclude = df_pairs.index[(df_pairs['remove_row'] == True) |                             (df_pairs['pt_r'] < 2.5) | (df_pairs['pt_r'] > 45) |                             (df_pairs['eta_r'] < -2.45) | (df_pairs['eta_r'] > 2.45) |                             (df_pairs['eta_e'] < -2.45) | (df_pairs['eta_e'] > 2.45)].tolist()
    
    indices_to_exclude = set(indices_to_exclude)
    additional_odd = set([x + 1 for x in indices_to_exclude if x % 2 == 0])
    additional_even = set([x - 1 for x in indices_to_exclude if x % 2 == 1])
    indices_to_exclude.update(additional_odd)
    indices_to_exclude.update(additional_even)
    
    clean_df = df_pairs.drop(indices_to_exclude)
    
    return clean_df


# In[3]:


def read_moun_dataset(filename):
    muon_pairs_dataset = pd.read_hdf(filename)

    if 'phi' in muon_pairs_dataset.columns:
        muon_pairs_dataset.rename(inplace=True, 
                                  columns={'phi':'phi_r','phie':'phi_e','eta':'eta_r','etae':'eta_e','pt':'pt_r'})

    has_true = False
    if set(['phi_t','eta_t','pt_t']).issubset(muon_pairs_dataset.columns):
        has_true = True

    return clear_pairs_dataset(muon_pairs_dataset), has_true


# In[4]:


def load_model(model_name):
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_name + '.h5')
    
    return model


# In[5]:


def parse_board_results(result, imgs_per_cluster, outputs):
    phi = np.array([x for i, x in enumerate(result) if i % outputs == 0])
    eta = np.array([x for i, x in enumerate(result) if (i - 1) % outputs == 0])
    pt = np.array([x for i, x in enumerate(result) if (i - 2) % outputs == 0])
        
    return phi, eta, pt


# In[6]:


def get_predictions_board(X_pairs, model_filepath, imgs_per_cluster=1000, sw=False):
    phi = []
    eta = []
    pt = []
    
    iterations = int(len(X_pairs) / imgs_per_cluster)
    
    #ie = microndla.MDLA()
    
    ie = fwdnxt.FWDNXT()
    ie.SetFlag('debug', 'wb')
    ie.SetFlag("options", "V")
    ie.SetFlag('imgs_per_cluster', str(imgs_per_cluster))
    swnresults = ie.Compile('4x1x1', model_filepath, 'save.bin', 1, 1)
    nresults = ie.Init('save.bin', '')
    result = np.ndarray(swnresults, dtype=np.float32)
    
    start = 0
    end = 0
    
    for iter_i in range(0, len(X_pairs), imgs_per_cluster):
        start = iter_i
        end = start + imgs_per_cluster
        print(iter_i)
        print(start)
        print(end)
        
        current_batch = np.ascontiguousarray(np.swapaxes(np.expand_dims(                            np.expand_dims(X_pairs[start:end, :], 0), 0), 0, 2), dtype=np.float32())
        
        print(current_batch.shape)
        print(current_batch[0][0][0])
        
        result = np.ndarray(swnresults, dtype=np.float32)
        
        ie.Run_sw(current_batch, result)
        
        cur_phi_pred, cur_eta_pred, cur_pt_pred = parse_board_results(result, imgs_per_cluster, outputs=3)
        
        phi.extend(cur_phi_pred)
        eta.extend(cur_eta_pred)
        pt.extend(cur_pt_pred)
        
    ie.Free()
     
    return phi, eta, pt


# In[7]:


def calculate_delta_phi(phi_approx, phi_true):
    return np.array([x - 2*math.pi if x > math.pi else x + 2*math.pi if x < -math.pi else x for x in phi_approx - phi_true])

def calculate_phi(phi_approx, delta_phi):
    """array = []
    for approx, dphi in zip(phi_approx, delta_phi):
        array.append(approx - dphi)
    array = np.reshape(array, (len(array),1))"""
    return np.array([x - 2*math.pi if x > math.pi else x + 2*math.pi if x < -math.pi else x for x in phi_approx - delta_phi])


# In[8]:


def print_stats(values):
    print(np.min(values), np.max(values))


# In[9]:


model_name = 'models/scouting_models/integer_scouting_1_4_2020_3layers'


# In[10]:


model = load_model(model_name)
model.summary()


# In[11]:


imgs_per_cluster = 1000
n = 1000000

muon_pairs_dataset, _ = read_moun_dataset('/home/dgolubov/scout_325172.hd5')
#muon_pairs_dataset = pd.read_csv('train_set_tight.csv')
muon_pairs_dataset = muon_pairs_dataset[:n]


# In[12]:


print(muon_pairs_dataset)


# In[13]:


# check if quality is always the same
def check_quality(dataset):
    if (dataset['qual'] == 12).all(): return True
    else:
        print(dataset[dataset['qual'] != 12]['qual'].shape)
        return False

print(check_quality(muon_pairs_dataset))


# In[14]:


def add_integer_values_to_df(df_base):
    #new_phi_r = np.array([x if x > 0 else x + 2*math.pi for x in df_base['phi_r']])
    new_phi_e = np.array([x if x > 0 else x + 2*math.pi for x in df_base['phi_e']])
    
    #df_base['int_phi_r'] = (new_phi_r / (2*math.pi / 576)).astype(int)
    df_base['int_phi_e'] = (new_phi_e / (2*math.pi / 576)).astype(int)
    
    #df_base['int_eta_r'] = (df_base['eta_r'] / 0.010875).astype(int)
    df_base['int_eta_e'] = (df_base['eta_e'] / 0.010875).astype(int)

    df_base['int_pt_r'] = (df_base['pt_r'] / 0.5).astype(int)
    df_base['charge'] = (df_base['charge']).astype(int)
    return df_base


# In[15]:


muon_pairs_dataset = add_integer_values_to_df(muon_pairs_dataset)

print_stats(muon_pairs_dataset['int_phi_e'])
print_stats(muon_pairs_dataset['int_eta_e'])
print_stats(muon_pairs_dataset['int_pt_r'])
print_stats(muon_pairs_dataset['charge'])


# In[16]:


X_pairs = np.array(muon_pairs_dataset[['int_phi_e','int_eta_e','int_pt_r','charge']],                   dtype=np.float32)


# In[17]:


for i in range(len(model.layers)):
    print(model.layers[i].__class__.__name__)
    for weight in model.layers[i].get_weights():
        print(weight.shape)
        print('min:', np.min(weight), 'max:', np.max(weight), 'mean:', np.mean(weight),'std:', np.std(weight))


# In[18]:


def get_predictions_python(X_pairs, model):
    Y_out = model.predict(X_pairs, batch_size=1024)
    
    phi = Y_out[0]
    eta = Y_out[1]
    pt = Y_out[2]
    
    return phi, eta, pt

def reconstruct_original_values(muon_pairs_dataset,                                 delta_phi_pred_delta_pairs, delta_eta_pred_delta_pairs, delta_pt_pred_delta_pairs):
    phi_pred_delta_pairs = calculate_phi(muon_pairs_dataset['phi_e'], delta_phi_pred_delta_pairs)    
    eta_pred_delta_pairs = np.array(muon_pairs_dataset['eta_e'] - delta_eta_pred_delta_pairs)
    pt_pred_delta_pairs = np.array(muon_pairs_dataset['pt_r'] - delta_pt_pred_delta_pairs)
    
    return phi_pred_delta_pairs, eta_pred_delta_pairs, pt_pred_delta_pairs


# In[19]:


delta_phi_python, delta_eta_python, delta_pt_python = get_predictions_python(X_pairs, model)
#y_predicted = get_predictions_python(X_pairs, model)
#print(y_predicted.shape)
delta_phi_python = np.squeeze(delta_phi_python)
delta_eta_python = np.squeeze(delta_eta_python)
delta_pt_python = np.squeeze(delta_pt_python)
phi, eta, pt = reconstruct_original_values(muon_pairs_dataset, delta_phi_python, delta_eta_python, delta_pt_python)
print(phi)
print(eta)
print(pt)


# In[ ]:


phi_board, eta_board, pt_board = get_predictions_board(X_pairs, model_name + '.onnx', imgs_per_cluster=1000)


# In[ ]:


res_python = np.stack((phi_python, eta_python, pt_python)).T
res_board = np.stack((phi_board, eta_board, pt_board)).T


# In[ ]:


np.savetxt('28_3_2020_scout_output_python.csv', res_python, delimiter=',')
np.savetxt('28_3_2020_scout_output_board.csv', res_board, delimiter=',')


# In[ ]:




