#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interpn
import pandas as pd
import seaborn as sns
import math
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Input, Activation, Multiply, Lambda
from keras.layers.normalization import BatchNormalization
import keras.optimizers
from keras import regularizers
import keras.backend as K
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import load_model, model_from_json
from IPython.display import clear_output
import time
import matplotlib
import keras
from scipy.signal import argrelextrema
import joblib
from scipy.stats import norm
import tensorflow as tf


# In[2]:


print(np.__version__)
print(tf.__version__)


# In[3]:


print(K.tensorflow_backend._get_available_gpus())


# In[4]:


def save_model(model_save_name, model):
    with open(model_save_name + '.json', 'w') as json_file:
        json_file.write(model.to_json())

    model.save_weights(model_save_name + '.h5')


# In[5]:


def load_model(model_name):
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_name + '.h5')
    
    return model


# In[6]:


def calculate_delta_phi(phi_approx, phi_true):
    return np.array([x - 2*math.pi if x > math.pi else x + 2*math.pi if x < -math.pi else x                           for x in phi_approx - phi_true])

def calculate_phi(phi_approx, delta_phi):
    return np.array([x - 2*math.pi if x > math.pi else x + 2*math.pi if x < -math.pi else x                           for x in phi_approx - delta_phi])


# In[7]:


models_dir = '../models/scouting_models/'
model_name = 'integer_scouting_7_4_2020_3layers'


# In[10]:


#load data - BARREL, ENDCAP, OVERLAP

#barrel
df_1= pd.read_csv('../../mu_barrel_tight_zb.csv', index_col=False, delimiter=',')
df_2= pd.read_csv('../../mu_barrel_tight_zb2018D.csv', index_col=False, delimiter=',')
#df_base = pd.concat([df_1, df_2], ignore_index=True)

# add the additional feature in the dataset (1 = barrel, 2 = endcap, 3 = overlap)
feature = 1
additional_array = feature*np.ones(df_1.shape[0], dtype=int)
df_1['dataset'] = additional_array

feature = 1
additional_array = feature*np.ones(df_2.shape[0], dtype=int)
df_2['dataset'] = additional_array

#endcap
df_3= pd.read_csv('../../mu_endcap_tight_zb.csv', index_col=False, delimiter=',')
df_4= pd.read_csv('../../mu_endcap_tight_zb2018D.csv', index_col=False, delimiter=',')
#df_base = pd.concat([df_3, df_4], ignore_index=True)

feature = 2
additional_array = feature*np.ones(df_3.shape[0], dtype=int)
df_3['dataset'] = additional_array

feature = 2
additional_array = feature*np.ones(df_4.shape[0], dtype=int)
df_4['dataset'] = additional_array

#overlap
df_5= pd.read_csv('../../mu_overlap_tight_zb.csv', index_col=False, delimiter=',')
df_6= pd.read_csv('../../mu_overlap_tight_zb2018D.csv', index_col=False, delimiter=',')
#df_base = pd.concat([df_5, df_6], ignore_index =True)

feature = 3
additional_array = feature*np.ones(df_5.shape[0], dtype=int)
df_5['dataset'] = additional_array

feature = 3
additional_array = feature*np.ones(df_6.shape[0], dtype=int)
df_6['dataset'] = additional_array

df_base = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6], ignore_index = True)

print(df_base.shape)


# In[11]:


#p_t restriction from 2.5 to 45 and quality = 12
df_base = df_base.drop(df_base[df_base.pTL1 < 2.5].index)
df_base = df_base.drop(df_base[df_base.pTL1 > 45].index)

df_base = df_base.drop(df_base[df_base.quality != 12].index)
df_base = df_base.drop(df_base[df_base.quality != 12].index)

print(df_base.shape)


# In[12]:


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


# In[13]:


def add_integer_values_to_df(df_base):
    #new_phi_r = np.array([x if x > 0 else x + 2*math.pi for x in df_base['phi_r']])
    new_phi_e = np.array([x if x > 0 else x + 2*math.pi for x in df_base['phiVtxL1']])
    
    #df_base['int_phi_r'] = (new_phi_r / (2*math.pi / 576)).astype(int)
    df_base['int_phi_e'] = (new_phi_e / (2*math.pi / 576)).astype(int)
    
    #df_base['int_eta_r'] = (df_base['eta_r'] / 0.010875).astype(int)
    df_base['int_eta_e'] = (df_base['etaVtxL1'] / 0.010875).astype(int)

    df_base['int_pt_r'] = (df_base['pTL1'] / 0.5).astype(int)
    return df_base


# In[14]:


df_base = add_integer_values_to_df(df_base)


# In[15]:


train_perc = 0.8
train_len = int(train_perc * len(df_base))
df_train = df_base[:train_len]
df_test = df_base[train_len:]

print(df_train.shape)
print(df_test.shape)


# In[ ]:


""" FLOAT VALUES AS INPUTS"""
#train data
X_train = np.array(df_train[['phiVtxL1', 'etaVtxL1', 'pTL1', 'quality', 'charge', 'dataset']])
Y_train = np.array(df_train[['deltaPhi', 'deltaEta', 'deltaPt']])

#test data
X_test = np.array(df_test[['phiVtxL1', 'etaVtxL1', 'pTL1', 'quality', 'charge', 'dataset']])
Y_test = np.array(df_test[['deltaPhi', 'deltaEta', 'deltaPt']])


# In[16]:


""" INTEGER VALUES AS INPUTS"""
X_train = np.array(df_train[['int_phi_e', 'int_eta_e', 'int_pt_r', 'charge']])
# for BARREL, ENDCAP AND OVERLAP dataset
Y_train = np.array(df_train[['deltaPhi', 'deltaEta', 'deltaPt']])

X_test = np.array(df_test[['int_phi_e', 'int_eta_e', 'int_pt_r','charge']])
# for BARREL, ENDCAP AND OVERLAP dataset
Y_test = np.array(df_test[['deltaPhi', 'deltaEta', 'deltaPt']])


# In[17]:


X_scaler = MinMaxScaler((0,1))
Y_scaler = StandardScaler((0,1))

#X_scaler = joblib.load(models_dir + model_name + '_X_scaler.dat')
#Y_scaler = joblib.load(models_dir + model_name + '_Y_scaler.dat')

X_train = X_scaler.fit_transform(X_train)
Y_train = Y_scaler.fit_transform(Y_train)

X_test = X_scaler.transform(X_test)
Y_test = Y_scaler.transform(Y_test)

joblib.dump(X_scaler, models_dir + model_name + '_X_scaler.dat')
joblib.dump(Y_scaler, models_dir + model_name + '_Y_scaler.dat')


# In[40]:


# helper code for scaling the dataset used for micron board inference
"""X_scaler = joblib.load(models_dir + model_name + '_X_scaler.dat')
X_pairs = pd.read_csv('../X_pairs_for_inference.csv')

X_pairs = np.array(X_pairs[['int_phi_e','int_eta_e','int_pt_r','charge']],dtype=np.float32)

X_test = X_scaler.transform(X_pairs)
print(X_test)

dataset = pd.DataFrame({'int_phi_e': X_test[:,0], 'int_eta_e': X_test[:,1], 'int_pt_r': X_test[:,2], 'charge': X_test[:,3]})

print(dataset)
dataset.to_csv('X_pairs_scaled.csv', index=False)
#Y_scaler = joblib.load(models_dir + model_name + '_Y_scaler.dat')"""


# In[18]:


#k_reg = regularizers.l2(0.001)
#a_reg = regularizers.l2(0.001)
k_reg = None
a_reg = None
constraint = keras.constraints.min_max_norm(0, 1)
hl_nodes = 32


# In[19]:


inputs = Input(shape=(4,), name='input_0')

hidden_layers = Dense(hl_nodes, name='dense_1', kernel_regularizer=k_reg, activity_regularizer=a_reg,                       kernel_constraint=constraint, bias_constraint=constraint)(inputs)
hidden_layers = BatchNormalization(name='bn_1', gamma_constraint=constraint,                     beta_constraint=constraint)(hidden_layers)
hidden_layers = Activation('relu', name='relu_1')(hidden_layers)

hidden_layers = Dense(hl_nodes, name='dense_2', kernel_regularizer=k_reg, activity_regularizer=a_reg,                      kernel_constraint=constraint, bias_constraint=constraint)(hidden_layers)
hidden_layers = BatchNormalization(name='bn_2', gamma_constraint=constraint,                     beta_constraint=constraint)(hidden_layers)
hidden_layers = Activation('relu', name='relu_2')(hidden_layers)

hidden_layers = Dense(hl_nodes, name='dense_3', kernel_regularizer=k_reg, activity_regularizer=a_reg,                      kernel_constraint=constraint, bias_constraint=constraint)(hidden_layers)
hidden_layers = BatchNormalization(name='bn_3', gamma_constraint=constraint,                     beta_constraint=constraint)(hidden_layers)
hidden_layers = Activation('relu', name='relu_3')(hidden_layers)

out_phi = Dense(1, name='phi', activation='linear')(hidden_layers)
out_eta = Dense(1, name='eta', activation='linear')(hidden_layers)
out_pt = Dense(1, name='pt', activation='linear')(hidden_layers)

#out = Dense(3, name='out', activation='linear')(hidden_layers)

model = Model(inputs=inputs, outputs=[out_phi, out_eta, out_pt])
#model = Model(inputs=inputs, outputs=[out])

model.summary()


# In[20]:


n_epochs = 500
batch_size = 2**13
opt = keras.optimizers.Adadelta()
losses = ['logcosh','logcosh', 'logcosh']
#losses = ['logcosh']

#losses = ['msle','msle', 'msle']
#loss_weights=[2, 1, 20]
loss_weights = [1,1,1]

callbacks = [keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]
#[Y_train[:, [0]], Y_train[:, [1]], Y_train[:, [2]]]
model.compile(loss=losses, optimizer=opt)
history=model.fit(X_train, [Y_train[:, [0]], Y_train[:, [1]], Y_train[:, [2]]], callbacks=callbacks,             epochs=n_epochs, batch_size=batch_size, validation_split=0.2, shuffle=True, verbose=1)
#history=model.fit(X_train, Y_train, callbacks=callbacks, \
#            epochs=n_epochs, batch_size=batch_size, validation_split=0.2, shuffle=True, verbose=1)
 
save_model(models_dir + model_name, model)


# In[ ]:


#ev_8192 = model.evaluate(X_test,[Y_test[:, [0]], Y_test[:, [1]], Y_test[:, [2]]], batch_size=8192)
#ev_2048 = model.evaluate(X_test,[Y_test[:, [0]], Y_test[:, [1]], Y_test[:, [2]]], batch_size=2048)
#ev_1024 = model.evaluate(X_test,[Y_test[:, [0]], Y_test[:, [1]], Y_test[:, [2]]], batch_size=1024)
#ev_128 = model.evaluate(X_test,[Y_test[:, [0]], Y_test[:, [1]], Y_test[:, [2]]], batch_size=128)
# ev_32 = model.evaluate(X_test,[Y_test[:, [0]], Y_test[:, [1]], Y_test[:, [2]]], batch_size=32)


# In[21]:


Y_out = model.predict(X_test, batch_size=1024)
Y_out = np.squeeze(Y_out)
Y_out = np.array(Y_out).T
print(Y_out.shape)
#print(Y_test.shape)

#save_model(models_dir + model_name, pruned_model)


# In[ ]:


""" used when inputs are INTEGER """
#from int inputs to float
input_phi = df_test['int_phi_e']
step = 0.0109
float_phi_e = np.array([x*step for x in input_phi])
print(float_phi_e.shape)

int_eta = df_test['int_eta_e']
eta_range_int = create_range_int_float(start=-2.436, stop=2.4360001, step=0.010875, before=-2.446875, after=2.446875)
new_eta_range = dict()
for key, val in eta_range_int.items():
    new_eta_range[key - 225] = val
float_eta_e = np.array([new_eta_range[int_value] for int_value in int_eta]) 
print(float_eta_e.shape)

int_pt = df_test['int_pt_r']
pt_range_int = create_range_int_float(start=0, stop=45, step=0.5, after=45)
float_pt_r = np.array([pt_range_int[int_value] for int_value in int_pt]) 
print(float_pt_r.shape)


# In[22]:


Y_pred = Y_scaler.inverse_transform(Y_out)
#Y_pred = Y_out.T
delta_phi_pred = Y_pred[:,0]
delta_eta_pred = Y_pred[:,1]
delta_pt_pred = Y_pred[:,2]

#calculate phi_pred,eta_pred,pt_pred
phi_pred = calculate_phi(df_test['phiVtxL1'], delta_phi_pred)
eta_pred = np.array(df_test['etaVtxL1'] - delta_eta_pred)
pt_pred = np.array(df_test['pTL1'] - delta_pt_pred)


#calculate difference between prediction and true values of phi,eta,pt
delta_phi_p = calculate_delta_phi(phi_pred, df_test['phiReco'])
delta_eta_p = np.array(eta_pred - df_test['etaReco'])
delta_pt_p = np.array((pt_pred - df_test['pTReco']))
delta_ptoverp = np.array((pt_pred - df_test['pTReco'])/df_test['pTReco'])

print(delta_phi_pred)


# In[23]:


#print('Baseline:')
#print(norm.fit(df_test['delta_phi_e']))
#print(norm.fit(df_test['delta_eta_e']))
#print(norm.fit(df_test['delta_pt_r']))

print('\nNeural Net:')
print(norm.fit(delta_phi_p))
print(norm.fit(delta_eta_p))
print(norm.fit(delta_pt_p))


# In[25]:


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


# In[26]:


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
range_pt_pred = get_range_99_perc(delta_ptoverp)
range_pt = max(range_pt, range_pt_pred)

range_pt = get_range_99_perc(delta_pt_p)


# In[27]:


""" BARREL, ENDCAP OR OVERLAP DATA """
#plotting difference between extrapolated - true values VS. predicted - true values
#values = delta phi, delta eta, delta pt
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
plt.hist(delta_ptoverp, bins=100, alpha=0.5, histtype='step', fill=False, range=(-range_pt,range_pt))
plt.title('delta pt distribution')
plt.legend(['reco - true', 'pred - true'])
plt.show()


# In[29]:


data = {
    'delta_phi_pred': delta_phi_p,
    'delta_eta_pred': delta_eta_p,
    'delta_pt_pred': delta_pt_p,
    'delta_ptoverp': delta_ptoverp
}

dataframe = pd.DataFrame (data, columns = ['delta_phi_pred','delta_eta_pred', 'delta_pt_pred', 'delta_ptoverp'])
dataframe.to_csv(r'NN_dataset_all_3layers.csv', index=False)


# In[ ]:


#################################


# In[ ]:


range_val_phi = 0.5

print('phi bins mean std')
sns.regplot(x=df_test['phi_t'], y=df_test['delta_phi_e'],             x_bins=10, fit_reg=None, ci='sd').set(ylim=(-range_val_phi,range_val_phi),title='phi (ext-true)/true')
plt.show()
sns.regplot(x=df_test['phi_t'], y=delta_phi_p, x_bins=10, fit_reg=None, ci='sd').set(                                                                    ylim=(-range_val_phi,range_val_phi),ylabel='',
                                                                    title='phi (pred-true)/true')
plt.show()


# In[ ]:


range_val_eta = 0.1

print('eta bins mean std')
sns.regplot(x=df_test['eta_t'], y=df_test['delta_eta_e'],             x_bins=10, fit_reg=None, ci='sd').set(ylim=(-range_val_eta,range_val_eta),title='eta (ext-true)/true')
plt.axhline(y=0, color='#000000', linestyle='--')
plt.show()
sns.regplot(x=df_test['eta_t'], y=delta_eta_p, x_bins=10, fit_reg=None, ci='sd').set(                                                                    ylim=(-range_val_eta,range_val_eta),ylabel='',
                                                                    title='eta (pred-true)/true')
plt.axhline(y=0, color='#000000', linestyle='--')
plt.show()


# In[ ]:


range_val_pt = 1

print('pt bins mean std')
sns.regplot(x=df_test['pt_t'], y=df_test['delta_pt_r'] / df_test['pt_t'],             x_bins=10, fit_reg=None, ci='sd').set(ylim=(-range_val_pt,range_val_pt),title='pt (reco-true)/true')
plt.axhline(y=0, color='#000000', linestyle='--')
plt.show()

sns.regplot(x=df_test['pt_t'], y=delta_pt_p / df_test['pt_t'], x_bins=10, fit_reg=None, ci='sd').set(                                                                    ylim=(-range_val_pt,range_val_pt),ylabel='',
                                                                    title='pt (pred-true)/true')
plt.axhline(y=0, color='#000000', linestyle='--')
plt.show()


# Masses plot

# In[ ]:


def get_pt_vectors(phi, eta, pt_val):
    theta = 2 * np.arctan(np.power(math.e, -eta))
    px = pt_val * np.cos(phi)
    py = pt_val * np.sin(phi)
    #ptz = pt_val * 2 * np.arctan(np.power(math.e, -eta))
    pz = pt_val / np.sin(theta) * np.cos(theta)
    
    return px, py, pz


def get_mass(E, p):
    return np.sqrt(E*E - p*p)


def get_e_p(muon_1, muon_2):
    px = muon_1[0] + muon_2[0]
    py = muon_1[1] + muon_2[1]
    pz = muon_1[2] + muon_2[2]
    
    E = np.sqrt(np.square(muon_1[0]) + np.square(muon_1[1]) + np.square(muon_1[2])) +         np.sqrt(np.square(muon_2[0]) + np.square(muon_2[1]) + np.square(muon_2[2])) + 0.01
    # Adding 0.01 to prevent E < p
    
    p = np.sqrt(np.square(px) + np.square(py) + np.square(pz))
    
    return E, p


def get_mass_combined(phi_m1, eta_m1, pt_m1, phi_m2, eta_m2, pt_m2):
    pt_vectors_even = get_pt_vectors(phi_m1, eta_m1, pt_m1)
    pt_vectors_odd = get_pt_vectors(phi_m2, eta_m2, pt_m2)
    E, p = get_e_p(pt_vectors_even, pt_vectors_odd)
    return get_mass(E, p)
    


# In[ ]:


muon_pairs_dataset = pd.read_hdf('/home/dgolubov/datasets/scout_325172.hd5')
if 'phi' in muon_pairs_dataset.columns:
    muon_pairs_dataset.rename(inplace=True, 
                              columns={'phi':'phi_r','phie':'phi_e','eta':'eta_r','etae':'eta_e','pt':'pt_r'})
    
has_true = False
if set(['phi_t','eta_t','pt_t']).issubset(muon_pairs_dataset.columns):
    has_true = True
    
print(has_true)


# In[ ]:


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


# In[ ]:


muon_pairs_dataset = clear_pairs_dataset(muon_pairs_dataset)


# In[ ]:


even_indices = np.arange(0, len(muon_pairs_dataset), 2)
odd_indices = np.arange(1, len(muon_pairs_dataset), 2)

muon_df_even = muon_pairs_dataset.iloc[even_indices]
muon_df_odd = muon_pairs_dataset.iloc[odd_indices]

print(len(muon_df_even))
print(len(muon_df_odd))


# In[ ]:


new_phi_r = np.array([x if x > 0 else x + 2*math.pi for x in muon_pairs_dataset['phi_r']])
new_phi_e = np.array([x if x > 0 else x + 2*math.pi for x in muon_pairs_dataset['phi_e']])

int_phi_r = get_int_values_from_detector(new_phi_r, range_len=576)
int_phi_e = get_int_values_from_detector(new_phi_e, range_len=576)

int_eta_r = get_int_values_from_detector(muon_pairs_dataset['eta_r'], eta_range)
int_eta_e = get_int_values_from_detector(muon_pairs_dataset['eta_e'], eta_range)

int_pt_r = get_int_values_from_detector(muon_pairs_dataset['pt_r'], pt_range)

muon_pairs_dataset['int_phi_r'] = int_phi_r
muon_pairs_dataset['int_phi_e'] = int_phi_e
muon_pairs_dataset['int_eta_r'] = int_eta_r
muon_pairs_dataset['int_eta_e'] = int_eta_e
muon_pairs_dataset['int_pt_r'] = int_pt_r


# In[ ]:


print_stats(muon_pairs_dataset['int_phi_r'])
print_stats(muon_pairs_dataset['int_phi_e'])
print_stats(muon_pairs_dataset['int_eta_r'])
print_stats(muon_pairs_dataset['int_eta_e'])
print_stats(muon_pairs_dataset['int_pt_r'])
print_stats(muon_pairs_dataset['qual'])
print_stats(muon_pairs_dataset['charge'])


# In[ ]:


if has_true:
    pt_vectors_even_t = get_pt_vectors(np.array(muon_df_even.phi_t), np.array(muon_df_even.eta_t), 
                                       np.array(muon_df_even.pt_t))
    pt_vectors_odd_t = get_pt_vectors(np.array(muon_df_odd.phi_t), np.array(muon_df_odd.eta_t),
                                      np.array(muon_df_odd.pt_t))
    E_t, p_t = get_e_p(pt_vectors_even_t, pt_vectors_odd_t)
    masses_true = get_mass(E_t, p_t)

#pt_vectors_even_r = get_pt_vectors(np.array(muon_df_even.phi_r), np.array(muon_df_even.eta_r), 
#                                   np.array(muon_df_even.pt_r))
#pt_vectors_odd_r = get_pt_vectors(np.array(muon_df_odd.phi_r), np.array(muon_df_odd.eta_r), 
#                                  np.array(muon_df_odd.pt_r))
#E_r, p_r = get_e_p(pt_vectors_even_r, pt_vectors_odd_r)
#masses_reco = get_mass(E_r, p_r)
masses_reco = get_mass_combined(np.array(muon_df_even.phi_r), np.array(muon_df_even.eta_r), 
                                np.array(muon_df_even.pt_r), np.array(muon_df_odd.phi_r), 
                                np.array(muon_df_odd.eta_r), np.array(muon_df_odd.pt_r))

#pt_vectors_even_e = get_pt_vectors(np.array(muon_df_even.phi_e), np.array(muon_df_even.eta_e), 
#                                   np.array(muon_df_even.pt_r))
#pt_vectors_odd_e = get_pt_vectors(np.array(muon_df_odd.phi_e), np.array(muon_df_odd.eta_e), 
#                                  np.array(muon_df_odd.pt_r))
#E_e, p_e = get_e_p(pt_vectors_even_e, pt_vectors_odd_e)
#masses_ext = get_mass(E_e, p_e)
masses_ext = get_mass_combined(np.array(muon_df_even.phi_e), np.array(muon_df_even.eta_e), 
                                np.array(muon_df_even.pt_r), np.array(muon_df_odd.phi_e), 
                                np.array(muon_df_odd.eta_e), np.array(muon_df_odd.pt_r))


# In[ ]:


if has_true:
    print((masses_true))
print((masses_reco))
print((masses_ext))


# In[ ]:


#X_pairs = np.array(muon_pairs_dataset[['phi_r', 'phi_e', 'eta_r', 'eta_e', 'pt_r', 'qual', 'charge']])
X_pairs = np.array(muon_pairs_dataset[['int_phi_r','int_phi_e','int_eta_r','int_eta_e','int_pt_r','qual','charge']])

X_pairs = X_scaler.transform(X_pairs)
Y_out_delta_pairs = model.predict(X_pairs, batch_size=4096)


# In[ ]:


#print(Y_out_delta_2_pairs.shape)
Y_out_delta_pairs = np.array(Y_out_delta_pairs).squeeze(-1).T
print(Y_out_delta_2_pairs.shape)
Y_out_delta_pairs = Y_scaler.inverse_transform(Y_out_delta_pairs)
print(Y_out_delta_pairs.shape)


# In[ ]:


# delta_phi_pred_delta_2_pairs = Y_out_delta_2_pairs[0]
# eta_pred_delta_2_pairs = Y_out_delta_2_pairs[1]
# pt_pred_delta_2_pairs = Y_out_delta_2_pairs[2]

# phi_pred_delta_2_pairs = calculate_phi(muon_pairs_dataset['phi_e'], delta_phi_pred_delta_2_pairs)

print(Y_out_delta_pairs[:,0])
print(Y_out_delta_pairs[:,1])
print(Y_out_delta_pairs[:,2])


# In[ ]:


phi_pred_delta_2_pairs = calculate_phi(muon_pairs_dataset['phi_e'], Y_out_delta_pairs[:,0])
eta_pred_delta_2_pairs = np.array(muon_pairs_dataset['eta_e'] - Y_out_delta_pairs[:,1])
pt_pred_delta_2_pairs = np.array(muon_pairs_dataset['pt_r'] - Y_out_delta_pairs[:,2])


# In[ ]:


phi_even = phi_pred_delta_2_pairs[even_indices]
#phi_even = np.delete(phi_even, indices_to_exclude)

phi_odd = phi_pred_delta_2_pairs[odd_indices]
#phi_odd = np.delete(phi_odd, indices_to_exclude)


eta_even = eta_pred_delta_2_pairs[even_indices]
#eta_even = np.delete(eta_even, indices_to_exclude)

eta_odd = eta_pred_delta_2_pairs[odd_indices]
#eta_odd = np.delete(eta_odd, indices_to_exclude)


pt_even = pt_pred_delta_2_pairs[even_indices]
#pt_even = np.delete(pt_even, indices_to_exclude)

pt_odd = pt_pred_delta_2_pairs[odd_indices]
#pt_odd = np.delete(pt_odd, indices_to_exclude)

pt_vectors_a_even = get_pt_vectors(phi_even, eta_even, pt_even)
pt_vectors_b_odd = get_pt_vectors(phi_odd, eta_odd, pt_odd)

E_p, p_p = get_e_p(pt_vectors_a_even, pt_vectors_b_odd)
masses_pred = get_mass(E_p, p_p)


# In[ ]:


if has_true:
    print(np.mean(masses_true))
print(np.mean(masses_reco))
print(np.mean(masses_ext))
print(np.mean(masses_pred))

if has_true:
    print(np.std(masses_true))
print(np.std(masses_reco))
print(np.std(masses_ext))
print(np.std(masses_pred))


# In[ ]:


if has_true:
    range_val = 10
    nb_diff = 100

    mass_residual_reco = np.array(masses_reco) - np.array(masses_true)
    mass_residual_ext = np.array(masses_ext) - np.array(masses_true)
    mass_residual_pred = np.array(masses_pred) - np.array(masses_true)

    #plt.hist(mass_residual_reco, nb_diff, histtype='step', fill=False, density=True, range=(-range_val,range_val))
    plt.hist(mass_residual_ext, nb_diff, histtype='step', fill=False, density=True, range=(-range_val,range_val))
    plt.hist(mass_residual_pred, nb_diff, histtype='step', fill=False, density=True, range=(-range_val,range_val))
    plt.title('Muon Mass Regression Results')
    plt.legend(['raw reconstruction','extrapolated','NN regression'])
    plt.xlabel('$M^{\mu\mu}_{RECO}$ - $M^{\mu\mu}_{GEN}$ (GeV)')
    plt.ylabel('Probability')
    plt.show()

    print("mass_residual_reco: %f +/- %f" % (np.mean(mass_residual_reco), np.std(mass_residual_reco)))
    print("mass_residual_ext: %f +/- %f" % (np.mean(mass_residual_ext), np.std(mass_residual_ext)))
    print("mass_residual_pred: %f +/- %f" % (np.mean(mass_residual_pred), np.std(mass_residual_pred)))


# In[ ]:


nb = 100
rng = 20
#plt.hist(np.array(masses_true), nb, range=(0.,rng), density=True, histtype='step', fill=False)
#plt.hist(np.array(masses_reco), nb, range=(0.,rng), density=True, histtype='step', fill=False)
plt.hist(np.array(masses_ext), nb, range=(0.,rng), density=True, histtype='step', fill=False)
plt.hist(np.array(masses_pred), nb, range=(0.,rng), density=True, histtype='step', fill=False)
plt.title('Muon Mass Regression Results')
plt.legend(['extrapolated','NN regression'])
plt.xlabel('$M^{\mu\mu}$')
plt.ylabel('Probability')
plt.show()


# In[ ]:


hist_values_mass_pred = np.histogram(np.array(masses_pred), nb, range=(0, rng), density=True)

plt.plot(hist_values_mass_pred[1][:-1], hist_values_mass_pred[0])
plt.show()
print(max(pt_even))
print(max(pt_odd))
local_max_indices = argrelextrema(hist_values_mass_pred[0], np.greater)
print('Local maxima at:')
for i in local_max_indices[0]:
    print('[%.2f - %.2f] : %.2f %%' %
          (hist_values_mass_pred[1][i], hist_values_mass_pred[1][i+1], 100*hist_values_mass_pred[0][i]))


# In[ ]:


for lay in model.layers:
    print(lay.__class__.__name__)
    for weight in lay.get_weights():
        w = weight.flatten()
        print(np.min(w), np.max(w), np.mean(w), np.std(w))


# In[ ]:


all_layers_outs = [[] for i in range(len(model.layers) - 1)]

for i in range(1, len(model.layers)):
    #print(i)
    new_out = model.layers[i].output
    #print(model.layers[i].weights)
    for weight in model.layers[i].get_weights():
        #plt.hist(weight.flatten(), 10)
        #plt.show()
        if model.layers[i].__class__.__name__ == 'Dense':
            if i >= len(model.layers) - 3:
                print('min:', np.min(weight), 'max:', np.max(weight), 'mean:',                   np.mean(weight),'std:', np.std(weight))
                #plt.hist(weight.flatten(), 100)
                #plt.show()
                continue
            next_layer_weights = model.layers[i + 1].get_weights()
            #print('fused')
            fused = None
            if len(weight.shape) == 2:
                fused = next_layer_weights[0] * weight / np.sqrt(next_layer_weights[3] + 0.001)
                print('min:', np.min(fused), 'max:', np.max(fused), 'mean:',                   np.mean(fused),'std:', np.std(fused))
            else:
                fused = next_layer_weights[0] * (weight - next_layer_weights[2]) /                 np.sqrt(next_layer_weights[3]+ 0.001) + next_layer_weights[1]
                print('min:', np.min(fused), 'max:', np.max(fused), 'mean:',                   np.mean(fused),'std:', np.std(fused))
            #plt.hist(fused.flatten(), 100)
            #plt.show()
        elif model.layers[i].__class__.__name__ != 'BatchNormalization':
            print('min:', np.min(weight), 'max:', np.max(weight), 'mean:',               np.mean(weight),'std:', np.std(weight))     

    int_model = Model(inputs=model.input, outputs=new_out)
    int_model.compile(loss='mse',optimizer=keras.optimizers.Adam())
    
    cur_input_res_int = int_model.predict(X_test, batch_size=8192)
    all_layers_outs[i - 1].append(cur_input_res_int)


# In[ ]:


all_layers_outs = [[] for i in range(len(model.layers) - 1)]
print(X_test.shape)
for i in range(1, len(model.layers)):
    #print(i)
    new_out = model.layers[i].output
    #print(model.layers[i].weights)
    for weight in model.layers[i].get_weights():
        #print(weight.shape)
        pass

    int_model = Model(inputs=model.input, outputs=new_out)
    int_model.compile(loss='mse',optimizer=keras.optimizers.Adam())
    
    cur_input_res_int = int_model.predict(X_test, batch_size=8192)
    all_layers_outs[i - 1].append(cur_input_res_int)


# In[ ]:


print(len(all_layers_outs))
for i, data in enumerate(all_layers_outs):
    data = np.array(data).flatten()
    print(model.layers[i + 1].name, data.shape)
    print('min:', np.min(data),'max:', np.max(data),'mean:', np.mean(data),'std:', np.std(data))
    if model.layers[i + 1].name.startswith('relu'):
        print(np.count_nonzero(data) / len(data) * 100)
    plt.hist(data, 100, histtype='step', density=True)
    plt.legend(['output of ' + model.layers[i + 1].name])
    plt.show()


# In[ ]:




